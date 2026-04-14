"""Kaggle-facing DBGDGM pipeline utilities.

This module keeps the Kaggle workflow aligned with the canonical MM-DBGDGM
package. It has been updated to support the Neurodegeneration Forecasting Module
and Weibull-AFT Time-to-Event pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from MM_DBGDGM.models.mm_dbgdgm import MM_DBGDGM
from MM_DBGDGM.training.losses import MM_DBGDGM_Loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class KaggleConfig:
    n_roi: int = 200
    seq_len: int = 50
    n_smri_features: int = 5
    latent_dim: int = 256
    num_classes: int = 4
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10
    annealing_epochs: int = 20
    num_workers: int = 0
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


class WindowedBrainDataset(Dataset):
    def __init__(self, fmri: np.ndarray, smri: np.ndarray, labels: np.ndarray):
        self.fmri = np.asarray(fmri, dtype=np.float32)
        self.smri = np.asarray(smri, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            'fmri': torch.from_numpy(self.fmri[idx]),
            'smri': torch.from_numpy(self.smri[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
        
        # MOCK ADNI BIOMARKERS AND SURVIVAL ENDPOINTS
        # Until arrays upload to Kaggle containing real target columns, we 
        # mock them to allow the forward/backward architecture pipeline to compile seamlessly.
        sample['hippo_vol'] = torch.randn(1) * 500 + 4000 
        sample['cortical_thinning'] = torch.rand(1) * 0.5 
        sample['dmn_conn'] = torch.rand(1) 
        sample['nss'] = torch.rand(1) * 100 
        sample['survival_times'] = torch.rand(3) * 10 + 0.1
        sample['survival_events'] = torch.randint(0, 2, (3,)).float()
        
        return sample


def validate_dbgdgm_arrays(fmri: np.ndarray, smri: np.ndarray, labels: np.ndarray, config: KaggleConfig) -> None:
    if fmri.ndim != 3:
        raise ValueError(f"Expected fMRI array with 3 dimensions, got shape {fmri.shape}")
    if fmri.shape[1:] != (config.n_roi, config.seq_len):
        raise ValueError(
            f"Expected DBGDGM fMRI windows shaped [N, {config.n_roi}, {config.seq_len}], got {fmri.shape}"
        )
    if smri.ndim != 2:
        raise ValueError(f"Expected sMRI array with 2 dimensions, got shape {smri.shape}")
    if smri.shape[1] != config.n_smri_features:
        raise ValueError(
            f"Expected {config.n_smri_features} sMRI features per sample, got {smri.shape[1]}"
        )
    if len(fmri) != len(smri) or len(fmri) != len(labels):
        raise ValueError("fMRI, sMRI, and labels arrays must have the same sample count")
    if not np.isfinite(fmri).all():
        raise ValueError("fMRI array contains NaN or inf values")
    if not np.isfinite(smri).all():
        raise ValueError("sMRI array contains NaN or inf values")


def load_kaggle_arrays(
    fmri_path: str,
    smri_path: str,
    labels_path: str,
    config: KaggleConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmri = np.load(fmri_path)
    smri = np.load(smri_path)
    labels = np.load(labels_path)
    validate_dbgdgm_arrays(fmri, smri, labels, config)
    return fmri.astype(np.float32), smri.astype(np.float32), labels.astype(np.int64)


def create_split_loaders(
    fmri: np.ndarray,
    smri: np.ndarray,
    labels: np.ndarray,
    config: KaggleConfig,
) -> Dict[str, DataLoader]:
    indices = np.arange(len(labels))
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices,
        labels,
        test_size=config.val_size + config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    temp_test_fraction = config.test_size / (config.val_size + config.test_size)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=temp_test_fraction,
        random_state=config.random_state,
        stratify=temp_labels,
    )

    datasets = {
        'train': WindowedBrainDataset(fmri[train_indices], smri[train_indices], labels[train_indices]),
        'val': WindowedBrainDataset(fmri[val_indices], smri[val_indices], labels[val_indices]),
        'test': WindowedBrainDataset(fmri[test_indices], smri[test_indices], labels[test_indices]),
    }

    return {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=name == 'train',
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=name == 'train',
        )
        for name, dataset in datasets.items()
    }


def build_model(config: KaggleConfig) -> MM_DBGDGM:
    return MM_DBGDGM(
        n_roi=config.n_roi,
        seq_len=config.seq_len,
        n_smri_features=config.n_smri_features,
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        use_gat_encoder=True,
        use_attention_fusion=True,
    )


def train_pipeline(loaders: Dict[str, DataLoader], config: KaggleConfig, output_dir: str) -> MM_DBGDGM:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    
    criterion = MM_DBGDGM_Loss(
        num_classes=config.num_classes,
        lambda_kl=0.1,
        lambda_align=0.1,
        lambda_recon=0.5,
        lambda_regression=0.5,
        lambda_survival=0.5
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        beta_annealing = min(1.0, epoch / config.annealing_epochs)
        
        # Training Phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(loaders['train'], desc=f"Epoch {epoch} [Train]"):
            fmri = batch['fmri'].to(device)
            smri = batch['smri'].to(device)
            labels = batch['label'].to(device)
            
            # Additional heads targets
            reg_targets = {
                'hippocampal_volume': batch['hippo_vol'].to(device),
                'cortical_thinning_rate': batch['cortical_thinning'].to(device),
                'dmn_connectivity': batch['dmn_conn'].to(device),
                'nss': batch['nss'].to(device)
            }
            surv_times = batch['survival_times'].to(device)
            surv_events = batch['survival_events'].to(device)

            optimizer.zero_grad()
            outputs = model(fmri, smri, return_all=True)
            losses = criterion(
                logits=outputs['logits'],
                targets=labels,
                mu=outputs['mu'],
                logvar=outputs['logvar'],
                fmri_recon=outputs['fmri_recon'],
                fmri_orig=fmri,
                smri_recon=outputs['smri_recon'],
                smri_orig=smri,
                z_fmri=outputs['z_fmri'],
                z_smri=outputs['z_smri'],
                degeneration_preds=outputs['degeneration'],
                regression_targets=reg_targets,
                survival_shape=outputs['survival']['shape'],
                survival_scale=outputs['survival']['scale'],
                survival_times=surv_times,
                survival_events=surv_events,
                beta_annealing=beta_annealing
            )
            
            total_loss = losses['total']
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            
        train_loss /= len(loaders['train'])
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in loaders['val']:
                fmri = batch['fmri'].to(device)
                smri = batch['smri'].to(device)
                labels = batch['label'].to(device)
                reg_targets = {k: v.to(device) for k, v in reg_targets.items()}
                surv_times = batch['survival_times'].to(device)
                surv_events = batch['survival_events'].to(device)

                outputs = model(fmri, smri, return_all=True)
                losses = criterion(
                    logits=outputs['logits'], targets=labels, mu=outputs['mu'], logvar=outputs['logvar'],
                    fmri_recon=outputs['fmri_recon'], fmri_orig=fmri, smri_recon=outputs['smri_recon'], 
                    smri_orig=smri, z_fmri=outputs['z_fmri'], z_smri=outputs['z_smri'],
                    degeneration_preds=outputs['degeneration'], regression_targets=reg_targets,
                    survival_shape=outputs['survival']['shape'], survival_scale=outputs['survival']['scale'],
                    survival_times=surv_times, survival_events=surv_events, beta_annealing=1.0
                )
                val_loss += losses['total'].item()
                
        val_loss /= len(loaders['val'])
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), Path(output_dir) / 'best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.patience:
                logger.info("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(Path(output_dir) / 'best_model.pth'))
    return model


@torch.no_grad()
def evaluate_split(model: MM_DBGDGM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    predictions = []
    probabilities = []
    labels = []

    for batch in loader:
        fmri = batch['fmri'].to(device)
        smri = batch['smri'].to(device)
        batch_labels = batch['label'].cpu().numpy()
        
        # Leveraging the deep clinical predict output wrapper
        # Since it returns a list of dictionaries per batch
        batch_reports = model.predict(fmri, smri)
        
        batch_preds = np.array([r['current_stage_prediction'] for r in batch_reports])
        batch_probs = np.array([r['stage_probabilities'] for r in batch_reports])
        
        predictions.append(batch_preds)
        probabilities.append(batch_probs)
        labels.append(batch_labels)

    predictions_array = np.concatenate(predictions, axis=0)
    probabilities_array = np.concatenate(probabilities, axis=0)
    labels_array = np.concatenate(labels, axis=0)

    metrics = {
        'accuracy': float(accuracy_score(labels_array, predictions_array)),
        'f1_macro': float(f1_score(labels_array, predictions_array, average='macro', zero_division=0)),
        'precision_macro': float(precision_score(labels_array, predictions_array, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(labels_array, predictions_array, average='macro', zero_division=0)),
    }
    try:
        metrics['roc_auc_macro'] = float(
            roc_auc_score(labels_array, probabilities_array, multi_class='ovr', average='macro')
        )
    except ValueError:
        metrics['roc_auc_macro'] = 0.0

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='DBGDGM Kaggle pipeline')
    parser.add_argument('--fmri', required=True, help='Path to fMRI .npy array')
    parser.add_argument('--smri', required=True, help='Path to sMRI .npy array')
    parser.add_argument('--labels', required=True, help='Path to labels .npy array')
    parser.add_argument('--output-dir', default='./kaggle_outputs', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = KaggleConfig()
    fmri, smri, labels = load_kaggle_arrays(
        args.fmri,
        args.smri,
        args.labels,
        config
    )
    
    loaders = create_split_loaders(fmri, smri, labels, config)
    model = train_pipeline(loaders, config, str(output_dir))
    device = next(model.parameters()).device
    metrics = evaluate_split(model, loaders['test'], device)

    with open(output_dir / 'kaggle_config.json', 'w', encoding='utf-8') as handle:
        json.dump(asdict(config), handle, indent=2)
    with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    logger.info('Saved Kaggle pipeline outputs to %s', output_dir)
    logger.info('Test metrics: %s', metrics)


if __name__ == '__main__':
    main()