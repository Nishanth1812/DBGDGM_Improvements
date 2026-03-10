"""Kaggle-facing DBGDGM pipeline utilities.

This module keeps the Kaggle workflow aligned with the canonical MM-DBGDGM
package instead of maintaining a second model implementation.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from MM_DBGDGM.models import MM_DBGDGM
from MM_DBGDGM.training import MM_DBGDGM_Loss, Trainer
from Preprocessing.src.utils.preprocessing_utils import extract_timeseries_windows


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
    learning_rate: float = 1e-3
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
        return {
            'fmri': torch.from_numpy(self.fmri[idx]),
            'smri': torch.from_numpy(self.smri[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


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


def convert_subject_timeseries_to_dbgdgm(
    fmri_subjects: np.ndarray,
    smri_subjects: np.ndarray,
    labels: np.ndarray,
    window_size: int = 50,
    window_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert subject-level [N, 200, T] arrays into DBGDGM windows."""
    fmri_subjects = np.asarray(fmri_subjects, dtype=np.float32)
    smri_subjects = np.asarray(smri_subjects, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if fmri_subjects.ndim != 3 or fmri_subjects.shape[1] != 200:
        raise ValueError("Subject-level fMRI input must have shape [N, 200, T]")
    if len(fmri_subjects) != len(smri_subjects) or len(fmri_subjects) != len(labels):
        raise ValueError("Subject-level fMRI, sMRI, and labels must have matching lengths")

    fmri_windows = []
    smri_windows = []
    label_windows = []

    for subject_idx in range(len(labels)):
        subject_windows = extract_timeseries_windows(
            fmri_subjects[subject_idx],
            window_size=window_size,
            window_step=window_step,
            standardize=True,
            dbgdgm_format=True,
        )
        fmri_windows.append(subject_windows)
        smri_windows.append(np.repeat(smri_subjects[subject_idx][np.newaxis, :], subject_windows.shape[0], axis=0))
        label_windows.append(np.full(subject_windows.shape[0], labels[subject_idx], dtype=np.int64))

    return (
        np.concatenate(fmri_windows, axis=0),
        np.concatenate(smri_windows, axis=0),
        np.concatenate(label_windows, axis=0),
    )


def load_kaggle_arrays(
    fmri_path: str,
    smri_path: str,
    labels_path: str,
    config: KaggleConfig,
    fmri_is_subject_timeseries: bool = False,
    window_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmri = np.load(fmri_path)
    smri = np.load(smri_path)
    labels = np.load(labels_path)

    if fmri_is_subject_timeseries:
        fmri, smri, labels = convert_subject_timeseries_to_dbgdgm(
            fmri,
            smri,
            labels,
            window_size=config.seq_len,
            window_step=window_step,
        )

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


def train_pipeline(loaders: Dict[str, DataLoader], config: KaggleConfig, output_dir: str) -> Tuple[MM_DBGDGM, Trainer]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    criterion = MM_DBGDGM_Loss(num_classes=config.num_classes).to(device)
    trainer = Trainer(model=model, criterion=criterion, device=device, output_dir=output_dir, seed=config.random_state)
    trainer.fit(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience,
        annealing_epochs=config.annealing_epochs,
    )
    return model, trainer


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
        batch_predictions, batch_probabilities = model.predict(fmri, smri)
        predictions.append(batch_predictions.cpu().numpy())
        probabilities.append(batch_probabilities.cpu().numpy())
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


@torch.no_grad()
def predict_arrays(model: MM_DBGDGM, fmri: np.ndarray, smri: np.ndarray, device: Optional[torch.device] = None) -> Dict[str, np.ndarray]:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    fmri_tensor = torch.as_tensor(fmri, dtype=torch.float32, device=device)
    smri_tensor = torch.as_tensor(smri, dtype=torch.float32, device=device)
    predictions, probabilities = model.predict(fmri_tensor, smri_tensor)
    latents = model.get_latent(fmri_tensor, smri_tensor)['z']
    return {
        'predictions': predictions.cpu().numpy(),
        'probabilities': probabilities.cpu().numpy(),
        'latents': latents.cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='DBGDGM Kaggle pipeline')
    parser.add_argument('--fmri', required=True, help='Path to fMRI .npy array')
    parser.add_argument('--smri', required=True, help='Path to sMRI .npy array')
    parser.add_argument('--labels', required=True, help='Path to labels .npy array')
    parser.add_argument('--output-dir', default='./kaggle_outputs', help='Output directory')
    parser.add_argument('--subject-timeseries', action='store_true', help='Interpret fMRI input as [N, 200, T] subject arrays')
    parser.add_argument('--window-step', type=int, default=1, help='Sliding window step when using subject timeseries')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = KaggleConfig()
    fmri, smri, labels = load_kaggle_arrays(
        args.fmri,
        args.smri,
        args.labels,
        config,
        fmri_is_subject_timeseries=args.subject_timeseries,
        window_step=args.window_step,
    )
    loaders = create_split_loaders(fmri, smri, labels, config)
    model, _ = train_pipeline(loaders, config, str(output_dir))
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