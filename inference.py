"""
Inference script for MM-DBGDGM model.
Loads trained model checkpoint and performs prediction on new data.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from MM_DBGDGM.models import MM_DBGDGM
from MM_DBGDGM.data import MultimodalBrainDataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Predictor:
    """
    Wrapper class for model inference.
    Handles checkpoint loading, prediction, and result formatting.
    """

    CLASS_NAMES = {0: 'CN', 1: 'eMCI', 2: 'lMCI', 3: 'AD'}
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        config_path: Optional[str] = None
    ):
        """
        Initialize predictor with trained model.
        
        Args:
            checkpoint_path: Path to saved model checkpoint (.pt file)
            device: Device to load model on ('cuda', 'cpu', or 'auto')
            config_path: Path to training config (optional, for model architecture)
        """
        self.checkpoint_path = checkpoint_path
        self.device = self._get_device(device)
        self.config = self._load_config(config_path) if config_path else None
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        logger.info(f"Using device: {self.device}")
        
        self.model = self._load_model()
        self.model.eval()
    
    @staticmethod
    def _get_device(device: str) -> str:
        """Determine compute device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self) -> MM_DBGDGM:
        """Load model architecture and checkpoint weights."""
        # Default model configuration
        model_config = {
            'n_roi': 200,
            'seq_len': 50,
            'n_smri_features': 5,
            'latent_dim': 256,
            'num_classes': 4,
            'use_gat_encoder': True,
            'use_attention_fusion': True,
            'dropout': 0.3,
        }
        
        # Override with config if provided
        if self.config and 'model' in self.config:
            model_config.update(self.config['model'])
        
        # Create model
        model = MM_DBGDGM(**model_config)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle both state dict and full checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        logger.info(f"Model loaded successfully. Total parameters: {self._count_parameters(model):,}")
        
        return model
    
    @staticmethod
    def _count_parameters(model: torch.nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def predict(
        self,
        fmri: np.ndarray,
        smri: np.ndarray,
        return_confidence: bool = True,
        return_latent: bool = False
    ) -> Dict:
        """
        Predict disease classification for single sample.
        
        Args:
            fmri: fMRI data, shape [200, 50]
            smri: sMRI features, shape [5]
            return_confidence: Return confidence scores
            return_latent: Return latent representation
        
        Returns:
            Dict with keys:
                - prediction: int (0-3)
                - class_name: str ('CN', 'eMCI', 'lMCI', 'AD')
                - confidence: float (0-1, if return_confidence=True)
                - probabilities: np.ndarray [4], if return_confidence=True
                - latent: np.ndarray [256], if return_latent=True
        """
        # Prepare inputs
        fmri_t = torch.FloatTensor(fmri[np.newaxis, :, :]).to(self.device)  # [1, 200, 50]
        smri_t = torch.FloatTensor(smri[np.newaxis, :]).to(self.device)     # [1, 5]
        
        with torch.no_grad():
            # Get predictions
            pred_class, probs = self.model.predict(fmri_t, smri_t)
            pred_class = pred_class.cpu().numpy()[0]
            probs = probs.cpu().numpy()[0]
            
            # Optionally get latent representation
            latent = None
            if return_latent:
                latent = self.model.get_latent(fmri_t, smri_t)['z'].cpu().numpy()[0]
        
        result = {
            'prediction': int(pred_class),
            'class_name': self.CLASS_NAMES[int(pred_class)],
        }
        
        if return_confidence:
            result['confidence'] = float(probs[pred_class])
            result['probabilities'] = probs
        
        if return_latent:
            result['latent'] = latent
        
        return result
    
    def predict_batch(
        self,
        fmri_batch: np.ndarray,
        smri_batch: np.ndarray,
        return_confidence: bool = True,
        return_latent: bool = False
    ) -> Dict:
        """
        Predict disease classification for batch of samples.
        
        Args:
            fmri_batch: Batch of fMRI data, shape [batch_size, 200, 50]
            smri_batch: Batch of sMRI features, shape [batch_size, 5]
            return_confidence: Return confidence scores
            return_latent: Return latent representations
        
        Returns:
            Dict with keys:
                - predictions: np.ndarray [batch_size]
                - class_names: List[str]
                - confidences: np.ndarray [batch_size], if return_confidence=True
                - probabilities: np.ndarray [batch_size, 4], if return_confidence=True
                - latents: np.ndarray [batch_size, 256], if return_latent=True
        """
        # Prepare batch inputs
        fmri_t = torch.FloatTensor(fmri_batch).to(self.device)  # [batch, 200, 50]
        smri_t = torch.FloatTensor(smri_batch).to(self.device)  # [batch, 5]
        
        with torch.no_grad():
            pred_classes, probs = self.model.predict(fmri_t, smri_t)
            pred_classes = pred_classes.cpu().numpy()
            probs = probs.cpu().numpy()  # [batch_size, 4]
            
            # Optionally get latents
            latents = None
            if return_latent:
                latents = self.model.get_latent(fmri_t, smri_t)['z'].cpu().numpy()
        
        # Convert class indices to names
        class_names = [self.CLASS_NAMES[int(p)] for p in pred_classes]
        
        result = {
            'predictions': pred_classes,
            'class_names': class_names,
        }
        
        if return_confidence:
            confidences = probs[np.arange(len(pred_classes)), pred_classes.astype(int)]
            result['confidences'] = confidences
            result['probabilities'] = probs
        
        if return_latent:
            result['latents'] = latents
        
        return result
    
    def predict_from_dataset(
        self,
        dataset: MultimodalBrainDataset,
        batch_size: int = 32,
        return_confidence: bool = True,
        return_latent: bool = False
    ) -> Dict:
        """
        Predict on entire dataset using DataLoader.
        
        Args:
            dataset: MultimodalBrainDataset instance
            batch_size: Batch size for prediction
            return_confidence: Return confidence scores
            return_latent: Return latent representations
        
        Returns:
            Dict with aggregated predictions and metrics
        """
        from torch.utils.data import DataLoader

        if len(dataset) == 0:
            raise ValueError("Cannot run inference on an empty dataset")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        all_predictions = []
        all_confidences = []
        all_probabilities = []
        all_latents = []
        all_labels = []
        
        logger.info(f"Running inference on {len(dataset)} samples...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):
            fmri = batch['fmri'].to(self.device)
            smri = batch['smri'].to(self.device)
            labels = batch['label'].cpu().numpy().flatten()
            
            with torch.no_grad():
                pred_classes, probs = self.model.predict(fmri, smri)
                pred_classes = pred_classes.cpu().numpy()
                probs = probs.cpu().numpy()
                
                if return_latent:
                    latents = self.model.get_latent(fmri, smri)['z'].cpu().numpy()
                    all_latents.append(latents)
            
            all_predictions.append(pred_classes)
            all_probabilities.append(probs)
            
            if return_confidence:
                confidences = probs[np.arange(len(pred_classes)), pred_classes.astype(int)]
                all_confidences.append(confidences)
            
            all_labels.append(labels)
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        probabilities = np.concatenate(all_probabilities, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        result = {
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities,
            'class_names': [self.CLASS_NAMES[int(p)] for p in predictions],
        }
        
        if return_confidence:
            confidences = np.concatenate(all_confidences, axis=0)
            result['confidences'] = confidences
        
        if return_latent:
            latents = np.concatenate(all_latents, axis=0)
            result['latents'] = latents
        
        return result


def main():
    """Main inference CLI."""
    parser = argparse.ArgumentParser(
        description='Inference script for MM-DBGDGM model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to training config (optional)'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Root directory of preprocessed data'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata CSV file (subject_id, timepoint, label)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cuda', 'cpu', 'auto'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./inference_results',
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--save-latents',
        action='store_true',
        help='Save latent representations'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        config_path=args.config
    )
    
    # Create dataset
    logger.info(f"Loading dataset from {args.dataset_root}")
    dataset = MultimodalBrainDataset(
        dataset_root=args.dataset_root,
        metadata_file=args.metadata,
        normalize_fmri=True,
        normalize_smri=True
    )
    logger.info(f"Dataset contains {len(dataset)} samples")
    
    # Run inference
    results = predictor.predict_from_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        return_confidence=True,
        return_latent=args.save_latents
    )
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.npy')
    np.save(predictions_path, results['predictions'])
    logger.info(f"Predictions saved to {predictions_path}")
    
    probabilities_path = os.path.join(args.output_dir, 'probabilities.npy')
    np.save(probabilities_path, results['probabilities'])
    logger.info(f"Probabilities saved to {probabilities_path}")
    
    if args.save_latents and 'latents' in results:
        latents_path = os.path.join(args.output_dir, 'latents.npy')
        np.save(latents_path, results['latents'])
        logger.info(f"Latents saved to {latents_path}")
    
    # Save summary JSON
    summary = {
        'total_samples': len(results['predictions']),
        'predictions_saved': predictions_path,
        'probabilities_saved': probabilities_path,
        'class_names': results['class_names'][:5] + (['...'] if len(results['class_names']) > 5 else []),
    }
    
    if 'confidences' in results:
        summary['mean_confidence'] = float(np.mean(results['confidences']))
        summary['min_confidence'] = float(np.min(results['confidences']))
        summary['max_confidence'] = float(np.max(results['confidences']))
    
    summary_path = os.path.join(args.output_dir, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")
    
    logger.info("Inference complete!")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
