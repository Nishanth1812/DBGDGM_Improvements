"""
Evaluation script for MM-DBGDGM model.
Computes comprehensive metrics including accuracy, F1, precision, recall, ROC-AUC, and confusion matrix.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report
)
from tqdm import tqdm

from inference import Predictor
from MM_DBGDGM.data import MultimodalBrainDataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate model performance on test set.
    Computes metrics, confusion matrix, and per-class statistics.
    """
    
    CLASS_NAMES = {0: 'CN', 1: 'eMCI', 2: 'lMCI', 3: 'AD'}
    
    def __init__(self, predictor: Predictor):
        """
        Initialize evaluator.
        
        Args:
            predictor: Predictor instance with loaded model
        """
        self.predictor = predictor
    
    def evaluate(
        self,
        dataset: MultimodalBrainDataset,
        batch_size: int = 32,
        return_latent: bool = False
    ) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: MultimodalBrainDataset instance
            batch_size: Batch size for inference
            return_latent: Return latent representations
        
        Returns:
            Dict with comprehensive metrics
        """
        logger.info(f"Evaluating on {len(dataset)} samples...")
        
        # Run inference
        results = self.predictor.predict_from_dataset(
            dataset=dataset,
            batch_size=batch_size,
            return_confidence=True,
            return_latent=return_latent
        )
        
        predictions = results['predictions']
        labels = results['labels']
        probabilities = results['probabilities']
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, labels, probabilities)
        
        # Add raw predictions
        metrics['predictions'] = predictions
        metrics['labels'] = labels
        metrics['probabilities'] = probabilities
        
        if return_latent and 'latents' in results:
            metrics['latents'] = results['latents']
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """Compute all evaluation metrics."""
        metrics = {}
        
        # Overall accuracy
        accuracy = accuracy_score(labels, predictions)
        metrics['accuracy'] = float(accuracy)
        
        # Macro and weighted averages
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        metrics['f1_macro'] = float(f1_macro)
        metrics['f1_weighted'] = float(f1_weighted)
        metrics['precision_macro'] = float(precision_macro)
        metrics['precision_weighted'] = float(precision_weighted)
        metrics['recall_macro'] = float(recall_macro)
        metrics['recall_weighted'] = float(recall_weighted)
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx in range(4):
            class_name = self.CLASS_NAMES[class_idx]
            class_mask = labels == class_idx
            
            if np.sum(class_mask) == 0:
                logger.warning(f"Class {class_name} not present in labels")
                continue
            
            class_labels = labels[class_mask]
            class_preds = predictions[class_mask]
            
            per_class_metrics[class_name] = {
                'count': int(np.sum(class_mask)),
                'accuracy': float(np.mean(class_preds == class_labels)),
                'precision': float(precision_recall_fscore_support(
                    labels,
                    predictions,
                    labels=[class_idx],
                    average=None,
                    zero_division=0,
                )[0][0]),
                'recall': float(precision_recall_fscore_support(
                    labels,
                    predictions,
                    labels=[class_idx],
                    average=None,
                    zero_division=0,
                )[1][0]),
                'f1': float(precision_recall_fscore_support(
                    labels,
                    predictions,
                    labels=[class_idx],
                    average=None,
                    zero_division=0,
                )[2][0]),
            }
        
        metrics['per_class'] = per_class_metrics
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3])
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC-AUC (one-vs-rest for multiclass)
        try:
            roc_auc_macro = roc_auc_score(
                labels, probabilities,
                multi_class='ovr',
                average='macro'
            )
            roc_auc_weighted = roc_auc_score(
                labels, probabilities,
                multi_class='ovr',
                average='weighted'
            )
            metrics['roc_auc_macro'] = float(roc_auc_macro)
            metrics['roc_auc_weighted'] = float(roc_auc_weighted)
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
        
        # Confidence statistics
        confidences = probabilities[np.arange(len(predictions)), predictions.astype(int)]
        metrics['confidence_mean'] = float(np.mean(confidences))
        metrics['confidence_std'] = float(np.std(confidences))
        metrics['confidence_min'] = float(np.min(confidences))
        metrics['confidence_max'] = float(np.max(confidences))
        
        # Correct vs incorrect confidence
        correct_mask = predictions == labels
        if np.sum(correct_mask) > 0:
            metrics['correct_confidence_mean'] = float(np.mean(confidences[correct_mask]))
        if np.sum(~correct_mask) > 0:
            metrics['incorrect_confidence_mean'] = float(np.mean(confidences[~correct_mask]))
        
        return metrics
    
    def print_report(self, metrics: Dict) -> str:
        """Generate formatted evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        report.append("\nOVERALL METRICS:")
        report.append(f"  Accuracy:           {metrics['accuracy']:.4f}")
        report.append(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
        report.append(f"  F1 (weighted):      {metrics['f1_weighted']:.4f}")
        report.append(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
        report.append(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
        
        if 'roc_auc_macro' in metrics:
            report.append(f"  ROC-AUC (macro):    {metrics['roc_auc_macro']:.4f}")
        
        # Per-class metrics
        report.append("\nPER-CLASS METRICS:")
        for class_name, class_metrics in metrics['per_class'].items():
            report.append(f"\n  {class_name}:")
            report.append(f"    Count:       {class_metrics['count']}")
            report.append(f"    Accuracy:    {class_metrics['accuracy']:.4f}")
            report.append(f"    Precision:   {class_metrics['precision']:.4f}")
            report.append(f"    Recall:      {class_metrics['recall']:.4f}")
            report.append(f"    F1:          {class_metrics['f1']:.4f}")
        
        # Confidence statistics
        report.append("\nCONFIDENCE STATISTICS:")
        report.append(f"  Mean:        {metrics['confidence_mean']:.4f}")
        report.append(f"  Std Dev:     {metrics['confidence_std']:.4f}")
        report.append(f"  Min:         {metrics['confidence_min']:.4f}")
        report.append(f"  Max:         {metrics['confidence_max']:.4f}")
        
        if 'correct_confidence_mean' in metrics:
            report.append(f"  Correct (mean):       {metrics['correct_confidence_mean']:.4f}")
        if 'incorrect_confidence_mean' in metrics:
            report.append(f"  Incorrect (mean):     {metrics['incorrect_confidence_mean']:.4f}")
        
        # Confusion matrix
        report.append("\nCONFUSION MATRIX:")
        cm = np.array(metrics['confusion_matrix'])
        class_names = ['CN', 'eMCI', 'lMCI', 'AD']
        
        report.append("         Predicted")
        report.append("         " + "  ".join([f"{name:>6}" for name in class_names]))
        for i, row in enumerate(cm):
            report.append("Actual " + f"{class_names[i]:>6} " + "  ".join([f"{val:>6}" for val in row]))
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def main():
    """Main evaluation CLI."""
    parser = argparse.ArgumentParser(
        description='Evaluation script for MM-DBGDGM model'
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
        help='Device to run evaluation on'
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
        default='./evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--save-latents',
        action='store_true',
        help='Save latent representations'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize predictor and evaluator
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        config_path=args.config
    )
    evaluator = Evaluator(predictor)
    
    # Create dataset
    logger.info(f"Loading test dataset from {args.dataset_root}")
    dataset = MultimodalBrainDataset(
        dataset_root=args.dataset_root,
        metadata_file=args.metadata,
        normalize_fmri=True,
        normalize_smri=True
    )
    logger.info(f"Test dataset contains {len(dataset)} samples")
    
    # Run evaluation
    metrics = evaluator.evaluate(
        dataset=dataset,
        batch_size=args.batch_size,
        return_latent=args.save_latents
    )
    
    # Print report
    report = evaluator.print_report(metrics)
    print(report)
    
    # Save report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    # Save detailed metrics as JSON
    metrics_clean = {k: v for k, v in metrics.items() 
                     if k not in ['predictions', 'labels', 'probabilities', 'latents']}
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save predictions and probabilities
    predictions_path = os.path.join(args.output_dir, 'predictions.npy')
    np.save(predictions_path, metrics['predictions'])
    logger.info(f"Predictions saved to {predictions_path}")
    
    probabilities_path = os.path.join(args.output_dir, 'probabilities.npy')
    np.save(probabilities_path, metrics['probabilities'])
    logger.info(f"Probabilities saved to {probabilities_path}")
    
    labels_path = os.path.join(args.output_dir, 'labels.npy')
    np.save(labels_path, metrics['labels'])
    logger.info(f"Labels saved to {labels_path}")
    
    if args.save_latents and 'latents' in metrics:
        latents_path = os.path.join(args.output_dir, 'latents.npy')
        np.save(latents_path, metrics['latents'])
        logger.info(f"Latents saved to {latents_path}")
    
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
