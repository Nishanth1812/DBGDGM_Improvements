"""
Visualization and analysis utilities for MM-DBGDGM inference results.
Includes confusion matrix plots, attention heatmaps, latent visualization, etc.
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class ResultsVisualizer:
    """
    Visualize model predictions, attention weights, latent representations, etc.
    """
    
    CLASS_NAMES = {0: 'CN', 1: 'eMCI', 2: 'lMCI', 3: 'AD'}
    CLASS_COLORS = {0: '#2E86AB', 1: '#A23B72', 2: '#F18F01', 3: '#C73E1D'}
    
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        output_path: str = 'confusion_matrix.png',
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            confusion_matrix: [4, 4] confusion matrix
            output_path: Path to save figure
            figsize: Figure size
        """
        class_names = ['CN', 'eMCI', 'lMCI', 'AD']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = ax.text(j, i, confusion_matrix[i, j],
                              ha='center', va='center', color='white' if confusion_matrix[i, j] > confusion_matrix.max() / 2 else 'black',
                              fontsize=12, fontweight='bold')
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix plot saved to {output_path}")
    
    @staticmethod
    def plot_class_distribution(
        labels: np.ndarray,
        predictions: np.ndarray,
        output_path: str = 'class_distribution.png',
        figsize: Tuple[int, int] = (10, 5)
    ) -> None:
        """
        Plot distribution of true vs predicted classes.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            output_path: Path to save figure
            figsize: Figure size
        """
        class_names = ['CN', 'eMCI', 'lMCI', 'AD']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # True distribution
        true_counts = np.bincount(labels, minlength=4)
        axes[0].bar(class_names, true_counts, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        axes[0].set_title('True Class Distribution')
        axes[0].set_ylabel('Count')
        
        # Predicted distribution
        pred_counts = np.bincount(predictions, minlength=4)
        axes[1].bar(class_names, pred_counts, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        axes[1].set_title('Predicted Class Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class distribution plot saved to {output_path}")
    
    @staticmethod
    def plot_confidence_scores(
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: Optional[np.ndarray] = None,
        output_path: str = 'confidence_scores.png',
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Plot confidence score distributions.
        
        Args:
            predictions: Predicted labels
            probabilities: Prediction probabilities [N, 4]
            labels: True labels (optional)
            output_path: Path to save figure
            figsize: Figure size
        """
        class_names = ['CN', 'eMCI', 'lMCI', 'AD']
        
        # Extract confidence scores (max probability for each prediction)
        confidences = probabilities[np.arange(len(predictions)), predictions]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Overall confidence distribution
        axes[0].hist(confidences, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Score Distribution')
        axes[0].legend()
        
        # Per-class confidence
        if labels is not None:
            correct_mask = predictions == labels
            axes[1].hist(confidences[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
            axes[1].hist(confidences[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='red')
            axes[1].set_xlabel('Confidence Score')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Confidence by Correctness')
            axes[1].legend()
        else:
            for i, class_name in enumerate(class_names):
                class_mask = predictions == i
                axes[1].hist(confidences[class_mask], bins=20, alpha=0.5, label=class_name)
            axes[1].set_xlabel('Confidence Score')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Confidence by Predicted Class')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confidence scores plot saved to {output_path}")
    
    @staticmethod
    def plot_latent_space_2d(
        latents: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        method: str = 'tsne',
        output_dir: str = './',
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot 2D projection of latent space using t-SNE or PCA.
        
        Args:
            latents: Latent representations [N, 256]
            labels: True labels
            predictions: Predicted labels
            method: 't-sne' or 'pca'
            output_dir: Directory to save figures
            figsize: Figure size
        """
        class_names = ['CN', 'eMCI', 'lMCI', 'AD']
        colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        print(f"Computing {method.upper()} projection...")
        
        if method.lower() == 'tsne':
            projector = TSNE(n_components=2, random_state=42, n_jobs=-1, verbose=1)
        elif method.lower() == 'pca':
            projector = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        latents_2d = projector.fit_transform(latents)
        
        # Plot: True labels
        fig, ax = plt.subplots(figsize=figsize)
        for i, class_name in enumerate(class_names):
            mask = labels == i
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                      c=colors_list[i], label=class_name, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Latent Space {method.upper()} - True Labels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        true_labels_path = os.path.join(output_dir, f'latent_space_{method}_true_labels.png')
        plt.savefig(true_labels_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Latent space plot (true labels) saved to {true_labels_path}")
        
        # Plot: Predicted labels
        fig, ax = plt.subplots(figsize=figsize)
        for i, class_name in enumerate(class_names):
            mask = predictions == i
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                      c=colors_list[i], label=class_name, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Latent Space {method.upper()} - Predicted Labels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        pred_labels_path = os.path.join(output_dir, f'latent_space_{method}_pred_labels.png')
        plt.savefig(pred_labels_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Latent space plot (predicted labels) saved to {pred_labels_path}")
        
        # Plot: Correctness
        fig, ax = plt.subplots(figsize=figsize)
        correct_mask = predictions == labels
        ax.scatter(latents_2d[correct_mask, 0], latents_2d[correct_mask, 1],
                  c='green', label='Correct', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.scatter(latents_2d[~correct_mask, 0], latents_2d[~correct_mask, 1],
                  c='red', label='Incorrect', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Latent Space {method.upper()} - Correctness')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        correctness_path = os.path.join(output_dir, f'latent_space_{method}_correctness.png')
        plt.savefig(correctness_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Latent space plot (correctness) saved to {correctness_path}")
    
    @staticmethod
    def plot_per_class_metrics(
        metrics: Dict,
        output_path: str = 'per_class_metrics.png',
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot per-class performance metrics.
        
        Args:
            metrics: Dict with 'per_class' key containing metrics per class
            output_path: Path to save figure
            figsize: Figure size
        """
        class_names = list(metrics['per_class'].keys())
        metric_names = ['precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(1, len(metric_names), figsize=figsize, sharey=True)
        
        for ax_idx, metric_name in enumerate(metric_names):
            values = [metrics['per_class'][cn][metric_name] for cn in class_names]
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            
            axes[ax_idx].bar(class_names, values, color=colors)
            axes[ax_idx].set_ylim([0, 1.0])
            axes[ax_idx].set_ylabel('Score')
            axes[ax_idx].set_title(f'{metric_name.capitalize()}')
            axes[ax_idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (cn, val) in enumerate(zip(class_names, values)):
                axes[ax_idx].text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Per-class metrics plot saved to {output_path}")


def visualize_results(
    results_dir: str,
    latents_available: bool = False,
    output_dir: Optional[str] = None
) -> None:
    """
    Create all visualizations from saved evaluation results.
    
    Args:
        results_dir: Directory containing evaluation results
        latents_available: Whether latents were saved
        output_dir: Directory to save visualizations (defaults to results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("Loading results...")
    predictions = np.load(os.path.join(results_dir, 'predictions.npy'))
    labels = np.load(os.path.join(results_dir, 'labels.npy'))
    probabilities = np.load(os.path.join(results_dir, 'probabilities.npy'))
    
    # Load metrics
    import json
    with open(os.path.join(results_dir, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = ResultsVisualizer()
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    visualizer.plot_confusion_matrix(
        cm,
        output_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Class distribution
    visualizer.plot_class_distribution(
        labels, predictions,
        output_path=os.path.join(output_dir, 'class_distribution.png')
    )
    
    # Confidence scores
    visualizer.plot_confidence_scores(
        predictions, probabilities, labels,
        output_path=os.path.join(output_dir, 'confidence_scores.png')
    )
    
    # Per-class metrics
    visualizer.plot_per_class_metrics(
        metrics,
        output_path=os.path.join(output_dir, 'per_class_metrics.png')
    )
    
    # Latent space visualizations
    if latents_available:
        print("Creating latent space visualizations...")
        latents = np.load(os.path.join(results_dir, 'latents.npy'))
        
        visualizer.plot_latent_space_2d(
            latents, labels, predictions,
            method='pca',
            output_dir=output_dir
        )
        
        visualizer.plot_latent_space_2d(
            latents, labels, predictions,
            method='tsne',
            output_dir=output_dir
        )
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory with evaluation results')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (defaults to results_dir)')
    parser.add_argument('--latents', action='store_true', help='Latents were saved')
    
    args = parser.parse_args()
    
    visualize_results(
        results_dir=args.results_dir,
        latents_available=args.latents,
        output_dir=args.output_dir
    )
