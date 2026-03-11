"""
Trainer class for MM-DBGDGM model.
Handles training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import json
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for MM-DBGDGM model with full training pipeline.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        output_dir: str = './outputs',
        seed: int = 42
    ):
        """
        Args:
            model: MM_DBGDGM model instance
            criterion: Loss function
            device: torch device
            output_dir: Directory to save checkpoints and logs
            seed: Random seed
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Training history
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized with device: {device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        beta_annealing: float = 1.0
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            optimizer: Optimizer
            epoch: Epoch number
            beta_annealing: KL annealing factor
        
        Returns:
            metrics dict with loss and accuracy
        """
        self.model.train()
        metrics = defaultdict(float)
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            fmri = batch['fmri'].to(self.device)
            smri = batch['smri'].to(self.device)
            targets = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(fmri, smri, return_all=True)
            
            # Compute loss
            loss_dict = self.criterion(
                logits=outputs['logits'],
                targets=targets,
                mu=outputs['mu'],
                logvar=outputs['logvar'],
                fmri_recon=outputs['fmri_recon'],
                fmri_orig=fmri,
                smri_recon=outputs['smri_recon'],
                smri_orig=smri,
                z_fmri=outputs['z_fmri'],
                z_smri=outputs['z_smri'],
                beta_annealing=beta_annealing
            )
            
            total_loss = loss_dict['total']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate metrics
            with torch.no_grad():
                for key in loss_dict:
                    if isinstance(loss_dict[key], torch.Tensor):
                        metrics[f'train_{key}'] += loss_dict[key].item()
                
                # Accuracy
                preds = outputs['predictions']
                acc = (preds == targets).float().mean()
                metrics['train_accuracy'] += acc.item()
            
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {total_loss.item():.4f}, "
                    f"Acc: {acc.item():.4f}"
                )

        if num_batches == 0:
            raise ValueError(
                "Training dataloader produced zero batches. Check dataset size, modality files, and drop_last settings."
            )
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return dict(metrics)
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
        beta_annealing: float = 1.0
    ) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation dataloader
            epoch: Epoch number
            beta_annealing: KL annealing factor
        
        Returns:
            metrics dict with loss and accuracy
        """
        self.model.eval()
        metrics = defaultdict(float)
        num_batches = 0
        
        all_preds = []
        all_targets = []
        
        for batch_idx, batch in enumerate(val_loader):
            # Move data to device
            fmri = batch['fmri'].to(self.device)
            smri = batch['smri'].to(self.device)
            targets = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(fmri, smri, return_all=True)
            
            # Compute loss
            loss_dict = self.criterion(
                logits=outputs['logits'],
                targets=targets,
                mu=outputs['mu'],
                logvar=outputs['logvar'],
                fmri_recon=outputs['fmri_recon'],
                fmri_orig=fmri,
                smri_recon=outputs['smri_recon'],
                smri_orig=smri,
                z_fmri=outputs['z_fmri'],
                z_smri=outputs['z_smri'],
                beta_annealing=beta_annealing
            )
            
            # Accumulate metrics
            for key in loss_dict:
                if isinstance(loss_dict[key], torch.Tensor):
                    metrics[f'val_{key}'] += loss_dict[key].item()
            
            # Accuracy
            preds = outputs['predictions']
            acc = (preds == targets).float().mean()
            metrics['val_accuracy'] += acc.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            num_batches += 1

        if num_batches == 0:
            raise ValueError(
                "Validation dataloader produced zero batches. Check dataset size and modality files."
            )
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return dict(metrics)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        annealing_epochs: int = 20
    ):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            annealing_epochs: Number of epochs for KL annealing
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        
        # Optimizer and scheduler
        optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # Training loop
        for epoch in range(num_epochs):
            # KL annealing
            if annealing_epochs <= 0:
                beta = 1.0
            else:
                beta = min(1.0, epoch / annealing_epochs)
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, epoch, beta_annealing=beta)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch, beta_annealing=beta)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_metrics['train_total']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.4f} | "
                f"Val Loss: {val_metrics['val_total']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
                f"β: {beta:.4f}"
            )
            
            # Store history
            for key, val in {**train_metrics, **val_metrics}.items():
                self.history[key].append(val)
            
            # Early stopping
            val_loss = val_metrics['val_total']
            val_acc = val_metrics['val_accuracy']
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, optimizer, 'best_loss')
                logger.info(f"✓ New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, optimizer, 'best_acc')
                logger.info(f"✓ New best validation accuracy: {val_acc:.4f}")
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping after {epoch} epochs")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, optimizer, f'epoch_{epoch}')
        
        logger.info("Training completed!")
        self._save_history()
    
    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        name: str = 'checkpoint'
    ):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"{name}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_history(self):
        """Save training history."""
        history_path = self.output_dir / 'training_history.json'
        
        # Convert to serializable format
        history_dict = {k: v for k, v in self.history.items()}
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Training history saved: {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}, Best val acc: {self.best_val_acc:.4f}")
