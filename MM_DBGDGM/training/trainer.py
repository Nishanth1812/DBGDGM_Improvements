"""
Trainer class for MM-DBGDGM model.
Handles training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import numpy as np
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Set
import logging
import json
from datetime import datetime
from collections import defaultdict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

logger = logging.getLogger("mm-dbgdgm-modal")


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
        seed: int = 42,
        frozen_module_names: Optional[List[str]] = None,
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

        # Keep the criterion on the same device as the model/data.
        self.criterion = criterion.to(device)
        self.num_classes = getattr(model, 'num_classes', 4)
        self.class_names = ['CN', 'eMCI', 'lMCI', 'AD'][:self.num_classes]
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Training history
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.frozen_module_names: Set[str] = set(frozen_module_names or [])
        self._progress_lock = threading.Lock()
        self._progress_state = {
            'phase': 'idle',
            'epoch': None,
            'batch': None,
            'total_batches': None,
            'details': None,
            'updated_at': time.perf_counter(),
        }
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        logger.info(f"Trainer initialized with device: {device}")
        logger.info(f"Output directory: {self.output_dir}")
        if self.frozen_module_names:
            logger.info(f"Frozen modules (eval mode during train): {sorted(self.frozen_module_names)}")

    def _gpu_memory_summary(self) -> str:
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            return ''

        try:
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            return f" | GPU mem alloc {allocated:.2f} GB | reserved {reserved:.2f} GB | peak {peak:.2f} GB"
        except Exception:
            return ''

    def _prepare_batch(self, batch):
        """Move a batch to the target device and build target dictionaries."""
        fmri = batch['fmri'].to(self.device)
        smri = batch['smri'].to(self.device)
        targets = batch['label'].to(self.device)

        regression_targets = {
            'hippocampal_volume': batch['hippo_vol'].to(self.device),
            'cortical_thinning_rate': batch['cortical_thinning'].to(self.device),
            'dmn_connectivity': batch['dmn_conn'].to(self.device),
            'nss': batch['nss'].to(self.device)
        }

        survival_times = batch['survival_times'].to(self.device)
        survival_events = batch['survival_events'].to(self.device)

        return fmri, smri, targets, regression_targets, survival_times, survival_events

    def _compute_loss_dict(
        self,
        outputs,
        fmri: torch.Tensor,
        smri: torch.Tensor,
        targets: torch.Tensor,
        regression_targets: Dict[str, torch.Tensor],
        survival_times: torch.Tensor,
        survival_events: torch.Tensor,
        beta_annealing: float
    ) -> Dict[str, torch.Tensor]:
        """Compute the full multimodal loss for a batch."""
        return self.criterion(
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
            degeneration_preds=outputs['degeneration'],
            regression_targets=regression_targets,
            survival_shape=outputs['survival']['shape'],
            survival_scale=outputs['survival']['scale'],
            survival_times=survival_times,
            survival_events=survival_events,
            beta_annealing=beta_annealing
        )

    def _set_progress(
        self,
        phase: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        total_batches: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        with self._progress_lock:
            self._progress_state = {
                'phase': phase,
                'epoch': epoch,
                'batch': batch,
                'total_batches': total_batches,
                'details': details,
                'updated_at': time.perf_counter(),
            }

    def _clear_progress(self) -> None:
        self._set_progress('idle')

    def _start_heartbeat(self, interval_seconds: float = 60.0) -> None:
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_stop.clear()

        def _heartbeat_loop() -> None:
            while not self._heartbeat_stop.wait(interval_seconds):
                with self._progress_lock:
                    state = dict(self._progress_state)

                if state['phase'] == 'idle':
                    continue

                elapsed = time.perf_counter() - state['updated_at']
                batch = state['batch']
                total_batches = state['total_batches']
                batch_text = f"{batch}/{total_batches}" if batch is not None and total_batches is not None else 'n/a'
                details = state['details'] or 'working'
                logger.info(
                    f"Heartbeat: still working after {elapsed:.0f}s | phase={state['phase']} | "
                    f"epoch={state['epoch']} | batch={batch_text} | {details}"
                )

        self._heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name='mm-dbgdgm-trainer-heartbeat',
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        self._heartbeat_thread = None
        self._clear_progress()

    def _enforce_frozen_modules(self) -> None:
        """Keep frozen modules in eval mode after global model.train() calls."""
        if not self.frozen_module_names:
            return

        for module_name in sorted(self.frozen_module_names):
            module = getattr(self.model, module_name, None)
            if module is None:
                logger.warning(f"Configured frozen module '{module_name}' was not found on model")
                continue
            module.eval()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        beta_annealing: float = 1.0,
        log_every_n_batches: int = 10,
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
        self._enforce_frozen_modules()
        metrics = defaultdict(float)
        num_batches = 0
        epoch_number = epoch + 1
        total_batches = len(train_loader)

        logger.info(f"Epoch {epoch_number}: training started with {total_batches} batches")
        
        for batch_idx, batch in enumerate(train_loader):
            batch_number = batch_idx + 1
            batch_start = time.perf_counter()

            self._set_progress(
                phase='train_prepare',
                epoch=epoch_number,
                batch=batch_number,
                total_batches=total_batches,
                details='moving batch to device',
            )
            fmri, smri, targets, regression_targets, survival_times, survival_events = self._prepare_batch(batch)
            prepare_done = time.perf_counter()
            
            # Forward pass
            self._set_progress(
                phase='train_forward',
                epoch=epoch_number,
                batch=batch_number,
                total_batches=total_batches,
                details='running forward pass',
            )
            outputs = self.model(fmri, smri, return_all=True)
            forward_done = time.perf_counter()
            
            # Compute loss
            self._set_progress(
                phase='train_loss',
                epoch=epoch_number,
                batch=batch_number,
                total_batches=total_batches,
                details='computing loss and backpropagation',
            )
            loss_dict = self._compute_loss_dict(
                outputs=outputs,
                fmri=fmri,
                smri=smri,
                targets=targets,
                regression_targets=regression_targets,
                survival_times=survival_times,
                survival_events=survival_events,
                beta_annealing=beta_annealing
            )
            
            total_loss = loss_dict['total']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            step_done = time.perf_counter()
            
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
            if batch_number == 1 or batch_number == total_batches or batch_number % max(1, log_every_n_batches) == 0:
                logger.info(
                    f"Epoch {epoch_number}: batch {batch_number}/{total_batches} done in {step_done - batch_start:.1f}s "
                    f"(prepare {prepare_done - batch_start:.1f}s, forward {forward_done - prepare_done:.1f}s, "
                    f"backward {step_done - forward_done:.1f}s) | Loss: {total_loss.item():.4f} | Acc: {acc.item():.4f}" \
                    f"{self._gpu_memory_summary()}"
                )

        if num_batches == 0:
            raise ValueError(
                "Training dataloader produced zero batches. Check dataset size, modality files, and drop_last settings."
            )
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        logger.info(f"Epoch {epoch_number}: training finished")
        
        return dict(metrics)
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
        beta_annealing: float = 1.0,
        log_every_n_batches: int = 10,
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
        epoch_number = epoch + 1
        total_batches = len(val_loader)

        logger.info(f"Epoch {epoch_number}: validation started with {total_batches} batches")
        
        for batch_idx, batch in enumerate(val_loader):
            batch_number = batch_idx + 1
            self._set_progress(
                phase='val_prepare',
                epoch=epoch_number,
                batch=batch_number,
                total_batches=total_batches,
                details='moving validation batch to device',
            )
            fmri, smri, targets, regression_targets, survival_times, survival_events = self._prepare_batch(batch)
            
            # Forward pass
            self._set_progress(
                phase='val_forward',
                epoch=epoch_number,
                batch=batch_number,
                total_batches=total_batches,
                details='running validation forward pass',
            )
            outputs = self.model(fmri, smri, return_all=True)
            
            # Compute loss
            self._set_progress(
                phase='val_loss',
                epoch=epoch_number,
                batch=batch_number,
                total_batches=total_batches,
                details='computing validation loss',
            )
            loss_dict = self._compute_loss_dict(
                outputs=outputs,
                fmri=fmri,
                smri=smri,
                targets=targets,
                regression_targets=regression_targets,
                survival_times=survival_times,
                survival_events=survival_events,
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

            if batch_number == 1 or batch_number == total_batches or batch_number % max(1, log_every_n_batches) == 0:
                logger.info(
                    f"Epoch {epoch_number}: validation batch {batch_number}/{total_batches} | "
                    f"Loss: {loss_dict['total'].item():.4f} | Acc: {acc.item():.4f}" \
                    f"{self._gpu_memory_summary()}"
                )
            
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

        logger.info(f"Epoch {epoch_number}: validation finished")
        
        return dict(metrics)

    @torch.no_grad()
    def test(
        self,
        test_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
        beta_annealing: float = 1.0,
        save_name: str = 'test_results.json',
        log_every_n_batches: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a held-out test set and persist the results.

        Args:
            test_loader: Test dataloader
            checkpoint_path: Optional checkpoint to load before evaluating
            beta_annealing: KL annealing factor used during evaluation
            save_name: Filename for the saved test report
        """
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()
        metrics = defaultdict(float)
        num_batches = 0

        all_preds = []
        all_targets = []
        all_probabilities = []
        total_batches = len(test_loader)

        logger.info(f"Testing started with {total_batches} batches")

        for batch_idx, batch in enumerate(test_loader):
            batch_number = batch_idx + 1
            self._set_progress(
                phase='test_prepare',
                epoch=None,
                batch=batch_number,
                total_batches=total_batches,
                details='moving test batch to device',
            )
            fmri, smri, targets, regression_targets, survival_times, survival_events = self._prepare_batch(batch)

            self._set_progress(
                phase='test_forward',
                epoch=None,
                batch=batch_number,
                total_batches=total_batches,
                details='running test forward pass',
            )
            outputs = self.model(fmri, smri, return_all=True)

            self._set_progress(
                phase='test_loss',
                epoch=None,
                batch=batch_number,
                total_batches=total_batches,
                details='computing test metrics',
            )
            loss_dict = self._compute_loss_dict(
                outputs=outputs,
                fmri=fmri,
                smri=smri,
                targets=targets,
                regression_targets=regression_targets,
                survival_times=survival_times,
                survival_events=survival_events,
                beta_annealing=beta_annealing
            )

            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    metrics[f'test_{key}'] += value.item()

            preds = outputs['predictions']
            probabilities = torch.softmax(outputs['logits'], dim=1)
            acc = (preds == targets).float().mean()
            metrics['test_accuracy'] += acc.item()

            if batch_number == 1 or batch_number == total_batches or batch_number % max(1, log_every_n_batches) == 0:
                logger.info(
                    f"Testing batch {batch_number}/{total_batches} | Loss: {loss_dict['total'].item():.4f} | "
                    f"Acc: {acc.item():.4f}{self._gpu_memory_summary()}"
                )

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

            num_batches += 1

        if num_batches == 0:
            raise ValueError(
                "Test dataloader produced zero batches. Check dataset size and modality files."
            )

        for key in metrics:
            metrics[key] /= num_batches

        y_true = np.asarray(all_targets)
        y_pred = np.asarray(all_preds)
        labels = list(range(self.num_classes))

        confusion = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        per_class_accuracy = {}
        for class_index, class_name in enumerate(self.class_names):
            row_total = confusion[class_index].sum()
            per_class_accuracy[class_name] = float(confusion[class_index, class_index] / row_total) if row_total > 0 else 0.0

        results: Dict[str, Any] = {
            **metrics,
            'overall_accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion.tolist(),
            'classification_report': report,
            'per_class_accuracy': per_class_accuracy,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'num_samples': len(all_targets),
            'num_batches': num_batches,
        }

        results_path = self.output_dir / save_name
        with open(results_path, 'w') as handle:
            json.dump(results, handle, indent=2)

        logger.info(f"Test results saved: {results_path}")
        logger.info(
            f"Test Accuracy: {results['overall_accuracy']:.4f} | "
            f"Macro F1: {results['macro_f1']:.4f} | "
            f"Weighted F1: {results['weighted_f1']:.4f}"
        )

        self._clear_progress()

        results['results_path'] = str(results_path)
        return results
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        annealing_epochs: int = 20,
        log_every_n_batches: int = 10,
        start_epoch: int = 0,
        resume_optimizer_state: Optional[Dict] = None,
        max_wall_time_seconds: Optional[float] = None
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
            start_epoch: Epoch to start training from (for resume capability)
            resume_optimizer_state: Optimizer state dict to resume from checkpoint
        """
        logger.info(f"Starting training for {num_epochs} epochs (starting from epoch {start_epoch})")
        logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        if max_wall_time_seconds is not None:
            logger.info(f"Wall-clock training budget: {max_wall_time_seconds / 3600:.2f} hours")
        
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters found. Check freeze settings before training.")

        trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
        logger.info(f"Trainable parameters for optimizer: {trainable_count:,}")

        # Optimizer and scheduler
        optimizer = Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Load optimizer state if resuming
        if resume_optimizer_state is not None:
            optimizer.load_state_dict(resume_optimizer_state)
            logger.info("Resumed optimizer state from checkpoint")
        
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # Step scheduler to current epoch if resuming
        for _ in range(start_epoch):
            scheduler.step()
        
        training_start = time.perf_counter()
        self._start_heartbeat()
        try:
            # Training loop
            for epoch in range(start_epoch, num_epochs):
                elapsed_seconds = time.perf_counter() - training_start
                if max_wall_time_seconds is not None and elapsed_seconds >= max_wall_time_seconds:
                    logger.info(
                        f"Stopping before epoch {epoch + 1} because the {max_wall_time_seconds / 3600:.2f} hour "
                        f"budget was reached"
                    )
                    break

                # KL annealing
                if annealing_epochs <= 0:
                    beta = 1.0
                else:
                    beta = min(1.0, epoch / annealing_epochs)

                self._set_progress(
                    phase='epoch',
                    epoch=epoch + 1,
                    batch=None,
                    total_batches=None,
                    details='starting epoch',
                )

                # Train
                train_metrics = self.train_epoch(
                    train_loader,
                    optimizer,
                    epoch,
                    beta_annealing=beta,
                    log_every_n_batches=log_every_n_batches,
                )

                # Validate
                val_metrics = self.validate(
                    val_loader,
                    epoch,
                    beta_annealing=beta,
                    log_every_n_batches=log_every_n_batches,
                )

                # Learning rate scheduling
                scheduler.step()

                # Log epoch results
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_metrics['train_total']:.4f}, "
                    f"Train Acc: {train_metrics['train_accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['val_total']:.4f}, "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
                    f"β: {beta:.4f} | Elapsed: {(time.perf_counter() - training_start) / 60:.1f} min"
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
                    self._save_best_component_models(
                        epoch=epoch,
                        metric_name='val_loss',
                        metric_value=val_loss,
                    )
                    logger.info(f"✓ New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self._save_checkpoint(epoch, optimizer, 'best_acc')
                    logger.info(f"✓ New best validation accuracy: {val_acc:.4f}")

                # Early stopping
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break

                # Save regular checkpoint
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(epoch, optimizer, f'epoch_{epoch}')

            logger.info("Training completed!")
            self._save_history()
        finally:
            self._stop_heartbeat()
    
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
            'component_state_dicts': self._collect_component_state_dicts(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _collect_component_state_dicts(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect component-level state dicts for transfer/fine-tuning workflows."""
        component_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}

        for module_name in ('fmri_encoder', 'smri_encoder'):
            module = getattr(self.model, module_name, None)
            if module is not None:
                component_state_dicts[module_name] = module.state_dict()

        return component_state_dicts

    def _save_best_component_models(
        self,
        epoch: int,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """Persist best component checkpoints for modality-specific fine-tuning."""
        timestamp = datetime.now().isoformat()
        component_state_dicts = self._collect_component_state_dicts()

        fmri_state_dict = component_state_dicts.get('fmri_encoder')
        if fmri_state_dict is not None:
            fmri_path = self.output_dir / 'best_fmri_model.pt'
            torch.save(
                {
                    'epoch': epoch,
                    'component_name': 'fmri_encoder',
                    'fmri_encoder_state_dict': fmri_state_dict,
                    'source_metric_name': metric_name,
                    'source_metric_value': float(metric_value),
                    'best_val_loss': self.best_val_loss,
                    'best_val_acc': self.best_val_acc,
                    'timestamp': timestamp,
                },
                fmri_path,
            )
            logger.info(f"Best fMRI checkpoint saved: {fmri_path}")

        smri_state_dict = component_state_dicts.get('smri_encoder')
        if smri_state_dict is not None:
            smri_path = self.output_dir / 'best_smri_model.pt'
            torch.save(
                {
                    'epoch': epoch,
                    'component_name': 'smri_encoder',
                    'smri_encoder_state_dict': smri_state_dict,
                    'source_metric_name': metric_name,
                    'source_metric_value': float(metric_value),
                    'best_val_loss': self.best_val_loss,
                    'best_val_acc': self.best_val_acc,
                    'timestamp': timestamp,
                },
                smri_path,
            )
            logger.info(f"Best sMRI checkpoint saved: {smri_path}")

        final_path = self.output_dir / 'best_final_model.pt'
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'component_state_dicts': component_state_dicts,
                'source_metric_name': metric_name,
                'source_metric_value': float(metric_value),
                'best_val_loss': self.best_val_loss,
                'best_val_acc': self.best_val_acc,
                'timestamp': timestamp,
            },
            final_path,
        )
        logger.info(f"Best final-model checkpoint saved: {final_path}")
    
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

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unsupported checkpoint format for: {checkpoint_path}")
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}, Best val acc: {self.best_val_acc:.4f}")
