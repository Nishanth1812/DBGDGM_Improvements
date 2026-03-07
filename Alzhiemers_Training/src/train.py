import inspect
import logging
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .dataset import data_loader


def _save_checkpoint(path, filename, model, optimizer, scaler,
                     epoch, temp, learning_rate,
                     best_nll, best_nll_train, best_label_score,
                     edge_nll, edge_aucroc, edge_ap, label_metrics,
                     embeddings, label):
    """Save a full, resumable checkpoint to disk."""
    clean = lambda d: {k: float(v) for k, v in d.items()}
    payload = {
        # ── model / optimiser ──────────────────────────────────────
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state':    scaler.state_dict(),
        # ── training progress ──────────────────────────────────────
        'epoch':           epoch,
        'temp':            float(temp),
        'learning_rate':   float(learning_rate),
        'best_nll':        float(best_nll),
        'best_nll_train':  float(best_nll_train),
        'best_label_score': float(best_label_score),
        # ── evaluation results at this checkpoint ──────────────────
        'metrics': {
            'edge_nll':    clean(edge_nll),
            'edge_aucroc': clean(edge_aucroc),
            'edge_ap':     clean(edge_ap),
            'label':       clean(label_metrics),
        },
        'embeddings': embeddings,
        'label': label,
    }
    torch.save(payload, Path(path) / filename)
    logging.info(f"Checkpoint saved → {Path(path) / filename}")


def train(model, dataset,
          save_path=Path.cwd() / "models",
          learning_rate=1e-4,
          temp=1.,
          temp_min=0.05,
          num_epochs=1001,
          anneal_rate=0.0003,
          batch_size=8,
          weight_decay=1e-4,
          valid_prop=0.1,
          test_prop=0.1,
          device=torch.device("cpu"),
          eval_every=20,
          eval_split=0.2,
          early_stopping_patience=100,
          classification_weight=1.0,
          amp_dtype="bf16",
          max_grad_norm=1.0,
          seed=42,
          resume_from=None,
          on_checkpoint=None,
          ):
    """
    Trains the improved DBGDGM model using the Oasis dataset.

    Early stopping monitors held-out valid NLL every `eval_every` epochs.
    Stops if no improvement for `early_stopping_patience` consecutive eval checks.

    Checkpoints saved:
      checkpoint_best_valid.pt  — best held-out validation NLL (use this for future retraining)
      checkpoint_best_train.pt  — best training NLL
      checkpoint_latest.pt      — most recent eval (for manual resume inspection)
    """

    # Create save path
    save_path = Path(save_path)
    try:
        save_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logging.warning(f"Save folder already exists at {save_path}. Existing checkpoints may be overwritten.")
    else:
        logging.info(f"Created save folder at {save_path}")

    logging.info(f"Model save path: {save_path}")

    # Initialize optimizer, scheduler & AMP scaler
    optimizer_kwargs = {
        'lr': learning_rate,
        'weight_decay': weight_decay,
    }
    if device.type == "cuda" and 'fused' in inspect.signature(torch.optim.AdamW).parameters:
        optimizer_kwargs['fused'] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    use_amp = device.type == "cuda"
    amp_dtype = amp_dtype.lower()
    amp_torch_dtype = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
    }.get(amp_dtype, torch.bfloat16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_torch_dtype == torch.float16)

    # Move model to device
    model.to(device)

    # Track best losses
    best_nll = float('inf')
    best_nll_train = float('inf')
    best_label_score = float('-inf')

    # Early stopping state
    patience_counter = 0
    early_stopped = False
    start_epoch = 0

    # ── Resume from checkpoint ───────────────────────────────────────────────
    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.exists():
            logging.warning(f"resume_from path not found: {resume_from}. Starting from scratch.")
        else:
            logging.info(f"Resuming training from checkpoint: {resume_from}")
            ckpt = torch.load(resume_from, map_location=device)
            ckpt_state = ckpt['model_state']

            ckpt_n    = ckpt_state['alpha_mean.weight'].shape[0]
            current_n = model.alpha_mean.weight.shape[0]

            if ckpt_n != current_n:
                raise ValueError(
                    f"Subject count mismatch: checkpoint={ckpt_n}, current={current_n}. "
                    "Refusing partial resume to avoid embedding misalignment and overfitting. "
                    "Start a fresh run (no --resume-from), or regenerate dataset with identical subject set/order."
                )
            else:
                model.load_state_dict(ckpt_state)
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                except (ValueError, KeyError, RuntimeError):
                    logging.warning("Optimizer state incompatible — starting with fresh optimizer.")


            if 'scaler_state' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state'])
            start_epoch    = ckpt['epoch'] + 1
            temp           = ckpt['temp']
            learning_rate  = ckpt['learning_rate']
            best_nll       = ckpt['best_nll']
            best_nll_train = ckpt['best_nll_train']
            best_label_score = ckpt.get('best_label_score', float('-inf'))
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            scheduler.last_epoch = start_epoch - 1
            logging.info(
                f"Resumed from epoch {ckpt['epoch']} | "
                f"best valid NLL={best_nll:.4f} | temp={temp:.4f} | lr={learning_rate:.6f}"
            )


    # Fixed held-out eval split
    labels = [sample[2] for sample in dataset]
    indices = np.arange(len(dataset))
    try:
        train_idx, eval_idx = train_test_split(
            indices,
            test_size=eval_split,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
    except ValueError:
        logging.warning('Falling back to a non-stratified subject split because one or more classes are too small.')
        train_idx, eval_idx = train_test_split(
            indices,
            test_size=eval_split,
            random_state=seed,
            shuffle=True,
        )

    train_dataset = [dataset[int(index)] for index in train_idx]
    eval_dataset = [dataset[int(index)] for index in eval_idx]

    class_counts = np.bincount([sample[2] for sample in train_dataset], minlength=model.num_classes).astype(np.float32)
    class_weights = np.ones(model.num_classes, dtype=np.float32)
    non_zero = class_counts > 0
    class_weights[non_zero] = class_counts[non_zero].sum() / (non_zero.sum() * class_counts[non_zero])
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    logging.info(f"Subject split: {len(train_dataset)} train / {len(eval_dataset)} eval (held-out)")
    logging.info(f"Early stopping patience: {early_stopping_patience} eval checks "
                 f"(= {early_stopping_patience * eval_every} epochs without improvement)")
    logging.info(f"Classification loss weight: {classification_weight:.3f}")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        logging.debug(f"Starting epoch {epoch}")
        random.shuffle(train_dataset)
        model.train()

        running_loss = {'nll': 0, 'kld_z': 0, 'kld_alpha': 0, 'kld_beta': 0, 'kld_phi': 0, 'classification': 0}

        for batch_graphs in data_loader(train_dataset, batch_size):
            optimizer.zero_grad()

            autocast_context = (
                torch.amp.autocast(device_type='cuda', dtype=amp_torch_dtype)
                if use_amp else nullcontext()
            )
            with autocast_context:
                batch_loss = model(batch_graphs,
                                   valid_prop=valid_prop,
                                   test_prop=test_prop,
                                   temp=temp,
                                   class_weights=class_weights)
                loss = (batch_loss['nll']
                        + 1.0  * batch_loss['kld_z']
                        + 0.1  * batch_loss['kld_alpha']
                        + 0.1  * batch_loss['kld_beta']
                        + 0.01 * batch_loss['kld_phi']) / len(batch_graphs)
                loss = loss + classification_weight * (batch_loss['classification'] / len(batch_graphs))

            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            for loss_name in running_loss:
                running_loss[loss_name] += batch_loss[loss_name].detach().item() / len(train_dataset)

        clean_loss = {k: round(float(v), 6) for k, v in running_loss.items()}
        logging.info(f"Epoch {epoch} | {clean_loss}")
        for h in logging.getLogger().handlers:
            h.flush()

        # ── Evaluation (every eval_every epochs) ────────────────────────────
        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                edge_nll, edge_aucroc, edge_ap = model.predict_auc_roc_precision(
                    eval_dataset,
                    valid_prop=valid_prop,
                    test_prop=test_prop)
                label_metrics = model.predict_label_metrics(
                    eval_dataset,
                    valid_prop=valid_prop,
                    test_prop=test_prop,
                )

            logging.info(
                f"Epoch {epoch} [eval | {len(eval_dataset)} held-out subjects] | "
                f"edge train nll {edge_nll['train']:.4f} aucroc {edge_aucroc['train']:.4f} ap {edge_ap['train']:.4f} | "
                f"edge valid nll {edge_nll['valid']:.4f} aucroc {edge_aucroc['valid']:.4f} ap {edge_ap['valid']:.4f} | "
                f"edge test nll {edge_nll['test']:.4f} aucroc {edge_aucroc['test']:.4f} ap {edge_ap['test']:.4f} | "
                f"label loss {label_metrics['loss']:.4f} acc {label_metrics['accuracy']:.4f} "
                f"bal_acc {label_metrics['balanced_accuracy']:.4f} macro_f1 {label_metrics['macro_f1']:.4f}"
            )

            save_valid = label_metrics['balanced_accuracy'] > best_label_score
            save_train = edge_nll['train'] < best_nll_train

            # Compute embeddings once if any checkpoint needs saving
            if save_valid or save_train:
                with torch.no_grad():
                    embeddings = model.predict_embeddings(train_dataset,
                                                          valid_prop=valid_prop,
                                                          test_prop=test_prop)
            else:
                embeddings = None

            # ── Best-valid checkpoint ────────────────────────────────────────
            if save_valid:
                logging.info(
                    f"✓ New best held-out balanced accuracy: {label_metrics['balanced_accuracy']:.4f} "
                    f"(was {best_label_score:.4f}). Saving best-valid checkpoint."
                )
                _save_checkpoint(save_path, "checkpoint_best_valid.pt",
                                  model, optimizer, scaler,
                                  epoch, temp, learning_rate,
                                  best_nll, best_nll_train, best_label_score,
                                  edge_nll, edge_aucroc, edge_ap, label_metrics,
                                  embeddings, label="best_valid")
                if on_checkpoint: on_checkpoint()
                best_nll = edge_nll['valid']
                best_label_score = label_metrics['balanced_accuracy']
                patience_counter = 0   # reset early-stopping counter
            else:
                patience_counter += 1
                logging.info(f"No held-out label improvement. Patience: {patience_counter}/{early_stopping_patience}")

            # ── Best-train checkpoint ────────────────────────────────────────
            if save_train:
                logging.info(f"✓ New best train edge NLL: {edge_nll['train']:.4f} (was {best_nll_train:.4f}). Saving best-train checkpoint.")
                _save_checkpoint(save_path, "checkpoint_best_train.pt",
                                  model, optimizer, scaler,
                                  epoch, temp, learning_rate,
                                  best_nll, best_nll_train, best_label_score,
                                  edge_nll, edge_aucroc, edge_ap, label_metrics,
                                  embeddings, label="best_train")
                if on_checkpoint: on_checkpoint()
                best_nll_train = edge_nll['train']

            # ── Latest checkpoint (always overwritten) ───────────────────────
            _save_checkpoint(save_path, "checkpoint_latest.pt",
                              model, optimizer, scaler,
                              epoch, temp, learning_rate,
                              best_nll, best_nll_train, best_label_score,
                              edge_nll, edge_aucroc, edge_ap, label_metrics,
                              embeddings, label="latest")
            if on_checkpoint: on_checkpoint()

            # ── Early stopping check ─────────────────────────────────────────
            if patience_counter >= early_stopping_patience:
                logging.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"No held-out label improvement for {early_stopping_patience} consecutive eval checks "
                    f"({patience_counter * eval_every} epochs)."
                )
                early_stopped = True
                break

        # ── Temperature annealing & LR scheduler step ───────────────────────
        if epoch % 10 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * epoch), temp_min)
            logging.debug(f"Gumbel temp → {temp:.5f}")
        scheduler.step()
        learning_rate = scheduler.get_last_lr()[0]
        logging.debug(f"Learning rate → {learning_rate:.6f}")

    if early_stopped:
        logging.info("Training ended early. Best checkpoints are saved and ready for future retraining.")
    else:
        logging.info("Training complete (full epochs). Best checkpoints are saved.")
