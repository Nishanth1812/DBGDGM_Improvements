import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from .dataset import data_loader


def _save_checkpoint(path, filename, model, optimizer, scaler,
                     epoch, temp, learning_rate,
                     best_nll, best_nll_train,
                     nll, aucroc, ap, embeddings, label):
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
        # ── evaluation results at this checkpoint ──────────────────
        'metrics': {
            'nll':    clean(nll),
            'aucroc': clean(aucroc),
            'ap':     clean(ap),
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
          batch_size=1,
          weight_decay=0.,
          valid_prop=0.1,
          test_prop=0.1,
          device=torch.device("cpu"),
          eval_every=20,
          eval_split=0.2,
          early_stopping_patience=100,
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
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Move model to device
    model.to(device)

    # Track best losses
    best_nll = float('inf')
    best_nll_train = float('inf')

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
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scaler.load_state_dict(ckpt['scaler_state'])
            start_epoch    = ckpt['epoch'] + 1
            temp           = ckpt['temp']
            learning_rate  = ckpt['learning_rate']
            best_nll       = ckpt['best_nll']
            best_nll_train = ckpt['best_nll_train']
            # Advance scheduler to match resumed epoch
            for _ in range(start_epoch):
                scheduler.step()
            logging.info(
                f"Resumed from epoch {ckpt['epoch']} | "
                f"best valid NLL={best_nll:.4f} | temp={temp:.4f} | lr={learning_rate:.6f}"
            )

    # Fixed held-out eval split
    shuffled = dataset[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - eval_split))
    train_dataset = shuffled[:split]
    eval_dataset  = shuffled[split:]
    logging.info(f"Subject split: {len(train_dataset)} train / {len(eval_dataset)} eval (held-out)")
    logging.info(f"Early stopping patience: {early_stopping_patience} eval checks "
                 f"(= {early_stopping_patience * eval_every} epochs without improvement)")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        logging.debug(f"Starting epoch {epoch}")
        random.shuffle(train_dataset)
        model.train()

        running_loss = {'nll': 0, 'kld_z': 0, 'kld_alpha': 0, 'kld_beta': 0, 'kld_phi': 0}

        for batch_graphs in data_loader(train_dataset, batch_size):
            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                batch_loss = model(batch_graphs,
                                   valid_prop=valid_prop,
                                   test_prop=test_prop,
                                   temp=temp)
                loss = (batch_loss['nll']
                        + 1.0  * batch_loss['kld_z']
                        + 0.1  * batch_loss['kld_alpha']
                        + 0.1  * batch_loss['kld_beta']
                        + 0.01 * batch_loss['kld_phi']) / len(batch_graphs)

            scaler.scale(loss).backward()
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
                nll, aucroc, ap = model.predict_auc_roc_precision(
                    eval_dataset,
                    valid_prop=valid_prop,
                    test_prop=test_prop)

            logging.info(
                f"Epoch {epoch} [eval | {len(eval_dataset)} held-out subjects] | "
                f"train nll {nll['train']:.4f} aucroc {aucroc['train']:.4f} ap {ap['train']:.4f} | "
                f"valid nll {nll['valid']:.4f} aucroc {aucroc['valid']:.4f} ap {ap['valid']:.4f} | "
                f"test  nll {nll['test']:.4f}  aucroc {aucroc['test']:.4f}  ap {ap['test']:.4f}"
            )

            save_valid = nll['valid'] < best_nll
            save_train = nll['train'] < best_nll_train

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
                logging.info(f"✓ New best valid NLL: {nll['valid']:.4f} (was {best_nll:.4f}). Saving best-valid checkpoint.")
                _save_checkpoint(save_path, "checkpoint_best_valid.pt",
                                  model, optimizer, scaler,
                                  epoch, temp, learning_rate,
                                  best_nll, best_nll_train,
                                  nll, aucroc, ap, embeddings, label="best_valid")
                if on_checkpoint: on_checkpoint()
                best_nll = nll['valid']
                patience_counter = 0   # reset early-stopping counter
            else:
                patience_counter += 1
                logging.info(f"No valid NLL improvement. Patience: {patience_counter}/{early_stopping_patience}")

            # ── Best-train checkpoint ────────────────────────────────────────
            if save_train:
                logging.info(f"✓ New best train NLL: {nll['train']:.4f} (was {best_nll_train:.4f}). Saving best-train checkpoint.")
                _save_checkpoint(save_path, "checkpoint_best_train.pt",
                                  model, optimizer, scaler,
                                  epoch, temp, learning_rate,
                                  best_nll, best_nll_train,
                                  nll, aucroc, ap, embeddings, label="best_train")
                if on_checkpoint: on_checkpoint()
                best_nll_train = nll['train']

            # ── Latest checkpoint (always overwritten) ───────────────────────
            _save_checkpoint(save_path, "checkpoint_latest.pt",
                              model, optimizer, scaler,
                              epoch, temp, learning_rate,
                              best_nll, best_nll_train,
                              nll, aucroc, ap, embeddings, label="latest")
            if on_checkpoint: on_checkpoint()

            # ── Early stopping check ─────────────────────────────────────────
            if patience_counter >= early_stopping_patience:
                logging.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"No valid NLL improvement for {early_stopping_patience} consecutive eval checks "
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
