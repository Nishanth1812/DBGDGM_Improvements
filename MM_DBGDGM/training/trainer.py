import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from preprocessing.fmri_pipeline import build_fmri_graphs
from preprocessing.smri_pipeline import build_structural_graph

from .losses import combined_loss


def train_one_epoch(model, train_loader, optimizer, device, epoch, warmup_epochs=20, lambda_vae=0.1, beta=1.0):
    model.train()
    total_loss = 0
    total_ce = 0
    total_kl = 0

    for batch in train_loader:
        fmri = batch["fmri"].to(device)
        smri = batch["smri"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        batch_graphs = []
        for b in range(fmri.size(0)):
            ts = fmri[b].cpu().numpy()
            graphs = build_fmri_graphs(ts)
            batch_graphs.append(graphs)

        smri_graphs = []
        for b in range(smri.size(0)):
            feats = smri[b].cpu().numpy()
            smri_graphs.append(build_structural_graph(feats))

        optimizer.zero_grad()
        
        fmri_graphs_batch = []
        smri_graphs_batch = []
        for b in range(fmri.size(0)):
            fmri_graphs_batch.append([g.to(device) for g in batch_graphs[b]])
            smri_graphs_batch.append(smri_graphs[b].to(device))

        out = model(fmri_graphs_batch, smri_graphs_batch, return_sample=True)

        loss, ce, kl = combined_loss(
            out["logits"],
            labels,
            out["mu"],
            out["logvar"],
            beta=beta,
            lambda_vae=lambda_vae,
            current_epoch=epoch,
            warmup_epochs=warmup_epochs,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * fmri.size(0)
        total_ce += ce.item() * fmri.size(0)
        total_kl += kl.item() * fmri.size(0)

    n = len(train_loader.dataset)
    return total_loss / n, total_ce / n, total_kl / n


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_unc = []

    for batch in val_loader:
        fmri = batch["fmri"].to(device)
        smri = batch["smri"].to(device)
        labels = batch["label"]

        fmri_graphs_batch = []
        smri_graphs_batch = []
        for b in range(fmri.size(0)):
            ts = fmri[b].cpu().numpy()
            fmri_graphs_batch.append([g.to(device) for g in build_fmri_graphs(ts)])

            feats = smri[b].cpu().numpy()
            smri_graphs_batch.append(build_structural_graph(feats).to(device))

        out = model(fmri_graphs_batch, smri_graphs_batch, return_sample=False)

        probs = torch.softmax(out["logits"], dim=-1)
        preds = out["logits"].argmax(dim=-1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_unc.append(out["uncertainty"].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_unc = torch.cat(all_unc).numpy()

    acc = accuracy_score(all_labels, all_preds)

    n_unique = len(np.unique(all_labels))
    if n_unique < 2:
        auc = float('nan')  # Can't compute AUC with only 1 class in val fold
    else:
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        except Exception:
            auc = float('nan')

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "auc": auc if not np.isnan(auc) else 0.0,  # store 0 for JSON-compatibility
        "auc_display": f"{auc:.3f}" if not np.isnan(auc) else "N/A (val fold has 1 class)",
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "uncertainty": all_unc,
    }


def train_fold(
    fold_i,
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    checkpoint_dir,
    warmup_epochs=20,
    lambda_vae=0.1,
    beta=2.0,
):
    best_val_score = -1.0
    patience_counter = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # init to starting weights

    for epoch in range(epochs):
        loss, ce, kl = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch, warmup_epochs=warmup_epochs, lambda_vae=lambda_vae, beta=beta,
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"  Fold {fold_i+1} | Epoch {epoch+1:3d} | "
            f"Loss: {loss:.4f} (CE: {ce:.4f}, KL: {kl:.4f}) | "
            f"Val Acc: {val_metrics['accuracy']:.3f} | Val AUC: {val_metrics['auc_display']}"
        )

        # Use accuracy for early stopping (robust when val fold has <2 classes)
        val_score = val_metrics["accuracy"]
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Save best checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"fold_{fold_i+1}_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"  Best Val Acc: {best_val_score:.3f} -> {ckpt_path}")

    # Restore best model for final evaluation
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, device)
    return final_metrics, best_state