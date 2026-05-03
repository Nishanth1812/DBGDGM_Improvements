#!/usr/bin/env python
"""
MM-DBGDGM ? Full pipeline entry point.

Usage:
    python run_experiment.py --data_dir data/synthetic/subjects \
                              --manifest data/synthetic/manifest.csv \
                              --epochs 100 \
                              --beta 2.0 \
                              --seed 42 \
                              --output_dir results/

Outputs to results/:
    fold_1/ ... fold_5/     (checkpoints)
    metrics_summary.json    (all CV metrics)
    figures/                (all plots)
    predictions.csv         (per-subject predicted class + uncertainty)
"""

import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from data.loaders import build_dataloaders, get_stratified_kfold_splits
from models.mm_dbgdgm import MM_DBGDGM
from training.trainer import train_fold
from training.evaluate import compute_metrics, print_metrics
from preprocessing.fmri_pipeline import build_fmri_graphs
from preprocessing.smri_pipeline import build_structural_graph
from visualisation.latent_space import plot_latent_space
from visualisation.uncertainty_plots import plot_uncertainty_distributions
from visualisation.attention_maps import plot_attention_maps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fmri_graphs(fmri_batch, device):
    """Build list of graph lists for a batch."""
    batch_size = fmri_batch.size(0)
    K = None
    all_graphs = []

    for b in range(batch_size):
        ts = fmri_batch[b].cpu().numpy()
        graphs = [g.to(device) for g in build_fmri_graphs(ts)]
        all_graphs.append(graphs)
        if K is None:
            K = len(graphs)

    return all_graphs


def collate_smri_graphs(smri_batch, device):
    """Build list of PyG Data objects for a batch."""
    graphs = []
    for b in range(smri_batch.size(0)):
        feats = smri_batch[b].cpu().numpy()
        graphs.append(build_structural_graph(feats).to(device))
    return graphs


def run(args):
    # Load config safely
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    config.setdefault("training", {})
    config.setdefault("model", {})
    config.setdefault("preprocessing", {})

    seed = args.seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Resolve paths
    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest)

    # Load data
    print("\n  Loading data...")
    loaders, splits = build_dataloaders(
        manifest_path=manifest_path,
        subjects_dir=data_dir,
        batch_size=args.batch_size,
        n_splits=args.k_folds,
        seed=seed,
    )

    # Detect number of classes actually present in the manifest
    _mf = pd.read_csv(manifest_path)
    num_classes = int(_mf["label"].nunique())
    print(f"\n  Classes detected: {num_classes} ({sorted(_mf['label'].unique().tolist())})") 

    # Training config
    epochs = args.epochs
    beta = args.beta
    lambda_vae = config["training"].get("lambda_vae", 0.1)
    warmup_epochs = config["training"].get("warmup_epochs", 20)
    patience = config["training"].get("patience", 15)
    lr = config["training"].get("lr", 1e-4)
    weight_decay = config["training"].get("weight_decay", 1e-5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print(f"\n  {'='*60}")
    print(f"  MM-DBGDGM Training - {args.k_folds}-Fold Cross-Validation")
    print(f"  {'='*60}")
    n_subjects = len(pd.read_csv(manifest_path))
    print(f"  Subjects: {n_subjects} | Epochs: {epochs} | beta={beta} | lambda_vae={lambda_vae}")
    print(f"  Warmup: {warmup_epochs} epochs | Patience: {patience}")
    print(f"  Output: {output_dir}")
    print(f"  {'='*60}\n")

    fold_results = []
    all_preds = []
    all_labels = []
    all_probs = []
    all_unc = []
    all_mu = []

    best_overall_acc = -1.0
    best_overall_state = None
    best_overall_fold = -1

    for fold_i in range(args.k_folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_i+1} / {args.k_folds}")
        print(f"{'='*60}")

        model = MM_DBGDGM(
            fmri_in_channels=1,
            smri_in_channels=4,
            gat_hidden_dim=config["model"].get("gat_hidden_dim", 32),
            gat_heads=config["model"].get("gat_heads", 8),
            lstm_hidden=config["model"].get("lstm_hidden_dim", 256),
            latent_dim=config["model"].get("latent_dim", 128),
            num_classes=num_classes,
            dropout=config["model"].get("dropout", 0.4),
        ).to(device)

        if torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
            # Note: Model forward must handle list inputs if using DataParallel with current trainer
            # model = torch.nn.DataParallel(model) 
            # For now, we stay on one device unless user confirms list splitting support

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_loader = loaders[fold_i]["train"]
        val_loader = loaders[fold_i]["val"]

        fold_metrics, best_state = train_fold(
            fold_i=fold_i,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            patience=patience,
            checkpoint_dir=output_dir / f"fold_{fold_i+1}",
            warmup_epochs=warmup_epochs,
            lambda_vae=lambda_vae,
            beta=beta,
        )

        fold_results.append(fold_metrics)

        all_preds.extend(fold_metrics["predictions"].tolist())
        all_labels.extend(fold_metrics["labels"].tolist())
        all_probs.append(fold_metrics["probabilities"])
        all_unc.extend(fold_metrics["uncertainty"].tolist())

        # Track best fold overall
        if fold_metrics["accuracy"] > best_overall_acc:
            best_overall_acc = fold_metrics["accuracy"]
            best_overall_state = best_state
            best_overall_fold = fold_i + 1

        print("\n  Fold summary:")
        print(f"    Accuracy: {fold_metrics['accuracy']:.3f}")
        print(f"    AUC:      {fold_metrics.get('auc_display', fold_metrics['auc'])}")
        print(f"    F1:       {fold_metrics['f1']:.3f}")

    # Aggregate results
    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_unc = np.array(all_unc)

    # ── Save best overall model ───────────────────────────────────────────────
    if best_overall_state is not None:
        best_model_path = output_dir / "best_model.pt"
        torch.save(best_overall_state, best_model_path)
        print(f"\n  Best model from Fold {best_overall_fold} (Acc={best_overall_acc:.3f}) saved to:")
        print(f"  {best_model_path}")
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("  CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    # Per-fold metrics
    print(f"\n  {'Fold':<6} {'Acc':>7} {'AUC':>7} {'F1':>7}")
    print("  " + "-" * 30)
    for i, fr in enumerate(fold_results):
        print(f"  {i+1:<6} {fr['accuracy']:>7.3f} {fr['auc']:>7.3f} {fr['f1']:>7.3f}")

    # Mean ? std
    accs = [fr["accuracy"] for fr in fold_results]
    aucs = [fr["auc"] for fr in fold_results]
    f1s = [fr["f1"] for fr in fold_results]
    print(f"\n  Mean:     {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"  AUC:      {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    print(f"  F1:       {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

    # 4-class overall metrics
    overall_acc = accuracy_score(all_labels, all_preds)
    try:
        overall_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except:
        overall_auc = 0.0
    overall_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"\n  Overall (pooled):")
    print(f"    Accuracy: {overall_acc:.3f}")
    print(f"    AUC:      {overall_auc:.3f}")
    print(f"    F1:       {overall_f1:.3f}")

    # Save metrics
    summary = {
        "n_folds": args.k_folds,
        "fold_accuracies": accs,
        "fold_aucs": aucs,
        "fold_f1s": f1s,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "overall_accuracy": float(overall_acc),
        "overall_auc": float(overall_auc),
        "overall_f1": float(overall_f1),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "probabilities": all_probs.tolist(),
        "uncertainty": all_unc.tolist(),
        "config": {
            "epochs": epochs,
            "beta": beta,
            "lambda_vae": lambda_vae,
            "warmup_epochs": warmup_epochs,
            "seed": seed,
        },
    }

    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Metrics saved: {output_dir / 'metrics_summary.json'}")

    # Visualisations
    print("\n  Generating visualisations...")

    # t-SNE latent space
    # We need ? vectors ? retrain a model and get ? for all subjects
    # For simplicity, use predictions and uncertainty
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Uncertainty distributions
    unc_by_class = {"CN": [], "MCI": [], "lMCI": [], "AD": []}
    for i, label in enumerate(all_labels):
        name = ["CN", "MCI", "lMCI", "AD"][label]
        unc_by_class[name].append(all_unc[i])

    plot_uncertainty_distributions(unc_by_class, save_path=fig_dir / "uncertainty_boxplot.png")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "subject_id": [f"sub-{i+1:03d}" for i in range(len(all_preds))],
        "true_label": all_labels,
        "pred_label": all_preds,
        "uncertainty": all_unc,
    })
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"  Predictions saved: {output_dir / 'predictions.csv'}")

    print(f"\n  All outputs in: {output_dir}")
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,  default="data/synthetic/subjects")
    parser.add_argument("--manifest",   type=str,  default="data/synthetic/subjects/manifest.csv")
    parser.add_argument("--epochs",      type=int,  default=100)
    parser.add_argument("--beta",       type=float, default=2.0)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--k_folds",   type=int,  default=5)
    parser.add_argument("--batch_size", type=int,  default=8)
    parser.add_argument("--output_dir", type=str,  default="results")
    parser.add_argument("--config",     type=str,  default="config.yaml")
    args = parser.parse_args()

    run(args)