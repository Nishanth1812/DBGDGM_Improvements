import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)


def compute_metrics(labels, predictions, probabilities, uncertainty, label_names=None):
    """Compute comprehensive metrics."""
    if label_names is None:
        label_names = ["CN", "MCI", "lMCI", "AD"]

    metrics = {}

    # 4-class metrics
    metrics["accuracy"] = accuracy_score(labels, predictions)
    try:
        metrics["auc_macro"] = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
    except:
        metrics["auc_macro"] = 0.0
    metrics["f1_macro"] = f1_score(labels, predictions, average="macro")

    # Per-class precision/recall
    for i, name in enumerate(label_names):
        metrics[f"precision_{name}"] = precision_score(labels, predictions, labels=[i], average="micro", zero_division=0)
        metrics[f"recall_{name}"] = recall_score(labels, predictions, labels=[i], average="micro", zero_division=0)

    # Binary CN vs MCI
    cn_emci_mask = (labels == 0) | (labels == 1)
    if cn_emci_mask.sum() > 0:
        cn_emci_labels = labels[cn_emci_mask]
        cn_emci_preds = predictions[cn_emci_mask]
        cn_emci_probs = probabilities[cn_emci_mask][:, [0, 1]]

        metrics["cn_vs_emci_acc"] = accuracy_score(cn_emci_labels, cn_emci_preds)
        try:
            metrics["cn_vs_emci_auc"] = roc_auc_score(cn_emci_labels, cn_emci_probs[:, 1])
        except:
            metrics["cn_vs_emci_auc"] = 0.0

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(labels, predictions).tolist()

    return metrics


def print_metrics(metrics, label_names=None):
    if label_names is None:
        label_names = ["CN", "MCI", "lMCI", "AD"]

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  AUC (macro):      {metrics['auc_macro']:.4f}")
    print(f"  F1 (macro):       {metrics['f1_macro']:.4f}")

    print("\n  Per-class precision / recall:")
    for i, name in enumerate(label_names):
        p = metrics.get(f"precision_{name}", 0)
        r = metrics.get(f"recall_{name}", 0)
        print(f"    {name:<6}: P={p:.3f}, R={r:.3f}")

    if "cn_vs_emci_acc" in metrics:
        print(f"\n  CN vs MCI binary:")
        print(f"    Accuracy: {metrics['cn_vs_emci_acc']:.4f}")
        print(f"    AUC:      {metrics['cn_vs_emci_auc']:.4f}")

    print("\n  Confusion matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"    Predicted?    CN   MCI  lMCI   AD")
    for i, name in enumerate(label_names):
        print(f"    True {name:<4}   {cm[i].tolist()}")
    print("=" * 60)