import matplotlib.pyplot as plt
import numpy as np


def plot_uncertainty_distributions(uncertainty_by_class, save_path=None):
    """
    Box plot of ?? distribution per diagnostic class.

    Args:
        uncertainty_by_class: dict {class_name: [unc_values]}
        save_path: path to save PNG
    """
    class_names = ["CN", "MCI", "AD"]
    data = [uncertainty_by_class.get(name, []) for name in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    bp = axes[0].boxplot(data, labels=class_names, patch_artist=True)
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Uncertainty ??")
    axes[0].set_title("Uncertainty Distribution by Diagnostic Class")
    axes[0].grid(axis="y", alpha=0.3)

    # Accuracy split by uncertainty
    all_data = [d for d in data if len(d) > 0]
    if all_data:
        median_unc = np.median(np.concatenate(all_data))
    else:
        median_unc = 0.5

    acc_low = 0.85  # placeholder
    acc_high = 0.65  # placeholder

    x = np.arange(2)
    bars = axes[1].bar(x, [acc_low, acc_high], color=["steelblue", "coral"], alpha=0.8, width=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Low ?? (< median)", "High ?? (> median)"])
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs. Uncertainty")
    axes[1].set_ylim(0, 1)
    axes[1].bar_label(bars, fmt="%.2f")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()


def plot_reliability_diagram(probs, labels, n_bins=10, save_path=None):
    """
    Reliability diagram: binned confidence vs. accuracy.

    Args:
        probs: (N, 4) predicted probabilities for true class
        labels: (N,) true labels
        n_bins: number of bins
        save_path: path to save PNG
    """
    class_names = ["CN", "MCI", "AD"]
    pred_class = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    accuracies = (pred_class == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(bin_confs, bin_accs, s=80, c="steelblue", zorder=3)
    ax.plot(bin_confs, bin_accs, "steelblue", linewidth=2)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()