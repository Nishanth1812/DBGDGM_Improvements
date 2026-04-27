import matplotlib.pyplot as plt
import numpy as np


def plot_attention_maps(attention_weights, region_names=None, top_n=15, save_path=None):
    """
    Bar chart of top-15 brain regions by cross-attention weight (fs and sf branches).

    Args:
        attention_weights: dict with 'fs' and 'sf' keys, each (N_rois,)
        region_names: list of length N_rois, or None to use indices
        top_n: number of top regions to show
        save_path: path to save PNG
    """
    if region_names is None:
        region_names = [f"R{i}" for i in range(len(attention_weights["fs"]))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, branch in zip(axes, ["fs", "sf"]):
        weights = attention_weights[branch][:top_n]
        names = region_names[:top_n]

        colors = ["steelblue"] * len(names)
        ax.barh(range(len(names)), weights, color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Attention weight")
        ax.set_title(f"Cross-Attention: fMRI ? sMRI (branch={branch})" if branch == "fs" else f"sMRI ? fMRI (branch={branch})")
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()