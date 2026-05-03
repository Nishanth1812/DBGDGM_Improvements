import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_latent_space(mu_vectors, labels, uncertainty=None, save_path=None):
    """
    t-SNE projection of latent ? vectors coloured by diagnostic label.

    Args:
        mu_vectors: (N, 128) latent means
        labels: (N,) int labels (0=CN, 1=MCI, 2=lMCI, 3=AD)
        uncertainty: (N,) scalar uncertainty ??, or None
        save_path: path to save PNG
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(mu_vectors) - 1))
    mu_2d = tsne.fit_transform(mu_vectors)

    class_names = ["CN", "MCI", "AD"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    markers = ["o", "s", "D"]

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = labels == i
        if uncertainty is not None:
            sizes = 30 + 100 * (uncertainty[mask] / uncertainty.max())
        else:
            sizes = 60
        ax.scatter(mu_2d[mask, 0], mu_2d[mask, 1], c=color, label=name, s=sizes, marker=marker, alpha=0.7, edgecolors="white")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Latent Space (?) ? MM-DBGDGM")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()