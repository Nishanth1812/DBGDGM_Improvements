import numpy as np
import torch
from torch_geometric.data import Data


def build_fmri_graphs(fmri_time_series: np.ndarray, window_size: int = 30, window_stride: int = 5, top_edge_percent: float = 0.20):
    """
    Convert fMRI time series (N_roi ? T) into a sequence of graph snapshots.

    Args:
        fmri_time_series: (N_roi, T) ROI time series
        window_size: sliding window length
        window_stride: step between windows
        top_edge_percent: fraction of edges to keep (top by absolute correlation)

    Returns:
        List of PyG Data objects, one per window
    """
    n_rois, n_times = fmri_time_series.shape
    graphs = []

    for start in range(0, n_times - window_size + 1, window_stride):
        window = fmri_time_series[:, start:start + window_size]

        # Pearson correlation matrix
        corr = np.corrcoef(window)
        np.fill_diagonal(corr, 0.0)

        abs_corr = np.abs(corr)
        threshold = np.percentile(abs_corr, 100 * (1 - top_edge_percent))
        mask = abs_corr >= threshold

        # Build edge index
        edge_index = np.argwhere(mask)
        edge_weight = corr[mask]

        # Node features: mean activation in window
        node_features = window.mean(axis=1, keepdims=True).astype(np.float32)

        data = Data(
            x=torch.from_numpy(node_features),
            edge_index=torch.from_numpy(edge_index.T).long(),
            edge_attr=torch.from_numpy(edge_weight).float(),
        )
        graphs.append(data)

    return graphs


def fmri_to_graphs(fmri_batch: torch.Tensor, window_size: int = 30, window_stride: int = 5, top_edge_percent: float = 0.20):
    """
    Process a batch of fMRI time series (B ? N_roi ? T) into list of graph lists.
    Returns list of length B, each containing K graph snapshots.
    """
    batch_size = fmri_batch.size(0)
    batch_graphs = []

    for b in range(batch_size):
        ts = fmri_batch[b].numpy()
        graphs = build_fmri_graphs(ts, window_size, window_stride, top_edge_percent)
        batch_graphs.append(graphs)

    return batch_graphs