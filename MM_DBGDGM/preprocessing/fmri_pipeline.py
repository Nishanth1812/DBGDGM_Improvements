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

    # If the time series is shorter than window_size, use the full series as one window
    actual_window = min(window_size, n_times)
    graphs = []

    for start in range(0, n_times - actual_window + 1, window_stride):
        window = fmri_time_series[:, start:start + actual_window]

        # Pearson correlation matrix; replace NaN (constant signal) with 0
        corr = np.corrcoef(window)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)

        abs_corr = np.abs(corr)
        threshold = np.percentile(abs_corr, 100 * (1 - top_edge_percent))
        mask = abs_corr >= threshold

        # Build edge index
        src, dst = np.where(mask)
        if len(src) == 0:
            # Ensure at least a self-loop to prevent empty graph crash
            src = np.arange(n_rois)
            dst = np.arange(n_rois)
        edge_index = np.stack([src, dst], axis=0)
        edge_weight = corr[mask] if corr[mask].size > 0 else np.zeros(len(src), dtype=np.float32)

        # Node features: mean activation in window
        node_features = window.mean(axis=1, keepdims=True).astype(np.float32)

        data = Data(
            x=torch.from_numpy(node_features),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_weight.astype(np.float32)),
        )
        graphs.append(data)

    # Guarantee at least one graph even for very short time series
    if len(graphs) == 0:
        node_features = fmri_time_series.mean(axis=1, keepdims=True).astype(np.float32)
        graphs.append(Data(
            x=torch.from_numpy(node_features),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros(0),
        ))

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