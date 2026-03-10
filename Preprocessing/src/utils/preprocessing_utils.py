"""
Common preprocessing utilities and helper functions.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy import signal
from scipy.ndimage import zoom
import warnings


def normalize_image(image: np.ndarray, 
                   method: str = "minmax",
                   axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize image to [0, 1] or [-1, 1] range.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    method : str
        Normalization method: 'minmax', 'zscore', 'robust'
    axis : Optional[int]
        Axis along which to normalize (None = global)
        
    Returns
    -------
    normalized : np.ndarray
        Normalized image
    """
    image = image.astype(np.float32)
    
    if method == "minmax":
        imin = np.min(image, axis=axis, keepdims=True)
        imax = np.max(image, axis=axis, keepdims=True)
        normalized = (image - imin) / (imax - imin + 1e-8)
        
    elif method == "zscore":
        mean = np.mean(image, axis=axis, keepdims=True)
        std = np.std(image, axis=axis, keepdims=True)
        normalized = (image - mean) / (std + 1e-8)
        
    elif method == "robust":
        # Robust scaling using quantiles
        q1 = np.percentile(image, 25, axis=axis, keepdims=True)
        q3 = np.percentile(image, 75, axis=axis, keepdims=True)
        median = np.median(image, axis=axis, keepdims=True)
        normalized = (image - median) / (q3 - q1 + 1e-8)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32)


def resample_image(image: np.ndarray,
                  source_spacing: Tuple[float, ...],
                  target_spacing: Tuple[float, ...],
                  order: int = 1) -> np.ndarray:
    """
    Resample image to new voxel spacing.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    source_spacing : Tuple[float, ...]
        Source voxel spacing (mm)
    target_spacing : Tuple[float, ...]
        Target voxel spacing (mm)
    order : int
        Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns
    -------
    resampled : np.ndarray
        Resampled image
    """
    # Calculate zoom factor for each dimension
    zoom_factors = np.array(source_spacing) / np.array(target_spacing)
    
    # Resample using scipy zoom
    resampled = zoom(image, zoom_factors, order=order, mode='constant', cval=0)
    
    return resampled.astype(np.float32)


def remove_outliers(image: np.ndarray, 
                   n_std: float = 3.0) -> np.ndarray:
    """
    Remove intensity outliers using z-score method.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    n_std : float
        Number of standard deviations for outlier threshold
        
    Returns
    -------
    cleaned : np.ndarray
        Image with outliers set to mean
    """
    mean = np.mean(image)
    std = np.std(image)
    threshold = n_std * std
    
    outlier_mask = np.abs(image - mean) > threshold
    cleaned = image.copy()
    cleaned[outlier_mask] = mean
    
    return cleaned


def despike(timeseries: np.ndarray, 
           threshold: float = 3.0) -> np.ndarray:
    """
    Remove spikes from fMRI timeseries.
    
    Uses median filter to detect and replace spikes.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series [N_voxels, T] or [T,]
    threshold : float
        Z-score threshold for spike detection
        
    Returns
    -------
    despiked : np.ndarray
        Despiked timeseries
    """
    if timeseries.ndim == 1:
        timeseries = timeseries[np.newaxis, :]
    
    despiked = timeseries.copy()
    
    for i in range(timeseries.shape[0]):
        ts = timeseries[i]
        
        # Use median filter for trend
        from scipy.ndimage import median_filter
        median = median_filter(ts, size=3)
        
        # Detect spikes
        residual = ts - median
        zscore = np.abs((residual - np.median(residual)) / (np.std(residual) + 1e-8))
        
        spike_mask = zscore > threshold
        
        # Replace spikes with interpolation
        if np.any(spike_mask):
            valid_idx = np.where(~spike_mask)[0]
            spike_idx = np.where(spike_mask)[0]
            
            for j in spike_idx:
                # Find nearest valid points
                nearest = valid_idx[np.abs(valid_idx - j).argmin()]
                despiked[i, j] = ts[nearest]
    
    return despiked.squeeze()


def apply_temporal_filter(timeseries: np.ndarray,
                         tr: float,
                         high_pass: float = 0.01,
                         low_pass: float = 0.1) -> np.ndarray:
    """
    Apply temporal bandpass filtering to fMRI timeseries.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series [N_voxels, T] or [T,]
    tr : float
        Repetition time in seconds
    high_pass : float
        High-pass cutoff in Hz
    low_pass : float
        Low-pass cutoff in Hz
        
    Returns
    -------
    filtered : np.ndarray
        Filtered timeseries
    """
    if timeseries.ndim == 1:
        timeseries = timeseries[np.newaxis, :]
    
    # Nyquist frequency
    nyquist = 1.0 / (2.0 * tr)
    
    # Normalize frequencies
    high_norm = high_pass / nyquist
    low_norm = low_pass / nyquist
    
    # Ensure valid range
    high_norm = np.clip(high_norm, 0.001, 0.999)
    low_norm = np.clip(low_norm, 0.001, 0.999)
    
    if high_norm >= low_norm:
        warnings.warn("High-pass frequency >= low-pass frequency. Using low-pass only.")
        low_norm = 0.999
        high_norm = 0.001
    
    # Design filter
    sos = signal.butter(4, [high_norm, low_norm], btype='band', output='sos')
    
    # Apply filter
    filtered = timeseries.copy()
    for i in range(timeseries.shape[0]):
        filtered[i] = signal.sosfilt(sos, timeseries[i])
    
    return filtered.squeeze()


def extract_timeseries_windows(timeseries: np.ndarray,
                              window_size: int,
                              window_step: int = 1,
                              standardize: bool = True,
                              dbgdgm_format: bool = True) -> np.ndarray:
    """
    Extract sliding windows from timeseries - DBGDGM-compatible output.
    
    Converts [N_ROI, T] to [N_samples, N_ROI, window_size]
    where N_samples = number of sliding windows.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Input timeseries [N_ROI, T]
    window_size : int
        Size of each window in TRs (typically 50 for DBGDGM)
    window_step : int
        Step size between windows (typically 1 for maximal overlap)
    standardize : bool
        Standardize each window independently (z-score normalization)
    dbgdgm_format : bool
        If True (default), return [N_samples, N_ROI, window_size]
        If False, return flattened [N_samples*N_ROI, window_size] (legacy)
        
    Returns
    -------
    windows : np.ndarray
        If dbgdgm_format=True: [N_samples, N_ROI, window_size]
                              Ready for DBGDGM fMRI encoder
        If dbgdgm_format=False: [N_samples*N_ROI, window_size] (legacy)
    """
    n_roi, n_timepoints = timeseries.shape
    
    # Extract windows: [N_ROI, N_windows, window_size]
    n_windows = (n_timepoints - window_size) // window_step + 1
    
    windows = np.zeros((n_roi, n_windows, window_size), dtype=np.float32)
    
    for w in range(n_windows):
        start = w * window_step
        end = start + window_size
        windows[:, w, :] = timeseries[:, start:end]
    
    # Standardize if requested
    if standardize:
        for w in range(n_windows):
            for r in range(n_roi):
                mean = np.mean(windows[r, w, :])
                std = np.std(windows[r, w, :])
                windows[r, w, :] = (windows[r, w, :] - mean) / (std + 1e-8)
    
    # Format output for DBGDGM
    if dbgdgm_format:
        # Transpose to [N_windows, N_ROI, window_size]
        windows = windows.transpose(1, 0, 2).astype(np.float32)
    else:
        # Legacy format: reshape to [N_ROI*N_windows, window_size]
        windows = windows.transpose(0, 1, 2).reshape(n_roi * n_windows, window_size)
    
    return windows.astype(np.float32)


def motion_scrubbing(timeseries: np.ndarray,
                    motion_params: np.ndarray,
                    threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove timepoints with excessive motion (motion scrubbing).
    
    Parameters
    ----------
    timeseries : np.ndarray
        fMRI timeseries [N_ROI, T]
    motion_params : np.ndarray
        Motion parameters [T, 6] (3 rotation + 3 translation)
    threshold : float
        Motion threshold in mm
        
    Returns
    -------
    scrubbed_ts : np.ndarray
        Scrubbed timeseries
    valid_mask : np.ndarray
        Boolean mask of valid timepoints
    """
    # Calculate frame-wise displacement (FD)
    # FD = sqrt(Δx^2 + Δy^2 + Δz^2 + (radius * Δα)^2 + (radius * Δβ)^2 + (radius * Δγ)^2)
    
    # Typical brain radius
    radius = 50  # mm
    
    # Calculate derivatives
    motion_deriv = np.diff(motion_params, axis=0, prepend=motion_params[0:1])
    
    # Convert rotations to distance
    rot_dist = radius * np.abs(motion_deriv[:, 3:6])
    trans_dist = np.abs(motion_deriv[:, 0:3])
    
    # Calculate FD
    fd = np.sum(np.concatenate([trans_dist, rot_dist], axis=1), axis=1)
    
    # Create mask
    valid_mask = fd < threshold
    
    # Scrub timeseries
    scrubbed_ts = timeseries[:, valid_mask]
    
    return scrubbed_ts, valid_mask


def standardize_timeseries(timeseries: np.ndarray) -> np.ndarray:
    """
    Z-score standardize timeseries.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Input timeseries [N, T]
        
    Returns
    -------
    standardized : np.ndarray
        Standardized timeseries
    """
    mean = np.mean(timeseries, axis=1, keepdims=True)
    std = np.std(timeseries, axis=1, keepdims=True)
    
    return (timeseries - mean) / (std + 1e-8)
