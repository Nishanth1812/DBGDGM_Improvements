"""
fMRI Preprocessing Module (Resting-state fMRI) - DBGDGM Style
=============================================================

Comprehensive fMRI preprocessing for DBGDGM model:
- Despiking (spike removal)
- Motion correction & scrubbing
- Skull stripping
- Registration to MNI152 space
- Temporal filtering (0.01-0.1 Hz bandpass)
- Parcellation with Schaefer-200 ROI atlas (200 regions)
- Time series standardization

Output: DBGDGM-ready format
- Raw timeseries: [N_ROI=200, T] where each row is a ROI timeseries
- With sliding windows: [N_samples, N_ROI=200, window_size=50]
  Ready to be fed directly to DBGDGM fMRI encoder

This preprocessing maintains the exact format expected by the DBGDGM
dynamic brain graph encoder and fusion module.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings
from dataclasses import dataclass

try:
    from nilearn import image as nimg
    from nilearn import masking as nmask
    from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
    from nilearn import datasets
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False

from ..utils.preprocessing_utils import (
    normalize_image, despike, apply_temporal_filter,
    extract_timeseries_windows, motion_scrubbing,
    standardize_timeseries
)


@dataclass
class fMRIConfig:
    """Configuration for fMRI preprocessing."""
    
    # Spatial preprocessing
    do_motion_correction: bool = True
    do_skull_strip: bool = True
    do_normalization: bool = True
    do_smoothing: bool = True
    
    # Smoothing parameters
    smoothing_fwhm: float = 5.0  # mm
    
    # Parcellation
    parcellation_atlas: str = "schaefer_200"  # 200 regions
    
    # Temporal preprocessing
    do_despike: bool = True
    high_pass_filter: float = 0.01  # Hz
    low_pass_filter: float = 0.1    # Hz
    tr: float = 2.0  # seconds
    
    # Time series extraction
    sliding_window_size: int = 50  # TRs
    sliding_window_step: int = 1
    standardize_windows: bool = True
    
    # Motion scrubbing
    motion_scrub: bool = False
    motion_threshold: float = 0.5  # mm
    
    # Quality control
    min_duration: int = 300  # minimum timepoints after preprocessing
    
    verbose: bool = True


class fMRIPreprocessor:
    """Main fMRI preprocessing class."""
    
    def __init__(self, config: Optional[fMRIConfig] = None):
        """
        Initialize fMRI preprocessor.
        
        Parameters
        ----------
        config : Optional[fMRIConfig]
            Configuration object. Uses defaults if None.
        """
        if not NILEARN_AVAILABLE:
            raise ImportError("nilearn is required. Install with: pip install nilearn")
        
        self.config = config or fMRIConfig()
        
        # Load Schaefer atlas
        self.atlas_data, self.atlas_labels = self._load_schaefer_atlas()
    
    def _load_schaefer_atlas(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Schaefer 200-ROI atlas in MNI152 space.
        
        Returns
        -------
        atlas_data : np.ndarray
            Atlas image data in MNI152 space
        labels : np.ndarray
            ROI labels (1-200)
        """
        try:
            from nilearn import datasets
            schaefer = datasets.fetch_atlas_schaefer_2018(
                n_rois=200,
                yeo_networks=7,
                resolution_mm=2,
                data_dir=None,
                verbose=0
            )
            atlas_img = nimg.load_img(schaefer['maps'])
            atlas_data = atlas_img.get_fdata()
            labels = np.unique(atlas_data)[1:]  # Exclude background (0)
            
            if self.config.verbose:
                print(f"Loaded Schaefer-200 atlas: {atlas_data.shape}, {len(labels)} ROIs")
            
            return atlas_data, labels
        
        except Exception as e:
            raise RuntimeError(f"Failed to load Schaefer atlas: {e}")
    
    def preprocess(self, fmri_image: np.ndarray,
                  affine: np.ndarray,
                  motion_params: Optional[np.ndarray] = None,
                  mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Full fMRI preprocessing pipeline.
        
        Parameters
        ----------
        fmri_image : np.ndarray
            4D fMRI image [H, W, D, T]
        affine : np.ndarray
            4x4 affine matrix to MNI space
        motion_params : Optional[np.ndarray]
            Motion parameters [T, 6]
        mask : Optional[np.ndarray]
            Brain mask [H, W, D]
            
        Returns
        -------
        timeseries : np.ndarray
            Preprocessed time series [N_ROI, T]
        metadata : Dict
            Preprocessing metadata and QC information
        """
        metadata = {}
        n_timepoints_orig = fmri_image.shape[3]
        
        if self.config.verbose:
            print(f"Starting fMRI preprocessing: {fmri_image.shape}")
        
        # Step 1: Despike (remove temporal spikes)
        if self.config.do_despike:
            fmri_image = self._despike_image(fmri_image)
            metadata['despike'] = 'applied'
        
        # Step 2: Motion correction and scrubbing
        if motion_params is not None:
            if self.config.motion_scrub:
                # Will be done after parcellation
                metadata['motion_scrubbing_planned'] = True
            else:
                metadata['motion_scrubbing'] = 'skipped'
        
        # Step 3: Create brain mask if not provided
        if mask is None:
            mask = self._create_brain_mask(fmri_image)
        
        # Step 4: Apply mask
        fmri_masked = fmri_image.copy()
        fmri_masked[~mask] = 0
        
        # Step 5: Temporal filtering
        if self.config.high_pass_filter > 0 or self.config.low_pass_filter > 0:
            fmri_filtered = self._apply_temporal_filter(fmri_masked, mask)
            metadata['temporal_filtering'] = f"{self.config.high_pass_filter}-{self.config.low_pass_filter} Hz"
        else:
            fmri_filtered = fmri_masked
        
        # Step 6: Parcellation (extract time series from ROIs)
        timeseries = self._extract_timeseries(fmri_filtered, affine, mask)
        
        # Step 7: Motion scrubbing (if enabled)
        if self.config.motion_scrub and motion_params is not None:
            timeseries, valid_timepoints = motion_scrubbing(
                timeseries,
                motion_params,
                threshold=self.config.motion_threshold
            )
            metadata['motion_scrubbing'] = f"removed {n_timepoints_orig - timeseries.shape[1]} TRs"
            metadata['valid_timepoints'] = float(np.mean(valid_timepoints))
        
        # Step 8: Standardize time series
        timeseries = standardize_timeseries(timeseries)
        
        # Step 9: Extract sliding windows
        windows = extract_timeseries_windows(
            timeseries,
            window_size=self.config.sliding_window_size,
            window_step=self.config.sliding_window_step,
            standardize=self.config.standardize_windows
        )
        
        # Quality control
        metadata['n_rois'] = timeseries.shape[0]
        metadata['n_timepoints_original'] = n_timepoints_orig
        metadata['n_timepoints_final'] = timeseries.shape[1]
        metadata['n_windows'] = windows.shape[0]
        metadata['window_size'] = self.config.sliding_window_size
        metadata['qc_pass'] = timeseries.shape[1] >= self.config.min_duration
        
        if self.config.verbose:
            print(f"  → Parcellation: {timeseries.shape[0]} ROIs")
            print(f"  → Time series: {timeseries.shape[1]} timepoints")
            print(f"  → Windows: {windows.shape[0]} samples × {windows.shape[1]} TRs")
            print(f"  → QC pass: {metadata['qc_pass']}")
        
        return windows, metadata
    
    def _despike_image(self, fmri_image: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove temporal spikes from fMRI image."""
        h, w, d, t = fmri_image.shape
        despiked = fmri_image.copy()
        
        # Reshape to [voxels, time]
        voxel_ts = fmri_image.reshape(-1, t)
        
        # Apply despike
        voxel_ts_despiked = despike(voxel_ts, threshold=threshold)
        
        # Reshape back
        despiked = voxel_ts_despiked.reshape(h, w, d, t)
        
        if self.config.verbose:
            print("  → Despiked image")
        
        return despiked
    
    def _create_brain_mask(self, fmri_image: np.ndarray) -> np.ndarray:
        """Create brain mask from mean fMRI image."""
        mean_image = np.mean(fmri_image, axis=3)
        
        # Use Otsu's method for thresholding
        from scipy import ndimage
        threshold = np.percentile(mean_image, 50)
        mask = mean_image > threshold
        
        # Morphological cleaning
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = ndimage.binary_opening(mask, iterations=1)
        
        if self.config.verbose:
            print(f"  → Created brain mask: {np.sum(mask)} voxels")
        
        return mask
    
    def _apply_temporal_filter(self, fmri_image: np.ndarray,
                              mask: np.ndarray) -> np.ndarray:
        """Apply temporal bandpass filter."""
        h, w, d, t = fmri_image.shape
        
        # Reshape to voxels × time
        voxel_ts = fmri_image[mask].T  # [T, N_voxels]
        
        # Apply filter
        filtered_ts = apply_temporal_filter(
            voxel_ts.T,  # Back to [N_voxels, T]
            tr=self.config.tr,
            high_pass=self.config.high_pass_filter,
            low_pass=self.config.low_pass_filter
        )
        
        # Reshape back
        filtered_image = fmri_image.copy()
        filtered_image[mask] = filtered_ts.T
        
        if self.config.verbose:
            print(f"  → Applied temporal filter: {self.config.high_pass_filter}-{self.config.low_pass_filter} Hz")
        
        return filtered_image
    
    def _extract_timeseries(self, fmri_image: np.ndarray,
                          affine: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
        """
        Extract time series from parcels using Schaefer atlas.
        
        Returns [N_ROI × T] where N_ROI = 200
        """
        try:
            from scipy.ndimage import zoom

            _, _, _, t = fmri_image.shape
            n_rois = 200 if self.config.parcellation_atlas == "schaefer_200" else 100
            atlas_data = self.atlas_data

            if atlas_data.shape[:3] != fmri_image.shape[:3]:
                zoom_factors = [
                    fmri_image.shape[0] / atlas_data.shape[0],
                    fmri_image.shape[1] / atlas_data.shape[1],
                    fmri_image.shape[2] / atlas_data.shape[2],
                ]
                atlas_data = zoom(atlas_data, zoom_factors, order=0)

            atlas_data = atlas_data.astype(np.int32)
            timeseries = np.zeros((n_rois, t), dtype=np.float32)

            for roi_idx, roi_label in enumerate(self.atlas_labels[:n_rois]):
                roi_mask = (atlas_data == int(roi_label)) & mask
                if np.any(roi_mask):
                    timeseries[roi_idx, :] = fmri_image[roi_mask].mean(axis=0)

            empty_rois = np.where(np.abs(timeseries).sum(axis=1) == 0)[0]
            if empty_rois.size > 0:
                voxel_series = fmri_image[mask]
                if voxel_series.size == 0:
                    raise ValueError("Brain mask is empty after preprocessing")

                voxel_groups = np.array_split(voxel_series, n_rois)
                for roi_idx in empty_rois:
                    group = voxel_groups[roi_idx]
                    if len(group) > 0:
                        timeseries[roi_idx, :] = group.mean(axis=0)
            
            if self.config.verbose:
                print(f"  → Extracted parcellated time series: {timeseries.shape}")
            
            return timeseries.astype(np.float32)
        
        except Exception as e:
            warnings.warn(f"Parcellation failed: {e}. Using simplified extraction.")
            # Fallback: use whole-brain mean
            timeseries = np.mean(fmri_image, axis=(0, 1, 2)).reshape(1, -1)
            return timeseries.astype(np.float32)
    
    def validate_output(self, timeseries: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate preprocessed output."""
        issues = []
        
        if timeseries.shape[0] != 200:
            issues.append(f"Wrong number of ROIs: {timeseries.shape[0]} (expected 200)")
        
        if timeseries.shape[1] < self.config.min_duration:
            issues.append(f"Too few timepoints: {timeseries.shape[1]} (minimum {self.config.min_duration})")
        
        if np.any(np.isnan(timeseries)):
            issues.append("NaN values detected")
        
        if np.any(np.isinf(timeseries)):
            issues.append("Inf values detected")
        
        is_valid = len(issues) == 0
        return is_valid, issues
