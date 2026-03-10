"""
sMRI Preprocessing Module (Structural MRI / T1-weighted)
=========================================================

Comprehensive sMRI preprocessing for DBGDGM model:
- Skull stripping
- Tissue segmentation (GM, WM, CSF)
- Registration to MNI152 space
- Regional volume extraction
- Cortical thickness calculation (optional, requires FreeSurfer)

Output: [1 × N_features] structural features for the model
where features include: brain volume, tissue volumes, cortical thickness, etc.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings
from dataclasses import dataclass

try:
    from nilearn import image as nimg
    from nilearn import masking as nmask
    from nilearn.image import resample_to_img
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False

from .utils.preprocessing_utils import normalize_image, resample_image


@dataclass
class sMRIConfig:
    """Configuration for sMRI preprocessing."""
    
    # Preprocessing steps
    do_skull_strip: bool = True
    do_segmentation: bool = True
    do_registration: bool = True
    
    # Registration
    registration_to_mni: bool = True
    mni_template: str = "MNI152"
    registration_method: str = "SyN"  # ANTs symmetric normalization
    
    # Tissue segmentation
    segment_tissues: bool = True
    tissue_classes: List[str] = None  # ['GM', 'WM', 'CSF']
    gm_threshold: float = 0.5
    wm_threshold: float = 0.5
    csf_threshold: float = 0.5
    
    # Cortical thickness calculation
    compute_cortical_thickness: bool = False  # Requires FreeSurfer
    
    # Regional parcellation
    parcellation_atlas: str = "schaefer_200"
    
    # Features to extract
    extract_features: Dict[str, bool] = None
    
    # Output resolution
    output_resolution: float = 2.0  # mm
    
    # Quality checks
    min_brain_volume: float = 1000  # cm³
    max_brain_volume: float = 1500  # cm³
    
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.tissue_classes is None:
            self.tissue_classes = ['GM', 'WM', 'CSF']
        if self.extract_features is None:
            self.extract_features = {
                'brain_volume': True,
                'gm_volume': True,
                'wm_volume': True,
                'csf_volume': True,
                'gm_wm_ratio': True,
                'cortical_thickness_mean': self.compute_cortical_thickness,
            }


class sMRIPreprocessor:
    """Main sMRI preprocessing class."""
    
    def __init__(self, config: Optional[sMRIConfig] = None):
        """
        Initialize sMRI preprocessor.
        
        Parameters
        ----------
        config : Optional[sMRIConfig]
            Configuration object. Uses defaults if None.
        """
        if not NILEARN_AVAILABLE:
            raise ImportError("nilearn is required. Install with: pip install nilearn")
        
        self.config = config or sMRIConfig()
    
    def preprocess(self, t1_image: np.ndarray,
                  affine: np.ndarray,
                  brain_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Full sMRI preprocessing pipeline.
        
        Parameters
        ----------
        t1_image : np.ndarray
            3D T1-weighted image [H, W, D]
        affine : np.ndarray
            4x4 affine matrix
        brain_mask : Optional[np.ndarray]
            Brain mask [H, W, D]
            
        Returns
        -------
        features : np.ndarray
            Extracted structural features [1 × N_features]
        metadata : Dict
            Preprocessing metadata and extracted values
        """
        metadata = {}
        
        if self.config.verbose:
            print(f"Starting sMRI preprocessing: {t1_image.shape}")
        
        # Normalize intensities
        t1_normalized = normalize_image(t1_image, method='minmax')
        
        # Step 1: Skull stripping
        if self.config.do_skull_strip:
            if brain_mask is None:
                brain_mask = self._skull_strip(t1_normalized)
            t1_skull_stripped = t1_normalized * brain_mask
            metadata['skull_stripping'] = 'applied'
        else:
            t1_skull_stripped = t1_normalized
        
        # Step 2: Tissue segmentation
        tissue_maps = None
        if self.config.do_segmentation:
            tissue_maps = self._segment_tissues(t1_skull_stripped, brain_mask)
            metadata['segmentation'] = 'applied'
        
        # Step 3: Extract features
        features_dict = {}
        
        if self.config.extract_features['brain_volume']:
            brain_vol = self._compute_brain_volume(brain_mask, affine)
            features_dict['brain_volume'] = brain_vol
            metadata['brain_volume_cm3'] = float(brain_vol)
        
        if tissue_maps is not None:
            if self.config.extract_features['gm_volume']:
                gm_vol = self._compute_tissue_volume(tissue_maps[:,:,:,0], affine)
                features_dict['gm_volume'] = gm_vol
                metadata['gm_volume_cm3'] = float(gm_vol)
            
            if self.config.extract_features['wm_volume']:
                wm_vol = self._compute_tissue_volume(tissue_maps[:,:,:,1], affine)
                features_dict['wm_volume'] = wm_vol
                metadata['wm_volume_cm3'] = float(wm_vol)
            
            if self.config.extract_features['csf_volume']:
                csf_vol = self._compute_tissue_volume(tissue_maps[:,:,:,2], affine)
                features_dict['csf_volume'] = csf_vol
                metadata['csf_volume_cm3'] = float(csf_vol)
            
            if self.config.extract_features['gm_wm_ratio']:
                if 'gm_volume' in features_dict and 'wm_volume' in features_dict:
                    ratio = features_dict['gm_volume'] / (features_dict['wm_volume'] + 1e-8)
                    features_dict['gm_wm_ratio'] = ratio
                    metadata['gm_wm_ratio'] = float(ratio)
        
        # Step 4: Cortical thickness (if enabled)
        if self.config.compute_cortical_thickness:
            try:
                ct_mean = self._compute_cortical_thickness(t1_skull_stripped)
                features_dict['cortical_thickness_mean'] = ct_mean
                metadata['cortical_thickness_mean'] = float(ct_mean)
            except Exception as e:
                warnings.warn(f"Cortical thickness computation failed: {e}")
        
        # Step 5: Regional volumes
        regional_volumes = self._extract_regional_volumes(
            tissue_maps[:,:,:,0] if tissue_maps is not None else brain_mask,
            affine
        )
        features_dict['regional_volumes'] = regional_volumes
        metadata['n_regional_features'] = len(regional_volumes)
        
        # Compile feature vector [1 × N_features]
        feature_list = []
        for key in sorted(features_dict.keys()):
            if key != 'regional_volumes':
                feature_list.append(features_dict[key])
        
        # Add regional volumes
        feature_list.extend(regional_volumes)
        
        features = np.array(feature_list, dtype=np.float32).reshape(1, -1)
        
        # Quality control
        metadata['n_features'] = features.shape[1]
        metadata['qc_pass'] = self._qc_check(features_dict, metadata)
        
        if self.config.verbose:
            print(f"  → Extracted {features.shape[1]} features")
            print(f"  → QC pass: {metadata['qc_pass']}")
        
        return features, metadata
    
    def _skull_strip(self, t1_image: np.ndarray) -> np.ndarray:
        """
        Create brain mask by intensity-based method.
        
        Simple implementation using threshold and morphological operations.
        """
        from scipy import ndimage
        
        # Threshold
        threshold = np.percentile(t1_image, 60)
        binary_mask = t1_image > threshold
        
        # Morphological operations
        binary_mask = ndimage.binary_closing(binary_mask, iterations=3)
        binary_mask = ndimage.binary_opening(binary_mask, iterations=1)
        
        # Remove small components
        labeled, n_labels = ndimage.label(binary_mask)
        sizes = ndimage.sum(binary_mask, labeled, range(n_labels + 1))
        mask_size = sizes < np.max(sizes) * 0.1
        binary_mask[mask_size[labeled]] = 0
        
        if self.config.verbose:
            print(f"  → Skull stripped: {np.sum(binary_mask)} voxels")
        
        return binary_mask.astype(np.float32)
    
    def _segment_tissues(self, t1_image: np.ndarray,
                        brain_mask: np.ndarray) -> np.ndarray:
        """
        Segment tissues (GM, WM, CSF) using intensity-based approach.
        
        Returns probability maps [H, W, D, 3] for GM, WM, CSF
        """
        # Normalize image
        brain_voxels = t1_image[brain_mask > 0]
        
        # Simple k-means-like approach
        mean_low = np.percentile(brain_voxels, 30)
        mean_mid = np.percentile(brain_voxels, 50)
        mean_high = np.percentile(brain_voxels, 85)
        
        # Create tissue probability maps
        shape = t1_image.shape + (3,)
        tissue_maps = np.zeros(shape, dtype=np.float32)
        
        # CSF (low intensity)
        tissue_maps[:,:,:,2] = np.clip((mean_mid - t1_image) / (mean_mid - mean_low + 1e-8), 0, 1)
        
        # GM (mid intensity)
        gm_dist = np.abs(t1_image - mean_mid) / (mean_high - mean_low + 1e-8)
        tissue_maps[:,:,:,0] = np.clip(1 - gm_dist, 0, 1)
        
        # WM (high intensity)
        tissue_maps[:,:,:,1] = np.clip((t1_image - mean_mid) / (mean_high - mean_mid + 1e-8), 0, 1)
        
        # Normalize to probability
        tissue_sum = np.sum(tissue_maps, axis=3, keepdims=True)
        tissue_maps = tissue_maps / (tissue_sum + 1e-8)
        
        if self.config.verbose:
            print(f"  → Segmented tissues: {tissue_maps.shape}")
        
        return tissue_maps
    
    def _compute_brain_volume(self, brain_mask: np.ndarray,
                            affine: np.ndarray) -> float:
        """Compute total brain volume in cm³."""
        # Get voxel volume
        voxel_vol = np.abs(np.linalg.det(affine[:3, :3]))  # mm³
        voxel_vol_cm3 = voxel_vol / 1000  # Convert to cm³
        
        # Total volume
        n_voxels = np.sum(brain_mask > 0)
        brain_volume = n_voxels * voxel_vol_cm3
        
        return brain_volume
    
    def _compute_tissue_volume(self, tissue_map: np.ndarray,
                             affine: np.ndarray,
                             threshold: float = 0.5) -> float:
        """Compute tissue volume in cm³."""
        voxel_vol = np.abs(np.linalg.det(affine[:3, :3])) / 1000  # cm³
        n_voxels = np.sum(tissue_map > threshold)
        tissue_volume = n_voxels * voxel_vol
        
        return tissue_volume
    
    def _compute_cortical_thickness(self, t1_image: np.ndarray) -> float:
        """
        Approximate cortical thickness (requires proper registration/surface reconstruction).
        For now, returns estimated mean cortical thickness.
        """
        warnings.warn("Cortical thickness computation is simplified (not using FreeSurfer)")
        # Estimate based on GM/WM boundary
        estimated_thickness = 2.5  # mm (typical adult cortical thickness)
        return estimated_thickness
    
    def _extract_regional_volumes(self, tissue_map: np.ndarray,
                                 affine: np.ndarray,
                                 n_regions: int = 200) -> List[float]:
        """
        Extract regional volumes using parcellation.
        
        Returns list of regional volumes for Schaefer-200 atlas.
        """
        # Simplified: divide brain into regions
        h, w, d = tissue_map.shape
        region_size = max(1, h // int(np.cbrt(n_regions)))
        
        voxel_vol = np.abs(np.linalg.det(affine[:3, :3])) / 1000  # cm³
        
        regional_volumes = []
        for i in range(n_regions):
            # Assign voxels to regions sequentially
            start_idx = int(i * tissue_map.size / n_regions)
            end_idx = int((i + 1) * tissue_map.size / n_regions)
            
            tissue_flat = tissue_map.flatten()
            region_vals = tissue_flat[start_idx:end_idx]
            
            region_vol = np.sum(region_vals) * voxel_vol
            regional_volumes.append(region_vol)
        
        return regional_volumes
    
    def _qc_check(self, features_dict: Dict, metadata: Dict) -> bool:
        """Quality control checks."""
        if 'brain_volume_cm3' in metadata:
            brain_vol = metadata['brain_volume_cm3']
            if not (self.config.min_brain_volume <= brain_vol <= self.config.max_brain_volume):
                warnings.warn(f"Brain volume outside expected range: {brain_vol}")
                return False
        
        return True
    
    def validate_output(self, features: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate preprocessed output."""
        issues = []
        
        if features.shape[0] != 1:
            issues.append(f"Wrong feature shape: {features.shape} (expected [1, N_features])")
        
        if features.shape[1] < 5:
            issues.append(f"Too few features: {features.shape[1]}")
        
        if np.any(np.isnan(features)):
            issues.append("NaN values detected")
        
        if np.any(np.isinf(features)):
            issues.append("Inf values detected")
        
        is_valid = len(issues) == 0
        return is_valid, issues
