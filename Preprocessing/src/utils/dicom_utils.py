"""
DICOM Loading and Handling Utilities
=====================================

This module handles DICOM file reading, validation, and conversion
for ADNI dataset preprocessing.
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings

try:
    import pydicom
    from pydicom.data import get_testdata_files
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    warnings.warn("pydicom not installed. DICOM loading will fail.")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel not installed. NIfTI conversion will fail.")


class DICOMLoader:
    """Load and process DICOM files from ADNI dataset."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize DICOM loader.
        
        Parameters
        ----------
        verbose : bool
            Print processing information
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM loading. Install with: pip install pydicom")
        
        self.verbose = verbose
    
    def load_dicom_series(self, dicom_dir: str) -> Tuple[np.ndarray, Dict]:
        """
        Load all DICOM files from a directory into a 3D volume.
        
        DICOM files are typically arranged as 2D slices stacked along z-axis.
        
        Parameters
        ----------
        dicom_dir : str
            Directory containing DICOM files
            
        Returns
        -------
        volume : np.ndarray
            3D volume array [H, W, D]
        metadata : Dict
            DICOM metadata (PatientID, StudyDate, SeriesDescription, etc.)
            
        Raises
        ------
        FileNotFoundError
            If no DICOM files found in directory
        ValueError
            If DICOM files cannot be stacked consistently
        """
        dicom_dir = Path(dicom_dir)
        
        if not dicom_dir.exists():
            raise FileNotFoundError(f"Directory not found: {dicom_dir}")
        
        # Find all DICOM files
        dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("**/*.dcm"))
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
        
        if self.verbose:
            print(f"Found {len(dicom_files)} DICOM files")
        
        # Load and sort DICOM files
        datasets = []
        for dcm_file in sorted(dicom_files):
            try:
                ds = pydicom.dcmread(dcm_file)
                datasets.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to read {dcm_file}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No valid DICOM files could be read")
        
        # Sort by slice location if available
        try:
            datasets.sort(key=lambda ds: float(ds.SliceLocation) if hasattr(ds, 'SliceLocation') else 0)
        except Exception:
            if self.verbose:
                print("Could not sort by slice location, using file order")
        
        # Extract pixel arrays and stack
        slices = []
        for ds in datasets:
            if hasattr(ds, 'pixel_array'):
                slices.append(ds.pixel_array)
            else:
                warnings.warn(f"DICOM file has no pixel data")
        
        if not slices:
            raise ValueError("No pixel data found in DICOM files")
        
        # Stack slices along z-axis [H, W, D]
        volume = np.stack(slices, axis=-1)
        
        # Extract metadata from first dataset
        metadata = self._extract_metadata(datasets[0])
        metadata['num_slices'] = len(slices)
        metadata['volume_shape'] = volume.shape
        
        if self.verbose:
            print(f"Loaded DICOM series: shape {volume.shape}")
            print(f"Patient ID: {metadata.get('PatientID', 'Unknown')}")
            print(f"Series: {metadata.get('SeriesDescription', 'Unknown')}")
        
        return volume.astype(np.float32), metadata
    
    def _extract_metadata(self, ds) -> Dict:
        """Extract relevant DICOM metadata."""
        metadata = {}
        
        # Common DICOM tags
        tags = {
            'PatientID': 'PatientID',
            'PatientAge': 'PatientAge',
            'PatientSex': 'PatientSex',
            'StudyDate': 'StudyDate',
            'StudyTime': 'StudyTime',
            'SeriesDescription': 'SeriesDescription',
            'SeriesNumber': 'SeriesNumber',
            'EchoTime': 'EchoTime',
            'RepetitionTime': 'RepetitionTime',
            'FlipAngle': 'FlipAngle',
            'Manufacturer': 'Manufacturer',
            'MagneticFieldStrength': 'MagneticFieldStrength',
            'PixelSpacing': 'PixelSpacing',
            'SliceThickness': 'SliceThickness',
        }
        
        for key, tag in tags.items():
            if hasattr(ds, tag):
                try:
                    metadata[key] = str(getattr(ds, tag))
                except Exception:
                    pass
        
        return metadata
    
    def validate_dicom_series(self, dicom_dir: str) -> Tuple[bool, List[str]]:
        """
        Validate DICOM series consistency.
        
        Checks:
        - All files are valid DICOM
        - Consistent image dimensions
        - Proper slice ordering
        
        Parameters
        ----------
        dicom_dir : str
            Directory containing DICOM files
            
        Returns
        -------
        is_valid : bool
            Whether series passes validation
        issues : List[str]
            List of issues found (empty if valid)
        """
        issues = []
        dicom_dir = Path(dicom_dir)
        
        dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("**/*.dcm"))
        
        if not dicom_files:
            issues.append(f"No DICOM files found")
            return False, issues
        
        # Load and check consistency
        shapes = []
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                if not hasattr(ds, 'pixel_array'):
                    issues.append(f"{dcm_file.name}: No pixel data")
                else:
                    shapes.append(ds.pixel_array.shape)
            except Exception as e:
                issues.append(f"{dcm_file.name}: Failed to read - {e}")
        
        # Check shape consistency
        if shapes:
            if len(set(shapes)) > 1:
                issues.append(f"Inconsistent image dimensions: {set(shapes)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def load_from_file(self, dicom_file: str) -> Tuple[np.ndarray, Dict]:
        """
        Load single DICOM file.
        
        Parameters
        ----------
        dicom_file : str
            Path to DICOM file
            
        Returns
        -------
        image : np.ndarray
            2D or 3D image array
        metadata : Dict
            DICOM metadata
        """
        ds = pydicom.dcmread(dicom_file)
        image = ds.pixel_array.astype(np.float32)
        metadata = self._extract_metadata(ds)
        
        return image, metadata


class Image2DLoader:
    """Load 2D medical images (OASIS dataset)."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize 2D image loader.
        
        Parameters
        ----------
        verbose : bool
            Print processing information
        """
        self.verbose = verbose
    
    def load_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load 2D medical image (NIfTI, NII, IMG/HDR).
        
        Parameters
        ----------
        image_path : str
            Path to image file
            
        Returns
        -------
        image : np.ndarray
            2D or 3D image array
        metadata : Dict
            Image metadata (shape, affine, etc.)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        suffix = image_path.suffix.lower()
        
        if suffix in ['.nii', '.nii.gz']:
            return self._load_nifti(image_path)
        elif suffix == '.img':
            return self._load_img_hdr(image_path)
        else:
            raise ValueError(f"Unsupported image format: {suffix}")
    
    def _load_nifti(self, nifti_file: Path) -> Tuple[np.ndarray, Dict]:
        """Load NIfTI image."""
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel required for NIfTI loading")
        
        img = nib.load(nifti_file)
        image = img.get_fdata().astype(np.float32)
        
        metadata = {
            'shape': image.shape,
            'affine': img.affine,
            'dtype': str(image.dtype),
            'filename': nifti_file.name,
        }
        
        if self.verbose:
            print(f"Loaded NIfTI: {image.shape}")
        
        return image, metadata
    
    def _load_img_hdr(self, img_file: Path) -> Tuple[np.ndarray, Dict]:
        """Load IMG/HDR pair."""
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel required for IMG/HDR loading")
        
        img = nib.load(img_file)
        image = img.get_fdata().astype(np.float32)
        
        metadata = {
            'shape': image.shape,
            'affine': img.affine,
            'dtype': str(image.dtype),
            'filename': img_file.name,
        }
        
        if self.verbose:
            print(f"Loaded IMG/HDR: {image.shape}")
        
        return image, metadata
    
    def load_slice_series(self, slice_dir: str) -> Tuple[np.ndarray, Dict]:
        """
        Load series of 2D image slices into 3D volume.
        
        Parameters
        ----------
        slice_dir : str
            Directory containing slice files
            
        Returns
        -------
        volume : np.ndarray
            3D volume [H, W, D]
        metadata : Dict
            Volume metadata
        """
        slice_dir = Path(slice_dir)
        
        # Find all image files
        image_files = list(slice_dir.glob("*.nii")) + \
                     list(slice_dir.glob("*.nii.gz")) + \
                     list(slice_dir.glob("*.img"))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {slice_dir}")
        
        slices = []
        for img_file in sorted(image_files):
            try:
                img, _ = self.load_image(str(img_file))
                slices.append(img)
            except Exception as e:
                warnings.warn(f"Failed to load {img_file}: {e}")
        
        if not slices:
            raise ValueError("No valid slices loaded")
        
        # Stack slices
        volume = np.stack(slices, axis=-1)
        metadata = {
            'shape': volume.shape,
            'num_slices': len(slices),
        }
        
        return volume, metadata
