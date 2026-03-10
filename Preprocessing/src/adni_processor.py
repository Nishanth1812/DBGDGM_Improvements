"""
ADNI (Alzheimer's Disease Neuroimaging Initiative) Dataset Processor
=====================================================================

ADNI dataset: DICOM format
- Multiple modalities: T1-weighted, rsfMRI (Resting-state fMRI)
- DICOM files organized by series
- Multiple timepoints per subject

Processing:
1. Extract DICOM files
2. Select fMRI and T1 series
3. Apply dedicated preprocessing for each modality
4. Extract time series for DBGDGM model
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import warnings
from dataclasses import dataclass
import json

from ..utils.dicom_utils import DICOMLoader
from ..fmri.fmri_preprocessing import fMRIPreprocessor, fMRIConfig
from ..smri.smri_preprocessing import sMRIPreprocessor, sMRIConfig


@dataclass
class ADNISubject:
    """ADNI subject information."""
    subject_id: str
    group: Optional[str] = None  # CN, MCI, AD
    diagnosis: Optional[str] = None  # Detailed diagnosis
    age: Optional[int] = None
    gender: Optional[str] = None
    apoe4_status: Optional[int] = None  # 0, 1, 2 copies
    timepoints: List[Dict] = None


class ADNIDatasetProcessor:
    """Process ADNI dataset for DBGDGM."""
    
    def __init__(self, data_dir: str, output_dir: str, verbose: bool = True):
        """
        Initialize ADNI processor.
        
        Parameters
        ----------
        data_dir : str
            Root directory of ADNI dataset
        output_dir : str
            Output directory for preprocessed data
        verbose : bool
            Print progress information
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'fmri').mkdir(exist_ok=True)
        (self.output_dir / 'smri').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
        self.dicom_loader = DICOMLoader(verbose=verbose)
        self.fmri_preprocessor = fMRIPreprocessor(config=fMRIConfig(verbose=verbose))
        self.smri_preprocessor = sMRIPreprocessor(config=sMRIConfig(verbose=verbose))
        
        self.subjects = []
        self.processed_subjects = []
    
    def discover_subjects(self) -> List[ADNISubject]:
        """
        Discover ADNI subjects in dataset.
        
        ADNI directory structure (typical):
        ADNI/
        ├── ADNI_XXX/
        │   ├── TIMEPOINT_X/
        │   │   ├── MODALITY_Y/
        │   │   │   ├── *.dcm
        """
        if self.verbose:
            print(f"Discovering ADNI subjects in {self.data_dir}")
        
        self.subjects = []
        
        # Find all subject directories
        for subject_dir in sorted(self.data_dir.glob("ADNI_*")):
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name
            subject = ADNISubject(subject_id=subject_id, timepoints=[])
            
            # Find timepoints
            for timepoint_dir in sorted(subject_dir.glob("*")):
                if not timepoint_dir.is_dir() or timepoint_dir.name.startswith('.'):
                    continue
                
                timepoint_info = {
                    'timepoint': timepoint_dir.name,
                    'path': str(timepoint_dir),
                    'modalities': {}
                }
                
                # Find modalities (fMRI, T1, etc.)
                for modality_dir in timepoint_dir.glob("*"):
                    if modality_dir.is_dir():
                        modality = modality_dir.name
                        timepoint_info['modalities'][modality] = str(modality_dir)
                
                subject.timepoints.append(timepoint_info)
            
            if subject.timepoints:
                self.subjects.append(subject)
        
        if self.verbose:
            print(f"Found {len(self.subjects)} subjects")
        
        return self.subjects
    
    def process_subject(self, subject: ADNISubject,
                       use_first_timepoint_only: bool = True) -> Dict:
        """
        Process single ADNI subject.
        
        Parameters
        ----------
        subject : ADNISubject
            Subject to process
        use_first_timepoint_only : bool
            Use only first timepoint (to avoid subjects repeated across timepoints)
        
        Returns
        -------
        metadata : Dict
            Processing metadata
        """
        if not subject.timepoints:
            warnings.warn(f"No timepoints for {subject.subject_id}")
            return None
        
        # Use first timepoint only
        if use_first_timepoint_only:
            timepoints = subject.timepoints[:1]
        else:
            timepoints = subject.timepoints
        
        subject_metadata = {
            'subject_id': subject.subject_id,
            'n_timepoints': len(timepoints),
            'timepoint_data': []
        }
        
        for tp in timepoints:
            tp_metadata = self._process_timepoint(subject.subject_id, tp)
            if tp_metadata:
                subject_metadata['timepoint_data'].append(tp_metadata)
        
        self.processed_subjects.append(subject_metadata)
        return subject_metadata
    
    def _process_timepoint(self, subject_id: str, timepoint_info: Dict) -> Optional[Dict]:
        """Process single timepoint."""
        tp_name = timepoint_info['timepoint']
        modalities = timepoint_info['modalities']
        
        if self.verbose:
            print(f"Processing {subject_id}/{tp_name}")
        
        tp_metadata = {
            'timepoint': tp_name,
            'modalities_found': [],
            'modalities_processed': []
        }
        
        # Process T1 (sMRI)
        t1_dirs = [modalities.get(m) for m in ['T1', 'MPRAGE', 't1', 'T1w'] if m in modalities]
        if t1_dirs:
            t1_metadata = self._process_t1(subject_id, tp_name, t1_dirs[0])
            if t1_metadata:
                tp_metadata['t1'] = t1_metadata
                tp_metadata['modalities_processed'].append('t1')
            tp_metadata['modalities_found'].append('T1')
        
        # Process fMRI (rsfMRI)
        fmri_dirs = [modalities.get(m) for m in ['REST', 'fMRI', 'rsfMRI', 'resting', 'rest'] if m in modalities]
        if fmri_dirs:
            fmri_metadata = self._process_fmri(subject_id, tp_name, fmri_dirs[0])
            if fmri_metadata:
                tp_metadata['fmri'] = fmri_metadata
                tp_metadata['modalities_processed'].append('fmri')
            tp_metadata['modalities_found'].append('fMRI')
        
        return tp_metadata if tp_metadata['modalities_processed'] else None
    
    def _process_t1(self, subject_id: str, timepoint: str, t1_dir: str) -> Optional[Dict]:
        """Process T1-weighted (sMRI) for a timepoint."""
        try:
            t1_path = Path(t1_dir)
            
            if self.verbose:
                print(f"  Processing T1: {t1_path}")
            
            # Load first DICOM series found
            dicom_dirs = [d for d in t1_path.glob("*") if d.is_dir()]
            if not dicom_dirs:
                # Check if current dir has DICOMs
                dicom_files = list(t1_path.glob("*.dcm"))
                if dicom_files:
                    dicom_dirs = [t1_path]
            
            if not dicom_dirs:
                warnings.warn(f"No DICOM directories found in {t1_dir}")
                return None
            
            # Load DICOM series
            t1_image, dicom_metadata = self.dicom_loader.load_dicom_series(str(dicom_dirs[0]))
            affine = np.eye(4)
            
            # Preprocess
            features, preproc_metadata = self.smri_preprocessor.preprocess(
                t1_image, affine
            )
            
            # Save
            output_dir = self.output_dir / 'smri' / subject_id / timepoint
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(output_dir / 'features.npy', features)
            with open(output_dir / 'metadata.json', 'w') as f:
                clean_metadata = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
                                for k, v in preproc_metadata.items()}
                json.dump(clean_metadata, f, indent=2)
            
            if self.verbose:
                print(f"    ✓ T1 features: {features.shape}")
            
            return {
                'features_shape': features.shape,
                'preprocessing': preproc_metadata
            }
        
        except Exception as e:
            warnings.warn(f"T1 processing failed for {subject_id}/{timepoint}: {e}")
            return None
    
    def _process_fmri(self, subject_id: str, timepoint: str, fmri_dir: str) -> Optional[Dict]:
        """Process resting-state fMRI for a timepoint."""
        try:
            fmri_path = Path(fmri_dir)
            
            if self.verbose:
                print(f"  Processing fMRI: {fmri_path}")
            
            # Load first DICOM series
            dicom_dirs = [d for d in fmri_path.glob("*") if d.is_dir()]
            if not dicom_dirs:
                dicom_files = list(fmri_path.glob("*.dcm"))
                if dicom_files:
                    dicom_dirs = [fmri_path]
            
            if not dicom_dirs:
                warnings.warn(f"No DICOM directories in {fmri_dir}")
                return None
            
            # Load DICOM series
            fmri_image, dicom_metadata = self.dicom_loader.load_dicom_series(str(dicom_dirs[0]))
            
            # Check if 4D
            if fmri_image.ndim < 4:
                if self.verbose:
                    print(f"  Warning: fMRI image is {fmri_image.ndim}D, expected 4D")
                # Simulate 4D by repeating slices
                fmri_image = np.stack([fmri_image] * 100, axis=-1)
            
            affine = np.eye(4)
            
            # Preprocess
            windows, preproc_metadata = self.fmri_preprocessor.preprocess(
                fmri_image, affine
            )
            
            # Save
            output_dir = self.output_dir / 'fmri' / subject_id / timepoint
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(output_dir / 'timeseries_windows.npy', windows)
            with open(output_dir / 'metadata.json', 'w') as f:
                clean_metadata = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
                                for k, v in preproc_metadata.items()}
                json.dump(clean_metadata, f, indent=2)
            
            if self.verbose:
                print(f"    ✓ fMRI windows: {windows.shape}")
            
            return {
                'windows_shape': windows.shape,
                'preprocessing': preproc_metadata
            }
        
        except Exception as e:
            warnings.warn(f"fMRI processing failed for {subject_id}/{timepoint}: {e}")
            return None
    
    def process_all_subjects(self) -> pd.DataFrame:
        """Process all discovered subjects."""
        if not self.subjects:
            self.discover_subjects()
        
        if self.verbose:
            print(f"\nProcessing {len(self.subjects)} subjects...")
        
        for subject in self.subjects:
            self.process_subject(subject)
        
        # Create summary
        summary_df = pd.DataFrame(self.processed_subjects)
        summary_df.to_csv(self.output_dir / 'metadata' / 'subjects_summary.csv', index=False)
        
        if self.verbose:
            print(f"\nProcessing complete: {len(self.processed_subjects)}/{len(self.subjects)} subjects")
        
        return summary_df
