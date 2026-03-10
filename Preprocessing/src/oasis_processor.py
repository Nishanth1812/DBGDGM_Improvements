"""
OASIS (Open Access Series of Imaging Studies) Dataset Processor
===============================================================

OASIS-1 dataset: 2D T1-weighted images
- 416 subjects (221 normal, 195 with dementia)
- Multiple scans per subject
- 2D axial slices (typically 128x128 or similar)

Processing:
1. Reconstruct 3D volume from 2D slices
2. Handle multiple scans per subject (group for no data leakage)
3. Apply sMRI and fMRI preprocessing
4. Extract features for DBGDGM model
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import warnings
from dataclasses import dataclass

from ..utils.dicom_utils import Image2DLoader
from .smri_preprocessing import sMRIPreprocessor, sMRIConfig


@dataclass
class OASISSubject:
    """OASIS subject information."""
    subject_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    cdr: Optional[float] = None  # Clinical Dementia Rating (0=normal, 0.5=very mild, 1=mild, 2=moderate, 3=severe)
    mmse: Optional[int] = None  # Mini-mental state exam score
    edu: Optional[int] = None  # Years of education
    sessions: List[Dict] = None


class OASISDatasetProcessor:
    """Process OASIS dataset for DBGDGM."""
    
    def __init__(self, data_dir: str, output_dir: str, verbose: bool = True):
        """
        Initialize OASIS processor.
        
        Parameters
        ----------
        data_dir : str
            Root directory of OASIS dataset
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
        (self.output_dir / 'smri').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
        self.image_loader = Image2DLoader(verbose=verbose)
        self.smri_preprocessor = sMRIPreprocessor(config=sMRIConfig(verbose=verbose))
        
        self.subjects = []
        self.processed_subjects = []
    
    def discover_subjects(self) -> List[OASISSubject]:
        """
        Discover subjects in OASIS dataset.
        
        OASIS directory structure:
        OAS1_XXXX_MR1/
        OAS1_XXXX_MR2/
        etc.
        """
        if self.verbose:
            print(f"Discovering OASIS subjects in {self.data_dir}")
        
        self.subjects = []
        
        # Find all OAS1_* directories
        for subject_dir in sorted(self.data_dir.glob("OAS1_*")):
            if not subject_dir.is_dir():
                continue
            
            # Extract subject_id and session from directory name
            # Format: OAS1_XXXX_MRX
            parts = subject_dir.name.split('_')
            if len(parts) >= 2:
                subject_id = f"{parts[0]}_{parts[1]}"  # OAS1_XXXX
                session_num = parts[2] if len(parts) > 2 else "MR1"
                
                # Look for already created subject
                subject = next((s for s in self.subjects if s.subject_id == subject_id), None)
                
                if subject is None:
                    subject = OASISSubject(subject_id=subject_id, sessions=[])
                    self.subjects.append(subject)
                
                # Add session
                session_info = {
                    'session': session_num,
                    'path': str(subject_dir),
                    'images': self._find_images(subject_dir)
                }
                subject.sessions.append(session_info) if subject.sessions else None
        
        if self.verbose:
            print(f"Found {len(self.subjects)} unique subjects")
        
        return self.subjects
    
    def _find_images(self, subject_dir: Path) -> List[Path]:
        """Find all image files in subject directory."""
        images = list(subject_dir.glob("**/*.nii")) + \
                list(subject_dir.glob("**/*.nii.gz")) + \
                list(subject_dir.glob("**/*.img"))
        return images
    
    def process_subject(self, subject: OASISSubject) -> Dict:
        """
        Process single OASIS subject.
        
        For multiple sessions, use only the first session to avoid data leakage.
        """
        if not subject.sessions:
            warnings.warn(f"No sessions found for {subject.subject_id}")
            return None
        
        # Use first session only
        first_session = subject.sessions[0]
        subject_dir = Path(first_session['path'])
        images = first_session['images']
        
        if not images:
            warnings.warn(f"No images found for {subject.subject_id}")
            return None
        
        if self.verbose:
            print(f"Processing {subject.subject_id} - {len(images)} images")
        
        try:
            # Load first image (would need to handle 3D reconstruction for multi-slices)
            image, metadata_img = self.image_loader.load_image(str(images[0]))
            
            # Get affine (typically identity for OASIS 2D images)
            affine = np.eye(4)
            
            # Preprocess sMRI
            features, preproc_metadata = self.smri_preprocessor.preprocess(
                image, affine
            )
            
            # Compile output
            subject_metadata = {
                'subject_id': subject.subject_id,
                'session': first_session['session'],
                'n_images': len(images),
                'image_shape': image.shape,
                'preprocessing': preproc_metadata,
                'features_shape': features.shape,
            }
            
            # Save preprocessed data
            output_subject_dir = self.output_dir / 'smri' / subject.subject_id / first_session['session']
            output_subject_dir.mkdir(parents=True, exist_ok=True)
            
            # Save features
            np.save(output_subject_dir / 'features.npy', features)
            
            # Save metadata
            import json
            with open(output_subject_dir / 'metadata.json', 'w') as f:
                # Convert non-serializable types
                clean_metadata = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
                                for k, v in subject_metadata.items()}
                json.dump(clean_metadata, f, indent=2)
            
            self.processed_subjects.append(subject_metadata)
            
            if self.verbose:
                print(f"  ✓ Saved features: {features.shape}")
            
            return subject_metadata
        
        except Exception as e:
            warnings.warn(f"Failed to process {subject.subject_id}: {e}")
            return None
    
    def process_all_subjects(self) -> pd.DataFrame:
        """Process all discovered subjects."""
        if not self.subjects:
            self.discover_subjects()
        
        if self.verbose:
            print(f"\nProcessing {len(self.subjects)} subjects...")
        
        for subject in self.subjects:
            self.process_subject(subject)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(self.processed_subjects)
        
        # Save summary
        summary_df.to_csv(self.output_dir / 'metadata' / 'subjects_summary.csv', index=False)
        
        if self.verbose:
            print(f"\nProcessing complete: {len(self.processed_subjects)}/{len(self.subjects)} subjects")
        
        return summary_df
    
    def load_processed_subject(self, subject_id: str, session: str = 'MR1') -> Tuple[np.ndarray, Dict]:
        """Load preprocessed features for a subject."""
        features_path = self.output_dir / 'smri' / subject_id / session / 'features.npy'
        metadata_path = self.output_dir / 'smri' / subject_id / session / 'metadata.json'
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        
        features = np.load(features_path)
        
        metadata = {}
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
        
        return features, metadata
