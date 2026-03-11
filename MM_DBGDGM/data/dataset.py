"""
Dataset loaders for preprocessed fMRI and sMRI data.
Handles both OASIS and ADNI datasets in DBGDGM format.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class MultimodalBrainDataset(Dataset):
    """
    Multimodal brain imaging dataset.
    Loads preprocessed fMRI and sMRI data in DBGDGM format.
    
    Expected directory structure:
    ```
    dataset_root/
    ├── fmri/
    │   └── {subject_id}/
    │       └── {timepoint}/
    │           └── fmri_windows_dbgdgm.npy  [N_samples, 200, 50]
    ├── smri/
    │   └── {subject_id}/
    │       └── {timepoint}/
    │           └── features.npy  [1, N_features]
    └── metadata/
        └── labels.csv  [subject_id, timepoint, label]
    ```
    """
    
    def __init__(
        self,
        dataset_root: str,
        metadata_file: str,
        modalities: list = ['fmri', 'smri'],
        normalize_fmri: bool = True,
        normalize_smri: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            dataset_root: Path to preprocessed dataset root
            metadata_file: CSV file with subject_id, timepoint, label columns
            modalities: List of modalities to load ['fmri', 'smri']
            normalize_fmri: Apply z-score normalization to fMRI
            normalize_smri: Apply min-max normalization to sMRI
            verbose: Print loading information
        """
        self.dataset_root = Path(dataset_root)
        self.metadata_file = Path(metadata_file)
        self.modalities = modalities
        self.normalize_fmri = normalize_fmri
        self.normalize_smri = normalize_smri
        self.verbose = verbose
        
        # Load metadata
        self.samples = self._load_metadata()
        
        if self.verbose:
            logger.info(f"Loaded {len(self.samples)} samples from {self.dataset_root}")
    
    def _load_metadata(self) -> List[Dict]:
        """Load sample information from metadata file."""
        samples = []
        
        try:
            import pandas as pd
            df = pd.read_csv(str(self.metadata_file))
            
            for _, row in df.iterrows():
                subject_id = str(row['subject_id'])
                timepoint = str(row['timepoint'])
                label = int(row['label'])
                
                samples.append({
                    'subject_id': subject_id,
                    'timepoint': timepoint,
                    'label': label
                })
            
            if self.verbose:
                logger.info(f"Loaded metadata for {len(samples)} samples")
        
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
        
        return samples
    
    def _load_fmri(self, subject_id: str, timepoint: str) -> Optional[torch.Tensor]:
        """Load fMRI data."""
        if 'fmri' not in self.modalities:
            return None
        
        fmri_file = self.dataset_root / 'fmri' / subject_id / timepoint / 'fmri_windows_dbgdgm.npy'
        
        if not fmri_file.exists():
            logger.warning(f"fMRI file not found: {fmri_file}")
            return None
        
        try:
            fmri_data = np.load(str(fmri_file)).astype(np.float32)
            
            # Expected shape: [N_samples, 200, 50]
            if fmri_data.ndim == 3 and fmri_data.shape[1] == 200 and fmri_data.shape[2] == 50:
                # Select first window if multiple samples, or average
                if fmri_data.shape[0] > 1:
                    # Take mean across windows or random selection
                    fmri_data = np.mean(fmri_data, axis=0, keepdims=True)[0]  # [200, 50]
                else:
                    fmri_data = fmri_data[0]  # [200, 50]
                
                # Normalize
                if self.normalize_fmri:
                    mean = np.mean(fmri_data)
                    std = np.std(fmri_data)
                    fmri_data = (fmri_data - mean) / (std + 1e-8)
                
                return torch.from_numpy(fmri_data)
            else:
                logger.warning(f"Unexpected fMRI shape: {fmri_data.shape}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load fMRI {fmri_file}: {e}")
            return None
    
    def _load_smri(self, subject_id: str, timepoint: str) -> Optional[torch.Tensor]:
        """Load sMRI data."""
        if 'smri' not in self.modalities:
            return None
        
        smri_file = self.dataset_root / 'smri' / subject_id / timepoint / 'features.npy'
        
        if not smri_file.exists():
            logger.warning(f"sMRI file not found: {smri_file}")
            return None
        
        try:
            smri_data = np.load(str(smri_file)).astype(np.float32)
            
            # Expected shape: [1, N_features]
            if smri_data.ndim == 2 and smri_data.shape[0] == 1:
                smri_data = smri_data[0]  # [N_features]
                
                # Normalize
                if self.normalize_smri:
                    min_val = np.min(smri_data)
                    max_val = np.max(smri_data)
                    smri_data = (smri_data - min_val) / (max_val - min_val + 1e-8)
                
                return torch.from_numpy(smri_data)
            else:
                logger.warning(f"Unexpected sMRI shape: {smri_data.shape}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load sMRI {smri_file}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                - fmri: [200, 50] (optional)
                - smri: [N_features] (optional)
                - label: scalar
                - subject_id: str
                - timepoint: str
        """
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        timepoint = sample['timepoint']
        label = sample['label']
        
        data = {
            'label': torch.tensor(label, dtype=torch.long),
            'subject_id': subject_id,
            'timepoint': timepoint
        }
        
        # Load modalities
        fmri = self._load_fmri(subject_id, timepoint)
        smri = self._load_smri(subject_id, timepoint)

        if 'fmri' in self.modalities and fmri is None:
            raise FileNotFoundError(
                f"Missing or invalid fMRI data for subject '{subject_id}' at timepoint '{timepoint}'"
            )
        if 'smri' in self.modalities and smri is None:
            raise FileNotFoundError(
                f"Missing or invalid sMRI data for subject '{subject_id}' at timepoint '{timepoint}'"
            )
        
        if fmri is not None:
            data['fmri'] = fmri
        if smri is not None:
            data['smri'] = smri
        
        return data


def create_dataloaders(
    dataset_root: str,
    train_metadata: str,
    val_metadata: str,
    test_metadata: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
    normalize: bool = True
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for train, val, and optionally test sets.
    
    Args:
        dataset_root: Path to dataset root
        train_metadata: Path to train metadata CSV
        val_metadata: Path to validation metadata CSV
        test_metadata: Path to test metadata CSV (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle_train: Shuffle training data
        normalize: Apply normalization
    
    Returns:
        dict with 'train', 'val', and optionally 'test' dataloaders
    """
    
    train_dataset = MultimodalBrainDataset(
        dataset_root=dataset_root,
        metadata_file=train_metadata,
        normalize_fmri=normalize,
        normalize_smri=normalize,
        verbose=True
    )
    
    val_dataset = MultimodalBrainDataset(
        dataset_root=dataset_root,
        metadata_file=val_metadata,
        normalize_fmri=normalize,
        normalize_smri=normalize,
        verbose=True
    )
    
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    }
    
    if test_metadata is not None:
        test_dataset = MultimodalBrainDataset(
            dataset_root=dataset_root,
            metadata_file=test_metadata,
            normalize_fmri=normalize,
            normalize_smri=normalize,
            verbose=True
        )
        
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    return dataloaders
