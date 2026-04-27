import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


class ADNISubjectDataset(Dataset):
    def __init__(self, manifest_path: Path, subjects_dir: Path, transform=None):
        self.manifest = pd.read_csv(manifest_path)
        self.subjects_dir = Path(subjects_dir)
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        subject_id = row["subject_id"]
        data = np.load(self.subjects_dir / f"{subject_id}.npz", allow_pickle=True)

        fmri = data["fmri"].astype(np.float32)
        smri = data["smri"].astype(np.float32)

        # Per-subject z-score normalisation on fMRI time series
        fmri_mean = fmri.mean(axis=1, keepdims=True)
        fmri_std = fmri.std(axis=1, keepdims=True) + 1e-8
        fmri = (fmri - fmri_mean) / fmri_std

        label = int(data["label"])

        return {
            "fmri": torch.from_numpy(fmri),
            "smri": torch.from_numpy(smri),
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": subject_id,
            "age": float(data["age"]),
            "sex": int(data["sex"]),
        }


def get_stratified_kfold_splits(manifest: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = manifest["label"].values
    splits = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        splits.append((train_idx, val_idx))
    return splits


def build_dataloaders(
    manifest_path: Path,
    subjects_dir: Path,
    batch_size: int = 8,
    n_splits: int = 5,
    seed: int = 42,
):
    dataset = ADNISubjectDataset(manifest_path, subjects_dir)
    manifest = dataset.manifest

    splits = get_stratified_kfold_splits(manifest, n_splits, seed)

    loaders = {}
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Weighted sampler for class imbalance
        train_labels = manifest.iloc[train_idx]["label"].values
        class_counts = np.bincount(train_labels, minlength=4)
        weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        loaders[fold_i] = {"train": train_loader, "val": val_loader}

    return loaders, splits