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

        # Label remapping to contiguous integers [0, N-1]
        # This prevents CUDA device-side asserts if labels are [0, 2, 3] and num_classes=3
        self.unique_labels = sorted(self.manifest["label"].unique().tolist())
        self.label_map = {orig: i for i, orig in enumerate(self.unique_labels)}
        self.inv_label_map = {i: orig for orig, i in self.label_map.items()}
        print(f"  [Dataset] Label mapping: {self.label_map}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        subject_id = row["subject_id"]
        cols = self.manifest.columns.tolist()

        if "fmri_path" in cols and "smri_path" in cols:
            fmri_path = str(row["fmri_path"])
            smri_path = str(row["smri_path"])
            if not Path(fmri_path).is_absolute():
                fmri_path = self.subjects_dir / fmri_path
            if not Path(smri_path).is_absolute():
                smri_path = self.subjects_dir / smri_path
                
            fmri = np.load(fmri_path).astype(np.float32)
            smri = np.load(smri_path).astype(np.float32)
            label = int(row["label"])
            age = float(row.get("age", 70.0))
            sex = int(row.get("sex", 0))
        else:
            data = np.load(self.subjects_dir / f"{subject_id}.npz", allow_pickle=True)
            fmri = data["fmri"].astype(np.float32)
            smri = data["smri"].astype(np.float32)
            label = int(data["label"])
            age = float(data["age"])
            sex = int(data["sex"])

        # Map to contiguous label
        mapped_label = self.label_map[label]

        # Per-subject z-score normalisation on fMRI time series
        fmri_mean = fmri.mean(axis=1, keepdims=True)
        fmri_std = fmri.std(axis=1, keepdims=True) + 1e-8
        fmri = (fmri - fmri_mean) / fmri_std

        return {
            "fmri": torch.from_numpy(fmri),
            "smri": torch.from_numpy(smri),
            "label": torch.tensor(mapped_label, dtype=torch.long),
            "orig_label": label,
            "subject_id": subject_id,
            "age": age,
            "sex": sex,
        }


def get_stratified_kfold_splits(manifest: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    # Ensure at least 2 samples per class for k-fold
    labels = manifest["label"].values
    counts = np.bincount(labels)
    valid_classes = np.where(counts >= n_splits)[0]
    if len(valid_classes) < len(np.unique(labels)):
        print(f"  [Warning] Some classes have < {n_splits} samples. StratifiedKFold might be unreliable.")
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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

        # Weighted sampler using mapped labels
        train_mapped_labels = [dataset.label_map[l] for l in manifest.iloc[train_idx]["label"].values]
        train_mapped_labels = np.array(train_mapped_labels)
        
        num_classes = len(dataset.unique_labels)
        class_counts = np.bincount(train_mapped_labels, minlength=num_classes)
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts[train_mapped_labels]
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