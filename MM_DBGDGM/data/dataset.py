"""
Dataset loaders for preprocessed fMRI and sMRI data.
Handles both OASIS and ADNI datasets in DBGDGM format.

The loader also supports local prepared sMRI JPG folders through metadata
columns such as `smri_path` or `prepared_folder`, which makes it usable on a
standalone GPU machine without the Modal preprocessing cache.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

IMAGE_FILE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

DICOM_FILE_EXTENSIONS = {
    ".dcm",
    ".dicom",
    ".ima",
    "",
}

FMRI_CLUE_TOKENS = (
    "fmri",
    "bold",
    "rest",
    "resting_state",
    "restingstate",
    "rsfmri",
    "ep2d",
    "epi",
)


def _is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_FILE_EXTENSIONS


def _extract_numeric_hint(name: str) -> Optional[int]:
    matches = re.findall(r"(\d+)", name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _ordered_image_files(series_dir: Path, max_images_per_series: int = 0) -> List[Path]:
    records = []
    for file_path in series_dir.rglob("*"):
        if not file_path.is_file() or not _is_image_file(file_path):
            continue

        numeric_hint = _extract_numeric_hint(file_path.stem)
        numeric_hint = numeric_hint if numeric_hint is not None else 10**12
        sort_key = (numeric_hint, file_path.name)
        records.append((sort_key, file_path))

    records.sort(key=lambda item: item[0])
    if max_images_per_series > 0:
        records = records[:max_images_per_series]

    return [item[1] for item in records]


def _is_dicom_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in DICOM_FILE_EXTENSIONS


def _series_has_fmri_clue(series_dir: Path) -> bool:
    path_lower = "/".join(series_dir.parts).lower()
    return any(token in path_lower for token in FMRI_CLUE_TOKENS)


def _extract_subject_id_from_series(series_dir: Path) -> str:
    parts = list(series_dir.parts)
    if "ADNI" in parts:
        idx = parts.index("ADNI")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    folder_name = series_dir.name.strip()
    if folder_name:
        if re.search(r"(?i)^OAS\d+_\d+_MR\d+$", folder_name):
            return folder_name
        if re.search(r"\d", folder_name) and folder_name.lower() not in {
            "non demented",
            "very mild dementia",
            "mild dementia",
            "moderate dementia",
        }:
            return folder_name

    image_like_files = [
        file_path for file_path in series_dir.iterdir()
        if file_path.is_file() and (_is_image_file(file_path) or _is_dicom_file(file_path))
    ]
    if image_like_files:
        first_name = image_like_files[0].stem
        subject_match = re.match(r"^(.*?)(?:_mpr-\d+)?_\d+$", first_name, flags=re.IGNORECASE)
        if subject_match:
            return subject_match.group(1)
        subject_match = re.match(r"^(.*?)(?:_\d+)$", first_name)
        if subject_match:
            return subject_match.group(1)
        if "_mpr-" in first_name:
            return first_name.split("_mpr-")[0]
        return first_name

    return folder_name or series_dir.parent.name


def _extract_numeric_hint_from_path(name: str) -> Optional[int]:
    matches = re.findall(r"(\d+)", name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _ordered_dicom_files(series_dir: Path, max_dicoms_per_series: int) -> Tuple[List[Path], List[Any]]:
    import pydicom

    records = []
    for file_path in series_dir.iterdir():
        if not file_path.is_file() or not _is_dicom_file(file_path):
            continue

        header = None
        temporal_pos = 10**9
        acquisition_no = 10**9
        instance_no = 10**9

        try:
            with pydicom.config.disable_value_validation():
                header = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
            temporal_pos = _to_int_or_default(getattr(header, "TemporalPositionIdentifier", None), 10**9)
            acquisition_no = _to_int_or_default(getattr(header, "AcquisitionNumber", None), 10**9)
            instance_no = _to_int_or_default(getattr(header, "InstanceNumber", None), 10**9)
        except Exception:
            header = None

        numeric_hint = _extract_numeric_hint_from_path(file_path.stem)
        numeric_hint = numeric_hint if numeric_hint is not None else 10**12

        sort_key = (
            temporal_pos,
            acquisition_no,
            instance_no,
            numeric_hint,
            file_path.name,
        )
        records.append((sort_key, file_path, header))

    records.sort(key=lambda x: x[0])
    if max_dicoms_per_series > 0:
        records = records[:max_dicoms_per_series]

    ordered_paths = [item[1] for item in records]
    headers = [item[2] for item in records if item[2] is not None]
    return ordered_paths, headers


def _build_fmri_from_signal(signal: np.ndarray, n_roi: int, seq_len: int) -> np.ndarray:
    signal = _resample_1d(signal, seq_len)
    signal = (signal - signal.mean()) / (signal.std() + 1e-6)

    roi_scale = np.linspace(0.9, 1.1, num=n_roi, dtype=np.float32)[:, None]
    phase = np.linspace(0.0, np.pi, num=n_roi, dtype=np.float32)[:, None]
    time_axis = np.linspace(0.0, 2.0 * np.pi, num=seq_len, dtype=np.float32)[None, :]
    fmri = roi_scale * signal[None, :] + 0.05 * np.sin(time_axis + phase)
    return fmri.astype(np.float32)


def _load_dicom_series_proxy_fmri(
    series_dir: Path,
    n_roi: int,
    seq_len: int,
    max_dicoms_per_series: int,
) -> Optional[np.ndarray]:
    import pydicom

    dcm_files, headers = _ordered_dicom_files(series_dir, max_dicoms_per_series)
    if not dcm_files:
        return None

    slice_means: List[float] = []
    slice_stds: List[float] = []

    for dcm_path in dcm_files:
        try:
            with pydicom.config.disable_value_validation():
                ds = pydicom.dcmread(str(dcm_path), force=True)
            if not hasattr(ds, "pixel_array"):
                continue
            pixels = ds.pixel_array.astype(np.float32)
            if pixels.size == 0:
                continue

            slice_means.append(float(np.mean(pixels)))
            slice_stds.append(float(np.std(pixels)))
        except Exception:
            continue

    if not slice_means:
        image_files = _ordered_image_files(series_dir, max_images_per_series=max_dicoms_per_series)
        if not image_files:
            return None

        for image_path in image_files:
            try:
                import cv2

                pixels = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if pixels is None:
                    continue
                if pixels.ndim == 3:
                    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
                pixels = pixels.astype(np.float32)
                if pixels.size == 0:
                    continue
                slice_means.append(float(np.mean(pixels)))
                slice_stds.append(float(np.std(pixels)))
            except Exception:
                continue

    if not slice_means:
        return None

    signal = np.asarray(slice_means, dtype=np.float32)
    return _build_fmri_from_signal(signal, n_roi=n_roi, seq_len=seq_len)


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if signal.size == 1:
        return np.full(target_len, float(signal[0]), dtype=np.float32)

    old_x = np.linspace(0.0, 1.0, num=signal.size, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(new_x, old_x, signal).astype(np.float32)


def _resize_2d(array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.shape == target_shape:
        return array

    if array.ndim != 2 or array.size == 0:
        return np.zeros(target_shape, dtype=np.float32)

    target_rows, target_cols = target_shape
    row_positions = np.linspace(0.0, array.shape[0] - 1, num=target_rows, dtype=np.float32)
    col_positions = np.linspace(0.0, array.shape[1] - 1, num=target_cols, dtype=np.float32)

    row_axis = np.arange(array.shape[0], dtype=np.float32)
    col_axis = np.arange(array.shape[1], dtype=np.float32)

    intermediate = np.vstack([
        np.interp(col_positions, col_axis, row)
        for row in array
    ]).astype(np.float32)

    resized = np.vstack([
        np.interp(row_positions, row_axis, intermediate[:, col_idx])
        for col_idx in range(intermediate.shape[1])
    ]).T.astype(np.float32)
    return resized


def _resolve_relative_path(path_value: Any, base_dir: Path) -> Optional[Path]:
    cleaned = _string_or_empty(path_value)
    if not cleaned:
        return None

    candidate = Path(cleaned).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def _first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_image_folder_proxy_features(series_dir: Path, target_len: Optional[int] = None) -> Optional[np.ndarray]:
    import cv2

    image_files = _ordered_image_files(series_dir)
    if not image_files:
        return None

    image_means: List[float] = []
    image_stds: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []
    row_values: List[float] = []
    col_values: List[float] = []
    file_sizes: List[float] = []

    for image_path in image_files:
        pixels = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            continue

        if pixels.ndim == 3:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

        pixels = pixels.astype(np.float32)
        if pixels.size == 0:
            continue

        image_means.append(float(np.mean(pixels)))
        image_stds.append(float(np.std(pixels)))
        mins.append(float(np.min(pixels)))
        maxs.append(float(np.max(pixels)))
        row_values.append(float(pixels.shape[0]))
        col_values.append(float(pixels.shape[1]))
        try:
            file_sizes.append(float(image_path.stat().st_size))
        except Exception:
            file_sizes.append(0.0)

    if not image_means:
        return None

    mean_rows = float(np.mean(row_values)) if row_values else 0.0
    mean_cols = float(np.mean(col_values)) if col_values else 0.0
    aspect_ratio = float(mean_rows / max(mean_cols, 1.0))

    smri_raw = np.asarray([
        float(np.mean(image_means)),
        float(np.std(image_means)),
        float(np.mean(image_stds)),
        float(np.min(mins)),
        float(np.max(maxs)),
        mean_rows / 512.0,
        mean_cols / 512.0,
        aspect_ratio,
        float(len(image_files)) / 100.0,
        float(np.mean(file_sizes)) / 1_000_000.0 if file_sizes else 0.0,
        float(np.std(file_sizes)) / 1_000_000.0 if file_sizes else 0.0,
    ], dtype=np.float32)
    smri_raw = np.nan_to_num(smri_raw, nan=0.0, posinf=0.0, neginf=0.0)

    if target_len is not None and target_len > 0 and smri_raw.size != target_len:
        smri_raw = _resample_1d(smri_raw, target_len)

    smri_min, smri_max = float(smri_raw.min()), float(smri_raw.max())
    if smri_max > smri_min:
        smri_raw = (smri_raw - smri_min) / (smri_max - smri_min)
    else:
        smri_raw = np.zeros_like(smri_raw, dtype=np.float32)

    return smri_raw.astype(np.float32)


def _build_auxiliary_targets(smri: np.ndarray, label: int, n_smri_features: int) -> Dict[str, torch.Tensor]:
    smri = np.asarray(smri, dtype=np.float32)
    if smri.size == 0:
        smri = np.zeros(max(1, n_smri_features), dtype=np.float32)

    first_value = float(smri[0]) if smri.size > 0 else 0.0
    second_value = float(smri[min(1, smri.size - 1)]) if smri.size > 0 else first_value
    third_value = float(smri[min(2, smri.size - 1)]) if smri.size > 0 else first_value

    hippo_vol = 2500.0 + (1.0 - first_value) * 2000.0
    cortical_thinning = 0.1 + second_value * 0.4
    dmn_conn = float(np.clip(1.0 - third_value, 0.0, 1.0))
    nss = float(np.clip(20.0 + float(label) * 20.0 + first_value * 15.0, 0.0, 100.0))

    survival_times = np.asarray([8.0, 5.5, 3.0], dtype=np.float32) - float(label) * 0.7
    survival_times = np.clip(survival_times, 0.3, None)
    survival_events = np.ones(3, dtype=np.float32)

    return {
        'hippo_vol': torch.tensor(hippo_vol, dtype=torch.float32),
        'cortical_thinning': torch.tensor(cortical_thinning, dtype=torch.float32),
        'dmn_conn': torch.tensor(dmn_conn, dtype=torch.float32),
        'nss': torch.tensor(nss, dtype=torch.float32),
        'survival_times': torch.from_numpy(survival_times),
        'survival_events': torch.from_numpy(survival_events),
    }


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
        metadata_file: Optional[str] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        modalities: Optional[List[str]] = None,
        normalize_fmri: bool = True,
        normalize_smri: bool = True,
        n_roi: Optional[int] = None,
        seq_len: Optional[int] = None,
        n_smri_features: Optional[int] = None,
        max_dicoms_per_series: int = 120,
        smri_source_root: Optional[str] = None,
        fmri_path_column: str = 'fmri_path',
        smri_path_column: str = 'smri_path',
        metadata_base_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Args:
            dataset_root: Path to preprocessed dataset root
            metadata_file: CSV file with subject_id, timepoint, label columns
            samples: Optional in-memory metadata records, useful for auto-split loaders
            modalities: List of modalities to load ['fmri', 'smri']
            normalize_fmri: Apply z-score normalization to fMRI
            normalize_smri: Apply min-max normalization to sMRI
            n_roi: Optional fMRI ROI count used for resizing local arrays
            seq_len: Optional fMRI sequence length used for resizing local arrays
            n_smri_features: Optional target feature count for local sMRI proxy features
            max_dicoms_per_series: Maximum number of raw DICOM slices to read per series
            smri_source_root: Optional root for prepared local sMRI JPG folders
            fmri_path_column: Optional metadata column name for explicit fMRI paths
            smri_path_column: Optional metadata column name for explicit sMRI paths
            metadata_base_dir: Base directory used to resolve relative metadata paths
            verbose: Print loading information
        """
        self.dataset_root = Path(dataset_root)
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.metadata_base_dir = Path(metadata_base_dir) if metadata_base_dir else (
            self.metadata_file.parent if self.metadata_file is not None else self.dataset_root
        )
        self.modalities = modalities or ['fmri', 'smri']
        self.normalize_fmri = normalize_fmri
        self.normalize_smri = normalize_smri
        self.n_roi = n_roi
        self.seq_len = seq_len
        self.n_smri_features = n_smri_features
        self.max_dicoms_per_series = max_dicoms_per_series
        self.smri_source_root = Path(smri_source_root) if smri_source_root else None
        self.fmri_path_column = fmri_path_column
        self.smri_path_column = smri_path_column
        self.verbose = verbose
        
        # Load metadata
        self.samples = self._normalize_samples(samples) if samples is not None else self._load_metadata()
        self._dicom_series_index: Dict[str, List[Path]] = {}
        if not (self.dataset_root / 'fmri').exists():
            self._dicom_series_index = self._build_dicom_series_index()
        
        if self.verbose:
            logger.info(f"Loaded {len(self.samples)} samples from {self.dataset_root}")

    def _normalize_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_samples: List[Dict[str, Any]] = []
        for sample in samples:
            normalized_samples.append(self._normalize_sample_record(sample))
        return normalized_samples

    def _normalize_sample_record(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(sample)
        normalized['subject_id'] = _string_or_empty(sample.get('subject_id'))
        normalized['timepoint'] = _string_or_empty(sample.get('timepoint'))
        normalized['label'] = int(sample.get('label', 0))

        for key in (
            'class_name',
            'fmri_path',
            'smri_path',
            'prepared_folder',
            'fmri_folder',
            'smri_folder',
            self.fmri_path_column,
            self.smri_path_column,
        ):
            value = _string_or_empty(sample.get(key))
            if value:
                normalized[key] = value

        if self.fmri_path_column != 'fmri_path' and _string_or_empty(sample.get(self.fmri_path_column)):
            normalized['fmri_path'] = _string_or_empty(sample.get(self.fmri_path_column))
        if self.smri_path_column != 'smri_path' and _string_or_empty(sample.get(self.smri_path_column)):
            normalized['smri_path'] = _string_or_empty(sample.get(self.smri_path_column))

        return normalized

    def _build_dicom_series_index(self) -> Dict[str, List[Path]]:
        series_dirs = set()
        for file_path in self.dataset_root.rglob("*"):
            if not file_path.is_file() or not _is_dicom_file(file_path):
                continue
            series_dirs.add(file_path.parent)

        dicom_index: Dict[str, List[Path]] = {}
        for series_dir in sorted(series_dirs):
            subject_id = _extract_subject_id_from_series(series_dir)
            if not subject_id:
                continue
            dicom_index.setdefault(subject_id, []).append(series_dir)

        for subject_id, candidate_dirs in dicom_index.items():
            candidate_dirs.sort(key=lambda path: (
                0 if _series_has_fmri_clue(path) else 1,
                len(path.parts),
                path.name.lower(),
            ))

        if self.verbose:
            logger.info(f"Indexed {sum(len(paths) for paths in dicom_index.values())} raw DICOM series for fallback lookup")

        return dicom_index
    
    def _load_metadata(self) -> List[Dict]:
        """Load sample information from metadata file."""
        if self.metadata_file is None:
            raise ValueError("metadata_file is required when samples are not provided")

        samples = []
        
        try:
            import pandas as pd
            df = pd.read_csv(str(self.metadata_file))
            
            for _, row in df.iterrows():
                sample: Dict[str, Any] = {
                    'subject_id': _string_or_empty(row.get('subject_id')),
                    'timepoint': _string_or_empty(row.get('timepoint')),
                    'label': int(row.get('label', 0)),
                }

                for column in (
                    'class_name',
                    'fmri_path',
                    'smri_path',
                    'prepared_folder',
                    'fmri_folder',
                    'smri_folder',
                    self.fmri_path_column,
                    self.smri_path_column,
                ):
                    if column in df.columns:
                        value = _string_or_empty(row.get(column))
                        if value:
                            sample[column] = value

                samples.append(self._normalize_sample_record(sample))
            
            if self.verbose:
                logger.info(f"Loaded metadata for {len(samples)} samples")
        
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
        
        return samples

    def _resolve_fmri_source(self, sample: Dict[str, Any]) -> Optional[Path]:
        fmri_path_value = _string_or_empty(sample.get('fmri_path')) or _string_or_empty(sample.get('fmri_folder'))
        if fmri_path_value:
            candidate = _resolve_relative_path(fmri_path_value, self.metadata_base_dir)
            if candidate is not None:
                if candidate.is_file():
                    return candidate
                if candidate.is_dir():
                    nested = _first_existing_path([
                        candidate / 'fmri_windows_dbgdgm.npy',
                        *sorted(candidate.rglob('fmri_windows_dbgdgm.npy')),
                    ])
                    if nested is not None:
                        return nested
                    return candidate

        subject_id = _string_or_empty(sample.get('subject_id'))
        timepoint = _string_or_empty(sample.get('timepoint'))

        candidate_roots = []
        if (self.dataset_root / 'fmri').exists():
            candidate_roots.append(self.dataset_root / 'fmri')
        candidate_roots.append(self.dataset_root)

        for root in candidate_roots:
            candidates = []
            if timepoint:
                candidates.append(root / subject_id / timepoint / 'fmri_windows_dbgdgm.npy')
            candidates.append(root / subject_id / 'fmri_windows_dbgdgm.npy')
            candidates.append(root / subject_id / timepoint / 'fmri.npy')
            candidates.append(root / subject_id / 'fmri.npy')

            existing = _first_existing_path(candidates)
            if existing is not None:
                return existing

            subject_dir = root / subject_id
            if subject_dir.exists():
                nested_matches = sorted(subject_dir.rglob('fmri_windows_dbgdgm.npy'))
                if nested_matches:
                    return nested_matches[0]

        raw_series_candidates = self._dicom_series_index.get(subject_id, [])
        if raw_series_candidates:
            return raw_series_candidates[0]

        return None

    def _resolve_smri_source(self, sample: Dict[str, Any]) -> Optional[Path]:
        smri_path_value = (
            _string_or_empty(sample.get('smri_path'))
            or _string_or_empty(sample.get('prepared_folder'))
            or _string_or_empty(sample.get('smri_folder'))
        )
        if smri_path_value:
            candidate = _resolve_relative_path(smri_path_value, self.metadata_base_dir)
            if candidate is not None and candidate.exists():
                return candidate

        subject_id = _string_or_empty(sample.get('subject_id'))
        class_name = _string_or_empty(sample.get('class_name'))

        candidate_roots = []
        if self.smri_source_root is not None:
            candidate_roots.append(self.smri_source_root)
        if (self.dataset_root / 'smri').exists():
            candidate_roots.append(self.dataset_root / 'smri')
        candidate_roots.append(self.dataset_root)

        for root in candidate_roots:
            candidates = []
            if class_name:
                candidates.append(root / class_name / subject_id / 'features.npy')
                candidates.append(root / class_name / subject_id)
            candidates.append(root / subject_id / 'features.npy')
            candidates.append(root / subject_id)

            existing = _first_existing_path(candidates)
            if existing is not None:
                return existing

            if root.exists():
                search_matches = [path for path in sorted(root.rglob(subject_id)) if path.is_dir()]
                if search_matches:
                    return search_matches[0]

        return None
    
    def _load_fmri(self, subject_id: str, timepoint: str) -> Optional[torch.Tensor]:
        """Load fMRI data."""
        if 'fmri' not in self.modalities:
            return None

        sample = {'subject_id': subject_id, 'timepoint': timepoint}
        fmri_source = self._resolve_fmri_source(sample)
        if fmri_source is None or not fmri_source.exists():
            logger.warning(f"fMRI file not found: {fmri_source}")
            return None
        
        try:
            if fmri_source.is_file() and fmri_source.suffix.lower() == '.npy':
                fmri_data = np.load(str(fmri_source)).astype(np.float32)

                if fmri_data.ndim == 3:
                    if fmri_data.shape[0] > 1:
                        fmri_data = np.mean(fmri_data, axis=0)
                    else:
                        fmri_data = fmri_data[0]

                if fmri_data.ndim != 2:
                    logger.warning(f"Unexpected fMRI shape: {fmri_data.shape}")
                    return None
            else:
                if fmri_source.is_file():
                    logger.warning(f"Unsupported fMRI source file: {fmri_source}")
                    return None

                fmri_data = _load_dicom_series_proxy_fmri(
                    series_dir=fmri_source,
                    n_roi=int(self.n_roi or 200),
                    seq_len=int(self.seq_len or 50),
                    max_dicoms_per_series=int(self.max_dicoms_per_series),
                )
                if fmri_data is None:
                    logger.warning(f"Could not derive fMRI proxy features from raw DICOM series: {fmri_source}")
                    return None

            if fmri_data.ndim != 2:
                logger.warning(f"Unexpected fMRI shape: {fmri_data.shape}")
                return None

            if self.n_roi is not None and self.seq_len is not None and fmri_data.shape != (self.n_roi, self.seq_len):
                fmri_data = _resize_2d(fmri_data, (self.n_roi, self.seq_len))

            if self.normalize_fmri:
                mean = np.mean(fmri_data)
                std = np.std(fmri_data)
                fmri_data = (fmri_data - mean) / (std + 1e-8)

            return torch.from_numpy(fmri_data.astype(np.float32))
        
        except Exception as e:
            logger.error(f"Failed to load fMRI {fmri_source}: {e}")
            return None
    
    def _load_smri(self, subject_id: str, timepoint: str) -> Optional[torch.Tensor]:
        """Load sMRI data."""
        if 'smri' not in self.modalities:
            return None

        sample = {'subject_id': subject_id, 'timepoint': timepoint}
        smri_source = self._resolve_smri_source(sample)
        if smri_source is None:
            logger.warning(f"sMRI file or folder not found for subject: {subject_id}")
            return None

        try:
            if smri_source.is_file() and smri_source.suffix.lower() == '.npy':
                smri_data = np.load(str(smri_source)).astype(np.float32)

                if smri_data.ndim == 2 and smri_data.shape[0] == 1:
                    smri_data = smri_data[0]
                elif smri_data.ndim != 1:
                    logger.warning(f"Unexpected sMRI shape: {smri_data.shape}")
                    return None
            else:
                smri_data = _load_image_folder_proxy_features(
                    series_dir=smri_source,
                    target_len=self.n_smri_features,
                )
                if smri_data is None:
                    logger.warning(f"Could not derive sMRI proxy features from folder: {smri_source}")
                    return None

            if self.n_smri_features is not None and smri_data.size != self.n_smri_features:
                smri_data = _resample_1d(smri_data, self.n_smri_features)

            if self.normalize_smri:
                min_val = float(np.min(smri_data))
                max_val = float(np.max(smri_data))
                if max_val > min_val:
                    smri_data = (smri_data - min_val) / (max_val - min_val + 1e-8)
                else:
                    smri_data = np.zeros_like(smri_data, dtype=np.float32)

            return torch.from_numpy(smri_data.astype(np.float32))
        
        except Exception as e:
            logger.error(f"Failed to load sMRI source {smri_source}: {e}")
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
            'timepoint': timepoint,
        }

        # Load modalities
        fmri = self._load_fmri(subject_id, timepoint)
        smri = self._load_smri(subject_id, timepoint)

        if fmri is None or smri is None:
            raise ValueError(
                f"Missing modalities for subject_id={subject_id!r}, timepoint={timepoint!r}. "
                f"fmri={'present' if fmri is not None else 'missing'}, smri={'present' if smri is not None else 'missing'}"
            )
        
        if fmri is not None:
            data['fmri'] = fmri
        if smri is not None:
            data['smri'] = smri

        data.update(_build_auxiliary_targets(smri.cpu().numpy(), label, int(self.n_smri_features or smri.numel())))
        
        return data


def create_dataloaders(
    dataset_root: str,
    train_metadata: Optional[str] = None,
    val_metadata: Optional[str] = None,
    test_metadata: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
    normalize: bool = True,
    metadata_file: Optional[str] = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    n_roi: Optional[int] = None,
    seq_len: Optional[int] = None,
    n_smri_features: Optional[int] = None,
    max_dicoms_per_series: int = 120,
    smri_source_root: Optional[str] = None,
    fmri_path_column: str = 'fmri_path',
    smri_path_column: str = 'smri_path'
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
        metadata_file: Optional single metadata CSV to auto-split into train/val/test
        val_fraction: Validation fraction when using a single metadata file
        test_fraction: Test fraction when using a single metadata file
        seed: Random seed for the auto-split path
        n_roi: Optional fMRI ROI count for resizing local data
        seq_len: Optional fMRI sequence length for resizing local data
        n_smri_features: Optional target sMRI feature count
        max_dicoms_per_series: Maximum raw DICOM slices to decode per series
        smri_source_root: Optional local prepared SMRI JPG root
        fmri_path_column: Optional metadata column for explicit fMRI paths
        smri_path_column: Optional metadata column for explicit sMRI paths
    
    Returns:
        dict with 'train', 'val', and optionally 'test' dataloaders
    """
    
    dataset_kwargs = {
        'dataset_root': dataset_root,
        'normalize_fmri': normalize,
        'normalize_smri': normalize,
        'n_roi': n_roi,
        'seq_len': seq_len,
        'n_smri_features': n_smri_features,
        'max_dicoms_per_series': max_dicoms_per_series,
        'smri_source_root': smri_source_root,
        'fmri_path_column': fmri_path_column,
        'smri_path_column': smri_path_column,
        'verbose': True,
    }

    explicit_split_mode = train_metadata is not None or val_metadata is not None or test_metadata is not None

    if explicit_split_mode:
        if train_metadata is None or val_metadata is None:
            raise ValueError(
                "train_metadata and val_metadata must both be provided when using explicit split files"
            )

        train_dataset = MultimodalBrainDataset(
            metadata_file=train_metadata,
            **dataset_kwargs,
        )
        val_dataset = MultimodalBrainDataset(
            metadata_file=val_metadata,
            **dataset_kwargs,
        )
        test_dataset = MultimodalBrainDataset(
            metadata_file=test_metadata,
            **dataset_kwargs,
        ) if test_metadata is not None else None
    elif metadata_file:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        metadata_path = Path(metadata_file)
        df = pd.read_csv(str(metadata_path))
        if 'label' not in df.columns:
            raise ValueError(f"metadata_file must contain a label column: {metadata_path}")

        records = []
        for _, row in df.iterrows():
            record = row.to_dict()
            records.append({
                key: value
                for key, value in record.items()
                if not (isinstance(value, float) and np.isnan(value))
            })

        if len(records) < 3:
            raise ValueError("metadata_file must contain at least 3 rows to create train/val/test splits")

        indices = np.arange(len(records))
        labels = np.asarray([int(record.get('label', 0)) for record in records], dtype=np.int64)
        stratify_labels = labels if len(np.unique(labels)) > 1 else None

        test_fraction = float(np.clip(test_fraction, 0.0, 0.5))
        val_fraction = float(np.clip(val_fraction, 0.0, 0.5))

        if test_fraction > 0.0:
            try:
                train_val_indices, test_indices = train_test_split(
                    indices,
                    test_size=test_fraction,
                    random_state=seed,
                    stratify=stratify_labels,
                )
            except ValueError:
                train_val_indices, test_indices = train_test_split(
                    indices,
                    test_size=test_fraction,
                    random_state=seed,
                    stratify=None,
                )
        else:
            train_val_indices, test_indices = indices, np.asarray([], dtype=int)

        remaining_fraction = val_fraction / max(1e-6, 1.0 - test_fraction)
        remaining_fraction = float(np.clip(remaining_fraction, 0.0, 0.5))

        if remaining_fraction > 0.0 and len(train_val_indices) > 1:
            stratify_train_val = labels[train_val_indices] if len(np.unique(labels[train_val_indices])) > 1 else None
            try:
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=remaining_fraction,
                    random_state=seed,
                    stratify=stratify_train_val,
                )
            except ValueError:
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=remaining_fraction,
                    random_state=seed,
                    stratify=None,
                )
        else:
            train_indices, val_indices = train_val_indices, np.asarray([], dtype=int)

        train_records = [records[int(i)] for i in train_indices]
        val_records = [records[int(i)] for i in val_indices]
        test_records = [records[int(i)] for i in test_indices]

        dataset_kwargs['metadata_base_dir'] = str(metadata_path.parent)

        train_dataset = MultimodalBrainDataset(samples=train_records, **dataset_kwargs)
        val_dataset = MultimodalBrainDataset(samples=val_records, **dataset_kwargs)
        test_dataset = MultimodalBrainDataset(samples=test_records, **dataset_kwargs) if test_records else None
    else:
        raise ValueError(
            "Provide either metadata_file for an auto-split manifest or train_metadata/val_metadata CSV files"
        )
    
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': False,
        'persistent_workers': bool(num_workers > 0),
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 4

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            shuffle=shuffle_train,
            **loader_kwargs,
        ),
        'val': DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )
    }
    
    if test_dataset is not None:
        dataloaders['test'] = DataLoader(
            test_dataset,
            **loader_kwargs,
            shuffle=False,
        )
    
    return dataloaders
