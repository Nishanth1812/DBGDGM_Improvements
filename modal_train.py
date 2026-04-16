"""
Modal-Optimized Training Entry Point for MM-DBGDGM

This module provides Serverless training on Modal with:
- GPU support (H100/A100)
- Volume-based data and checkpoint persistence
- Distributed training capabilities
- Full integration with existing MM-DBGDGM pipeline

Usage:
    modal run modal_train.py --config configs/default_config.yaml --data-path /path/to/data
    
    Or programmatically:
    from modal_train import train
    train.remote()
"""

import os
import sys
import json
import argparse
import yaml
import hashlib
import shutil
import zipfile
import re
import time
import warnings
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from datetime import datetime

import modal
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Add MM_DBGDGM to path for imports
sys.path.insert(0, str(Path(__file__).parent / "MM_DBGDGM"))
# Also add root path for Modal containers (where files are copied)
sys.path.insert(0, "/root/MM_DBGDGM")

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

# Create Modal app
app = modal.App(name="mm-dbgdgm-training")

# Setup volumes for data persistence
data_volume = modal.Volume.from_name("mm-dbgdgm-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("mm-dbgdgm-checkpoints", create_if_missing=True)
logs_volume = modal.Volume.from_name("mm-dbgdgm-logs", create_if_missing=True)

# Mount points in the container
MOUNT_PATH = Path("/data")
CHECKPOINT_DIR = Path("/checkpoints")
LOGS_DIR = Path("/logs")

# GPU configuration
# Best value for 50-epoch training: L40S ($1.95/hr) - excellent cost/performance
try:
    GPU_CONFIG = modal.gpu.L40S.with_count(1)
except AttributeError:
    GPU_CONFIG = "L40S"

STAGE_NAMES = ["CN", "eMCI", "lMCI", "AD"]
CACHE_SOURCE_KIND_DRIVE = "drive_dicom"
CACHE_SOURCE_KIND_UPLOADED_BUNDLE = "uploaded_bundle_dicom"
CACHE_SOURCE_KIND = CACHE_SOURCE_KIND_DRIVE
DEFAULT_UPLOADED_BUNDLE_NAME = "prepared_dicom_bundle.zip"
SERIES_SAMPLE_CACHE_VERSION = 1
PYPICOM_UID_WARNING_PATTERN = r"Invalid value for VR UI:.*"

REQUIRED_SAMPLE_KEYS = {
    'fmri',
    'smri',
    'label',
    'hippo_vol',
    'cortical_thinning',
    'dmn_conn',
    'nss',
    'survival_times',
    'survival_events',
}

# Alternative options (if you need different tradeoffs):
# GPU_CONFIG = modal.gpu.A100(memory="40GB").with_count(1)  # A100-40GB ($2.10/hr, ~10% faster)
# GPU_CONFIG = modal.gpu.A100(memory="80GB").with_count(1)  # A100-80GB ($2.50/hr, same speed as 40GB)
# GPU_CONFIG = modal.gpu.L4.with_count(1)                   # L4 ($0.80/hr, for quick testing only)
# GPU_CONFIG = modal.gpu.T4.with_count(1)                   # T4 ($0.59/hr, very cheap, quite slow)

# Image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "gdown>=5.2.0",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "nilearn>=0.10.0",
        "nibabel>=5.0.0",
        "pydicom>=2.4.0",
        "networkx>=3.2",
        "numpy>=1.26",
        "scipy>=1.12",
        "scikit-learn>=1.4",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "matplotlib>=3.8",
        "tensorboard>=2.15",
        "tqdm>=4.66",
        "opencv-python-headless>=4.9"
    ])
    .add_local_dir(Path(__file__).parent / "MM_DBGDGM", remote_path="/root/MM_DBGDGM")
)

# ============================================================================
# LOGGER SETUP
# ============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    _configure_pydicom_warning_filters()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("mm-dbgdgm-modal")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s - %(name)s - %(levelname)s] %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class _SuppressPydicomUidWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Invalid value for VR UI:" not in record.getMessage()


def _configure_pydicom_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=PYPICOM_UID_WARNING_PATTERN,
        category=UserWarning,
    )
    pydicom_logger = logging.getLogger("pydicom")
    if not any(isinstance(filter_obj, _SuppressPydicomUidWarningFilter) for filter_obj in pydicom_logger.filters):
        pydicom_logger.addFilter(_SuppressPydicomUidWarningFilter())


@contextmanager
def log_phase(logger: logging.Logger, message: str):
    """Log the start and end time for a high-level training phase."""
    start = time.perf_counter()
    logger.info(f"{message} started")
    try:
        yield
    finally:
        logger.info(f"{message} finished in {time.perf_counter() - start:.1f}s")


class ADNIRealTensorDataset(Dataset):
    """Tensor-backed dataset that matches the trainer batch contract."""

    def __init__(self, samples: List[Dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a 1D array to target length."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if signal.size == 1:
        return np.full(target_len, float(signal[0]), dtype=np.float32)

    old_x = np.linspace(0.0, 1.0, num=signal.size, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(new_x, old_x, signal).astype(np.float32)


def _extract_subject_id(series_dir: Path) -> str:
    """Extract subject id from folder shape .../ADNI/<subject>/..."""
    parts = list(series_dir.parts)
    if "ADNI" in parts:
        idx = parts.index("ADNI")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    folder_name = series_dir.name.strip()
    if folder_name:
        if re.search(r"(?i)^OAS\d+_\d+_MR\d+$", folder_name):
            return folder_name
        if re.search(r"\d", folder_name) and folder_name.lower() not in {"non demented", "very mild dementia", "mild dementia", "moderate dementia"}:
            return folder_name

    image_like_files = [
        file_path for file_path in series_dir.iterdir()
        if file_path.is_file() and (_is_image_file(file_path) or file_path.suffix.lower() == ".dcm" or file_path.suffix == "")
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


def _canonical_subject_key(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().casefold() if ch.isalnum())


def _deterministic_label(subject_id: str, num_classes: int) -> int:
    """Fallback label assignment when diagnosis labels are unavailable."""
    digest = hashlib.md5(subject_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_classes


def _load_optional_label_map(
    csv_path: Optional[str],
    logger: logging.Logger,
    base_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Load optional subject-to-label mapping if provided."""
    if not csv_path:
        return {}

    label_path = Path(csv_path)
    label_file = label_path
    if base_dir is not None and not label_path.is_absolute():
        label_file = base_dir / label_path
    if not label_file.exists():
        if base_dir is not None and not label_path.is_absolute():
            matches = list(base_dir.rglob(label_path.name))
            if matches:
                label_file = matches[0]
        if not label_file.exists():
            logger.warning(f"labels_csv not found at {label_file}; using deterministic fallback labels")
            return {}

    try:
        import pandas as pd

        df = pd.read_csv(label_file)
        subject_col_candidates = ["subject_id", "Subject", "PTID", "RID"]
        label_col_candidates = ["label", "Label", "diagnosis", "Diagnosis", "DX"]

        subject_col = next((c for c in subject_col_candidates if c in df.columns), None)
        label_col = next((c for c in label_col_candidates if c in df.columns), None)

        if subject_col is None or label_col is None:
            logger.warning(
                f"labels_csv is missing expected columns (found: {list(df.columns)}); using deterministic fallback labels"
            )
            return {}

        stage_map = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}
        label_map: Dict[str, int] = {}
        for _, row in df.iterrows():
            subject = str(row[subject_col]).strip()
            raw_label = row[label_col]
            try:
                label_value = int(raw_label)
            except Exception:
                label_value = stage_map.get(str(raw_label).strip().upper(), 0)
            label_map[subject] = label_value
            canonical_subject = _canonical_subject_key(subject)
            if canonical_subject and canonical_subject not in label_map:
                label_map[canonical_subject] = label_value

        logger.info(f"Loaded {len(label_map)} labels from {label_file}")
        return label_map
    except Exception as exc:
        logger.warning(f"Failed to parse labels_csv ({exc}); using deterministic fallback labels")
        return {}


def _download_drive_folder(drive_folder_url: str, download_dir: Path, logger: logging.Logger) -> List[Path]:
    """Download all files from Google Drive folder and return ZIP paths."""
    download_dir.mkdir(parents=True, exist_ok=True)

    existing_zips = sorted(download_dir.rglob("*.zip"))
    if existing_zips:
        logger.info(f"Reusing {len(existing_zips)} previously downloaded zip files")
        return existing_zips

    logger.info(
        f"Downloading dataset folder from Google Drive: {drive_folder_url} "
        f"-> {download_dir}"
    )
    import gdown

    gdown.download_folder(
        url=drive_folder_url,
        output=str(download_dir),
        quiet=False,
        use_cookies=False,
    )

    zip_files = sorted(download_dir.rglob("*.zip"))
    if not zip_files:
        raise RuntimeError(
            f"No zip files found after download. Verify folder accessibility: {drive_folder_url}"
        )

    logger.info(f"Drive download complete: found {len(zip_files)} zip files in {download_dir}")
    return zip_files


def _cleanup_legacy_drive_artifacts(pipeline_root: Path, logger: logging.Logger) -> None:
    """Remove persistent download/extract artifacts from older runs."""
    for legacy_name in ("downloads", "extracted"):
        legacy_path = pipeline_root / legacy_name
        if legacy_path.exists():
            logger.info(f"Removing legacy persistent artifact directory: {legacy_path}")
            shutil.rmtree(legacy_path, ignore_errors=True)


def _sample_looks_valid(sample: Any) -> bool:
    """Return True when cached sample payload has the expected tensor keys."""
    return isinstance(sample, dict) and REQUIRED_SAMPLE_KEYS.issubset(sample.keys())


def _atomic_torch_save(payload: Any, target_path: Path) -> None:
    """Safely write torch payloads to disk without leaving partial files."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    torch.save(payload, temp_path)
    temp_path.replace(target_path)


def _compute_series_sample_cache_path(
    series_dir: Path,
    extract_root: Path,
    source_identifier: str,
    source_kind: str,
    label: int,
    n_roi: int,
    seq_len: int,
    n_smri_features: int,
    max_dicoms_per_series: int,
    cache_dir: Path,
) -> Path:
    """Build a deterministic cache path for one preprocessed series sample."""
    try:
        series_identifier = str(series_dir.relative_to(extract_root))
    except Exception:
        series_identifier = str(series_dir)

    key_source = "|".join([
        source_kind,
        str(SERIES_SAMPLE_CACHE_VERSION),
        source_identifier,
        series_identifier,
        str(label),
        str(n_roi),
        str(seq_len),
        str(n_smri_features),
        str(max_dicoms_per_series),
    ])
    sample_key = hashlib.md5(key_source.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{sample_key}.pt"


def _compute_drive_zip_cache_dir(source_identifier: str) -> Path:
    """Return the persistent volume directory used to cache raw ZIP files."""
    cache_dir = MOUNT_PATH / "adni_drive_pipeline" / "raw_zips"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _resolve_uploaded_bundle_zip_files(
    uploaded_bundle_name: str,
    download_dir: Path,
    logger: logging.Logger,
) -> List[Path]:
    """Resolve one uploaded bundle ZIP from the Modal raw_zips cache."""
    requested_name = str(uploaded_bundle_name or DEFAULT_UPLOADED_BUNDLE_NAME).strip()
    requested_stem = Path(requested_name).stem

    logger.info(f"Scanning Modal raw ZIP cache at {download_dir} for uploaded bundle '{requested_name}'")
    zip_files = sorted(download_dir.rglob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No ZIP files found in Modal raw zip cache: {download_dir}")
    logger.info(f"Found {len(zip_files)} ZIP candidate(s) in Modal raw ZIP cache")

    candidate_names = [requested_name]
    if not requested_name.lower().endswith(".zip"):
        candidate_names.append(f"{requested_name}.zip")
    candidate_names.append(DEFAULT_UPLOADED_BUNDLE_NAME)

    for candidate_name in candidate_names:
        candidate_stem = Path(candidate_name).stem
        matches = [
            zip_path for zip_path in zip_files
            if zip_path.name == candidate_name or zip_path.stem == candidate_stem
        ]
        if matches:
            selected = matches[0]
            if len(matches) > 1:
                logger.warning(
                    f"Multiple ZIPs matched uploaded bundle name '{requested_name}'; using {selected.name}"
                )
            logger.info(f"Using uploaded bundle ZIP from Modal volume: {selected}")
            return [selected]

    if requested_name == DEFAULT_UPLOADED_BUNDLE_NAME and len(zip_files) == 1:
        logger.info(f"Using only ZIP in Modal raw cache: {zip_files[0]}")
        return zip_files

    available = ", ".join(zip_path.name for zip_path in zip_files[:10])
    if len(zip_files) > 10:
        available += ", ..."
    raise FileNotFoundError(
        f"Could not find uploaded bundle '{requested_name}' under {download_dir}. Available ZIPs: {available}"
    )


def _extract_zip_files(zip_files: List[Path], extract_root: Path, logger: logging.Logger) -> None:
    """Extract zip files once using marker files for idempotency."""
    extract_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {len(zip_files)} zip files into {extract_root}")

    for zip_path in zip_files:
        target_dir = extract_root / zip_path.stem
        marker = target_dir / ".extracted"
        if marker.exists():
            logger.info(f"Skipping already extracted archive: {zip_path.name}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        archive_size_mb = zip_path.stat().st_size / 1_000_000.0 if zip_path.exists() else 0.0
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            total_members = len(members)
            total_uncompressed_mb = sum(member.file_size for member in members) / 1_000_000.0
            logger.info(
                f"Extracting {zip_path.name} ({archive_size_mb:.1f} MB compressed, "
                f"{total_uncompressed_mb:.1f} MB uncompressed, {total_members} entries) -> {target_dir}"
            )

            progress_interval = max(1, total_members // 10)
            for member_index, member in enumerate(members, start=1):
                zf.extract(member, target_dir)
                if member_index == 1 or member_index == total_members or member_index % progress_interval == 0:
                    logger.info(
                        f"Extraction progress for {zip_path.name}: {member_index}/{total_members} entries extracted"
                    )

        marker.touch()
        logger.info(f"Finished extracting {zip_path.name} into {target_dir}")

    logger.info(f"Extraction complete: contents available under {extract_root}")


def _find_series_dirs_with_dicoms(root_dir: Path) -> List[Path]:
    """Return all directories containing DICOM or image files."""
    series_dirs = set()
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".dcm" or suffix == "" or _is_image_file(file_path):
            series_dirs.add(file_path.parent)

    return sorted(series_dirs)


def _select_candidate_series(
    series_dirs: List[Path],
    use_filename_clues: bool,
    strict_fmri_clues: bool,
    logger: logging.Logger,
) -> List[Path]:
    """Filter candidate series using fMRI clues when enabled."""
    if not use_filename_clues:
        return series_dirs

    clue_dirs = [series_dir for series_dir in series_dirs if _series_has_fmri_clue(series_dir)]
    if clue_dirs:
        logger.info(
            f"Selected {len(clue_dirs)} series with fMRI filename/folder clues out of {len(series_dirs)} total"
        )
        return clue_dirs

    if strict_fmri_clues:
        raise RuntimeError("No series matched fMRI clues in folder/file names while strict_fmri_clues=true")

    logger.info("No explicit fMRI clues found; falling back to all detected DICOM series")
    return series_dirs


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

IMAGE_FILE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def _is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_FILE_EXTENSIONS


def _extract_numeric_hint(name: str) -> Optional[int]:
    """Extract trailing numeric clue from names like I240811 / IMG_000123."""
    matches = re.findall(r"(\d+)", name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _series_has_fmri_clue(series_dir: Path) -> bool:
    """Check whether folder names indicate likely fMRI/BOLD acquisition."""
    path_lower = "/".join(series_dir.parts).lower()
    return any(token in path_lower for token in FMRI_CLUE_TOKENS)


def _to_int_or_default(value: Any, default_value: int) -> int:
    try:
        return int(value)
    except Exception:
        return default_value


def _get_series_file_groups(series_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Return DICOM-like files and image files from a series directory."""
    dicom_files: List[Path] = []
    image_files: List[Path] = []

    for file_path in series_dir.iterdir():
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".dcm" or suffix == "":
            dicom_files.append(file_path)
        elif _is_image_file(file_path):
            image_files.append(file_path)

    return sorted(dicom_files), sorted(image_files)


def _ordered_dicom_files(series_dir: Path, max_dicoms_per_series: int) -> Tuple[List[Path], List[Any]]:
    """
    Return DICOM files ordered by robust clues:
    1) TemporalPositionIdentifier / AcquisitionNumber / InstanceNumber
    2) Numeric cues in file name (e.g., I240811)
    3) Lexicographic file name fallback
    """
    import pydicom

    records = []
    for file_path in series_dir.iterdir():
        if not file_path.is_file() or not (file_path.suffix.lower() == ".dcm" or file_path.suffix == ""):
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

        numeric_hint = _extract_numeric_hint(file_path.stem)
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


def _ordered_image_files(series_dir: Path, max_images_per_series: int) -> List[Path]:
    """Return JPG/PNG/etc. files ordered by filename numeric hint and lexicographic fallback."""
    records = []
    for file_path in series_dir.iterdir():
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


def _build_sample_from_dicom_series(
    series_dir: Path,
    label: int,
    n_roi: int,
    seq_len: int,
    n_smri_features: int,
    max_dicoms_per_series: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Convert one DICOM series folder into model-ready tensors."""
    import pydicom

    dcm_files, headers = _ordered_dicom_files(series_dir, max_dicoms_per_series)
    if not dcm_files:
        return None

    slice_means = []
    slice_stds = []
    mins = []
    maxs = []

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
            mins.append(float(np.min(pixels)))
            maxs.append(float(np.max(pixels)))
        except Exception:
            continue

    if not slice_means:
        return None

    def _header_float_values(attribute: str) -> List[float]:
        values: List[float] = []
        for header in headers:
            raw = getattr(header, attribute, None)
            if raw is None:
                continue
            try:
                if isinstance(raw, (list, tuple)):
                    values.append(float(raw[0]))
                else:
                    values.append(float(raw))
            except Exception:
                continue
        return values

    tr_values = _header_float_values("RepetitionTime")
    te_values = _header_float_values("EchoTime")
    slice_thickness_values = _header_float_values("SliceThickness")
    row_values = _header_float_values("Rows")
    col_values = _header_float_values("Columns")

    pixel_spacing_values: List[float] = []
    for header in headers:
        raw_spacing = getattr(header, "PixelSpacing", None)
        if raw_spacing is None:
            continue
        try:
            pixel_spacing_values.append(float(raw_spacing[0]))
            if len(raw_spacing) > 1:
                pixel_spacing_values.append(float(raw_spacing[1]))
        except Exception:
            continue

    signal = _resample_1d(np.asarray(slice_means, dtype=np.float32), seq_len)
    signal = (signal - signal.mean()) / (signal.std() + 1e-6)

    # Build [n_roi, seq_len] tensor from temporal signal.
    roi_scale = np.linspace(0.9, 1.1, num=n_roi, dtype=np.float32)[:, None]
    phase = np.linspace(0.0, np.pi, num=n_roi, dtype=np.float32)[:, None]
    time_axis = np.linspace(0.0, 2.0 * np.pi, num=seq_len, dtype=np.float32)[None, :]
    fmri = roi_scale * signal[None, :] + 0.05 * np.sin(time_axis + phase)

    clue_text = ("/".join(series_dir.parts) + " " + " ".join(p.name for p in dcm_files)).lower()
    clue_rest = 1.0 if "rest" in clue_text else 0.0
    clue_bold = 1.0 if "bold" in clue_text else 0.0
    clue_epi = 1.0 if "epi" in clue_text or "ep2d" in clue_text else 0.0
    clue_fmri = 1.0 if "fmri" in clue_text else 0.0
    clue_rs = 1.0 if "rsfmri" in clue_text or "resting_state" in clue_text else 0.0

    smri_raw = np.asarray([
        float(np.mean(slice_means)),
        float(np.std(slice_means)),
        float(np.mean(slice_stds)),
        float(np.min(mins)),
        float(np.max(maxs)),
        float(np.mean(tr_values)) if tr_values else 0.0,
        float(np.mean(te_values)) if te_values else 0.0,
        float(np.mean(slice_thickness_values)) if slice_thickness_values else 0.0,
        float(np.mean(row_values) / 512.0) if row_values else 0.0,
        float(np.mean(col_values) / 512.0) if col_values else 0.0,
        float(np.mean(pixel_spacing_values)) if pixel_spacing_values else 0.0,
        float(_series_has_fmri_clue(series_dir)),
        clue_rest,
        clue_bold,
        clue_epi,
        clue_fmri,
        clue_rs,
    ], dtype=np.float32)
    smri_raw = np.nan_to_num(smri_raw, nan=0.0, posinf=0.0, neginf=0.0)

    smri = _resample_1d(smri_raw, n_smri_features) if n_smri_features != smri_raw.size else smri_raw
    smri_min, smri_max = float(smri.min()), float(smri.max())
    if smri_max > smri_min:
        smri = (smri - smri_min) / (smri_max - smri_min)
    else:
        smri = np.zeros_like(smri, dtype=np.float32)

    hippo_vol = 2500.0 + (1.0 - float(smri[0])) * 2000.0
    cortical_thinning = 0.1 + float(smri[min(1, n_smri_features - 1)]) * 0.4
    dmn_conn = float(np.clip(1.0 - float(smri[min(2, n_smri_features - 1)]), 0.0, 1.0))
    nss = float(np.clip(20.0 + label * 20.0 + float(smri[0]) * 15.0, 0.0, 100.0))

    survival_times = np.asarray([8.0, 5.5, 3.0], dtype=np.float32) - label * 0.7
    survival_times = np.clip(survival_times, 0.3, None)
    survival_events = np.ones(3, dtype=np.float32)

    return {
        'fmri': torch.from_numpy(fmri.astype(np.float32)),
        'smri': torch.from_numpy(smri.astype(np.float32)),
        'label': torch.tensor(label, dtype=torch.long),
        'hippo_vol': torch.tensor(hippo_vol, dtype=torch.float32),
        'cortical_thinning': torch.tensor(cortical_thinning, dtype=torch.float32),
        'dmn_conn': torch.tensor(dmn_conn, dtype=torch.float32),
        'nss': torch.tensor(nss, dtype=torch.float32),
        'survival_times': torch.from_numpy(survival_times),
        'survival_events': torch.from_numpy(survival_events),
    }


def _build_sample_from_image_series(
    series_dir: Path,
    label: int,
    n_roi: int,
    seq_len: int,
    n_smri_features: int,
    max_images_per_series: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Convert one JPG/PNG series folder into model-ready tensors for demo use."""
    import cv2

    image_files = _ordered_image_files(series_dir, max_images_per_series)
    if not image_files:
        return None

    image_means = []
    image_stds = []
    mins = []
    maxs = []
    row_values = []
    col_values = []
    file_sizes = []

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

    signal = _resample_1d(np.asarray(image_means, dtype=np.float32), seq_len)
    signal = (signal - signal.mean()) / (signal.std() + 1e-6)

    roi_scale = np.linspace(0.9, 1.1, num=n_roi, dtype=np.float32)[:, None]
    phase = np.linspace(0.0, np.pi, num=n_roi, dtype=np.float32)[:, None]
    time_axis = np.linspace(0.0, 2.0 * np.pi, num=seq_len, dtype=np.float32)[None, :]
    fmri = roi_scale * signal[None, :] + 0.05 * np.sin(time_axis + phase)

    clue_text = ("/".join(series_dir.parts) + " " + " ".join(p.name for p in image_files)).lower()
    clue_rest = 1.0 if "rest" in clue_text else 0.0
    clue_bold = 1.0 if "bold" in clue_text else 0.0
    clue_epi = 1.0 if "epi" in clue_text or "ep2d" in clue_text else 0.0
    clue_fmri = 1.0 if "fmri" in clue_text else 0.0
    clue_rs = 1.0 if "rsfmri" in clue_text or "resting_state" in clue_text else 0.0

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
        float(_series_has_fmri_clue(series_dir)),
        clue_rest,
        clue_bold,
        clue_epi,
        clue_fmri,
        clue_rs,
    ], dtype=np.float32)
    smri_raw = np.nan_to_num(smri_raw, nan=0.0, posinf=0.0, neginf=0.0)

    smri = _resample_1d(smri_raw, n_smri_features) if n_smri_features != smri_raw.size else smri_raw
    smri_min, smri_max = float(smri.min()), float(smri.max())
    if smri_max > smri_min:
        smri = (smri - smri_min) / (smri_max - smri_min)
    else:
        smri = np.zeros_like(smri, dtype=np.float32)

    hippo_vol = 2500.0 + (1.0 - float(smri[0])) * 2000.0
    cortical_thinning = 0.1 + float(smri[min(1, n_smri_features - 1)]) * 0.4
    dmn_conn = float(np.clip(1.0 - float(smri[min(2, n_smri_features - 1)]), 0.0, 1.0))
    nss = float(np.clip(20.0 + label * 20.0 + float(smri[0]) * 15.0, 0.0, 100.0))

    survival_times = np.asarray([8.0, 5.5, 3.0], dtype=np.float32) - label * 0.7
    survival_times = np.clip(survival_times, 0.3, None)
    survival_events = np.ones(3, dtype=np.float32)

    return {
        'fmri': torch.from_numpy(fmri.astype(np.float32)),
        'smri': torch.from_numpy(smri.astype(np.float32)),
        'label': torch.tensor(label, dtype=torch.long),
        'hippo_vol': torch.tensor(hippo_vol, dtype=torch.float32),
        'cortical_thinning': torch.tensor(cortical_thinning, dtype=torch.float32),
        'dmn_conn': torch.tensor(dmn_conn, dtype=torch.float32),
        'nss': torch.tensor(nss, dtype=torch.float32),
        'survival_times': torch.from_numpy(survival_times),
        'survival_events': torch.from_numpy(survival_events),
    }


def _build_sample_from_series(
    series_dir: Path,
    label: int,
    n_roi: int,
    seq_len: int,
    n_smri_features: int,
    max_dicoms_per_series: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build a training/inference sample from either DICOM or JPG/PNG series."""
    dicom_files, image_files = _get_series_file_groups(series_dir)
    if dicom_files:
        sample = _build_sample_from_dicom_series(
            series_dir=series_dir,
            label=label,
            n_roi=n_roi,
            seq_len=seq_len,
            n_smri_features=n_smri_features,
            max_dicoms_per_series=max_dicoms_per_series,
        )
        if sample is not None:
            return sample

    if image_files:
        return _build_sample_from_image_series(
            series_dir=series_dir,
            label=label,
            n_roi=n_roi,
            seq_len=seq_len,
            n_smri_features=n_smri_features,
            max_images_per_series=max_dicoms_per_series,
        )

    return None


def _split_series_for_inference(
    series_dirs: List[Path],
    seed: int,
    holdout_ratio: float,
) -> Tuple[List[Path], List[Path]]:
    """Split candidate series into train pool and deterministic inference holdout."""
    if not series_dirs:
        return [], []

    holdout_ratio = float(np.clip(holdout_ratio, 0.0, 0.5))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(series_dirs))
    rng.shuffle(idx)

    if holdout_ratio <= 0.0:
        return [series_dirs[int(i)] for i in idx], []

    # Keep at least 3 samples for train/val/test downstream split.
    max_holdout = max(0, len(series_dirs) - 3)
    if max_holdout == 0:
        return [series_dirs[int(i)] for i in idx], []

    n_holdout = max(1, int(np.floor(len(series_dirs) * holdout_ratio)))
    n_holdout = min(n_holdout, max_holdout)

    holdout_ids = idx[:n_holdout]
    train_ids = idx[n_holdout:]

    train_series = [series_dirs[int(i)] for i in train_ids]
    holdout_series = [series_dirs[int(i)] for i in holdout_ids]
    return train_series, holdout_series


def _persist_inference_holdout_artifacts(
    inference_series_dirs: List[Path],
    output_prefix: str,
    cache_key: str,
    holdout_ratio: float,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Persist a copy of inference holdout series plus a reusable manifest in volume."""
    holdout_root = MOUNT_PATH / "adni_drive_pipeline" / "inference_holdout" / cache_key
    holdout_root.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    copied = 0
    missing_source_dirs = 0

    for idx, source_dir in enumerate(inference_series_dirs, start=1):
        if not source_dir.exists():
            missing_source_dirs += 1
            logger.warning(f"Inference holdout source is missing and will be skipped: {source_dir}")
            continue

        subject_id = _extract_subject_id(source_dir)
        series_hash = hashlib.md5(str(source_dir).encode("utf-8")).hexdigest()[:10]
        holdout_dir = holdout_root / f"{idx:05d}_{subject_id}_{series_hash}"

        if not holdout_dir.exists():
            shutil.copytree(source_dir, holdout_dir)
            copied += 1

        entries.append({
            'index': idx - 1,
            'subject_id': subject_id,
            'source_series_dir': str(source_dir),
            'holdout_series_dir': str(holdout_dir),
        })

    manifest_payload = {
        'output_prefix': output_prefix,
        'cache_key': cache_key,
        'holdout_ratio': float(holdout_ratio),
        'holdout_root': str(holdout_root),
        'num_series': len(entries),
        'num_missing_source_dirs': missing_source_dirs,
        'entries': entries,
        'updated_at': datetime.now().isoformat(),
    }

    holdout_manifest_path = holdout_root / "manifest.json"
    with open(holdout_manifest_path, 'w') as f:
        json.dump(manifest_payload, f, indent=2)

    output_dir = CHECKPOINT_DIR / output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest_path = output_dir / "inference_holdout_manifest.json"
    with open(checkpoint_manifest_path, 'w') as f:
        json.dump(manifest_payload, f, indent=2)

    logger.info(
        "Inference holdout prepared: "
        f"series={len(entries)}, copied_now={copied}, holdout_root={holdout_root}"
    )

    return {
        'manifest_path': str(checkpoint_manifest_path),
        'holdout_root': str(holdout_root),
        'num_series': len(entries),
        'num_missing_source_dirs': missing_source_dirs,
    }


def _load_saved_inference_holdout_series(
    output_prefix: str,
    logger: logging.Logger,
) -> Tuple[List[Path], Optional[Path], Optional[Path]]:
    """Load saved holdout manifest and return reusable series directories."""
    manifest_path = CHECKPOINT_DIR / output_prefix / "inference_holdout_manifest.json"
    if not manifest_path.exists():
        return [], None, None

    try:
        with open(manifest_path) as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to read inference holdout manifest at {manifest_path}: {exc}")
        return [], manifest_path, None

    series_dirs: List[Path] = []
    for entry in payload.get('entries', []):
        holdout_dir_str = str(entry.get('holdout_series_dir', '')).strip()
        source_dir_str = str(entry.get('source_series_dir', '')).strip()

        holdout_dir = Path(holdout_dir_str) if holdout_dir_str else None
        source_dir = Path(source_dir_str) if source_dir_str else None

        if holdout_dir is not None and holdout_dir.exists():
            series_dirs.append(holdout_dir)
        elif source_dir is not None and source_dir.exists():
            series_dirs.append(source_dir)

    holdout_root_str = str(payload.get('holdout_root', '')).strip()
    holdout_root = Path(holdout_root_str) if holdout_root_str else None

    if series_dirs:
        logger.info(f"Loaded {len(series_dirs)} series from saved inference holdout manifest")
    else:
        logger.warning(
            f"Inference holdout manifest found at {manifest_path} but no reusable series paths exist"
        )

    return series_dirs, manifest_path, holdout_root


def _update_holdout_manifest_with_results(
    manifest_path: Optional[Path],
    predictions_path: Path,
    logger: logging.Logger,
) -> None:
    """Record latest inference predictions path in holdout manifest for quick local reuse."""
    if manifest_path is None or not manifest_path.exists():
        return

    try:
        with open(manifest_path) as f:
            payload = json.load(f)

        payload['latest_predictions_path'] = str(predictions_path)
        payload['latest_predictions_timestamp'] = datetime.now().isoformat()

        with open(manifest_path, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not update holdout manifest with predictions path ({exc})")


def _split_samples(
    samples: List[Dict[str, torch.Tensor]],
    seed: int,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """Deterministic 80/10/10 split with safeguards for small datasets."""
    if len(samples) < 3:
        raise ValueError("Need at least 3 samples for train/val/test split")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)

    n_total = len(samples)
    n_test = max(1, int(0.1 * n_total))
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_test - n_val

    if n_train < 1:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    train_ids = idx[:n_train]
    val_ids = idx[n_train:n_train + n_val]
    test_ids = idx[n_train + n_val:]

    train_samples = [samples[i] for i in train_ids]
    val_samples = [samples[i] for i in val_ids]
    test_samples = [samples[i] for i in test_ids]
    return train_samples, val_samples, test_samples


def _create_real_dataloaders(
    samples: List[Dict[str, torch.Tensor]],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from tensor-backed samples."""
    train_samples, val_samples, test_samples = _split_samples(samples, seed)
    train_drop_last = len(train_samples) >= batch_size
    pin_memory = torch.cuda.is_available()

    train_loader_kwargs: Dict[str, Any] = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'drop_last': train_drop_last,
        'pin_memory': pin_memory,
    }
    eval_loader_kwargs: Dict[str, Any] = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'drop_last': False,
        'pin_memory': pin_memory,
    }

    if num_workers > 0:
        train_loader_kwargs['persistent_workers'] = True
        train_loader_kwargs['prefetch_factor'] = 2
        eval_loader_kwargs['persistent_workers'] = True
        eval_loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        ADNIRealTensorDataset(train_samples),
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        ADNIRealTensorDataset(val_samples),
        **eval_loader_kwargs,
    )
    test_loader = DataLoader(
        ADNIRealTensorDataset(test_samples),
        **eval_loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def _prepare_real_dataloaders_from_drive(
    config: Dict[str, Any],
    source_identifier: str,
    output_prefix: str,
    logger: logging.Logger,
    source_kind: str = CACHE_SOURCE_KIND_DRIVE,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Resolve ZIPs from Drive or Modal volume, then preprocess and return dataloaders + holdout metadata."""
    data_cfg = config['data']
    model_cfg = config['model']

    batch_size = int(data_cfg.get('batch_size', 16))
    num_workers = int(data_cfg.get('num_workers', 0))
    max_series = int(data_cfg.get('max_series', 2000))
    max_dicoms_per_series = int(data_cfg.get('max_dicoms_per_series', 120))
    seed = int(data_cfg.get('seed', 42))
    inference_holdout_ratio = float(data_cfg.get('inference_holdout_ratio', 0.02))
    inference_holdout_ratio = float(np.clip(inference_holdout_ratio, 0.0, 0.5))
    use_filename_clues = bool(data_cfg.get('use_filename_clues', True))
    strict_fmri_clues = bool(data_cfg.get('strict_fmri_clues', False))
    use_series_cache = bool(data_cfg.get('use_series_cache', True))

    n_roi = int(model_cfg['n_roi'])
    seq_len = int(model_cfg['seq_len'])
    n_smri_features = int(model_cfg['n_smri_features'])
    num_classes = int(model_cfg['num_classes'])

    pipeline_root = MOUNT_PATH / "adni_drive_pipeline"
    cache_dir = pipeline_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    series_cache_dir = cache_dir / f"series_samples_v{SERIES_SAMPLE_CACHE_VERSION}"
    series_cache_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_drive_artifacts(pipeline_root, logger)

    processing_root = Path("/tmp") / "adni_drive_pipeline_work"
    shutil.rmtree(processing_root, ignore_errors=True)
    extract_root = processing_root / "extracted"
    download_dir = _compute_drive_zip_cache_dir(source_identifier)
    logger.info(
        f"Using ephemeral processing root {processing_root} for source resolution/extraction; "
        f"persistent preprocessed caches will be written to {cache_dir}"
    )
    logger.info(f"Persistent raw ZIP cache: {download_dir}")
    logger.info(f"Source kind: {source_kind} | source identifier: {source_identifier}")
    logger.info("Pipeline stages: source resolution -> extraction -> series discovery -> holdout split -> preprocessing -> cache write")
    logger.info(
        f"Per-series preprocessing cache: {'enabled' if use_series_cache else 'disabled'} "
        f"({series_cache_dir})"
    )

    cache_key_src = "|".join([
        source_kind,
        source_identifier,
        str(n_roi),
        str(seq_len),
        str(n_smri_features),
        str(max_series),
        str(max_dicoms_per_series),
        str(seed),
        f"{inference_holdout_ratio:.6f}",
        str(use_filename_clues),
        str(strict_fmri_clues),
    ])
    cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"samples_{cache_key}.pt"
    logger.info(f"Real-data cache key: {cache_key}")

    if cache_path.exists():
        logger.info(f"Found cached samples candidate at {cache_path}")
        cached_payload = torch.load(cache_path, map_location="cpu")
        cache_version = cached_payload.get('version') if isinstance(cached_payload, dict) else None
        cache_source_kind = cached_payload.get('source_kind') if isinstance(cached_payload, dict) else None

        if isinstance(cached_payload, dict) and cache_version == 3 and cache_source_kind == source_kind:
            logger.info(
                f"Cache hit: loading {len(cached_payload.get('samples', []))} preprocessed samples "
                f"from {cache_path} (source_kind={cache_source_kind})"
            )
            samples = cached_payload.get('samples', [])
            inference_series_dirs = [
                Path(p) for p in cached_payload.get('inference_series_dirs', []) if isinstance(p, str)
            ]
        else:
            logger.warning(
                "Cache exists but is not a compatible real-data cache; rebuilding from the configured source. "
                f"version={cache_version}, source_kind={cache_source_kind!r}"
            )
            samples = []
            inference_series_dirs = []

        if samples:
            holdout_info = _persist_inference_holdout_artifacts(
                inference_series_dirs=inference_series_dirs,
                output_prefix=output_prefix,
                cache_key=cache_key,
                holdout_ratio=inference_holdout_ratio,
                logger=logger,
            )
            train_loader, val_loader, test_loader = _create_real_dataloaders(
                samples,
                batch_size,
                num_workers,
                seed,
            )
            return train_loader, val_loader, test_loader, holdout_info

    if source_kind == CACHE_SOURCE_KIND_UPLOADED_BUNDLE:
        zip_files = _resolve_uploaded_bundle_zip_files(source_identifier, download_dir, logger)
    else:
        zip_files = _download_drive_folder(source_identifier, download_dir, logger)
    logger.info(f"Resolved {len(zip_files)} ZIP archive(s) from source; beginning extraction")
    _extract_zip_files(zip_files, extract_root, logger)

    series_dirs = _find_series_dirs_with_dicoms(extract_root)
    if not series_dirs:
        raise RuntimeError("No DICOM series found after extracting ZIP files")

    logger.info(f"Discovered {len(series_dirs)} candidate series directories after extraction")

    series_dirs = _select_candidate_series(
        series_dirs=series_dirs,
        use_filename_clues=use_filename_clues,
        strict_fmri_clues=strict_fmri_clues,
        logger=logger,
    )

    logger.info(f"Series remaining after clue filtering: {len(series_dirs)}")

    if max_series > 0:
        series_dirs = series_dirs[:max_series]

    series_dirs, inference_series_dirs = _split_series_for_inference(
        series_dirs=series_dirs,
        seed=seed,
        holdout_ratio=inference_holdout_ratio,
    )

    logger.info(
        f"Inference holdout reserved {len(inference_series_dirs)} series "
        f"({inference_holdout_ratio * 100:.2f}%), training pool has {len(series_dirs)} series"
    )

    if len(series_dirs) < 3:
        raise RuntimeError(
            "Not enough series left for train/val/test after inference holdout split. "
            f"remaining={len(series_dirs)}, holdout={len(inference_series_dirs)}"
        )

    label_map = _load_optional_label_map(data_cfg.get('labels_csv'), logger, base_dir=extract_root)

    preprocess_workers = int(data_cfg.get('preprocess_workers', max(4, min(8, os.cpu_count() or 4))))
    preprocess_workers = max(1, preprocess_workers)
    logger.info(
        f"Preprocessing {len(series_dirs)} DICOM series into model-ready tensors using {preprocess_workers} workers"
    )

    work_items: List[Tuple[Path, int, Path]] = []
    for series_dir in series_dirs:
        subject_id = _extract_subject_id(series_dir)
        label = label_map.get(subject_id)
        if label is None:
            label = label_map.get(_canonical_subject_key(subject_id))
        if label is None:
            label = _deterministic_label(subject_id, num_classes)
        label = int(np.clip(label, 0, num_classes - 1))
        series_sample_cache_path = _compute_series_sample_cache_path(
            series_dir=series_dir,
            extract_root=extract_root,
            source_identifier=source_identifier,
            source_kind=source_kind,
            label=label,
            n_roi=n_roi,
            seq_len=seq_len,
            n_smri_features=n_smri_features,
            max_dicoms_per_series=max_dicoms_per_series,
            cache_dir=series_cache_dir,
        )
        work_items.append((series_dir, label, series_sample_cache_path))

    samples_buffer: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(work_items)
    work_items_to_process: List[Tuple[int, Path, int, Path]] = []
    skipped = 0
    cached_hits = 0
    cached_misses = 0

    for index, (series_dir, label, series_sample_cache_path) in enumerate(work_items):
        if use_series_cache and series_sample_cache_path.exists():
            try:
                cached_sample = torch.load(series_sample_cache_path, map_location='cpu')
                if _sample_looks_valid(cached_sample):
                    samples_buffer[index] = cached_sample
                    cached_hits += 1
                    continue
                logger.warning(f"Invalid cached sample payload found, rebuilding: {series_sample_cache_path}")
            except Exception as exc:
                logger.warning(f"Failed to read cached sample {series_sample_cache_path}: {exc}")

        cached_misses += 1
        work_items_to_process.append((index, series_dir, label, series_sample_cache_path))

    logger.info(
        f"Series cache lookup finished: hits={cached_hits}, misses={cached_misses}, total={len(work_items)}"
    )

    valid_samples = cached_hits
    preprocess_start = time.perf_counter()

    if work_items_to_process:
        progress_interval = max(10, len(work_items_to_process) // 20)
        with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
            future_to_index: Dict[Any, Tuple[int, Path, Path]] = {}
            for index, series_dir, label, series_sample_cache_path in work_items_to_process:
                future = executor.submit(
                    _build_sample_from_series,
                    series_dir=series_dir,
                    label=label,
                    n_roi=n_roi,
                    seq_len=seq_len,
                    n_smri_features=n_smri_features,
                    max_dicoms_per_series=max_dicoms_per_series,
                )
                future_to_index[future] = (index, series_dir, series_sample_cache_path)

            for completed_count, future in enumerate(as_completed(future_to_index), start=1):
                index, series_dir, series_sample_cache_path = future_to_index[future]
                try:
                    sample = future.result()
                except Exception as exc:
                    skipped += 1
                    logger.warning(f"Preprocessing failed for {series_dir}: {exc}")
                    continue

                if sample is None:
                    skipped += 1
                else:
                    samples_buffer[index] = sample
                    valid_samples += 1
                    if use_series_cache:
                        try:
                            _atomic_torch_save(sample, series_sample_cache_path)
                        except Exception as exc:
                            logger.warning(f"Could not persist series cache {series_sample_cache_path}: {exc}")

                if completed_count == 1 or completed_count % progress_interval == 0 or completed_count == len(work_items_to_process):
                    elapsed = time.perf_counter() - preprocess_start
                    logger.info(
                        f"Preprocessing progress: {completed_count}/{len(work_items_to_process)} uncached series | "
                        f"valid samples={valid_samples}/{len(work_items)} | "
                        f"skipped={skipped} | cache_hits={cached_hits} | elapsed={elapsed:.1f}s"
                    )
    else:
        logger.info("All required series were loaded from per-series cache; no preprocessing needed")

    samples = [sample for sample in samples_buffer if sample is not None]

    if len(samples) < 3:
        raise RuntimeError(
            f"Insufficient samples after preprocessing. valid={len(samples)}, skipped={skipped}"
        )

    cache_payload = {
        'version': 3,
        'source_kind': source_kind,
        'samples': samples,
        'inference_series_dirs': [str(p) for p in inference_series_dirs],
        'inference_holdout_ratio': inference_holdout_ratio,
        'cache_key': cache_key,
        'saved_at': datetime.now().isoformat(),
    }
    torch.save(cache_payload, cache_path)
    logger.info(
        f"Saved source-derived cache to {cache_path} "
        f"(source_kind={source_kind}, samples={len(samples)}, skipped={skipped}, "
        f"inference_holdout={len(inference_series_dirs)})"
    )

    holdout_info = _persist_inference_holdout_artifacts(
        inference_series_dirs=inference_series_dirs,
        output_prefix=output_prefix,
        cache_key=cache_key,
        holdout_ratio=inference_holdout_ratio,
        logger=logger,
    )

    train_loader, val_loader, test_loader = _create_real_dataloaders(samples, batch_size, num_workers, seed)
    shutil.rmtree(processing_root, ignore_errors=True)
    logger.info(f"Cleaned ephemeral processing root {processing_root}")
    return train_loader, val_loader, test_loader, holdout_info


def _load_model_from_checkpoint(
    config_path: str,
    output_prefix: str,
    checkpoint_name: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any], Path]:
    """Load model + config + checkpoint for inference utilities."""
    cfg_path = Path(config_path) if Path(config_path).is_absolute() else Path(__file__).parent / config_path
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    from models.mm_dbgdgm import MM_DBGDGM

    model = MM_DBGDGM(
        n_roi=model_cfg['n_roi'],
        seq_len=model_cfg['seq_len'],
        n_smri_features=model_cfg['n_smri_features'],
        latent_dim=model_cfg['latent_dim'],
        num_classes=model_cfg['num_classes'],
        dropout=model_cfg['dropout'],
        use_attention_fusion=model_cfg['use_attention_fusion'],
        num_fusion_heads=model_cfg['num_fusion_heads'],
        num_fusion_iterations=model_cfg['num_fusion_iterations']
    ).to(device)

    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_absolute():
        checkpoint_path = CHECKPOINT_DIR / output_prefix / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, config, checkpoint_path


def _resolve_checkpoint_reference(checkpoint_reference: str, output_prefix: str) -> Path:
    """Resolve a checkpoint path from absolute or common relative locations."""
    checkpoint_reference = checkpoint_reference.strip()
    if not checkpoint_reference:
        raise ValueError("checkpoint_reference cannot be empty")

    requested_path = Path(checkpoint_reference)
    candidates: List[Path] = []

    if requested_path.is_absolute():
        candidates.append(requested_path)
    else:
        candidates.append(requested_path)
        candidates.append(CHECKPOINT_DIR / output_prefix / requested_path)
        candidates.append(CHECKPOINT_DIR / requested_path)

    expanded_candidates: List[Path] = []
    for candidate in candidates:
        expanded_candidates.append(candidate)
        if candidate.suffix.lower() != ".pt":
            expanded_candidates.append(candidate.with_suffix(".pt"))

    seen: set[str] = set()
    for candidate in expanded_candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in expanded_candidates)
    raise FileNotFoundError(
        f"Could not resolve checkpoint '{checkpoint_reference}'. Tried:\n{searched}"
    )


def _extract_module_state_dict_from_checkpoint(
    checkpoint: Any,
    module_name: str,
    explicit_state_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Extract a module state dict from different checkpoint formats."""
    if not isinstance(checkpoint, dict):
        return None

    if explicit_state_key:
        explicit_state_dict = checkpoint.get(explicit_state_key)
        if isinstance(explicit_state_dict, dict):
            return explicit_state_dict

    component_state_dicts = checkpoint.get('component_state_dicts')
    if isinstance(component_state_dicts, dict):
        component_state = component_state_dicts.get(module_name)
        if isinstance(component_state, dict):
            return component_state

    if isinstance(checkpoint.get(module_name), dict):
        return checkpoint[module_name]

    state_dict_container = checkpoint.get('model_state_dict') if isinstance(checkpoint.get('model_state_dict'), dict) else checkpoint
    prefix = f"{module_name}."
    filtered_state = {
        key[len(prefix):]: value
        for key, value in state_dict_container.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if filtered_state:
        return filtered_state

    # Support raw module-only state_dict checkpoints (already un-prefixed keys).
    if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint

    return None


def _load_pretrained_fmri_encoder(
    model: nn.Module,
    checkpoint_reference: str,
    output_prefix: str,
    device: torch.device,
    logger: logging.Logger,
) -> Path:
    """Load only the fMRI encoder weights into the current model."""
    checkpoint_path = _resolve_checkpoint_reference(checkpoint_reference, output_prefix)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    fmri_state_dict = _extract_module_state_dict_from_checkpoint(
        checkpoint=checkpoint,
        module_name='fmri_encoder',
        explicit_state_key='fmri_encoder_state_dict',
    )
    if fmri_state_dict is None:
        raise ValueError(
            f"Checkpoint does not contain fMRI encoder weights: {checkpoint_path}"
        )

    load_result = model.fmri_encoder.load_state_dict(fmri_state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning(f"fMRI preload missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}")
    if load_result.unexpected_keys:
        logger.warning(
            f"fMRI preload unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}"
        )

    logger.info(f"Loaded pretrained fMRI encoder from: {checkpoint_path}")
    return checkpoint_path


def _freeze_named_module(model: nn.Module, module_name: str, logger: logging.Logger) -> bool:
    """Freeze all parameters in a named top-level module."""
    module = getattr(model, module_name, None)
    if module is None:
        logger.warning(f"Requested freeze for missing module '{module_name}'")
        return False

    for parameter in module.parameters():
        parameter.requires_grad = False
    return True


def _freeze_all_modules_except(
    model: nn.Module,
    trainable_modules: List[str],
) -> List[str]:
    """Freeze all top-level modules except an allow-list of trainable modules."""
    trainable_set = set(trainable_modules)
    frozen_module_names: List[str] = []

    for module_name, module in model.named_children():
        should_train = module_name in trainable_set
        for parameter in module.parameters():
            parameter.requires_grad = should_train
        if not should_train:
            frozen_module_names.append(module_name)

    return frozen_module_names


def _volume_path_exists(volume: modal.Volume, remote_path: str) -> bool:
    remote_path = remote_path.strip().lstrip("/")
    if not remote_path:
        return False

    remote_dir = str(Path(remote_path).parent).replace("\\", "/")
    try:
        entries = volume.listdir(f"/{remote_dir}")
    except Exception:
        return False

    target_name = Path(remote_path).name
    for entry in entries:
        entry_path = str(getattr(entry, "path", "")).replace("\\", "/")
        if entry_path.endswith(f"/{target_name}") or entry_path == remote_path:
            return True
    return False


def _seed_bundle_into_volume(
    bundle_zip_path: Path,
    summary_path: Optional[Path],
    volume_name: str,
    remote_base_root: str,
    logger: logging.Logger,
) -> None:
    import modal._utils.blob_utils as blob_utils

    blob_utils.HEALTHY_R2_UPLOAD_PERCENTAGE = 1.0

    logger.info(f"Checking Modal volume '{volume_name}' for an existing seeded bundle")
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    remote_bundle_path = f"{remote_base_root}/raw_zips/{bundle_zip_path.name}"
    remote_summary_path = f"{remote_base_root}/manifests/{summary_path.name}" if summary_path is not None else None
    bundle_size_mb = bundle_zip_path.stat().st_size / 1_000_000.0 if bundle_zip_path.exists() else 0.0
    summary_size_kb = summary_path.stat().st_size / 1_000.0 if summary_path is not None and summary_path.exists() else 0.0

    bundle_exists = _volume_path_exists(volume, remote_bundle_path)
    summary_exists = bool(remote_summary_path and _volume_path_exists(volume, remote_summary_path))
    if bundle_exists and (remote_summary_path is None or summary_exists):
        logger.info(
            f"Bundle already exists in Modal volume at {remote_bundle_path}; skipping upload"
        )
        return

    logger.info(
        f"Seeding Modal volume '{volume_name}' with local bundle: {bundle_zip_path} ({bundle_size_mb:.1f} MB)"
    )
    logger.info(f"  raw zip -> {remote_bundle_path}")
    if remote_summary_path is not None and summary_path is not None and summary_path.exists():
        logger.info(f"  summary -> {remote_summary_path} ({summary_size_kb:.1f} KB)")

    with volume.batch_upload(force=True) as batch:
        logger.info("Uploading bundle zip into Modal volume")
        batch.put_file(str(bundle_zip_path), remote_bundle_path)
        logger.info("Bundle zip upload staged")
        if summary_path is not None and summary_path.exists():
            logger.info("Uploading bundle summary into Modal volume")
            batch.put_file(str(summary_path), remote_summary_path)
            logger.info("Bundle summary upload staged")

    logger.info(f"Modal volume bundle seed upload completed: {remote_bundle_path}")

# ============================================================================
# MODAL FUNCTIONS
# ============================================================================

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        str(MOUNT_PATH): data_volume,
        str(CHECKPOINT_DIR): checkpoint_volume,
        str(LOGS_DIR): logs_volume
    },
    timeout=86400  # 24 hours
)
def train(
    config_path: str = "MM_DBGDGM/configs/default_config.yaml",
    data_split_info: Optional[Dict[str, str]] = None,
    resume_checkpoint: Optional[str] = None,
    output_prefix: str = "mm_dbgdgm",
    pretrained_fmri_checkpoint: str = "",
    freeze_fmri_encoder: bool = False,
    freeze_smri_encoder: bool = False,
    train_only_smri: bool = False,
    num_epochs_override: Optional[int] = None,
    uploaded_bundle_name: str = "",
    prefer_uploaded_bundle: bool = True,
) -> Dict[str, Any]:
    logger = setup_logging(LOGS_DIR)
    logger.info("=" * 80)
    logger.info("Starting MM-DBGDGM Modal Training")
    logger.info("=" * 80)

    with log_phase(logger, "Environment setup"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            logger.info("Enabled CUDA TF32 and high matmul precision for faster training")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

    # Load config
    # Handle config path - resolves relative to /root in container
    with log_phase(logger, "Config load"):
        if Path(config_path).is_absolute():
            config_full_path = Path(config_path)
        else:
            config_full_path = Path("/root") / config_path

        with open(config_full_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from: {config_full_path}")

    training_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    configured_pretrained_fmri = str(training_cfg.get('pretrained_fmri_checkpoint', '') or '').strip()
    effective_pretrained_fmri_checkpoint = pretrained_fmri_checkpoint.strip() or configured_pretrained_fmri
    freeze_fmri_encoder_effective = bool(training_cfg.get('freeze_fmri_encoder', False) or freeze_fmri_encoder)
    freeze_smri_encoder_effective = bool(training_cfg.get('freeze_smri_encoder', False) or freeze_smri_encoder)
    train_only_smri_effective = bool(training_cfg.get('train_only_smri', False) or train_only_smri)
    configured_uploaded_bundle_name = str(data_cfg.get('uploaded_bundle_name', '') or '').strip()
    effective_uploaded_bundle_name = (
        uploaded_bundle_name.strip()
        or str(data_split_info.get('uploaded_bundle_name', '') if data_split_info else '').strip()
        or configured_uploaded_bundle_name
    )

    if train_only_smri_effective:
        freeze_fmri_encoder_effective = True
        freeze_smri_encoder_effective = False

    prefer_uploaded_bundle_effective = bool(prefer_uploaded_bundle)

    logger.info(f"Pretrained fMRI checkpoint: {effective_pretrained_fmri_checkpoint or '<none>'}")
    logger.info(f"Freeze fMRI encoder: {freeze_fmri_encoder_effective}")
    logger.info(f"Freeze sMRI encoder: {freeze_smri_encoder_effective}")
    logger.info(f"Train only sMRI encoder: {train_only_smri_effective}")
    logger.info(f"Uploaded bundle name: {effective_uploaded_bundle_name or '<none>'}")
    logger.info(f"Prefer uploaded bundle source: {prefer_uploaded_bundle_effective}")

    # Import training components (done after path setup)
    with log_phase(logger, "Importing training modules"):
        from models.mm_dbgdgm import MM_DBGDGM
        from training.trainer import Trainer
        from training.losses import MM_DBGDGM_Loss
        from data.adni_loader import get_adni_dataloaders

    # ========================================================================
    # DATA LOADING
    # ========================================================================
    inference_holdout_info: Dict[str, Any] = {
        'manifest_path': None,
        'holdout_root': None,
        'num_series': 0,
        'num_missing_source_dirs': 0,
    }

    with log_phase(logger, "Dataset loading"):
        # Use mock data if no real data provided, or real data from mounted volume
        use_mock = data_cfg.get('mock_data', True)

        if use_mock:
            logger.info("Using mock data for training")
            train_loader, val_loader, test_loader = get_adni_dataloaders(
                batch_size=data_cfg['batch_size'],
                mock_data=True,
                include_test=True
            )
        else:
            drive_folder_url = None
            drive_folder_url_source = "config default"
            if data_split_info is not None:
                drive_folder_url = data_split_info.get('drive_folder_url')
                drive_folder_url_source = str(data_split_info.get('drive_folder_url_source', 'CLI/config')).strip() or 'CLI/config'
            if not drive_folder_url:
                drive_folder_url = data_cfg.get('drive_folder_url')
                drive_folder_url_source = "config default"

            if prefer_uploaded_bundle_effective and effective_uploaded_bundle_name:
                logger.info(f"Real-data mode selected; uploaded bundle source: {effective_uploaded_bundle_name}")
                if drive_folder_url:
                    logger.info("Drive folder URL will be used only as a fallback if the uploaded bundle is missing")
                logger.info("Preparing real data from uploaded Modal bundle ZIP")
                try:
                    train_loader, val_loader, test_loader, inference_holdout_info = _prepare_real_dataloaders_from_drive(
                        config=config,
                        source_identifier=effective_uploaded_bundle_name,
                        output_prefix=output_prefix,
                        logger=logger,
                        source_kind=CACHE_SOURCE_KIND_UPLOADED_BUNDLE,
                    )
                except FileNotFoundError as exc:
                    if drive_folder_url:
                        logger.warning(
                            f"Uploaded bundle not found or not usable ({exc}); falling back to Google Drive ZIP folder"
                        )
                        train_loader, val_loader, test_loader, inference_holdout_info = _prepare_real_dataloaders_from_drive(
                            config=config,
                            source_identifier=drive_folder_url,
                            output_prefix=output_prefix,
                            logger=logger,
                            source_kind=CACHE_SOURCE_KIND_DRIVE,
                        )
                    else:
                        raise
            elif drive_folder_url:
                logger.info(f"Real-data mode selected; Drive folder URL source: {drive_folder_url_source}")
                logger.info(f"Drive folder URL: {drive_folder_url}")
                logger.info("Preparing real data from Google Drive ZIP folder")
                train_loader, val_loader, test_loader, inference_holdout_info = _prepare_real_dataloaders_from_drive(
                    config=config,
                    source_identifier=drive_folder_url,
                    output_prefix=output_prefix,
                    logger=logger,
                    source_kind=CACHE_SOURCE_KIND_DRIVE,
                )
            else:
                raise ValueError(
                    "Real-data mode was requested but neither an uploaded bundle nor drive_folder_url was provided. "
                    "Refusing to fall back to mock data."
                )

        logger.info(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
            f"Test batches: {len(test_loader)}"
        )

    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    with log_phase(logger, "Model initialization"):
        model = MM_DBGDGM(
            n_roi=config['model']['n_roi'],
            seq_len=config['model']['seq_len'],
            gru_hidden=config['model'].get('gru_hidden', 128),
            n_smri_features=config['model']['n_smri_features'],
            latent_dim=config['model']['latent_dim'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout'],
            use_attention_fusion=config['model']['use_attention_fusion'],
            num_fusion_heads=config['model']['num_fusion_heads'],
            num_fusion_iterations=config['model']['num_fusion_iterations']
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")

        # Move to device
        model = model.to(device)

    # ========================================================================
    # LOSS & TRAINER INITIALIZATION
    # ========================================================================
    with log_phase(logger, "Trainer setup"):
        criterion = MM_DBGDGM_Loss(
            lambda_kl=config['training']['loss_weights']['lambda_kl'],
            lambda_align=config['training']['loss_weights']['lambda_align'],
            lambda_recon=config['training']['loss_weights']['lambda_recon'],
            lambda_regression=config['training']['loss_weights']['lambda_regression'],
            lambda_survival=config['training']['loss_weights']['lambda_survival'],
            fmri_recon_weight=config['training']['recon_weights']['fmri'],
            smri_recon_weight=config['training']['recon_weights']['smri']
        )

        # Setup output directories
        output_dir = CHECKPOINT_DIR / output_prefix
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            model=model,
            criterion=criterion,
            device=device,
            output_dir=str(output_dir),
            seed=42
        )
    
    # ========================================================================
    # RESUME FROM CHECKPOINT (if provided)
    # ========================================================================
    start_epoch = 0
    resume_optimizer_state = None
    resume_loaded = False
    loaded_pretrained_fmri_path: Optional[str] = None

    if resume_checkpoint:
        resolved_resume_checkpoint = _resolve_checkpoint_reference(resume_checkpoint, output_prefix)
        logger.info(f"Resuming from checkpoint: {resolved_resume_checkpoint}")
        checkpoint = torch.load(resolved_resume_checkpoint, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
            resume_optimizer_state = checkpoint.get('optimizer_state_dict', None)
            if 'trainer_history' in checkpoint:
                trainer.history = checkpoint['trainer_history']
            if 'best_val_loss' in checkpoint:
                trainer.best_val_loss = checkpoint['best_val_loss']
            if 'best_val_acc' in checkpoint:
                trainer.best_val_acc = checkpoint['best_val_acc']
        else:
            model.load_state_dict(checkpoint)
            logger.warning(
                "Resume checkpoint contained weights only (no optimizer/history). "
                "Continuing from epoch 0."
            )

        resume_loaded = True
        logger.info(f"Resumed from epoch {start_epoch} with best_val_loss={trainer.best_val_loss:.4f}")

    if effective_pretrained_fmri_checkpoint and not resume_loaded:
        loaded_pretrained_fmri_path = str(
            _load_pretrained_fmri_encoder(
                model=model,
                checkpoint_reference=effective_pretrained_fmri_checkpoint,
                output_prefix=output_prefix,
                device=device,
                logger=logger,
            )
        )
    elif effective_pretrained_fmri_checkpoint and resume_loaded:
        logger.info("Resume checkpoint provided; skipping separate pretrained fMRI load")

    frozen_module_names: List[str] = []
    if train_only_smri_effective:
        frozen_module_names = _freeze_all_modules_except(model, trainable_modules=['smri_encoder'])
        logger.info("Train-only-sMRI mode enabled: only smri_encoder parameters will be trainable")
    else:
        if freeze_fmri_encoder_effective:
            if _freeze_named_module(model, 'fmri_encoder', logger):
                frozen_module_names.append('fmri_encoder')
                logger.info("fMRI encoder parameters frozen for training")

        if freeze_smri_encoder_effective:
            if _freeze_named_module(model, 'smri_encoder', logger):
                frozen_module_names.append('smri_encoder')
                logger.info("sMRI encoder parameters frozen for training")

    trainer.frozen_module_names = set(frozen_module_names)
    if trainer.frozen_module_names:
        logger.info(f"Frozen modules for this run: {sorted(trainer.frozen_module_names)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters after transfer setup - Total: {total_params:,}, Trainable: {trainable_params:,}")

    if trainable_params == 0:
        raise ValueError("No trainable parameters remain after applying freeze settings")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    num_epochs = int(config['training']['num_epochs'])
    if num_epochs_override is not None:
        num_epochs = int(num_epochs_override)
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    patience = int(config['training'].get('early_stopping_patience', 10))
    annealing_epochs = int(config['training'].get('kl_annealing_epochs', 20))
    max_wall_time_seconds = float(config['training'].get('max_wall_time_seconds', 5400))
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"LR: {learning_rate}, Weight Decay: {weight_decay}, Patience: {patience}")
    logger.info(f"Training wall-clock budget: {max_wall_time_seconds / 3600:.2f} hours")
    
    test_results: Dict[str, Any] = {}

    try:
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            annealing_epochs=annealing_epochs,
            start_epoch=start_epoch,
            resume_optimizer_state=resume_optimizer_state,
            max_wall_time_seconds=max_wall_time_seconds
        )
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    # ========================================================================
    # SAVE FINAL MODEL & HISTORY
    # ========================================================================
    final_model_path = output_dir / f"{output_prefix}_final.pt"
    final_history_path = output_dir / f"{output_prefix}_history.json"
    with log_phase(logger, "Final artifact save"):
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'trainer_history': trainer.history,
            'final_epoch': num_epochs,
            'training_timestamp': datetime.now().isoformat()
        }, final_model_path)

        # Save history as JSON
        history_dict = {k: [float(v) if isinstance(v, float) else v for v in vals]
                        for k, vals in trainer.history.items()}
        with open(final_history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Training history saved to: {final_history_path}")

    # Evaluate the best checkpoint on the held-out test set after saving the final state.
    best_checkpoint_path = output_dir / "best_loss.pt"
    with log_phase(logger, "Held-out test evaluation"):
        if best_checkpoint_path.exists():
            logger.info(f"Evaluating best checkpoint on test set: {best_checkpoint_path}")
            test_results = trainer.test(
                test_loader=test_loader,
                checkpoint_path=str(best_checkpoint_path),
                save_name='test_results.json'
            )
        else:
            logger.warning("Best checkpoint not found, evaluating current model weights on the test set")
            test_results = trainer.test(
                test_loader=test_loader,
                save_name='test_results.json'
            )

    best_fmri_checkpoint_path = output_dir / "best_fmri_model.pt"
    best_smri_checkpoint_path = output_dir / "best_smri_model.pt"
    best_final_checkpoint_path = output_dir / "best_final_model.pt"
    
    # Prepare results
    results = {
        'status': 'completed',
        'model_path': str(final_model_path),
        'history_path': str(final_history_path),
        'inference_holdout_manifest_path': inference_holdout_info.get('manifest_path'),
        'inference_holdout_root': inference_holdout_info.get('holdout_root'),
        'inference_holdout_count': inference_holdout_info.get('num_series', 0),
        'config': config,
        'best_val_loss': float(trainer.best_val_loss),
        'best_val_acc': float(trainer.best_val_acc),
        'best_fmri_model_path': str(best_fmri_checkpoint_path) if best_fmri_checkpoint_path.exists() else None,
        'best_smri_model_path': str(best_smri_checkpoint_path) if best_smri_checkpoint_path.exists() else None,
        'best_final_model_path': str(best_final_checkpoint_path) if best_final_checkpoint_path.exists() else None,
        'pretrained_fmri_checkpoint_loaded': loaded_pretrained_fmri_path,
        'freeze_fmri_encoder': freeze_fmri_encoder_effective,
        'freeze_smri_encoder': freeze_smri_encoder_effective,
        'train_only_smri': train_only_smri_effective,
        'test_results_path': test_results.get('results_path'),
        'test_accuracy': test_results.get('overall_accuracy'),
        'test_macro_f1': test_results.get('macro_f1'),
        'test_weighted_f1': test_results.get('weighted_f1'),
        'test_per_class_accuracy': test_results.get('per_class_accuracy'),
        'test_confusion_matrix': test_results.get('confusion_matrix'),
        'final_epoch': num_epochs,
        'training_timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Results: {json.dumps(results, indent=2)}")
    return results


@app.function(
    image=image,
    volumes={
        str(LOGS_DIR): logs_volume
    }
)
def fetch_logs(output_prefix: str = "mm_dbgdgm") -> str:
    """Fetch and display training logs."""
    log_files = sorted(LOGS_DIR.glob("*.log"))
    if not log_files:
        return "No logs found"
    
    latest_log = log_files[-1]
    with open(latest_log) as f:
        return f.read()


@app.function(
    image=image,
    volumes={
        str(CHECKPOINT_DIR): checkpoint_volume
    }
)
def list_checkpoints(output_prefix: str = "mm_dbgdgm") -> Dict[str, str]:
    """List available checkpoints."""
    checkpoint_path = CHECKPOINT_DIR / output_prefix
    checkpoints = {}
    
    if checkpoint_path.exists():
        for ckpt in checkpoint_path.glob("*.pt"):
            checkpoints[ckpt.name] = str(ckpt)
    
    return checkpoints


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        str(MOUNT_PATH): data_volume,
        str(CHECKPOINT_DIR): checkpoint_volume,
        str(LOGS_DIR): logs_volume
    },
    timeout=3600,
)
def predict_single_sample(
    output_prefix: str = "mm_dbgdgm",
    checkpoint_name: str = "best_loss.pt",
    config_path: str = "MM_DBGDGM/configs/default_config.yaml",
    series_dir: str = "",
    drive_folder_url: str = "",
    series_index: int = 0,
    use_saved_holdout: bool = True,
) -> Dict[str, Any]:
    """
    Run best-model inference on one sample and return stage prediction + probabilities.

    If series_dir is empty, one series is auto-selected from the saved holdout manifest
    (if available) or from downloaded/extracted Drive data.
    """
    logger = setup_logging(LOGS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Prediction device: {device}")
    model, config, checkpoint_path = _load_model_from_checkpoint(
        config_path=config_path,
        output_prefix=output_prefix,
        checkpoint_name=checkpoint_name,
        device=device,
    )

    data_cfg = config['data']
    model_cfg = config['model']

    # Resolve sample series path.
    selected_series_dir: Optional[Path] = None
    selection_source = "explicit_series_dir"
    if series_dir:
        selected_series_dir = Path(series_dir)
        if not selected_series_dir.exists():
            raise FileNotFoundError(f"Provided series_dir does not exist: {selected_series_dir}")
    else:
        if use_saved_holdout:
            holdout_series_dirs, _, _ = _load_saved_inference_holdout_series(output_prefix, logger)
            if holdout_series_dirs:
                index = int(np.clip(series_index, 0, len(holdout_series_dirs) - 1))
                selected_series_dir = holdout_series_dirs[index]
                selection_source = "saved_holdout_manifest"

        if selected_series_dir is None:
            selection_source = "drive_scan"
            chosen_drive_url = drive_folder_url or data_cfg.get('drive_folder_url', '')
            if not chosen_drive_url:
                raise ValueError(
                    "series_dir is empty and no drive_folder_url is available. "
                    "Run training first to create the saved 2% holdout or pass an explicit series_dir."
                )

            processing_root = Path("/tmp") / "adni_drive_predict_work"
            shutil.rmtree(processing_root, ignore_errors=True)
            download_dir = _compute_drive_zip_cache_dir(chosen_drive_url)
            extract_root = processing_root / "extracted"

            zip_files = _download_drive_folder(chosen_drive_url, download_dir, logger)
            _extract_zip_files(zip_files, extract_root, logger)

            series_dirs = _find_series_dirs_with_dicoms(extract_root)
            if not series_dirs:
                raise RuntimeError("No DICOM series found for prediction")

            series_dirs = _select_candidate_series(
                series_dirs=series_dirs,
                use_filename_clues=bool(data_cfg.get('use_filename_clues', True)),
                strict_fmri_clues=bool(data_cfg.get('strict_fmri_clues', False)),
                logger=logger,
            )
            if not series_dirs:
                raise RuntimeError("No candidate DICOM series available for prediction")

            index = int(np.clip(series_index, 0, len(series_dirs) - 1))
            selected_series_dir = series_dirs[index]

    sample = _build_sample_from_series(
        series_dir=selected_series_dir,
        label=0,
        n_roi=int(model_cfg['n_roi']),
        seq_len=int(model_cfg['seq_len']),
        n_smri_features=int(model_cfg['n_smri_features']),
        max_dicoms_per_series=int(data_cfg.get('max_dicoms_per_series', 120)),
    )
    if sample is None:
        raise RuntimeError(f"Could not build model sample from DICOM series: {selected_series_dir}")

    fmri = sample['fmri'].unsqueeze(0).to(device)
    smri = sample['smri'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(fmri, smri, return_all=False)
        probs = torch.softmax(outputs['logits'], dim=1)[0].detach().cpu().tolist()
        pred_id = int(outputs['predictions'][0].item())

    stage_names = STAGE_NAMES[:int(model_cfg['num_classes'])]
    pred_stage = stage_names[pred_id] if pred_id < len(stage_names) else f"class_{pred_id}"

    if pred_stage == "CN":
        health_binary = "Healthy/Control-like"
    elif pred_stage == "AD":
        health_binary = "Alzheimer-like"
    else:
        health_binary = "MCI/Intermediate (not a strict Healthy vs AD endpoint)"

    result = {
        'checkpoint_used': str(checkpoint_path),
        'series_dir_used': str(selected_series_dir),
        'selection_source': selection_source,
        'predicted_class_id': pred_id,
        'predicted_stage': pred_stage,
        'stage_probabilities': {stage_names[i]: float(probs[i]) for i in range(len(stage_names))},
        'healthy_vs_ad_summary': health_binary,
        'ad_probability': float(probs[3]) if len(probs) > 3 else None,
        'cn_probability': float(probs[0]) if len(probs) > 0 else None,
        'timestamp': datetime.now().isoformat(),
    }

    if 'processing_root' in locals():
        shutil.rmtree(processing_root, ignore_errors=True)

    logger.info(f"Prediction result: {json.dumps(result, indent=2)}")
    return result


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        str(MOUNT_PATH): data_volume,
        str(CHECKPOINT_DIR): checkpoint_volume,
        str(LOGS_DIR): logs_volume
    },
    timeout=86400,
)
def predict_unseen_dataset(
    output_prefix: str = "mm_dbgdgm",
    checkpoint_name: str = "best_loss.pt",
    config_path: str = "MM_DBGDGM/configs/default_config.yaml",
    unseen_drive_folder_url: str = "",
    series_root: str = "",
    output_file_name: str = "unseen_predictions.json",
    max_series: int = 0,
    use_saved_holdout: bool = True,
) -> Dict[str, Any]:
    """
    Run inference over all unseen DICOM series and persist predictions.

    Provide either:
    - unseen_drive_folder_url (Drive folder with ZIP files), or
    - series_root (already extracted root folder path).

    If neither is provided and use_saved_holdout=true, the function uses the
    saved 2% holdout manifest from training when available.
    """
    logger = setup_logging(LOGS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Unseen-dataset inference device: {device}")

    model, config, checkpoint_path = _load_model_from_checkpoint(
        config_path=config_path,
        output_prefix=output_prefix,
        checkpoint_name=checkpoint_name,
        device=device,
    )

    data_cfg = config['data']
    model_cfg = config['model']
    stage_names = STAGE_NAMES[:int(model_cfg['num_classes'])]

    series_dirs: List[Path] = []
    search_root: Optional[Path] = None
    holdout_manifest_path: Optional[Path] = None
    source_mode = "explicit_source"

    if series_root:
        search_root = Path(series_root)
        if not search_root.exists():
            raise FileNotFoundError(f"series_root does not exist: {search_root}")
        source_mode = "series_root"
    elif unseen_drive_folder_url:
        source_url = unseen_drive_folder_url

        processing_root = Path("/tmp") / "adni_unseen_inference_work"
        shutil.rmtree(processing_root, ignore_errors=True)
        download_dir = _compute_drive_zip_cache_dir(source_url)
        extract_root = processing_root / "extracted"

        zip_files = _download_drive_folder(source_url, download_dir, logger)
        _extract_zip_files(zip_files, extract_root, logger)
        search_root = extract_root
        source_mode = "unseen_drive_folder_url"
    else:
        if use_saved_holdout:
            series_dirs, holdout_manifest_path, holdout_root = _load_saved_inference_holdout_series(
                output_prefix,
                logger,
            )
            if series_dirs:
                search_root = holdout_root if holdout_root is not None else series_dirs[0].parent
                source_mode = "saved_holdout_manifest"

        if not series_dirs:
            source_mode = "config_drive_folder_url"
            source_url = unseen_drive_folder_url or data_cfg.get('drive_folder_url', '')
            if not source_url:
                raise ValueError("No unseen data source provided. Set unseen_drive_folder_url or series_root")

            processing_root = Path("/tmp") / "adni_unseen_inference_work"
            shutil.rmtree(processing_root, ignore_errors=True)
            download_dir = _compute_drive_zip_cache_dir(source_url)
            extract_root = processing_root / "extracted"

            zip_files = _download_drive_folder(source_url, download_dir, logger)
            _extract_zip_files(zip_files, extract_root, logger)
            search_root = extract_root

    if not series_dirs:
        series_dirs = _find_series_dirs_with_dicoms(search_root)
        if not series_dirs:
            raise RuntimeError(f"No DICOM series found under: {search_root}")

        series_dirs = _select_candidate_series(
            series_dirs=series_dirs,
            use_filename_clues=bool(data_cfg.get('use_filename_clues', True)),
            strict_fmri_clues=bool(data_cfg.get('strict_fmri_clues', False)),
            logger=logger,
        )
        if not series_dirs:
            raise RuntimeError("No candidate unseen series after clue filtering")

    if max_series > 0:
        series_dirs = series_dirs[:max_series]

    predictions: List[Dict[str, Any]] = []
    skipped = 0

    for idx, one_series_dir in enumerate(series_dirs, start=1):
        sample = _build_sample_from_series(
            series_dir=one_series_dir,
            label=0,
            n_roi=int(model_cfg['n_roi']),
            seq_len=int(model_cfg['seq_len']),
            n_smri_features=int(model_cfg['n_smri_features']),
            max_dicoms_per_series=int(data_cfg.get('max_dicoms_per_series', 120)),
        )
        if sample is None:
            skipped += 1
            continue

        fmri = sample['fmri'].unsqueeze(0).to(device)
        smri = sample['smri'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(fmri, smri, return_all=False)
            probs = torch.softmax(outputs['logits'], dim=1)[0].detach().cpu().tolist()
            pred_id = int(outputs['predictions'][0].item())

        pred_stage = stage_names[pred_id] if pred_id < len(stage_names) else f"class_{pred_id}"
        if pred_stage == "CN":
            health_binary = "Healthy/Control-like"
        elif pred_stage == "AD":
            health_binary = "Alzheimer-like"
        else:
            health_binary = "MCI/Intermediate"

        predictions.append({
            'series_dir': str(one_series_dir),
            'subject_id': _extract_subject_id(one_series_dir),
            'predicted_class_id': pred_id,
            'predicted_stage': pred_stage,
            'healthy_vs_ad_summary': health_binary,
            'stage_probabilities': {stage_names[i]: float(probs[i]) for i in range(len(stage_names))},
            'ad_probability': float(probs[3]) if len(probs) > 3 else None,
            'cn_probability': float(probs[0]) if len(probs) > 0 else None,
        })

        if idx % 50 == 0:
            logger.info(f"Inference progress: {idx}/{len(series_dirs)} | predicted={len(predictions)} | skipped={skipped}")

    if not predictions:
        raise RuntimeError("No valid samples produced predictions from unseen dataset")

    stage_counts: Dict[str, int] = {}
    binary_counts: Dict[str, int] = {}
    for item in predictions:
        stage_name = item['predicted_stage']
        stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1

        binary_name = item['healthy_vs_ad_summary']
        binary_counts[binary_name] = binary_counts.get(binary_name, 0) + 1

    out_dir = CHECKPOINT_DIR / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    if not output_file_name:
        output_file_name = f"unseen_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    elif not output_file_name.lower().endswith(".json"):
        output_file_name = f"{output_file_name}.json"

    out_path = out_dir / output_file_name
    payload = {
        'checkpoint_used': str(checkpoint_path),
        'search_root': str(search_root),
        'source_mode': source_mode,
        'holdout_manifest_path': str(holdout_manifest_path) if holdout_manifest_path else None,
        'num_series_discovered': len(series_dirs),
        'num_predictions': len(predictions),
        'num_skipped': skipped,
        'stage_counts': stage_counts,
        'healthy_vs_ad_counts': binary_counts,
        'predictions': predictions,
        'timestamp': datetime.now().isoformat(),
    }

    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    _update_holdout_manifest_with_results(holdout_manifest_path, out_path, logger)

    if 'processing_root' in locals():
        shutil.rmtree(processing_root, ignore_errors=True)

    logger.info(f"Saved unseen-dataset predictions to: {out_path}")
    return {
        'status': 'completed',
        'checkpoint_used': str(checkpoint_path),
        'predictions_path': str(out_path),
        'source_mode': source_mode,
        'holdout_manifest_path': str(holdout_manifest_path) if holdout_manifest_path else None,
        'num_series_discovered': len(series_dirs),
        'num_predictions': len(predictions),
        'num_skipped': skipped,
        'stage_counts': stage_counts,
        'healthy_vs_ad_counts': binary_counts,
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

@app.local_entrypoint()
def main(
    config: str = "MM_DBGDGM/configs/default_config.yaml",
    output_prefix: str = "mm_dbgdgm",
    resume_from: Optional[str] = None,
    pretrained_fmri_checkpoint: str = "",
    freeze_fmri_encoder: bool = False,
    freeze_smri_encoder: bool = False,
    train_only_smri: bool = False,
    num_epochs: Optional[int] = None,
    use_mock_data: bool = True,
    drive_folder_url: str = "",
    uploaded_bundle_name: str = "",
    seed_bundle_path: str = "",
    seed_bundle_summary_path: str = "",
    predict_only: bool = False,
    predict_dataset: bool = False,
    sample_series_path: str = "",
    unseen_series_root: str = "",
    checkpoint_name: str = "best_loss.pt",
    sample_index: int = 0,
    max_infer_series: int = 0,
    unseen_output_file: str = "unseen_predictions.json",
    use_saved_holdout: bool = True,
):
    """
    Command-line interface for Modal training.
    
    Examples:
        modal run modal_train.py --config MM_DBGDGM/configs/default_config.yaml
        modal run modal_train.py --output-prefix my_experiment --use-mock-data false
        modal run modal_train.py --use-mock-data false --drive-folder-url "https://drive.google.com/drive/folders/..."
        modal run modal_train.py --freeze-smri-encoder true
        modal run modal_train.py --num-epochs 100
        modal run modal_train.py --pretrained-fmri-checkpoint best_fmri_model.pt --freeze-fmri-encoder true
        modal run modal_train.py --pretrained-fmri-checkpoint /checkpoints/mm_dbgdgm/best_fmri_model.pt --train-only-smri true
        modal run modal_train.py --uploaded-bundle-name prepared_dicom_bundle.zip --use-mock-data false
        modal run modal_train.py --seed-bundle-path "C:/Users/Devab/Downloads/dicom_bundle/prepared_dicom_bundle.zip" --use-mock-data false
        modal run modal_train.py --predict-only true --checkpoint-name best_loss.pt --sample-index 0
        modal run modal_train.py --predict-dataset true --checkpoint-name best_loss.pt --drive-folder-url "https://drive.google.com/drive/folders/<UNSEEN_FOLDER>"
        modal run modal_train.py --predict-dataset true --checkpoint-name best_loss.pt --use-saved-holdout true
    """
    local_logger = setup_logging(LOGS_DIR)
    local_logger.info("=" * 80)
    local_logger.info("MM-DBGDGM Modal Training Launcher")
    local_logger.info("=" * 80)
    local_logger.info(f"Config: {config}")
    local_logger.info(f"Output Prefix: {output_prefix}")
    local_logger.info(f"Resume From: {resume_from}")
    local_logger.info(f"Pretrained fMRI Checkpoint: {pretrained_fmri_checkpoint or '<none>'}")
    local_logger.info(f"Freeze fMRI Encoder: {freeze_fmri_encoder}")
    local_logger.info(f"Freeze sMRI Encoder: {freeze_smri_encoder}")
    local_logger.info(f"Train Only sMRI Encoder: {train_only_smri}")
    local_logger.info(f"Num Epochs Override: {num_epochs if num_epochs is not None else '<config default>'}")
    local_logger.info(f"Use Mock Data: {use_mock_data}")
    local_logger.info(f"Seed Bundle Path: {seed_bundle_path or '<none>'}")
    local_logger.info(f"Seed Bundle Summary Path: {seed_bundle_summary_path or '<auto>'}")
    config_path = Path(config)
    config_drive_folder_url = ""
    config_uploaded_bundle_name = ""
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config_payload = yaml.safe_load(config_file) or {}
            config_drive_folder_url = str(config_payload.get("data", {}).get("drive_folder_url", "")).strip()
            config_uploaded_bundle_name = str(config_payload.get("data", {}).get("uploaded_bundle_name", "")).strip()
        except Exception:
            config_drive_folder_url = ""
            config_uploaded_bundle_name = ""

    effective_drive_folder_url = drive_folder_url.strip() or config_drive_folder_url
    effective_uploaded_bundle_name = uploaded_bundle_name.strip() or config_uploaded_bundle_name
    drive_folder_url_source = "CLI override" if drive_folder_url.strip() else ("config default" if config_drive_folder_url else "none")
    local_logger.info(f"Drive Folder URL Source: {drive_folder_url_source}")
    local_logger.info(f"Drive Folder URL Effective: {effective_drive_folder_url or '<none>'}")
    local_logger.info(f"Uploaded Bundle Name Effective: {effective_uploaded_bundle_name or '<none>'}")
    if use_mock_data:
        training_data_mode = "mock"
    elif effective_uploaded_bundle_name:
        training_data_mode = "real uploaded bundle data (bundle -> extract -> preprocess -> cache)"
    else:
        training_data_mode = "real drive data (download -> preprocess -> cache)"
    local_logger.info(f"Training Data Mode: {training_data_mode}")
    local_logger.info(f"Predict Only: {predict_only}")
    local_logger.info(f"Predict Dataset: {predict_dataset}")
    local_logger.info(f"Use Saved Holdout: {use_saved_holdout}")
    local_logger.info("=" * 80)

    if predict_dataset:
        results = predict_unseen_dataset.remote(
            output_prefix=output_prefix,
            checkpoint_name=checkpoint_name,
            config_path=config,
            unseen_drive_folder_url=drive_folder_url,
            series_root=unseen_series_root,
            output_file_name=unseen_output_file,
            max_series=max_infer_series,
            use_saved_holdout=use_saved_holdout,
        )
        local_logger.info("Unseen Dataset Inference Result:")
        local_logger.info("=" * 80)
        print(json.dumps(results, indent=2))
        return

    if predict_only:
        results = predict_single_sample.remote(
            output_prefix=output_prefix,
            checkpoint_name=checkpoint_name,
            config_path=config,
            series_dir=sample_series_path,
            drive_folder_url=drive_folder_url,
            series_index=sample_index,
            use_saved_holdout=use_saved_holdout,
        )
        local_logger.info("Prediction Result:")
        local_logger.info("=" * 80)
        print(json.dumps(results, indent=2))
        return

    prefer_uploaded_bundle = True
    if not use_mock_data and seed_bundle_path.strip():
        bundle_path = Path(seed_bundle_path).expanduser()
        summary_path = Path(seed_bundle_summary_path).expanduser() if seed_bundle_summary_path.strip() else None
        if not bundle_path.exists():
            raise FileNotFoundError(f"Seed bundle zip not found: {bundle_path}")
        if summary_path is None:
            default_summary = bundle_path.with_name("bundle_summary.json")
            summary_path = default_summary if default_summary.exists() else None

        try:
            with log_phase(local_logger, "Modal volume bundle seeding"):
                local_logger.info(f"Seeding Modal volume from local bundle: {bundle_path}")
                if summary_path is not None:
                    local_logger.info(f"Seeding Modal volume summary: {summary_path}")
                _seed_bundle_into_volume(
                    bundle_zip_path=bundle_path,
                    summary_path=summary_path,
                    volume_name="mm-dbgdgm-data",
                    remote_base_root="adni_drive_pipeline",
                    logger=local_logger,
                )
        except Exception as exc:
            prefer_uploaded_bundle = False
            local_logger.warning(
                f"Bundle seed failed ({exc}); falling back to Google Drive download and volume caching for this run"
            )
    
    # Call remote function
    local_logger.info("Dispatching remote training request")
    results = train.remote(
        config_path=config,
        data_split_info=None if use_mock_data else {
            "drive_folder_url": effective_drive_folder_url,
            "drive_folder_url_source": drive_folder_url_source,
            "uploaded_bundle_name": effective_uploaded_bundle_name,
        },
        resume_checkpoint=resume_from,
        output_prefix=output_prefix,
        pretrained_fmri_checkpoint=pretrained_fmri_checkpoint,
        freeze_fmri_encoder=freeze_fmri_encoder,
        freeze_smri_encoder=freeze_smri_encoder,
        train_only_smri=train_only_smri,
        num_epochs_override=num_epochs,
        uploaded_bundle_name=effective_uploaded_bundle_name,
        prefer_uploaded_bundle=prefer_uploaded_bundle,
    )
    local_logger.info("Remote training request completed")
    
    local_logger.info("Training Results:")
    local_logger.info("=" * 80)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
