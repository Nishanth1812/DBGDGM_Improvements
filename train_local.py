"""Local MM-DBGDGM training entrypoint for standalone GPU machines.

This script keeps the existing model/trainer stack, but removes the Modal-only
assumptions so it can run on a DigitalOcean AMD GPU droplet with local data.
It supports either split metadata CSVs or a single manifest that is split
in-memory into train/validation/test sets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
import os
import json
import logging
import re
import shutil
import sys
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from MM_DBGDGM.data.dataset import create_dataloaders
from MM_DBGDGM.models.mm_dbgdgm import MM_DBGDGM
from MM_DBGDGM.training.losses import MM_DBGDGM_Loss
from MM_DBGDGM.training.trainer import Trainer
from prepare_smri_jpg_dataset import DEFAULT_CLASS_LABELS, build_dataset as build_smri_dataset


def _coalesce(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _resolve_optional_path(raw_value: Any, base_dir: Path) -> Optional[Path]:
    if raw_value is None:
        return None

    if isinstance(raw_value, Path):
        candidate = raw_value.expanduser()
    else:
        text = str(raw_value).strip()
        if not text:
            return None
        candidate = Path(text).expanduser()

    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def _resolve_required_path(raw_value: Any, base_dir: Path, label: str) -> Path:
    candidate = _resolve_optional_path(raw_value, base_dir)
    if candidate is None:
        raise ValueError(f"{label} is required")
    return candidate


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mm-dbgdgm-local")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")

        file_handler = logging.FileHandler(output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        trainer_logger = logging.getLogger("mm-dbgdgm-modal")
        trainer_logger.setLevel(logging.INFO)
        trainer_logger.propagate = False
        trainer_logger.addHandler(file_handler)
        trainer_logger.addHandler(console_handler)

        dataset_logger = logging.getLogger("MM_DBGDGM.data.dataset")
        dataset_logger.setLevel(logging.INFO)
        dataset_logger.propagate = False
        dataset_logger.addHandler(file_handler)
        dataset_logger.addHandler(console_handler)

    return logger


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def detect_device(preferred_device: str, logger: logging.Logger) -> torch.device:
    preferred_device = preferred_device.lower().strip()

    if preferred_device == "cpu":
        logger.info("Using CPU device")
        return torch.device("cpu")

    if preferred_device not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported device value: {preferred_device}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            logger.info(f"Using AMD GPU via ROCm {hip_version}")
        else:
            logger.info("Using CUDA GPU")
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
                logger.info("Enabled CUDA TF32 and high matmul precision")
            except Exception as exc:
                logger.warning(f"Could not enable CUDA performance flags: {exc}")

        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except Exception:
            pass
        return device

    if preferred_device == "cuda":
        raise RuntimeError("CUDA/ROCm device requested but torch.cuda.is_available() is false")

    logger.info("Using CPU device")
    return torch.device("cpu")


def resolve_checkpoint_reference(checkpoint_reference: str, output_dir: Path, base_dir: Path) -> Path:
    checkpoint_reference = checkpoint_reference.strip()
    if not checkpoint_reference:
        raise ValueError("checkpoint_reference cannot be empty")

    requested_path = Path(checkpoint_reference).expanduser()
    candidates: List[Path] = []

    if requested_path.is_absolute():
        candidates.append(requested_path)
    else:
        candidates.append(requested_path)
        candidates.append(base_dir / requested_path)
        candidates.append(output_dir / requested_path)
        candidates.append(output_dir / requested_path.name)

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
    raise FileNotFoundError(f"Could not resolve checkpoint '{checkpoint_reference}'. Tried:\n{searched}")


def extract_module_state_dict_from_checkpoint(
    checkpoint: Any,
    module_name: str,
    explicit_state_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        return None

    if explicit_state_key:
        explicit_state_dict = checkpoint.get(explicit_state_key)
        if isinstance(explicit_state_dict, dict):
            return explicit_state_dict

    component_state_dicts = checkpoint.get("component_state_dicts")
    if isinstance(component_state_dicts, dict):
        component_state = component_state_dicts.get(module_name)
        if isinstance(component_state, dict):
            return component_state

    module_state = checkpoint.get(module_name)
    if isinstance(module_state, dict):
        return module_state

    model_state_dict = checkpoint.get("model_state_dict")
    state_dict_container = model_state_dict if isinstance(model_state_dict, dict) else checkpoint
    prefix = f"{module_name}."
    filtered_state = {
        key[len(prefix):]: value
        for key, value in state_dict_container.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if filtered_state:
        return filtered_state

    if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()) and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        return checkpoint

    return None


def load_pretrained_fmri_encoder(
    model: torch.nn.Module,
    checkpoint_reference: str,
    output_dir: Path,
    base_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> Path:
    checkpoint_path = resolve_checkpoint_reference(checkpoint_reference, output_dir, base_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    fmri_state_dict = extract_module_state_dict_from_checkpoint(
        checkpoint=checkpoint,
        module_name="fmri_encoder",
        explicit_state_key="fmri_encoder_state_dict",
    )
    if fmri_state_dict is None:
        raise ValueError(f"Checkpoint does not contain fMRI encoder weights: {checkpoint_path}")

    load_result = model.fmri_encoder.load_state_dict(fmri_state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning(f"fMRI preload missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}")
    if load_result.unexpected_keys:
        logger.warning(f"fMRI preload unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}")

    logger.info(f"Loaded pretrained fMRI encoder from: {checkpoint_path}")
    return checkpoint_path


def freeze_named_module(model: torch.nn.Module, module_name: str, logger: logging.Logger) -> bool:
    module = getattr(model, module_name, None)
    if module is None:
        logger.warning(f"Requested freeze for missing module '{module_name}'")
        return False

    for parameter in module.parameters():
        parameter.requires_grad = False
    return True


def freeze_all_modules_except(model: torch.nn.Module, trainable_modules: List[str]) -> List[str]:
    trainable_set = set(trainable_modules)
    frozen_module_names: List[str] = []

    for module_name, module in model.named_children():
        should_train = module_name in trainable_set
        for parameter in module.parameters():
            parameter.requires_grad = should_train
        if not should_train:
            frozen_module_names.append(module_name)

    return frozen_module_names


def build_output_dir(base_output_dir: Optional[Path]) -> Path:
    if base_output_dir is not None:
        return base_output_dir.expanduser()
    return Path("local_results") / datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_raw_work_dir() -> Path:
    return ROOT_DIR / ".cache" / "raw_inputs"


def _source_signature(path: Path) -> Dict[str, Any]:
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "size": int(stat_result.st_size),
    }


def _marker_path(target_dir: Path) -> Path:
    return target_dir / ".source.json"


def _load_marker(target_dir: Path) -> Optional[Dict[str, Any]]:
    marker_file = _marker_path(target_dir)
    if not marker_file.exists():
        return None

    try:
        return json.loads(marker_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_marker(target_dir: Path, payload: Dict[str, Any]) -> None:
    _marker_path(target_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _archive_reuse_matches(zip_path: Path, target_dir: Path) -> bool:
    marker = _load_marker(target_dir)
    if not marker:
        return False
    expected = {
        "kind": "zip_archive",
        "source": _source_signature(zip_path),
    }
    return marker == expected


def _extract_zip_archive(zip_path: Path, target_dir: Path, logger: logging.Logger) -> None:
    expected_marker = {
        "kind": "zip_archive",
        "source": _source_signature(zip_path),
    }

    if target_dir.exists() and _archive_reuse_matches(zip_path, target_dir):
        logger.info(f"Reusing previously extracted archive: {zip_path.name} -> {target_dir}")
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_size_mb = zip_path.stat().st_size / 1_000_000.0
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        members = [member for member in zip_file.infolist() if not member.is_dir()]
        total_members = len(members)
        total_uncompressed_mb = sum(member.file_size for member in members) / 1_000_000.0

        logger.info(
            f"Extracting {zip_path.name}: {total_members} files, "
            f"{zip_size_mb:.1f} MB compressed, {total_uncompressed_mb:.1f} MB uncompressed -> {target_dir}"
        )

        if total_members == 0:
            logger.info(f"Archive {zip_path.name} contains no extractable files")
            return

        progress_interval = max(1, total_members // 20)
        for member_index, member in enumerate(members, start=1):
            zip_file.extract(member, target_dir)
            if member_index == 1 or member_index == total_members or member_index % progress_interval == 0:
                percent = (member_index / total_members) * 100.0
                logger.info(
                    f"Extracted {zip_path.name}: {member_index}/{total_members} files ({percent:.1f}%)"
                )

    _write_marker(target_dir, expected_marker)
    logger.info(f"Finished extracting {zip_path.name} to {target_dir}")


def _require_existing_zip_path(raw_value: Any, base_dir: Path, label: str) -> Path:
    candidate = _resolve_required_path(raw_value, base_dir, label)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is not a file: {candidate}")
    if candidate.suffix.lower() != ".zip":
        raise ValueError(f"{label} must be a .zip file: {candidate}")
    return candidate


def _find_labels_csv(root_dir: Path) -> Optional[Path]:
    direct = root_dir / "labels.csv"
    if direct.exists():
        return direct

    matches = sorted(root_dir.rglob("labels.csv"))
    return matches[0] if matches else None


def _find_prepared_smri_root(extracted_smri_root: Path) -> Optional[Path]:
    labels_csv = _find_labels_csv(extracted_smri_root)
    if labels_csv is not None:
        return labels_csv.parent

    for candidate_root in [extracted_smri_root, *[path for path in extracted_smri_root.rglob("*") if path.is_dir()]]:
        if all((candidate_root / class_name).is_dir() for class_name, _ in DEFAULT_CLASS_LABELS):
            return candidate_root

    return None


def _find_any_labels_csv(*roots: Path) -> Optional[Path]:
    for root in roots:
        labels_csv = _find_labels_csv(root)
        if labels_csv is not None:
            return labels_csv
    return None


def _prepared_smri_reuse_matches(prepared_smri_root: Path, raw_smri_root: Path, labels_csv: Optional[Path]) -> bool:
    marker = _load_marker(prepared_smri_root)
    if not marker:
        return False

    expected_labels_signature = _source_signature(labels_csv) if labels_csv is not None and labels_csv.exists() else None

    expected_train_local_marker = {
        "kind": "prepared_smri_dataset",
        "source_root": str(raw_smri_root.resolve()),
        "source_root_signature": _source_signature(raw_smri_root),
        "labels_csv_signature": expected_labels_signature,
    }
    if marker == expected_train_local_marker:
        return True

    # Backward/side-channel compatibility: `prepare_smri_jpg_dataset.py` stores
    # source metadata without the train-local marker envelope.
    expected_builder_marker = {
        "input_root": str(raw_smri_root.resolve()),
        "input_root_signature": _source_signature(raw_smri_root),
        "labels_csv_signature": expected_labels_signature,
    }
    return marker == expected_builder_marker


def _has_dicom_files(root_dir: Path) -> bool:
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix in {".dcm", ".dicom", ".ima", ""}:
            return True
    return False


def _prepare_raw_zip_inputs(
    dicom_bundle_zip: Optional[Path],
    smri_zip: Optional[Path],
    work_dir: Path,
    logger: logging.Logger,
) -> Tuple[Path, Path, Path]:
    if dicom_bundle_zip is None and smri_zip is None:
        raise ValueError("At least one raw zip input is required")

    work_dir.mkdir(parents=True, exist_ok=True)
    extract_root = work_dir / "extracted_inputs"
    extract_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing raw ZIP inputs under {work_dir}")

    dicom_extract_root = extract_root / "dicom_bundle"
    smri_extract_root = extract_root / "smri_bundle"

    extract_jobs: List[Tuple[Path, Path]] = []
    if dicom_bundle_zip is not None:
        extract_jobs.append((dicom_bundle_zip, dicom_extract_root))
    if smri_zip is not None:
        extract_jobs.append((smri_zip, smri_extract_root))

    logger.info(f"Extracting {len(extract_jobs)} zip file(s) in parallel")
    with ThreadPoolExecutor(max_workers=min(2, len(extract_jobs))) as executor:
        futures = [executor.submit(_extract_zip_archive, zip_path, target_dir, logger) for zip_path, target_dir in extract_jobs]
        for future in futures:
            future.result()

    if dicom_bundle_zip is None:
        raise ValueError("dicom_bundle_zip is required when using raw zip mode")

    dicom_root = dicom_extract_root
    while True:
        if _has_dicom_files(dicom_root):
            break
        child_dirs = [path for path in sorted(dicom_root.iterdir()) if path.is_dir()]
        if len(child_dirs) != 1:
            break
        dicom_root = child_dirs[0]

    if smri_zip is None:
        raise ValueError("smri_zip is required when using raw zip mode")

    prepared_smri_root = _find_prepared_smri_root(smri_extract_root)
    if prepared_smri_root is None:
        raw_smri_root = smri_extract_root
        labels_csv = _find_any_labels_csv(dicom_extract_root, smri_extract_root)
        if labels_csv is None:
            logger.warning(
                "No labels.csv found in the raw inputs; raw ADNI SMRI preparation will require labels from the archive"
            )
        else:
            logger.info(f"Using labels CSV for raw SMRI preparation: {labels_csv}")
        prepared_smri_root = work_dir / "prepared_smri_dataset"
        if prepared_smri_root.exists() and _prepared_smri_reuse_matches(prepared_smri_root, raw_smri_root, labels_csv):
            logger.info(f"Reusing prepared SMRI dataset at {prepared_smri_root}")
        else:
            logger.info(f"Rebuilding prepared SMRI dataset from raw folders: {raw_smri_root}")
            build_smri_dataset(
                input_root=raw_smri_root,
                output_root=prepared_smri_root,
                transfer_mode="hardlink",
                overwrite=True,
                labels_csv=labels_csv,
                logger=logger,
            )

    labels_csv = prepared_smri_root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Could not locate labels.csv in prepared SMRI root: {prepared_smri_root}")

    logger.info(f"Resolved raw DICOM root: {dicom_root}")
    logger.info(f"Resolved prepared SMRI root: {prepared_smri_root}")
    logger.info(f"Resolved labels.csv: {labels_csv}")
    return dicom_root, prepared_smri_root, labels_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MM-DBGDGM locally on a GPU machine")
    parser.add_argument("--config", type=Path, default=Path("MM_DBGDGM/configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dicom-bundle-zip", type=Path, default=None)
    parser.add_argument("--smri-zip", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--train-metadata", type=Path, default=None)
    parser.add_argument("--val-metadata", type=Path, default=None)
    parser.add_argument("--test-metadata", type=Path, default=None)
    parser.add_argument("--smri-source-root", type=Path, default=None)
    parser.add_argument("--allow-unaligned-pairing", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--pretrained-fmri-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-fmri-encoder", action="store_true")
    parser.add_argument("--freeze-smri-encoder", action="store_true")
    parser.add_argument("--train-only-smri", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    config_path = args.config.expanduser()
    if not config_path.is_absolute() and not config_path.exists():
        config_path = (ROOT_DIR / config_path).resolve()
    else:
        config_path = config_path.resolve()

    path_base_dir = ROOT_DIR
    config = load_config(config_path)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    loss_weights_cfg = training_cfg.get("loss_weights", {})
    recon_weights_cfg = training_cfg.get("recon_weights", {})

    output_dir = build_output_dir(args.output_dir)
    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("Starting MM-DBGDGM local training")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output directory: {output_dir}")

    device = detect_device(args.device, logger)
    logger.info(f"Using device: {device}")

    raw_zip_mode = args.dicom_bundle_zip is not None or args.smri_zip is not None
    if raw_zip_mode:
        if args.dicom_bundle_zip is None or args.smri_zip is None:
            raise ValueError("Both --dicom-bundle-zip and --smri-zip are required in raw zip mode")

        dicom_bundle_zip = _require_existing_zip_path(args.dicom_bundle_zip, path_base_dir, "dicom_bundle_zip")
        smri_zip = _require_existing_zip_path(args.smri_zip, path_base_dir, "smri_zip")
        work_dir = _resolve_optional_path(args.work_dir, path_base_dir) or _default_raw_work_dir()

        logger.info(f"Raw zip mode enabled")
        logger.info(f"DICOM bundle zip: {dicom_bundle_zip}")
        logger.info(f"SMRI zip: {smri_zip}")
        logger.info(f"Work dir: {work_dir}")

        dataset_root, prepared_smri_root, metadata_csv = _prepare_raw_zip_inputs(
            dicom_bundle_zip=dicom_bundle_zip,
            smri_zip=smri_zip,
            work_dir=work_dir,
            logger=logger,
        )
        metadata_file = metadata_csv
        smri_source_root = prepared_smri_root
        train_metadata = None
        val_metadata = None
        test_metadata = None
    else:
        dataset_root = _resolve_required_path(
            args.dataset_root if args.dataset_root is not None else data_cfg.get("dataset_root"),
            path_base_dir,
            "dataset_root",
        )
        metadata_file = _resolve_optional_path(
            args.metadata_file if args.metadata_file is not None else data_cfg.get("metadata_file"),
            path_base_dir,
        )
        train_metadata = _resolve_optional_path(
            args.train_metadata if args.train_metadata is not None else data_cfg.get("train_metadata"),
            path_base_dir,
        )
        val_metadata = _resolve_optional_path(
            args.val_metadata if args.val_metadata is not None else data_cfg.get("val_metadata"),
            path_base_dir,
        )
        test_metadata = _resolve_optional_path(
            args.test_metadata if args.test_metadata is not None else data_cfg.get("test_metadata"),
            path_base_dir,
        )
        smri_source_root = _resolve_optional_path(
            args.smri_source_root if args.smri_source_root is not None else data_cfg.get("smri_source_root"),
            path_base_dir,
        )
    resume_checkpoint = _resolve_optional_path(args.resume_from, path_base_dir)
    pretrained_fmri_checkpoint = _resolve_optional_path(args.pretrained_fmri_checkpoint, path_base_dir)

    batch_size = int(_coalesce(args.batch_size, training_cfg.get("batch_size"), data_cfg.get("batch_size"), default=16))
    num_workers = int(_coalesce(args.num_workers, training_cfg.get("num_workers"), data_cfg.get("num_workers"), default=4))
    num_epochs = int(_coalesce(args.num_epochs, training_cfg.get("num_epochs"), default=50))
    learning_rate = float(_coalesce(training_cfg.get("learning_rate"), default=1e-4))
    weight_decay = float(_coalesce(training_cfg.get("weight_decay"), default=1e-5))
    patience = int(_coalesce(training_cfg.get("early_stopping_patience"), training_cfg.get("patience"), default=10))
    annealing_epochs = int(_coalesce(training_cfg.get("kl_annealing_epochs"), training_cfg.get("annealing_epochs"), default=20))
    batch_log_interval = int(_coalesce(training_cfg.get("batch_log_interval"), default=10))
    max_wall_time_seconds = float(_coalesce(training_cfg.get("max_wall_time_seconds"), default=5400))
    seed = int(_coalesce(args.seed, training_cfg.get("seed"), data_cfg.get("seed"), default=42))
    val_fraction = float(_coalesce(args.val_fraction, data_cfg.get("val_fraction"), default=0.1))
    test_fraction = float(_coalesce(args.test_fraction, data_cfg.get("test_fraction"), default=0.1))
    normalize = bool(data_cfg.get("normalize", True))
    fmri_path_column = str(data_cfg.get("fmri_path_column", "fmri_path"))
    smri_path_column = str(data_cfg.get("smri_path_column", "smri_path"))
    allow_unaligned_pairing = bool(_coalesce(args.allow_unaligned_pairing, data_cfg.get("allow_unaligned_pairing"), default=False))
    max_dicoms_per_series = int(_coalesce(data_cfg.get("max_dicoms_per_series"), default=120))

    if raw_zip_mode:
        cpu_count = os.cpu_count() or 20
        num_workers = max(num_workers, min(32, max(8, cpu_count - 2)))

    if metadata_file is None and train_metadata is None and val_metadata is None:
        raise ValueError(
            "Provide either metadata_file for an auto-split manifest or train_metadata/val_metadata CSV files"
        )

    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Metadata file: {metadata_file if metadata_file is not None else '<none>'}")
    logger.info(f"Train metadata: {train_metadata if train_metadata is not None else '<none>'}")
    logger.info(f"Val metadata: {val_metadata if val_metadata is not None else '<none>'}")
    logger.info(f"Test metadata: {test_metadata if test_metadata is not None else '<none>'}")
    logger.info(f"SMRI source root: {smri_source_root if smri_source_root is not None else '<none>'}")
    logger.info(f"fMRI path column: {fmri_path_column} | sMRI path column: {smri_path_column}")
    logger.info(f"Allow unaligned pairing: {allow_unaligned_pairing}")
    logger.info(f"Max DICOM slices per series: {max_dicoms_per_series}")
    logger.info(f"Batch size: {batch_size} | Workers: {num_workers} | Epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate} | Weight decay: {weight_decay}")
    logger.info(f"Val fraction: {val_fraction} | Test fraction: {test_fraction}")
    logger.info(f"Batch log interval: every {batch_log_interval} batch(es)")
    logger.info(f"Seed: {seed}")
    logger.info(f"Max wall-clock budget: {max_wall_time_seconds / 60:.1f} minutes")

    logger.info("Initializing model")
    model_kwargs = {
        "n_roi": int(model_cfg.get("n_roi", 200)),
        "seq_len": int(model_cfg.get("seq_len", 50)),
        "gru_hidden": int(model_cfg.get("gru_hidden", 128)),
        "gru_layers": int(model_cfg.get("gru_layers", 2)),
        "n_smri_features": int(model_cfg.get("n_smri_features", 5)),
        "max_dicoms_per_series": max_dicoms_per_series,
        "use_gat_encoder": bool(model_cfg.get("use_gat_encoder", True)),
        "latent_dim": int(model_cfg.get("latent_dim", 256)),
        "num_classes": int(model_cfg.get("num_classes", 4)),
        "use_attention_fusion": bool(model_cfg.get("use_attention_fusion", True)),
        "num_fusion_heads": int(model_cfg.get("num_fusion_heads", 4)),
        "num_fusion_iterations": int(model_cfg.get("num_fusion_iterations", 2)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }

    constructor_signature = inspect.signature(MM_DBGDGM.__init__)
    supported_keys = {key for key in constructor_signature.parameters.keys() if key != "self"}
    filtered_model_kwargs = {key: value for key, value in model_kwargs.items() if key in supported_keys}
    dropped_model_kwargs = sorted(key for key in model_kwargs.keys() if key not in supported_keys)
    if dropped_model_kwargs:
        logger.warning(
            f"Dropping unsupported MM_DBGDGM args: {dropped_model_kwargs}. "
            f"Supported keys are: {sorted(supported_keys)}"
        )

    model = MM_DBGDGM(**filtered_model_kwargs).to(device)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")

    frozen_module_names: List[str] = []
    if args.train_only_smri:
        frozen_module_names = freeze_all_modules_except(model, trainable_modules=["smri_encoder"])
        logger.info("Train-only-sMRI mode enabled: only smri_encoder parameters remain trainable")
    else:
        if args.freeze_fmri_encoder and freeze_named_module(model, "fmri_encoder", logger):
            frozen_module_names.append("fmri_encoder")
        if args.freeze_smri_encoder and freeze_named_module(model, "smri_encoder", logger):
            frozen_module_names.append("smri_encoder")

    if pretrained_fmri_checkpoint is not None and not resume_checkpoint:
        logger.info(f"Loading pretrained fMRI encoder from {pretrained_fmri_checkpoint}")
        load_pretrained_fmri_encoder(
            model=model,
            checkpoint_reference=str(pretrained_fmri_checkpoint),
            output_dir=output_dir,
            base_dir=path_base_dir,
            device=device,
            logger=logger,
        )

    logger.info("Building loss function")
    criterion = MM_DBGDGM_Loss(
        num_classes=int(model_cfg.get("num_classes", 4)),
        lambda_kl=float(_coalesce(loss_weights_cfg.get("lambda_kl"), training_cfg.get("lambda_kl"), default=0.1)),
        lambda_align=float(_coalesce(loss_weights_cfg.get("lambda_align"), training_cfg.get("lambda_align"), default=0.1)),
        lambda_recon=float(_coalesce(loss_weights_cfg.get("lambda_recon"), training_cfg.get("lambda_recon"), default=0.1)),
        lambda_regression=float(_coalesce(loss_weights_cfg.get("lambda_regression"), training_cfg.get("lambda_regression"), default=0.1)),
        lambda_survival=float(_coalesce(loss_weights_cfg.get("lambda_survival"), training_cfg.get("lambda_survival"), default=0.1)),
        fmri_recon_weight=float(_coalesce(recon_weights_cfg.get("fmri"), training_cfg.get("fmri_recon_weight"), default=2.0)),
        smri_recon_weight=float(_coalesce(recon_weights_cfg.get("smri"), training_cfg.get("smri_recon_weight"), default=1.0)),
    ).to(device)

    logger.info("Creating dataloaders")
    dataloaders = create_dataloaders(
        dataset_root=str(dataset_root),
        train_metadata=str(train_metadata) if train_metadata is not None else None,
        val_metadata=str(val_metadata) if val_metadata is not None else None,
        test_metadata=str(test_metadata) if test_metadata is not None else None,
        metadata_file=str(metadata_file) if metadata_file is not None else None,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
        normalize=normalize,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        n_roi=int(model_cfg.get("n_roi", 200)),
        seq_len=int(model_cfg.get("seq_len", 50)),
        n_smri_features=int(model_cfg.get("n_smri_features", 5)),
        max_dicoms_per_series=max_dicoms_per_series,
        smri_source_root=str(smri_source_root) if smri_source_root is not None else None,
        fmri_path_column=fmri_path_column,
        smri_path_column=smri_path_column,
        allow_unaligned_pairing=allow_unaligned_pairing,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders.get("test")

    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset) if test_loader is not None else 0
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    test_batches = len(test_loader) if test_loader is not None else 0

    logger.info(f"Dataset samples | train={train_samples} | val={val_samples} | test={test_samples}")
    logger.info(f"Loader batches  | train={train_batches} | val={val_batches} | test={test_batches}")

    min_train_batches = int(_coalesce(training_cfg.get("min_train_batches"), default=16))
    if train_samples > 0 and train_batches < min_train_batches:
        adjusted_batch_size = max(1, min(batch_size, max(1, train_samples // min_train_batches)))
        if adjusted_batch_size < batch_size:
            logger.warning(
                f"Train loader only has {train_batches} batches; reducing batch size from {batch_size} to {adjusted_batch_size} "
                f"to target at least {min_train_batches} batches per epoch"
            )
            logger.info("Rebuilding dataloaders with adjusted batch size")
            batch_size = adjusted_batch_size
            dataloaders = create_dataloaders(
                dataset_root=str(dataset_root),
                train_metadata=str(train_metadata) if train_metadata is not None else None,
                val_metadata=str(val_metadata) if val_metadata is not None else None,
                test_metadata=str(test_metadata) if test_metadata is not None else None,
                metadata_file=str(metadata_file) if metadata_file is not None else None,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle_train=True,
                normalize=normalize,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed,
                n_roi=int(model_cfg.get("n_roi", 200)),
                seq_len=int(model_cfg.get("seq_len", 50)),
                n_smri_features=int(model_cfg.get("n_smri_features", 5)),
                max_dicoms_per_series=max_dicoms_per_series,
                smri_source_root=str(smri_source_root) if smri_source_root is not None else None,
                fmri_path_column=fmri_path_column,
                smri_path_column=smri_path_column,
                allow_unaligned_pairing=allow_unaligned_pairing,
            )
            train_loader = dataloaders["train"]
            val_loader = dataloaders["val"]
            test_loader = dataloaders.get("test")
            logger.info(f"Adjusted loader batches | train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader) if test_loader is not None else 0}")

            training_cfg["batch_size"] = batch_size
            training_cfg["num_workers"] = num_workers
            training_cfg["batch_log_interval"] = batch_log_interval

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    if test_loader is not None:
        logger.info(f"Test batches: {len(test_loader)}")

    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        device=device,
        output_dir=str(output_dir),
        seed=seed,
        frozen_module_names=frozen_module_names,
    )
    logger.info(
        "Trainer initialized; next the trainer will build the optimizer, scheduler, and heartbeat thread"
    )

    start_epoch = 0
    resume_optimizer_state = None
    if resume_checkpoint is not None:
        resolved_resume_checkpoint = resolve_checkpoint_reference(str(resume_checkpoint), output_dir, path_base_dir)
        logger.info(f"Resuming from checkpoint: {resolved_resume_checkpoint}")
        logger.info("Loading checkpoint state")
        checkpoint = torch.load(resolved_resume_checkpoint, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            resume_optimizer_state = checkpoint.get("optimizer_state_dict")
            if "trainer_history" in checkpoint:
                trainer.history = checkpoint["trainer_history"]
            if "best_val_loss" in checkpoint:
                trainer.best_val_loss = float(checkpoint["best_val_loss"])
            if "best_val_acc" in checkpoint:
                trainer.best_val_acc = float(checkpoint["best_val_acc"])
        else:
            model.load_state_dict(checkpoint)
            logger.warning("Resume checkpoint only contained model weights; continuing from epoch 0")

        logger.info(f"Resumed from epoch {start_epoch} with best_val_loss={trainer.best_val_loss:.4f}")

    logger.info(
        f"Starting training loop | epochs={num_epochs} | start_epoch={start_epoch} | "
        f"patience={patience} | annealing_epochs={annealing_epochs} | "
        f"batch_log_interval={batch_log_interval}"
    )
    logger.info("Handoff to trainer.fit beginning now")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        annealing_epochs=annealing_epochs,
        log_every_n_batches=batch_log_interval,
        start_epoch=start_epoch,
        resume_optimizer_state=resume_optimizer_state,
        max_wall_time_seconds=max_wall_time_seconds,
    )
    logger.info("trainer.fit returned; training loop has completed or exited early")

    final_model_path = output_dir / "final.pt"
    final_history_path = output_dir / "history.json"
    config_snapshot_path = output_dir / "config.yaml"
    logger.info("Writing final config snapshot")
    config_snapshot_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    logger.info("Saving final model checkpoint")
    with final_model_path.open("wb") as final_model_file:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "trainer_history": trainer.history,
                "final_epoch": num_epochs,
                "training_timestamp": datetime.now().isoformat(),
            },
            final_model_file,
        )

    history_dict = {
        key: [float(value) if isinstance(value, (float, int)) else value for value in values]
        for key, values in trainer.history.items()
    }
    logger.info("Saving training history")
    with final_history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history_dict, history_file, indent=2)

    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Training history saved to: {final_history_path}")
    logger.info(f"Config snapshot saved to: {config_snapshot_path}")

    test_results: Dict[str, Any] = {}
    if test_loader is not None and len(test_loader) > 0:
        best_checkpoint_path = output_dir / "best_model.pt"
        if best_checkpoint_path.exists():
            logger.info(f"Evaluating best checkpoint on the held-out test set: {best_checkpoint_path}")
            test_results = trainer.test(
                test_loader=test_loader,
                checkpoint_path=str(best_checkpoint_path),
                save_name="test_results.json",
            )
        else:
            logger.info("Best checkpoint not found, evaluating current model weights on the test set")
            test_results = trainer.test(
                test_loader=test_loader,
                save_name="test_results.json",
            )
    else:
        logger.info("No test loader available; skipping held-out test evaluation")

    results = {
        "status": "completed",
        "output_dir": str(output_dir),
        "model_path": str(final_model_path),
        "history_path": str(final_history_path),
        "device": str(device),
        "best_val_loss": float(trainer.best_val_loss),
        "best_val_acc": float(trainer.best_val_acc),
        "final_epoch": num_epochs,
        "test_results_path": test_results.get("results_path"),
        "test_accuracy": test_results.get("overall_accuracy"),
        "test_macro_f1": test_results.get("macro_f1"),
        "test_weighted_f1": test_results.get("weighted_f1"),
        "test_per_class_accuracy": test_results.get("per_class_accuracy"),
        "test_confusion_matrix": test_results.get("confusion_matrix"),
        "training_timestamp": datetime.now().isoformat(),
    }

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Run summary saved to: {summary_path}")
    logger.info(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
