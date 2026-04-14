"""Prepare JPG-based SMRI data, build a Modal-compatible cache, and upload both to Modal.

This is the local workflow for cheaper reruns:

1. Rebuild the root `Data/` JPG folders into a subject-level prepared dataset.
2. Build the same `samples_*.pt` cache file that `modal_train.py` would create.
3. Upload both artifacts into the persistent `mm-dbgdgm-data` Modal volume.

The cached file is the main cost saver. Once it exists on the volume, the Modal
trainer should load it and skip the expensive preprocessing/download step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal
import numpy as np
import torch
import yaml

from prepare_smri_jpg_dataset import build_dataset as build_smri_dataset


def _read_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _compute_cache_key(config: Dict[str, Any], drive_folder_url: str) -> str:
    data_cfg = config["data"]
    model_cfg = config["model"]
    cache_key_src = "|".join([
        drive_folder_url,
        str(int(model_cfg["n_roi"])),
        str(int(model_cfg["seq_len"])),
        str(int(model_cfg["n_smri_features"])),
        str(int(data_cfg.get("max_series", 2000))),
        str(int(data_cfg.get("max_dicoms_per_series", 120))),
        str(int(data_cfg.get("seed", 42))),
        f"{float(data_cfg.get('inference_holdout_ratio', 0.02)):.6f}",
        str(bool(data_cfg.get("use_filename_clues", True))),
        str(bool(data_cfg.get("strict_fmri_clues", False))),
    ])
    return hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()[:16]


def _load_label_map(labels_csv_path: Path) -> Dict[str, int]:
    if not labels_csv_path.exists():
        return {}

    label_map: Dict[str, int] = {}
    with labels_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            subject_id = str(row.get("subject_id", "")).strip()
            if not subject_id:
                continue
            label_value = int(row.get("label", 0))
            label_map[subject_id] = label_value

    return label_map


def _class_label_from_path(series_dir: Path) -> Optional[int]:
    class_name = series_dir.parent.name.strip().lower()
    mapping = {
        "non demented": 0,
        "very mild dementia": 1,
        "mild dementia": 2,
        "moderate dementia": 3,
    }
    return mapping.get(class_name)


def _build_samples_from_prepared_root(
    prepared_root: Path,
    config: Dict[str, Any],
    labels_csv: Path,
) -> Tuple[List[Dict[str, torch.Tensor]], List[str]]:
    from modal_train import _build_sample_from_series, _extract_subject_id, _split_series_for_inference

    data_cfg = config["data"]
    model_cfg = config["model"]
    label_map = _load_label_map(labels_csv)

    n_roi = int(model_cfg["n_roi"])
    seq_len = int(model_cfg["seq_len"])
    n_smri_features = int(model_cfg["n_smri_features"])
    max_dicoms_per_series = int(data_cfg.get("max_dicoms_per_series", 120))
    num_classes = int(model_cfg.get("num_classes", 4))
    seed = int(data_cfg.get("seed", 42))
    holdout_ratio = float(data_cfg.get("inference_holdout_ratio", 0.02))
    holdout_ratio = float(np.clip(holdout_ratio, 0.0, 0.5))

    from modal_train import _find_series_dirs_with_dicoms

    series_dirs = _find_series_dirs_with_dicoms(prepared_root)
    series_dirs = [series_dir for series_dir in series_dirs if series_dir.is_dir()]

    if int(data_cfg.get("max_series", 0)) > 0:
        series_dirs = series_dirs[: int(data_cfg.get("max_series", 0))]

    train_series_dirs, inference_series_dirs = _split_series_for_inference(
        series_dirs=series_dirs,
        seed=seed,
        holdout_ratio=holdout_ratio,
    )

    samples: List[Dict[str, torch.Tensor]] = []
    for index, series_dir in enumerate(train_series_dirs, start=1):
        subject_id = _extract_subject_id(series_dir)
        label = label_map.get(subject_id)
        if label is None:
            label = _class_label_from_path(series_dir)
        if label is None:
            label = int(subject_id[-1]) % num_classes if subject_id else 0

        sample = _build_sample_from_series(
            series_dir=series_dir,
            label=int(np.clip(label, 0, num_classes - 1)),
            n_roi=n_roi,
            seq_len=seq_len,
            n_smri_features=n_smri_features,
            max_dicoms_per_series=max_dicoms_per_series,
        )
        if sample is None:
            continue

        samples.append(sample)

    return samples, []


def _save_cache_payload(
    output_root: Path,
    cache_key: str,
    samples: List[Dict[str, torch.Tensor]],
    inference_series_dirs: List[str],
    holdout_ratio: float,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / f"samples_{cache_key}.pt"

    cache_payload = {
        "version": 2,
        "samples": samples,
        "inference_series_dirs": inference_series_dirs,
        "inference_holdout_ratio": float(holdout_ratio),
        "cache_key": cache_key,
        "saved_at": __import__("datetime").datetime.now().isoformat(),
    }
    torch.save(cache_payload, cache_path)
    return cache_path


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                zip_file.write(file_path, file_path.relative_to(source_dir))


def _upload_to_volume(
    volume_name: str,
    local_prepared_zip: Path,
    local_cache_path: Path,
    remote_prepared_zip: str,
    remote_cache_root: str,
) -> None:
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_prepared_zip), remote_prepared_zip)
        batch.put_file(str(local_cache_path), f"{remote_cache_root}/{local_cache_path.name}")

    volume.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare JPG sMRI data and upload Modal training inputs.")
    parser.add_argument("--config", type=Path, default=Path("MM_DBGDGM/configs/default_config.yaml"))
    parser.add_argument("--raw-data-root", type=Path, default=Path("Data"))
    parser.add_argument("--prepared-smri-root", type=Path, default=Path("prepared_smri_dataset"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("modal_training_artifacts"))
    parser.add_argument("--drive-folder-url", type=str, default="", help="Drive URL used only for the cache key.")
    parser.add_argument("--volume-name", type=str, default="mm-dbgdgm-data")
    parser.add_argument(
        "--remote-base-root",
        type=str,
        default="adni_drive_pipeline",
        help="Volume-relative root used by the trainer and cache lookup.",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize the prepared JPG dataset locally.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace any existing local artifacts.")
    parser.add_argument("--skip-smri-build", action="store_true", help="Reuse an existing prepared SMRI dataset.")
    parser.add_argument("--skip-upload", action="store_true", help="Build local artifacts only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_config(args.config)

    if not args.skip_smri_build:
        build_smri_dataset(
            input_root=args.raw_data_root,
            output_root=args.prepared_smri_root,
            transfer_mode=args.transfer_mode,
            overwrite=args.overwrite,
        )

    prepared_root = args.prepared_smri_root
    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared SMRI dataset not found: {prepared_root}")

    labels_csv = prepared_root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found in prepared dataset: {labels_csv}")

    drive_folder_url = args.drive_folder_url or str(config["data"].get("drive_folder_url", ""))
    cache_key = _compute_cache_key(config, drive_folder_url)

    remote_prepared_zip = f"{args.remote_base_root}/prepared_smri_dataset.zip"
    remote_cache_root = f"{args.remote_base_root}/cache"

    artifacts_root = args.artifacts_root
    artifacts_root.mkdir(parents=True, exist_ok=True)

    samples, remote_inference_series_dirs = _build_samples_from_prepared_root(
        prepared_root=prepared_root,
        config=config,
        labels_csv=labels_csv,
    )

    holdout_ratio = float(config["data"].get("inference_holdout_ratio", 0.02))
    holdout_ratio = float(np.clip(holdout_ratio, 0.0, 0.5))

    prepared_zip_path = artifacts_root / "prepared_smri_dataset.zip"
    _zip_directory(prepared_root, prepared_zip_path)

    cache_path = _save_cache_payload(
        output_root=artifacts_root / "cache",
        cache_key=cache_key,
        samples=samples,
        inference_series_dirs=remote_inference_series_dirs,
        holdout_ratio=holdout_ratio,
    )

    summary = {
        "prepared_smri_root": str(prepared_root),
        "cache_path": str(cache_path),
        "prepared_zip_path": str(prepared_zip_path),
        "cache_key": cache_key,
        "sample_count": len(samples),
        "inference_holdout_count": len(remote_inference_series_dirs),
        "remote_prepared_zip": remote_prepared_zip,
        "remote_cache_root": remote_cache_root,
        "volume_name": args.volume_name,
        "upload_requested": not args.skip_upload,
    }
    summary_path = artifacts_root / "training_inputs_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))

    if args.skip_upload:
        return

    _upload_to_volume(
        volume_name=args.volume_name,
        local_prepared_zip=prepared_zip_path,
        local_cache_path=cache_path,
        remote_prepared_zip=remote_prepared_zip,
        remote_cache_root=remote_cache_root,
    )

    print(json.dumps({"uploaded": True, "volume_name": args.volume_name}, indent=2))


if __name__ == "__main__":
    main()