#!/usr/bin/env python3
"""Prepare fMRI + sMRI DICOM folders into MM-DBGDGM-ready layout.

This script pairs subjects across fMRI/sMRI roots, copies source folders into
an output dataset folder, and writes labels.csv plus lightweight proxy features.

Proxy outputs:
- fmri.npy : (90, 200) built from per-slice intensity statistics.
- features.npy : (90, 4) global intensity stats repeated across 90 ROIs.

If you already have real preprocessed arrays, you can disable proxy generation
and point labels.csv to your own files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pydicom
except Exception:  # pragma: no cover - optional in some environments
    pydicom = None

# Expected model shapes
N_ROIS = 90
N_TIMEPOINTS = 200
N_SMRI_FEATURES = 4

# Heuristic tokens to auto-detect modality roots
FMRI_TOKENS = ("fmri", "bold", "rest", "rsfmri", "epi", "functional")
SMRI_TOKENS = ("smri", "mprage", "t1", "structural", "t1w")

# Default label mapping (case-insensitive)
DEFAULT_LABEL_MAP = {
    "cn": 0,
    "emci": 1,
    "lmci": 2,
    "mci": 1,
    "ad": 3,
}


@dataclass
class SubjectRecord:
    subject_id: str
    label: int
    fmri_src: Path
    smri_src: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair and preprocess fMRI/sMRI DICOM data for MM-DBGDGM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root folder containing input data")
    parser.add_argument("--fmri-root", type=Path, default=None, help="Explicit fMRI root folder")
    parser.add_argument("--smri-root", type=Path, default=None, help="Explicit sMRI root folder")
    parser.add_argument("--output-root", type=Path, default=None, help="Output folder (default: input_root/<auto>)")
    parser.add_argument("--label-map", type=str, default=None, help="JSON string or path to JSON file mapping labels")
    parser.add_argument("--allow-unknown-labels", action="store_true", help="Skip subjects with unknown labels")
    parser.add_argument("--skip-fmri-npy", action="store_true", help="Do not generate fmri.npy")
    parser.add_argument("--skip-smri-features", action="store_true", help="Do not generate features.npy")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing outputs")
    parser.add_argument("--accept-any-files", action="store_true", help="Treat non-DICOM files as valid slices")
    return parser.parse_args()


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "")


def load_label_map(label_map_arg: Optional[str]) -> Dict[str, int]:
    if not label_map_arg:
        return DEFAULT_LABEL_MAP.copy()
    candidate = Path(label_map_arg)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = label_map_arg
    loaded = json.loads(text)
    return {str(k).lower(): int(v) for k, v in loaded.items()}


def find_modality_root(input_root: Path, tokens: Tuple[str, ...]) -> Optional[Path]:
    for path in sorted(input_root.rglob("*")):
        if path.is_dir() and any(token in path.name.lower() for token in tokens):
            return path
    return None


def read_labels_from_csvs(root: Path, label_map: Dict[str, int]) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    csv_files = sorted(root.rglob("*.csv"))
    for csv_path in csv_files:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                fields = {field.lower(): field for field in reader.fieldnames}
                subject_field = None
                label_field = None
                for candidate in ("subject_id", "subjectid", "sub_id", "rid", "id"):
                    if candidate in fields:
                        subject_field = fields[candidate]
                        break
                for candidate in ("label", "class", "diagnosis", "group"):
                    if candidate in fields:
                        label_field = fields[candidate]
                        break
                if not subject_field or not label_field:
                    continue
                for row in reader:
                    subject_id = str(row.get(subject_field, "")).strip()
                    raw_label = str(row.get(label_field, "")).strip()
                    if not subject_id or not raw_label:
                        continue
                    label_key = _normalize_label(raw_label)
                    if label_key not in label_map:
                        continue
                    labels[subject_id] = int(label_map[label_key])
        except Exception:
            continue
    return labels


def _list_candidate_dirs(root: Path, subject_id: str) -> List[Path]:
    subject_lower = subject_id.lower()
    matches = []
    for path in root.rglob("*"):
        if path.is_dir() and subject_lower in path.name.lower():
            matches.append(path)
    return matches


def _list_candidate_files(root: Path, subject_id: str) -> List[Path]:
    subject_lower = subject_id.lower()
    matches = []
    for path in root.rglob("*"):
        if path.is_file() and subject_lower in path.name.lower():
            matches.append(path)
    return matches


def resolve_subject_folder(root: Path, subject_id: str, accept_any_files: bool) -> Optional[Path]:
    dir_matches = _list_candidate_dirs(root, subject_id)
    if dir_matches:
        return sorted(dir_matches, key=lambda p: len(str(p)))[0]

    file_matches = _list_candidate_files(root, subject_id)
    if not file_matches:
        return None

    # Use parent folder of the first matched file
    parent = sorted(file_matches, key=lambda p: len(str(p)))[0].parent
    if accept_any_files:
        return parent

    if any(is_dicom_file(path) for path in parent.iterdir() if path.is_file()):
        return parent
    return None


def is_dicom_file(path: Path) -> bool:
    if path.suffix.lower() == ".dcm":
        return True
    if pydicom is None:
        return False
    try:
        with path.open("rb") as handle:
            prefix = handle.read(132)
        return b"DICM" in prefix
    except Exception:
        return False


def load_dicom_pixel_data(path: Path) -> Optional[np.ndarray]:
    if pydicom is None:
        return None
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=False, force=True)
        pixel_array = ds.pixel_array.astype(np.float32)
        return pixel_array
    except Exception:
        return None


def scan_dicom_files(folder: Path, accept_any_files: bool) -> List[Path]:
    files = [path for path in folder.rglob("*") if path.is_file()]
    dicoms = [path for path in files if is_dicom_file(path)]
    if dicoms:
        return sorted(dicoms)
    if accept_any_files:
        return sorted(files)
    return []


def compute_proxy_smri_features(dicom_files: List[Path]) -> np.ndarray:
    values: List[float] = []
    for path in dicom_files:
        pixel = load_dicom_pixel_data(path)
        if pixel is None:
            values.append(float(path.stat().st_size))
        else:
            values.append(float(pixel.mean()))
    if not values:
        values = [0.0]
    values_arr = np.array(values, dtype=np.float32)
    mean_val = float(values_arr.mean())
    std_val = float(values_arr.std() + 1e-6)
    p10 = float(np.percentile(values_arr, 10))
    p90 = float(np.percentile(values_arr, 90))
    row = np.array([mean_val, std_val, p10, p90], dtype=np.float32)
    features = np.tile(row, (N_ROIS, 1))
    return features


def compute_proxy_fmri_timeseries(dicom_files: List[Path]) -> np.ndarray:
    values: List[float] = []
    for path in dicom_files:
        pixel = load_dicom_pixel_data(path)
        if pixel is None:
            values.append(float(path.stat().st_size))
        else:
            values.append(float(pixel.mean()))
    if not values:
        values = [0.0]
    values_arr = np.array(values, dtype=np.float32)
    # Resample to N_TIMEPOINTS
    if values_arr.size == 1:
        series = np.repeat(values_arr, N_TIMEPOINTS)
    else:
        x_old = np.linspace(0.0, 1.0, num=values_arr.size)
        x_new = np.linspace(0.0, 1.0, num=N_TIMEPOINTS)
        series = np.interp(x_new, x_old, values_arr)
    # Repeat across ROIs and z-score
    fmri = np.tile(series, (N_ROIS, 1))
    fmri = (fmri - fmri.mean(axis=1, keepdims=True)) / (fmri.std(axis=1, keepdims=True) + 1e-8)
    return fmri.astype(np.float32)


def build_subject_records(
    fmri_root: Path,
    smri_root: Path,
    label_map: Dict[str, int],
    allow_unknown_labels: bool,
    accept_any_files: bool,
) -> List[SubjectRecord]:
    fmri_labels = read_labels_from_csvs(fmri_root, label_map)
    smri_labels = read_labels_from_csvs(smri_root, label_map)

    all_subjects = set(fmri_labels) | set(smri_labels)
    records: List[SubjectRecord] = []
    for subject_id in sorted(all_subjects):
        label = fmri_labels.get(subject_id, smri_labels.get(subject_id))
        if label is None:
            if allow_unknown_labels:
                continue
            else:
                continue

        fmri_folder = resolve_subject_folder(fmri_root, subject_id, accept_any_files)
        smri_folder = resolve_subject_folder(smri_root, subject_id, accept_any_files)
        if fmri_folder is None or smri_folder is None:
            continue
        records.append(SubjectRecord(subject_id=subject_id, label=label, fmri_src=fmri_folder, smri_src=smri_folder))
    return records


def prepare_output_root(input_root: Path, output_root: Optional[Path]) -> Path:
    if output_root is not None:
        return output_root
    base = input_root / "mm_dbgdgm_prepared"
    if not base.exists():
        return base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_root / f"mm_dbgdgm_prepared_{timestamp}"


def copy_subject_folder(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    label_map = load_label_map(args.label_map)

    fmri_root = args.fmri_root
    smri_root = args.smri_root
    if fmri_root is None:
        fmri_root = find_modality_root(input_root, FMRI_TOKENS)
    if smri_root is None:
        smri_root = find_modality_root(input_root, SMRI_TOKENS)

    if fmri_root is None or smri_root is None:
        print("[ERROR] Could not detect fmri_root or smri_root. Provide --fmri-root and --smri-root.")
        return 2

    fmri_root = fmri_root.expanduser().resolve()
    smri_root = smri_root.expanduser().resolve()

    print(f"fMRI root: {fmri_root}")
    print(f"sMRI root: {smri_root}")

    records = build_subject_records(
        fmri_root=fmri_root,
        smri_root=smri_root,
        label_map=label_map,
        allow_unknown_labels=args.allow_unknown_labels,
        accept_any_files=args.accept_any_files,
    )

    if not records:
        print("[ERROR] No paired subjects found. Check CSVs and folder names.")
        return 3

    output_root = prepare_output_root(input_root, args.output_root).resolve()
    fmri_out_root = output_root / "fmri"
    smri_out_root = output_root / "smri"
    labels_csv = output_root / "labels.csv"

    print(f"Output root: {output_root}")
    print(f"Subjects paired: {len(records)}")

    if args.dry_run:
        for record in records[:5]:
            print(f"  {record.subject_id} | label={record.label}")
        print("Dry run only. No files were written.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for record in records:
        fmri_dst = fmri_out_root / record.subject_id
        smri_dst = smri_out_root / record.subject_id
        copy_subject_folder(record.fmri_src, fmri_dst)
        copy_subject_folder(record.smri_src, smri_dst)

        fmri_path = fmri_dst
        smri_path = smri_dst

        fmri_dicom_files = scan_dicom_files(fmri_dst, args.accept_any_files)
        smri_dicom_files = scan_dicom_files(smri_dst, args.accept_any_files)

        if not args.skip_fmri_npy:
            fmri_npy = fmri_dst / "fmri.npy"
            fmri_array = compute_proxy_fmri_timeseries(fmri_dicom_files)
            np.save(str(fmri_npy), fmri_array)
            fmri_path = fmri_npy

        if not args.skip_smri_features:
            features_npy = smri_dst / "features.npy"
            smri_features = compute_proxy_smri_features(smri_dicom_files)
            np.save(str(features_npy), smri_features)
            smri_path = features_npy

        rows.append({
            "subject_id": record.subject_id,
            "timepoint": "T0",
            "label": str(record.label),
            "fmri_path": str(fmri_path.resolve()),
            "smri_path": str(smri_path.resolve()),
        })

    with labels_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "timepoint", "label", "fmri_path", "smri_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved labels.csv: {labels_csv}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

