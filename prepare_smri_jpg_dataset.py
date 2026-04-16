"""Prepare a subject-level SMRI proxy dataset from class-folder JPGs.

Input layout expected:

    Data/
        Non Demented/
        Very mild Dementia/
        Mild Dementia/
        Moderate Dementia/

The script can rebuild either:

1. Class-folder JPG datasets like:

    Data/
        Non Demented/
        Very mild Dementia/
        Mild Dementia/
        Moderate Dementia/

2. Raw ADNI subject trees like:

    Data/
        ADNI/
            002_S_0295/
                ...

In class-folder mode, JPGs are grouped by subject prefix extracted from the
filename. In raw ADNI mode, JPGs are grouped by subject id extracted from the
directory tree and labels are loaded from a labels CSV when available.

The dataset is rebuilt as:

    prepared_smri_dataset/
        labels.csv
        dataset_summary.json
        Non Demented/
            OAS1_0001_MR1/
                ...jpg files...
        Very mild Dementia/
            OAS1_0003_MR1/
                ...jpg files...
        ...

The resulting folder can be zipped and uploaded into the existing Modal
training flow. The included labels.csv maps subject_id -> numeric class label.
"""

from __future__ import annotations

import argparse
import csv
import logging
import json
import os
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_CLASS_LABELS: List[Tuple[str, int]] = [
    ("Non Demented", 0),
    ("Very mild Dementia", 1),
    ("Mild Dementia", 2),
    ("Moderate Dementia", 3),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima", ""}
logger = logging.getLogger("smri-prep")


def _normalize_folder_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").replace("-", " ").split()).casefold()


def _canonical_subject_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _is_dicom_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in DICOM_EXTENSIONS


def _resolve_class_dir(input_root: Path, class_name: str) -> Path | None:
    aliases = {
        "Non Demented": ("Non Demented", "Non-Demented", "Non_Demented", "NonDemented", "CN", "Control", "Healthy Control"),
        "Very mild Dementia": ("Very mild Dementia", "Very Mild Dementia", "Very-mild Dementia", "eMCI", "EMCI", "MCI"),
        "Mild Dementia": ("Mild Dementia", "Mild-dementia", "lMCI", "LMCI"),
        "Moderate Dementia": ("Moderate Dementia", "Moderate-dementia", "AD", "Alzheimer", "Demented"),
    }

    candidate_names = (class_name, *aliases.get(class_name, ()))
    normalized_candidates = {_normalize_folder_name(name) for name in candidate_names}

    direct_candidate = input_root / class_name
    if direct_candidate.is_dir():
        return direct_candidate

    for candidate_name in candidate_names[1:]:
        candidate_dir = input_root / candidate_name
        if candidate_dir.is_dir():
            return candidate_dir

    matched_dirs: List[Path] = []
    for child_dir in input_root.rglob("*"):
        if child_dir.is_dir() and _normalize_folder_name(child_dir.name) in normalized_candidates:
            matched_dirs.append(child_dir)

    if not matched_dirs:
        return None

    matched_dirs.sort(key=lambda path: (len(path.relative_to(input_root).parts), str(path).lower()))
    return matched_dirs[0]


def _extract_subject_id_from_path(image_path: Path) -> str:
    parts = list(image_path.parts)
    if "ADNI" in parts:
        index = parts.index("ADNI")
        if index + 1 < len(parts):
            return parts[index + 1]

    parent_name = image_path.parent.name.strip()
    if parent_name:
        return parent_name

    return _infer_subject_id(image_path)


def _find_subject_root_from_path(image_path: Path, input_root: Path) -> Path:
    parts = list(image_path.parts)
    if "ADNI" in parts:
        index = parts.index("ADNI")
        if index + 1 < len(parts):
            subject_root = Path(*parts[: index + 2])
            if subject_root.exists():
                return subject_root

    return image_path.parent


def _load_label_map(labels_csv_path: Path | None, log: logging.Logger) -> Dict[str, int]:
    if labels_csv_path is None:
        return {}

    if not labels_csv_path.exists():
        log.warning(f"labels CSV not found at {labels_csv_path}; labels must come from the source tree")
        return {}

    label_map: Dict[str, int] = {}
    with labels_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            subject_id = str(row.get("subject_id", row.get("Subject", row.get("PTID", row.get("RID", ""))))).strip()
            if not subject_id:
                continue

            raw_label = row.get("label", row.get("Label", row.get("diagnosis", row.get("Diagnosis", row.get("DX", "")))))
            try:
                label_map[subject_id] = int(raw_label)
                continue
            except Exception:
                pass

            label_text = _normalize_folder_name(raw_label)
            stage_map = {
                "cn": 0,
                "non demented": 0,
                "control": 0,
                "healthy control": 0,
                "emci": 1,
                "very mild dementia": 1,
                "mci": 1,
                "lmci": 2,
                "mild dementia": 2,
                "ad": 3,
                "alzheimer": 3,
                "moderate dementia": 3,
            }
            if label_text in stage_map:
                label_map[subject_id] = stage_map[label_text]

            canonical_subject_id = _canonical_subject_key(subject_id)
            if canonical_subject_id and canonical_subject_id not in label_map:
                label_map[canonical_subject_id] = label_map.get(subject_id, stage_map.get(label_text, 0))

    log.info(f"Loaded {len(label_map)} subject label(s) from {labels_csv_path}")
    return label_map


def _lookup_subject_label(subject_id: str, label_map: Dict[str, int]) -> Optional[int]:
    direct_label = label_map.get(subject_id)
    if direct_label is not None:
        return direct_label
    return label_map.get(_canonical_subject_key(subject_id))


def _class_name_for_label(label: int) -> str:
    label_to_class = {class_label: class_name for class_name, class_label in DEFAULT_CLASS_LABELS}
    if label not in label_to_class:
        raise ValueError(f"Unsupported label {label}; expected one of {sorted(label_to_class)}")
    return label_to_class[label]


def _copy_subject_images(
    subject_images: List[Path],
    subject_root: Path,
    prepared_subject_dir: Path,
    transfer_mode: str,
) -> None:
    for image_path in subject_images:
        try:
            relative_path = image_path.relative_to(subject_root)
            destination = prepared_subject_dir / relative_path
        except Exception:
            destination = prepared_subject_dir / image_path.name
        _safe_transfer(image_path, destination, transfer_mode)


def _collect_subject_media(subject_root: Path) -> List[Path]:
    return sorted(
        file_path
        for file_path in subject_root.rglob("*")
        if file_path.is_file() and (_is_image_file(file_path) or _is_dicom_file(file_path))
    )


def _load_subject_proxy_features(subject_root: Path, max_files: int = 120) -> Optional[np.ndarray]:
    media_files = _collect_subject_media(subject_root)
    if not media_files:
        return None

    media_files = media_files[:max_files]

    slice_means: List[float] = []
    slice_stds: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []
    row_values: List[float] = []
    col_values: List[float] = []
    file_sizes: List[float] = []

    try:
        import pydicom
    except Exception:
        pydicom = None

    try:
        import cv2
    except Exception:
        cv2 = None

    for media_path in media_files:
        pixels: Optional[np.ndarray] = None

        if _is_dicom_file(media_path) and pydicom is not None:
            try:
                with pydicom.config.disable_value_validation():
                    ds = pydicom.dcmread(str(media_path), force=True)
                if hasattr(ds, "pixel_array"):
                    pixels = ds.pixel_array.astype(np.float32)
            except Exception:
                pixels = None
        elif cv2 is not None and _is_image_file(media_path):
            try:
                pixels = cv2.imread(str(media_path), cv2.IMREAD_UNCHANGED)
                if pixels is not None and pixels.ndim == 3:
                    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
                if pixels is not None:
                    pixels = pixels.astype(np.float32)
            except Exception:
                pixels = None

        if pixels is None or pixels.size == 0:
            continue

        slice_means.append(float(np.mean(pixels)))
        slice_stds.append(float(np.std(pixels)))
        mins.append(float(np.min(pixels)))
        maxs.append(float(np.max(pixels)))
        row_values.append(float(pixels.shape[0]))
        col_values.append(float(pixels.shape[1] if pixels.ndim >= 2 else 1.0))
        try:
            file_sizes.append(float(media_path.stat().st_size))
        except Exception:
            file_sizes.append(0.0)

    if not slice_means:
        return None

    mean_rows = float(np.mean(row_values)) if row_values else 0.0
    mean_cols = float(np.mean(col_values)) if col_values else 0.0
    aspect_ratio = float(mean_rows / max(mean_cols, 1.0))

    features = np.asarray([
        float(np.mean(slice_means)),
        float(np.std(slice_means)),
        float(np.mean(slice_stds)),
        float(np.min(mins)),
        float(np.max(maxs)),
        mean_rows / 512.0,
        mean_cols / 512.0,
        aspect_ratio,
        float(len(media_files)) / 100.0,
        float(np.mean(file_sizes)) / 1_000_000.0 if file_sizes else 0.0,
        float(np.std(file_sizes)) / 1_000_000.0 if file_sizes else 0.0,
    ], dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def _infer_subject_id(image_path: Path) -> str:
    """Infer a subject-level identifier from an image filename."""

    stem = image_path.stem

    match = re.match(r"^(?P<subject>.+?)_mpr-\d+_\d+$", stem, flags=re.IGNORECASE)
    if match:
        return match.group("subject")

    match = re.match(r"^(?P<subject>.+?)_\d+$", stem)
    if match and "_" in match.group("subject"):
        return match.group("subject")

    return stem


def _safe_transfer(src: Path, dst: Path, mode: str) -> None:
    """Copy, hardlink, or symlink a file into the output tree."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    shutil.copy2(src, dst)


def _collect_images(folder: Path) -> List[Path]:
    return sorted(
        file_path
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                zip_file.write(file_path, file_path.relative_to(source_dir))


def build_dataset(
    input_root: Path,
    output_root: Path,
    transfer_mode: str,
    overwrite: bool,
    labels_csv: Path | None = None,
    logger: logging.Logger | None = None,
) -> Dict[str, object]:
    log = logger or logging.getLogger("smri-prep")

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    log.info(
        f"Preparing SMRI dataset | input_root={input_root} | output_root={output_root} | "
        f"transfer_mode={transfer_mode} | overwrite={overwrite} | labels_csv={labels_csv if labels_csv else '<none>'}"
    )

    class_labels = dict(DEFAULT_CLASS_LABELS)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    class_summary: Dict[str, Dict[str, int]] = {}
    total_images = 0
    total_subjects = 0
    label_map = _load_label_map(labels_csv, log)

    missing_class_dirs: List[str] = []
    resolved_class_dirs: Dict[str, Path] = {}

    for class_name, label in DEFAULT_CLASS_LABELS:
        class_dir = _resolve_class_dir(input_root, class_name)
        if class_dir is None:
            log.warning(f"Could not resolve class folder for '{class_name}' under {input_root}")
            missing_class_dirs.append(class_name)
            continue

        resolved_class_dirs[class_name] = class_dir

        log.info(f"Scanning class '{class_name}' from {class_dir}")

        grouped_images: Dict[str, List[Path]] = defaultdict(list)
        class_images = _collect_images(class_dir)
        log.info(f"Found {len(class_images)} image files under {class_dir}")

        for image_index, image_path in enumerate(class_images, start=1):
            subject_id = _infer_subject_id(image_path)
            grouped_images[subject_id].append(image_path)
            if image_index == 1 or image_index == len(class_images) or image_index % 500 == 0:
                log.info(
                    f"Class '{class_name}': indexed {image_index}/{len(class_images)} images -> {len(grouped_images)} subjects"
                )

        prepared_class_dir = output_root / class_name
        prepared_class_dir.mkdir(parents=True, exist_ok=True)

        class_image_count = 0
        for subject_id in sorted(grouped_images):
            subject_images = sorted(grouped_images[subject_id])
            if not subject_images:
                continue

            prepared_subject_dir = prepared_class_dir / subject_id
            prepared_subject_dir.mkdir(parents=True, exist_ok=True)

            for image_path in subject_images:
                _safe_transfer(image_path, prepared_subject_dir / image_path.name, transfer_mode)

            image_count = len(subject_images)
            class_image_count += image_count
            total_images += image_count
            total_subjects += 1

            if total_subjects == 1 or total_subjects % 50 == 0:
                log.info(
                    f"Prepared {total_subjects} subjects so far | current class='{class_name}' | "
                    f"subject='{subject_id}' | images={image_count}"
                )

            rows.append(
                {
                    "subject_id": subject_id,
                    "class_name": class_name,
                    "label": label,
                    "image_count": image_count,
                    "source_folder": str(class_dir),
                    "smri_path": str(prepared_subject_dir.relative_to(output_root)),
                    "prepared_folder": str(prepared_subject_dir.relative_to(output_root)),
                }
            )

        class_summary[class_name] = {
            "label": label,
            "subjects": len(grouped_images),
            "images": class_image_count,
        }

        log.info(
            f"Finished class '{class_name}' | subjects={len(grouped_images)} | images={class_image_count}"
        )

    if missing_class_dirs and not resolved_class_dirs:
        if not label_map:
            available_dirs = [str(path) for path in sorted(path for path in input_root.rglob("*") if path.is_dir())[:50]]
            raise FileNotFoundError(
                "Could not resolve the expected class folders under "
                f"{input_root}: {missing_class_dirs}. Available directories include: {available_dirs}. "
                "This input looks like a raw ADNI subject tree, so a labels CSV is required."
            )

        log.info("No class folders were found; switching to raw ADNI subject-tree mode")
        raw_groups: Dict[str, List[Path]] = defaultdict(list)
        raw_subject_roots: Dict[str, Path] = {}

        for media_path in _collect_subject_media(input_root):
            subject_id = _extract_subject_id_from_path(media_path)
            raw_groups[subject_id].append(media_path)
            raw_subject_roots.setdefault(subject_id, _find_subject_root_from_path(media_path, input_root))

        if not raw_groups:
            raise FileNotFoundError(f"No image files were found under {input_root}")

        rows.clear()
        class_summary.clear()
        total_images = 0
        total_subjects = 0

        for subject_index, subject_id in enumerate(sorted(raw_groups), start=1):
            subject_media = sorted(raw_groups[subject_id])
            if not subject_media:
                continue

            label = _lookup_subject_label(subject_id, label_map)
            if label is None:
                raise FileNotFoundError(
                    f"Missing label for subject_id={subject_id!r} in {labels_csv if labels_csv else 'labels CSV'}"
                )

            label = int(label)
            class_name = _class_name_for_label(label)
            subject_root = raw_subject_roots.get(subject_id, input_root)
            prepared_subject_dir = output_root / class_name / subject_id
            prepared_subject_dir.mkdir(parents=True, exist_ok=True)

            proxy_features = _load_subject_proxy_features(subject_root)
            if proxy_features is None:
                raise FileNotFoundError(f"Could not derive proxy features from subject root: {subject_root}")

            np.save(prepared_subject_dir / "features.npy", proxy_features.astype(np.float32))

            image_count = len(subject_media)
            total_images += image_count
            total_subjects += 1

            if total_subjects == 1 or total_subjects % 25 == 0:
                log.info(
                    f"Prepared {total_subjects} raw ADNI subjects so far | subject='{subject_id}' | "
                    f"label={label} ({class_name}) | media_files={image_count} | features=features.npy"
                )

            rows.append(
                {
                    "subject_id": subject_id,
                    "class_name": class_name,
                    "label": label,
                    "image_count": image_count,
                    "source_folder": str(subject_root),
                    "smri_path": str((prepared_subject_dir / "features.npy").relative_to(output_root)),
                    "prepared_folder": str(prepared_subject_dir.relative_to(output_root)),
                }
            )

        for class_name, label in DEFAULT_CLASS_LABELS:
            class_subjects = [row for row in rows if row["class_name"] == class_name]
            class_summary[class_name] = {
                "label": label,
                "subjects": len(class_subjects),
                "images": sum(int(row["image_count"]) for row in class_subjects),
            }

        missing_class_dirs = []

    labels_csv = output_root / "labels.csv"
    with labels_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["subject_id", "class_name", "label", "image_count", "source_folder", "smri_path", "prepared_folder"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "class_labels": class_labels,
        "total_subjects": total_subjects,
        "total_images": total_images,
        "class_summary": class_summary,
        "labels_csv": str(labels_csv),
    }
    summary_path = output_root / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info(
        f"SMRI dataset prepared | total_subjects={total_subjects} | total_images={total_images} | "
        f"labels_csv={labels_csv} | summary={summary_path}"
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a subject-level SMRI JPG dataset.")
    parser.add_argument("--input-root", type=Path, default=Path("Data"), help="Root folder containing the diagnosis folders.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prepared_smri_dataset"),
        help="Destination root for the reorganized dataset.",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize files in the output tree.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output root if it already exists.")
    parser.add_argument("--create-zip", action="store_true", help="Also create a zip archive of the prepared dataset.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional zip output path. Defaults to <output-root>.zip when --create-zip is set.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional labels CSV for raw ADNI trees when class folders are not present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s - %(levelname)s] %(message)s")

    summary = build_dataset(
        args.input_root,
        args.output_root,
        args.transfer_mode,
        args.overwrite,
        labels_csv=args.labels_csv,
    )

    logger.info(json.dumps(summary, indent=2))

    if args.create_zip:
        zip_path = args.zip_path or args.output_root.with_suffix(".zip")
        _zip_directory(args.output_root, zip_path)
        logger.info(json.dumps({"zip_path": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()