"""Prepare a subject-level SMRI proxy dataset from class-folder JPGs.

Input layout expected:

    Data/
        Non Demented/
        Very mild Dementia/
        Mild Dementia/
        Moderate Dementia/

The script groups JPGs by subject prefix extracted from the filename, then
rebuilds the dataset as:

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
import json
import os
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_CLASS_LABELS: List[Tuple[str, int]] = [
    ("Non Demented", 0),
    ("Very mild Dementia", 1),
    ("Mild Dementia", 2),
    ("Moderate Dementia", 3),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _normalize_folder_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").replace("-", " ").split()).casefold()


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


def build_dataset(input_root: Path, output_root: Path, transfer_mode: str, overwrite: bool) -> Dict[str, object]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

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

    missing_class_dirs: List[str] = []

    for class_name, label in DEFAULT_CLASS_LABELS:
        class_dir = _resolve_class_dir(input_root, class_name)
        if class_dir is None:
            missing_class_dirs.append(class_name)
            continue

        grouped_images: Dict[str, List[Path]] = defaultdict(list)
        for image_path in _collect_images(class_dir):
            subject_id = _infer_subject_id(image_path)
            grouped_images[subject_id].append(image_path)

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

    if missing_class_dirs:
        available_dirs = [str(path) for path in sorted(path for path in input_root.rglob("*") if path.is_dir())[:50]]
        raise FileNotFoundError(
            "Could not resolve the expected class folders under "
            f"{input_root}: {missing_class_dirs}. Available directories include: {available_dirs}"
        )

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(args.input_root, args.output_root, args.transfer_mode, args.overwrite)

    print(json.dumps(summary, indent=2))

    if args.create_zip:
        zip_path = args.zip_path or args.output_root.with_suffix(".zip")
        _zip_directory(args.output_root, zip_path)
        print(json.dumps({"zip_path": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()