#!/usr/bin/env python3
"""Create balanced train/val/test metadata from extracted ADNI-style folders.

The script scans extracted folders, pairs fMRI and sMRI by subject ID, infers class
labels from folder names, holds out one random sample for inference, balances the
remaining set across labels, and writes split CSV files.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


LOGGER = logging.getLogger("preprocess_balanced_splits")

FMRI_TOKENS = (
    "fmri", "bold", "rest", "rsfmri", "ep2d", "epi", "functional", "asl", "pcasl", "resting",
)
SMRI_TOKENS = (
    "mprage", "t1", "structural", "spgr", "t1w", "bravo", "axial", "sagittal", "coronal",
)

LABEL_RULES: List[Tuple[int, List[str]]] = [
    (3, ["alzheimer", "alzheimers", "/ad/", "_ad_", "\\ad\\", "dementia"]),
    (2, ["lmci", "late_mci", "late-mci", "late mci"]),
    (1, ["emci", "early_mci", "early-mci", "early mci", "/mci/", "_mci_", "\\mci\\"]),
    (0, ["cognitively_normal", "cogn_normal", "normal_control", "/cn/", "_cn_", "\\cn\\"]),
]

LABEL_NAME_TO_ID = {
    "cn": 0,
    "normal": 0,
    "control": 0,
    "emci": 1,
    "early_mci": 1,
    "early-mci": 1,
    "lmci": 2,
    "late_mci": 2,
    "late-mci": 2,
    "ad": 3,
    "alzheimers": 3,
    "alzheimer": 3,
    "dementia": 3,
}

SUBJECT_ID_COLUMNS = ("subject_id", "ptid", "subject", "participant_id", "id")
LABEL_COLUMNS = ("label", "diagnosis", "dx", "group", "class")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced metadata splits for MM-DBGDGM")
    parser.add_argument("--extracted-root", type=Path, default=Path("/root/mm_dbgdgm_inputs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/mm_dbgdgm_inputs/preprocessed"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-count", type=int, default=1)
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional CSV with subject-level labels")
    parser.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(asctime)s] [%(levelname)s] %(message)s")


def counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda kv: kv[0])}


def get_subject_id(path: Path) -> str:
    match = re.search(r"(\d{3})[_.\-]S[_.\-](\d{4})", str(path))
    if not match:
        return ""
    return f"{match.group(1)}_S_{match.group(2)}"


def normalize_subject_id(raw_value: object) -> str:
    text = str(raw_value).strip()
    match = re.search(r"(\d{3})[_.\-]S[_.\-](\d{4})", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return f"{match.group(1)}_S_{match.group(2)}"


def parse_label_value(raw_value: object) -> Optional[int]:
    if raw_value is None:
        return None

    if isinstance(raw_value, (int, float)):
        parsed_int = int(raw_value)
        return parsed_int if parsed_int in {0, 1, 2, 3} else None

    text = str(raw_value).strip().lower()
    if not text:
        return None

    if text.isdigit():
        parsed_int = int(text)
        return parsed_int if parsed_int in {0, 1, 2, 3} else None

    text_compact = text.replace(" ", "_")
    for key, label_id in LABEL_NAME_TO_ID.items():
        if key in text_compact:
            return label_id
    return None


def find_first_matching_column(columns: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    lowered_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered_to_original:
            return lowered_to_original[candidate]
    return None


def discover_labels_csv(extracted_root: Path) -> Optional[Path]:
    common_names = ["labels.csv", "metadata.csv", "manifest.csv"]
    for name in common_names:
        direct = extracted_root / name
        if direct.exists() and direct.is_file():
            return direct

    for name in common_names:
        matches = sorted(extracted_root.rglob(name))
        if matches:
            return matches[0]
    return None


def load_label_lookup(labels_csv: Path) -> Tuple[Dict[str, int], Dict[str, object]]:
    LOGGER.info("Loading labels from CSV: %s", labels_csv)
    df = pd.read_csv(labels_csv)

    sid_col = find_first_matching_column(df.columns, SUBJECT_ID_COLUMNS)
    label_col = find_first_matching_column(df.columns, LABEL_COLUMNS)

    if sid_col is None or label_col is None:
        raise ValueError(
            f"Could not detect subject/label columns in {labels_csv}. "
            f"Columns found: {list(df.columns)}"
        )

    lookup: Dict[str, int] = {}
    skipped_rows = 0
    for _, row in df.iterrows():
        sid = normalize_subject_id(row[sid_col])
        label = parse_label_value(row[label_col])
        if not sid or label is None:
            skipped_rows += 1
            continue
        lookup[sid] = label

    diagnostics = {
        "labels_csv": str(labels_csv),
        "rows_total": int(len(df)),
        "rows_loaded": int(len(lookup)),
        "rows_skipped": int(skipped_rows),
        "subject_column": sid_col,
        "label_column": label_col,
        "label_distribution": counter_to_dict(Counter(lookup.values())),
    }
    LOGGER.info("Loaded label lookup rows: %s", diagnostics["rows_loaded"])
    LOGGER.info("Label lookup distribution: %s", diagnostics["label_distribution"])
    return lookup, diagnostics


def infer_label(path: Path) -> int:
    normalized = str(path).lower().replace("\\", "/")
    for label, keywords in LABEL_RULES:
        if any(keyword in normalized for keyword in keywords):
            return label
    return 0


def infer_label_with_match(path: Path) -> Tuple[int, bool]:
    normalized = str(path).lower().replace("\\", "/")
    for label, keywords in LABEL_RULES:
        if any(keyword in normalized for keyword in keywords):
            return label, True
    return 0, False


def iter_leaf_dirs(root: Path) -> Iterable[Path]:
    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        try:
            has_file = any(child.is_file() for child in candidate.iterdir())
        except PermissionError:
            continue
        if has_file:
            yield candidate


def pair_modalities(
    extracted_root: Path,
    label_lookup: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    fmri_map: Dict[str, List[Path]] = defaultdict(list)
    smri_map: Dict[str, List[Path]] = defaultdict(list)
    label_map: Dict[str, int] = {}
    label_default_subjects: set[str] = set()

    scanned_leaf_dirs = 0
    skipped_missing_subject_id = 0
    heuristic_as_fmri = 0
    heuristic_as_smri = 0
    labels_from_csv = 0
    labels_from_path = 0

    for folder in iter_leaf_dirs(extracted_root):
        scanned_leaf_dirs += 1
        sid = get_subject_id(folder)
        if not sid:
            skipped_missing_subject_id += 1
            continue

        text = str(folder).lower()
        is_fmri = any(token in text for token in FMRI_TOKENS)
        is_smri = any(token in text for token in SMRI_TOKENS)

        if is_fmri and not is_smri:
            fmri_map[sid].append(folder)
        elif is_smri and not is_fmri:
            smri_map[sid].append(folder)
        elif is_fmri and is_smri:
            fmri_map[sid].append(folder)
        else:
            file_count = sum(1 for child in folder.iterdir() if child.is_file())
            if file_count > 50:
                fmri_map[sid].append(folder)
                heuristic_as_fmri += 1
            else:
                smri_map[sid].append(folder)
                heuristic_as_smri += 1

        if sid not in label_map:
            if label_lookup is not None and sid in label_lookup:
                label_map[sid] = int(label_lookup[sid])
                labels_from_csv += 1
            else:
                inferred_label, matched = infer_label_with_match(folder)
                label_map[sid] = inferred_label
                labels_from_path += 1
                if not matched:
                    label_default_subjects.add(sid)

    paired_ids = sorted(set(fmri_map) & set(smri_map))
    fmri_only_ids = sorted(set(fmri_map) - set(smri_map))
    smri_only_ids = sorted(set(smri_map) - set(fmri_map))

    LOGGER.info("Leaf folders scanned: %s", scanned_leaf_dirs)
    LOGGER.info("Folders skipped (no subject id): %s", skipped_missing_subject_id)
    LOGGER.info("Subjects with fMRI candidates: %s", len(fmri_map))
    LOGGER.info("Subjects with sMRI candidates: %s", len(smri_map))
    LOGGER.info("Paired subjects (both modalities): %s", len(paired_ids))
    LOGGER.info("Heuristic-only modality assignment -> fMRI: %s | sMRI: %s", heuristic_as_fmri, heuristic_as_smri)
    if fmri_only_ids:
        LOGGER.warning("fMRI-only subjects: %s (examples: %s)", len(fmri_only_ids), fmri_only_ids[:5])
    if smri_only_ids:
        LOGGER.warning("sMRI-only subjects: %s (examples: %s)", len(smri_only_ids), smri_only_ids[:5])

    rows: List[Dict[str, object]] = []
    for sid in paired_ids:
        fmri_path = fmri_map[sid][0]
        smri_path = smri_map[sid][0]
        fmri_npy = fmri_path / "fmri.npy"

        rows.append(
            {
                "subject_id": sid,
                "timepoint": "T0",
                "label": int(label_map.get(sid, 0)),
                "fmri_path": str(fmri_npy if fmri_npy.exists() else fmri_path),
                "smri_path": str(smri_path),
            }
        )

    paired_label_counts = Counter(int(label_map.get(row["subject_id"], 0)) for row in rows)
    LOGGER.info("Paired label distribution: %s", counter_to_dict(paired_label_counts))
    LOGGER.info("Label source usage -> csv: %s | path-inference: %s", labels_from_csv, labels_from_path)
    if label_default_subjects:
        LOGGER.warning(
            "Subjects with default label=0 (no label keyword match): %s (examples: %s)",
            len(label_default_subjects),
            sorted(label_default_subjects)[:5],
        )

    diagnostics = {
        "leaf_dirs_scanned": scanned_leaf_dirs,
        "skipped_missing_subject_id": skipped_missing_subject_id,
        "heuristic_as_fmri": heuristic_as_fmri,
        "heuristic_as_smri": heuristic_as_smri,
        "subjects_with_fmri": len(fmri_map),
        "subjects_with_smri": len(smri_map),
        "paired_subjects": len(paired_ids),
        "fmri_only_subjects": len(fmri_only_ids),
        "smri_only_subjects": len(smri_only_ids),
        "label_default_subjects": len(label_default_subjects),
        "paired_label_distribution": counter_to_dict(paired_label_counts),
        "labels_from_csv": labels_from_csv,
        "labels_from_path": labels_from_path,
    }
    return rows, diagnostics


def per_class_split_count(class_size: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if class_size < 3:
        raise ValueError("Need at least 3 samples per class to build train/val/test splits.")

    train = int(class_size * train_ratio)
    val = int(class_size * val_ratio)
    test = class_size - train - val

    if train == 0:
        train = 1
    if val == 0:
        val = 1
    if test == 0:
        test = 1

    while train + val + test > class_size:
        if train >= val and train >= test and train > 1:
            train -= 1
        elif val >= test and val > 1:
            val -= 1
        elif test > 1:
            test -= 1
        else:
            break

    while train + val + test < class_size:
        train += 1

    return train, val, test


def build_balanced_splits(
    rows: List[Dict[str, object]],
    seed: int,
    inference_count: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    rng = random.Random(seed)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No paired subjects were found.")

    if inference_count < 1:
        raise ValueError("inference_count must be >= 1")
    if len(df) <= inference_count:
        raise ValueError("Not enough samples to set aside inference sample(s).")

    inference_indices = rng.sample(list(df.index), k=inference_count)
    inference_df = df.loc[inference_indices].reset_index(drop=True)
    remaining_df = df.drop(index=inference_indices).reset_index(drop=True)

    before_holdout_counts = Counter(df["label"]) if "label" in df.columns else Counter()
    after_holdout_counts = Counter(remaining_df["label"]) if "label" in remaining_df.columns else Counter()
    inference_counts = Counter(inference_df["label"]) if "label" in inference_df.columns else Counter()

    LOGGER.info("Label distribution before inference holdout: %s", counter_to_dict(before_holdout_counts))
    LOGGER.info("Label distribution after inference holdout: %s", counter_to_dict(after_holdout_counts))
    LOGGER.info("Inference holdout distribution: %s", counter_to_dict(inference_counts))

    class_groups = {
        int(label): group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for label, group in remaining_df.groupby("label")
    }
    if len(class_groups) < 2:
        raise ValueError(
            "Need at least two classes for balanced splitting. "
            f"Observed labels after inference holdout: {counter_to_dict(after_holdout_counts)}. "
            "This usually means folder names do not encode diagnosis labels for LABEL_RULES."
        )

    min_count = min(len(group) for group in class_groups.values())
    if min_count < 3:
        raise ValueError(
            "At least 3 samples per class are required after holding inference sample(s). "
            f"Observed per-class counts: {counter_to_dict(Counter(remaining_df['label']))}"
        )

    trimmed_groups = {label: group.iloc[:min_count].copy() for label, group in class_groups.items()}
    balanced_df = pd.concat(trimmed_groups.values(), ignore_index=True)

    train_chunks: List[pd.DataFrame] = []
    val_chunks: List[pd.DataFrame] = []
    test_chunks: List[pd.DataFrame] = []

    train_count, val_count, test_count = per_class_split_count(min_count, train_ratio, val_ratio)
    LOGGER.info(
        "Balanced per-class split counts -> train: %s, val: %s, test: %s (per class)",
        train_count,
        val_count,
        test_count,
    )

    for _, group in sorted(trimmed_groups.items()):
        train_chunks.append(group.iloc[:train_count])
        val_chunks.append(group.iloc[train_count : train_count + val_count])
        test_chunks.append(group.iloc[train_count + val_count : train_count + val_count + test_count])

    train_df = pd.concat(train_chunks, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_chunks, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_chunks, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    diagnostics = {
        "label_distribution_before_holdout": counter_to_dict(before_holdout_counts),
        "label_distribution_after_holdout": counter_to_dict(after_holdout_counts),
        "inference_distribution": counter_to_dict(inference_counts),
        "balanced_distribution": counter_to_dict(Counter(balanced_df["label"])),
        "train_distribution": counter_to_dict(Counter(train_df["label"])),
        "val_distribution": counter_to_dict(Counter(val_df["label"])),
        "test_distribution": counter_to_dict(Counter(test_df["label"])),
    }

    return balanced_df, train_df, val_df, test_df, inference_df, diagnostics


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    ratio_total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    extracted_root = args.extracted_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not extracted_root.exists():
        raise FileNotFoundError(f"Extracted root not found: {extracted_root}")

    LOGGER.info("Starting preprocessing")
    LOGGER.info("Extracted root: %s", extracted_root)
    LOGGER.info("Output root: %s", output_root)

    labels_csv_path: Optional[Path] = None
    label_lookup: Optional[Dict[str, int]] = None
    label_lookup_diagnostics: Dict[str, object] = {}
    if args.labels_csv is not None:
        labels_csv_path = args.labels_csv.expanduser().resolve()
    else:
        labels_csv_path = discover_labels_csv(extracted_root)

    if labels_csv_path is not None and labels_csv_path.exists():
        try:
            label_lookup, label_lookup_diagnostics = load_label_lookup(labels_csv_path)
        except Exception as exc:
            LOGGER.warning("Could not load labels CSV (%s). Falling back to path inference. Error: %s", labels_csv_path, exc)
    else:
        LOGGER.warning("No labels CSV found. Falling back to path-based label inference.")

    rows, pairing_diagnostics = pair_modalities(extracted_root, label_lookup=label_lookup)

    try:
        balanced_df, train_df, val_df, test_df, inference_df, split_diagnostics = build_balanced_splits(
            rows=rows,
            seed=args.seed,
            inference_count=args.inference_count,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    except Exception:
        failure_summary = {
            "extracted_root": str(extracted_root),
            "output_root": str(output_root),
            "label_lookup_diagnostics": label_lookup_diagnostics,
            "pairing_diagnostics": pairing_diagnostics,
            "hint": "If all labels are 0, update LABEL_RULES or provide label-aware folder naming.",
        }
        failure_path = output_root / "split_failure_diagnostics.json"
        failure_path.write_text(json.dumps(failure_summary, indent=2), encoding="utf-8")
        LOGGER.error("Wrote failure diagnostics to %s", failure_path)
        raise

    all_path = output_root / "all_balanced_metadata.csv"
    train_path = output_root / "train_metadata.csv"
    val_path = output_root / "val_metadata.csv"
    test_path = output_root / "test_metadata.csv"
    inference_path = output_root / "inference_sample.csv"

    balanced_df.to_csv(all_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    inference_df.to_csv(inference_path, index=False)

    summary = {
        "extracted_root": str(extracted_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "inference_count": args.inference_count,
        "total_paired_subjects": len(rows),
        "balanced_subjects": len(balanced_df),
        "train_subjects": len(train_df),
        "val_subjects": len(val_df),
        "test_subjects": len(test_df),
        "label_distribution": {
            "train": {str(k): int(v) for k, v in Counter(train_df["label"]).items()},
            "val": {str(k): int(v) for k, v in Counter(val_df["label"]).items()},
            "test": {str(k): int(v) for k, v in Counter(test_df["label"]).items()},
            "inference": {str(k): int(v) for k, v in Counter(inference_df["label"]).items()},
        },
        "diagnostics": {
            "label_lookup": label_lookup_diagnostics,
            "pairing": pairing_diagnostics,
            "splits": split_diagnostics,
        },
        "files": {
            "all_balanced_metadata": str(all_path),
            "train_metadata": str(train_path),
            "val_metadata": str(val_path),
            "test_metadata": str(test_path),
            "inference_sample": str(inference_path),
        },
    }

    summary_path = output_root / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

