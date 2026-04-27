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
from typing import Dict, Iterable, List, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced metadata splits for MM-DBGDGM")
    parser.add_argument("--extracted-root", type=Path, default=Path("/root/mm_dbgdgm_inputs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/mm_dbgdgm_inputs/preprocessed"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-count", type=int, default=1)
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


def pair_modalities(extracted_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    fmri_map: Dict[str, List[Path]] = defaultdict(list)
    smri_map: Dict[str, List[Path]] = defaultdict(list)
    label_map: Dict[str, int] = {}
    label_default_subjects: set[str] = set()

    scanned_leaf_dirs = 0
    skipped_missing_subject_id = 0
    heuristic_as_fmri = 0
    heuristic_as_smri = 0

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
            inferred_label, matched = infer_label_with_match(folder)
            label_map[sid] = inferred_label
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
    rows, pairing_diagnostics = pair_modalities(extracted_root)

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

