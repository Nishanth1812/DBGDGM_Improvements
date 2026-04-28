#!/usr/bin/env python3
"""Create train/val/test metadata from extracted ADNI-style folders.

- Default mode: requires at least two classes and creates class-balanced splits.
- Fallback mode (--skip-balancing): writes all paired subjects to train split only.
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

SUBJECT_ID_COLUMNS = (
    "subject_id", "ptid", "subject", "participant_id", "participant", "id", "image_id",
)
LABEL_COLUMNS = (
    "label", "diagnosis", "dx", "dx_bl", "group", "class", "research_group",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata splits for MM-DBGDGM")
    parser.add_argument("--extracted-root", type=Path, default=Path("/root/mm_dbgdgm_inputs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/mm_dbgdgm_inputs/preprocessed"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-count", type=int, default=1)
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional CSV with subject-level labels")
    parser.add_argument("--skip-balancing", action="store_true", help="Skip balancing and write all data to train split")
    parser.add_argument("--balance-mode", type=str, default="trim", choices=("trim", "upsample"),
                        help="How to balance classes: trim (reduce to smallest) or upsample (oversample minorities)")
    parser.add_argument("--max-cn-ratio", type=float, default=None,
                        help="Cap CN (label 0) to this fraction of total training set (e.g., 0.5 = max 50%%).")
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
        label_id = int(raw_value)
        return label_id if label_id in {0, 1, 2, 3} else None

    text = str(raw_value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        label_id = int(text)
        return label_id if label_id in {0, 1, 2, 3} else None

    text_compact = text.replace(" ", "_")
    for key, label_id in LABEL_NAME_TO_ID.items():
        if key in text_compact:
            return label_id
    return None


def find_first_matching_column(columns: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    lowered_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered_to_original:
            return lowered_to_original[candidate]
    return None


def discover_label_csv_candidates(extracted_root: Path) -> List[Path]:
    candidate_names = ["labels.csv", "metadata.csv", "manifest.csv", "diagnosis.csv", "dx.csv"]
    search_roots: List[Path] = [extracted_root]
    if extracted_root.parent != extracted_root:
        search_roots.append(extracted_root.parent)

    discovered: Dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue

        for name in candidate_names:
            direct = root / name
            if direct.exists() and direct.is_file():
                discovered[str(direct.resolve())] = direct.resolve()

        for pattern in ("*label*.csv", "*meta*.csv", "*manifest*.csv", "*diag*.csv", "*dx*.csv"):
            for match in root.rglob(pattern):
                if match.is_file():
                    discovered[str(match.resolve())] = match.resolve()

    return sorted(discovered.values())


def load_label_lookup(labels_csv: Path) -> Tuple[Dict[str, int], Dict[str, object]]:
    df = pd.read_csv(labels_csv)

    sid_col = find_first_matching_column(df.columns, SUBJECT_ID_COLUMNS)
    label_col = find_first_matching_column(df.columns, LABEL_COLUMNS)
    if sid_col is None or label_col is None:
        raise ValueError(f"Could not detect subject/label columns in {labels_csv}. Columns: {list(df.columns)}")

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
    return lookup, diagnostics


def resolve_label_lookup(
    extracted_root: Path,
    explicit_labels_csv: Optional[Path],
    paired_subject_ids: List[str],
) -> Tuple[Optional[Dict[str, int]], Dict[str, object]]:
    if explicit_labels_csv is not None:
        explicit_path = explicit_labels_csv.expanduser().resolve()
        if not explicit_path.exists():
            raise FileNotFoundError(f"labels CSV does not exist: {explicit_path}")

        lookup, diagnostics = load_label_lookup(explicit_path)
        overlap = sum(1 for sid in paired_subject_ids if sid in lookup)
        diagnostics["paired_subject_overlap"] = int(overlap)
        LOGGER.info("Using explicit labels CSV: %s (overlap %s/%s)", explicit_path, overlap, len(paired_subject_ids))
        return lookup, {"selected": diagnostics, "candidates": [diagnostics]}

    candidates = discover_label_csv_candidates(extracted_root)
    if not candidates:
        LOGGER.warning("No candidate labels CSV found near extracted data.")
        return None, {"selected": {}, "candidates": []}

    LOGGER.info("Found %s candidate metadata CSV file(s)", len(candidates))
    evaluated: List[Dict[str, object]] = []
    best_lookup: Optional[Dict[str, int]] = None
    best_diag: Optional[Dict[str, object]] = None
    best_score: Tuple[int, int, int] = (-1, -1, -1)

    for candidate in candidates:
        try:
            lookup, diag = load_label_lookup(candidate)
        except Exception as exc:
            evaluated.append({"labels_csv": str(candidate), "error": str(exc)})
            continue

        overlap = sum(1 for sid in paired_subject_ids if sid in lookup)
        class_count = len(set(lookup.values()))
        loaded_rows = int(diag.get("rows_loaded", 0))
        score = (class_count, overlap, loaded_rows)

        diag = dict(diag)
        diag["paired_subject_overlap"] = int(overlap)
        diag["class_count"] = int(class_count)
        diag["score"] = list(score)
        evaluated.append(diag)

        if score > best_score:
            best_score = score
            best_lookup = lookup
            best_diag = diag

    if best_lookup is None:
        return None, {"selected": {}, "candidates": evaluated}

    LOGGER.info(
        "Selected labels CSV: %s | classes=%s | overlap=%s | loaded_rows=%s",
        best_diag.get("labels_csv"),
        best_diag.get("class_count"),
        best_diag.get("paired_subject_overlap"),
        best_diag.get("rows_loaded"),
    )
    return best_lookup, {"selected": best_diag, "candidates": evaluated}


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
    LOGGER.info("Leaf folders scanned: %s", scanned_leaf_dirs)
    LOGGER.info("Folders skipped (no subject id): %s", skipped_missing_subject_id)
    LOGGER.info("Paired subjects (both modalities): %s", len(paired_ids))
    LOGGER.info("Heuristic modality assignment -> fMRI: %s | sMRI: %s", heuristic_as_fmri, heuristic_as_smri)
    LOGGER.info("Paired label distribution: %s", counter_to_dict(paired_label_counts))
    LOGGER.info("Label source usage -> csv: %s | path-inference: %s", labels_from_csv, labels_from_path)
    if label_default_subjects:
        LOGGER.warning("Subjects with default label=0 (examples: %s)", sorted(label_default_subjects)[:5])

    diagnostics = {
        "leaf_dirs_scanned": scanned_leaf_dirs,
        "skipped_missing_subject_id": skipped_missing_subject_id,
        "paired_subjects": len(paired_ids),
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

    train = max(train, 1)
    val = max(val, 1)
    test = max(test, 1)

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


def apply_cn_sample_limit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_cn_ratio: Optional[float],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Limit CN (label 0) samples in training set (by ratio) and redistribute excess to val/test."""
    diagnostics: Dict[str, object] = {}
    
    if max_cn_ratio is None or max_cn_ratio <= 0 or max_cn_ratio > 1:
        return train_df, val_df, test_df, diagnostics
    
    cn_mask_train = train_df["label"] == 0
    cn_count_train = cn_mask_train.sum()
    
    # Calculate max_cn_samples from ratio based on total training set size
    max_cn_samples = int(len(train_df) * max_cn_ratio)
    
    diagnostics["cn_limit_applied"] = True
    diagnostics["cn_ratio_limit"] = float(max_cn_ratio)
    diagnostics["cn_train_before_limit"] = int(cn_count_train)
    diagnostics["cn_limit_samples"] = int(max_cn_samples)
    
    if cn_count_train <= max_cn_samples:
        diagnostics["cn_removed"] = 0
        return train_df, val_df, test_df, diagnostics
    
    # Separate CN and non-CN samples from training set
    cn_samples_train = train_df[cn_mask_train].reset_index(drop=True)
    non_cn_samples_train = train_df[~cn_mask_train].reset_index(drop=True)
    
    # Keep only max_cn_samples CN samples in training
    rng = random.Random(seed)
    cn_indices_to_keep = sorted(rng.sample(range(len(cn_samples_train)), max_cn_samples))
    cn_samples_to_keep = cn_samples_train.iloc[cn_indices_to_keep].reset_index(drop=True)
    cn_samples_to_redistribute = cn_samples_train.drop(cn_indices_to_keep).reset_index(drop=True)
    
    # Rebuild training set
    train_df_new = pd.concat([non_cn_samples_train, cn_samples_to_keep], ignore_index=True)
    train_df_new = train_df_new.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Redistribute excess CN samples to val and test
    cn_redistributed = len(cn_samples_to_redistribute)
    val_addition = cn_redistributed // 2
    test_addition = cn_redistributed - val_addition
    
    cn_for_val = cn_samples_to_redistribute.iloc[:val_addition].reset_index(drop=True)
    cn_for_test = cn_samples_to_redistribute.iloc[val_addition:].reset_index(drop=True)
    
    val_df_new = pd.concat([val_df, cn_for_val], ignore_index=True)
    test_df_new = pd.concat([test_df, cn_for_test], ignore_index=True)
    
    val_df_new = val_df_new.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df_new = test_df_new.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    diagnostics["cn_removed"] = int(cn_redistributed)
    diagnostics["cn_added_to_val"] = int(val_addition)
    diagnostics["cn_added_to_test"] = int(test_addition)
    diagnostics["cn_train_after_limit"] = int(max_cn_samples)
    
    LOGGER.info(
        "CN sample limiting applied: kept %s in train, redistributed %s (val: %s | test: %s)",
        max_cn_samples, cn_redistributed, val_addition, test_addition
    )
    
    return train_df_new, val_df_new, test_df_new, diagnostics


def build_balanced_splits(
    rows: List[Dict[str, object]],
    seed: int,
    inference_count: int,
    train_ratio: float,
    val_ratio: float,
    balance_mode: str = "trim",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    rng = random.Random(seed)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No paired subjects were found.")

    if inference_count < 1:
        raise ValueError("inference_count must be >= 1")
    if len(df) <= inference_count:
        raise ValueError("Not enough samples to set aside inference sample(s).")

    mutable_counts = Counter(df["label"])
    available_indices = list(df.index)
    inference_indices: List[int] = []
    for _ in range(inference_count):
        eligible = [idx for idx in available_indices if mutable_counts[int(df.loc[idx, "label"])] > 1]
        pool = eligible if eligible else available_indices
        chosen = rng.choice(pool)
        inference_indices.append(chosen)
        chosen_label = int(df.loc[chosen, "label"])
        mutable_counts[chosen_label] -= 1
        available_indices.remove(chosen)

    inference_df = df.loc[inference_indices].reset_index(drop=True)
    remaining_df = df.drop(index=inference_indices).reset_index(drop=True)

    after_holdout_counts = Counter(remaining_df["label"])
    # group by label without shuffling yet
    class_groups = {int(label): group.reset_index(drop=True) for label, group in remaining_df.groupby("label")}
    if len(class_groups) < 2:
        raise ValueError(
            "Need at least two classes for balanced splitting. "
            f"Observed labels after inference holdout: {counter_to_dict(after_holdout_counts)}"
        )

    # Balance according to chosen mode
    if balance_mode == "trim":
        min_count = min(len(group) for group in class_groups.values())
        if min_count < 3:
            raise ValueError(
                "At least 3 samples per class are required after inference holdout. "
                f"Observed: {counter_to_dict(after_holdout_counts)}"
            )
        trimmed_groups = {label: group.sample(frac=1.0, random_state=seed).iloc[:min_count].copy() for label, group in class_groups.items()}
        balanced_df = pd.concat(trimmed_groups.values(), ignore_index=True)
        per_class_size = min_count
        working_groups = trimmed_groups
    elif balance_mode == "upsample":
        max_count = max(len(group) for group in class_groups.values())
        target = max(3, max_count)
        upsampled_groups = {}
        for label, group in class_groups.items():
            if len(group) >= target:
                upsampled = group.sample(n=target, replace=False, random_state=seed).reset_index(drop=True)
            else:
                upsampled = group.sample(n=target, replace=True, random_state=seed).reset_index(drop=True)
            upsampled_groups[label] = upsampled
        balanced_df = pd.concat(upsampled_groups.values(), ignore_index=True)
        per_class_size = target
        working_groups = upsampled_groups
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    train_count, val_count, test_count = per_class_split_count(per_class_size, train_ratio, val_ratio)
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, group in sorted(working_groups.items()):
        train_parts.append(group.iloc[:train_count])
        val_parts.append(group.iloc[train_count : train_count + val_count])
        test_parts.append(group.iloc[train_count + val_count : train_count + val_count + test_count])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    diagnostics = {
        "mode": "balanced",
        "label_distribution_before_holdout": counter_to_dict(Counter(df["label"])),
        "label_distribution_after_holdout": counter_to_dict(after_holdout_counts),
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

    rows_without_lookup, _ = pair_modalities(extracted_root, label_lookup=None)
    paired_subject_ids = [str(row["subject_id"]) for row in rows_without_lookup]

    label_lookup_diagnostics: Dict[str, object] = {}
    label_lookup: Optional[Dict[str, int]] = None
    try:
        label_lookup, label_lookup_diagnostics = resolve_label_lookup(
            extracted_root=extracted_root,
            explicit_labels_csv=args.labels_csv,
            paired_subject_ids=paired_subject_ids,
        )
    except Exception as exc:
        LOGGER.warning("Label CSV resolution failed: %s", exc)

    rows, pairing_diagnostics = pair_modalities(extracted_root, label_lookup=label_lookup)
    df = pd.DataFrame(rows)

    if args.skip_balancing:
        LOGGER.warning("--skip-balancing enabled: writing all paired subjects to train split")
        if df.empty:
            raise ValueError("No paired subjects found; cannot create metadata splits.")

        shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        inference_n = min(max(args.inference_count, 1), len(shuffled) - 1) if len(shuffled) > 1 else 0
        inference_df = shuffled.iloc[:inference_n].copy()
        remaining = shuffled.iloc[inference_n:].copy()

        if len(remaining) < 3:
            train_df = remaining.copy()
            val_df = remaining.copy()
            test_df = remaining.copy()
        else:
            val_n = max(1, int(len(remaining) * args.val_ratio))
            test_n = max(1, int(len(remaining) * args.test_ratio))
            train_n = len(remaining) - val_n - test_n
            if train_n < 1:
                train_n = 1
                if val_n > test_n and val_n > 1:
                    val_n -= 1
                elif test_n > 1:
                    test_n -= 1

            train_df = remaining.iloc[:train_n].copy()
            val_df = remaining.iloc[train_n : train_n + val_n].copy()
            test_df = remaining.iloc[train_n + val_n : train_n + val_n + test_n].copy()

            if val_df.empty:
                val_df = train_df.iloc[:1].copy()
            if test_df.empty:
                test_df = train_df.iloc[:1].copy()

        balanced_df = remaining.copy()
        split_diagnostics = {
            "mode": "skip_balancing",
            "reason": "single_class_or_missing_labels",
            "remaining_after_inference": int(len(remaining)),
        }
    else:
        balanced_df, train_df, val_df, test_df, inference_df, split_diagnostics = build_balanced_splits(
            rows=rows,
            seed=args.seed,
            inference_count=args.inference_count,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            balance_mode=args.balance_mode,
        )
    
    # Apply CN sample limiting if specified
    cn_limit_diagnostics: Dict[str, object] = {}
    if not args.skip_balancing and args.max_cn_ratio is not None:
        train_df, val_df, test_df, cn_limit_diagnostics = apply_cn_sample_limit(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            max_cn_ratio=args.max_cn_ratio,
            seed=args.seed,
        )
        split_diagnostics["cn_limiting"] = cn_limit_diagnostics

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
        "skip_balancing": args.skip_balancing,
        "total_paired_subjects": len(rows),
        "balanced_subjects": len(balanced_df),
        "train_subjects": len(train_df),
        "val_subjects": len(val_df),
        "test_subjects": len(test_df),
        "label_distribution": {
            "train": counter_to_dict(Counter(train_df["label"])) if not train_df.empty else {},
            "val": counter_to_dict(Counter(val_df["label"])) if not val_df.empty else {},
            "test": counter_to_dict(Counter(test_df["label"])) if not test_df.empty else {},
            "inference": counter_to_dict(Counter(inference_df["label"])) if not inference_df.empty else {},
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

