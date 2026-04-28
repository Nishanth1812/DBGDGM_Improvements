#!/usr/bin/env python3
"""Build a subject-level labels CSV from fMRI/sMRI download metadata.

This script is robust to common ADNI column names and will merge labels from
both CSVs, preferring non-null labels with the best coverage.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger("build_labels_from_downloads")

SUBJECT_ID_COLUMNS = (
    "subject_id",
    "ptid",
    "subject",
    "participant_id",
    "participant",
    "id",
    "image_id",
    "rid",
)

LABEL_COLUMNS = (
    "label",
    "diagnosis",
    "dx",
    "dx_bl",
    "group",
    "class",
    "research_group",
)

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

SUBJECT_ID_REGEX = re.compile(r"(\d{3})[_.\-]S[_.\-](\d{4})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labels.csv from download metadata")
    parser.add_argument("--fmri-csv", type=Path, required=True, help="Path to FMRI download CSV")
    parser.add_argument("--smri-csv", type=Path, required=True, help="Path to SMRI download CSV")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output labels CSV path")
    parser.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(asctime)s] [%(levelname)s] %(message)s")


def find_first_matching_column(columns: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    lowered_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered_to_original:
            return lowered_to_original[candidate]
    return None


def normalize_subject_id(raw_value: object) -> str:
    text = str(raw_value).strip()
    match = SUBJECT_ID_REGEX.search(text)
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


def build_lookup(df: pd.DataFrame, source_name: str) -> Tuple[Dict[str, int], Dict[str, object]]:
    sid_col = find_first_matching_column(df.columns, SUBJECT_ID_COLUMNS)
    label_col = find_first_matching_column(df.columns, LABEL_COLUMNS)
    if sid_col is None or label_col is None:
        raise ValueError(
            f"{source_name}: could not detect subject/label columns. Columns: {list(df.columns)}"
        )

    lookup: Dict[str, int] = {}
    skipped = 0
    label_conflicts = 0
    for _, row in df.iterrows():
        sid = normalize_subject_id(row[sid_col])
        label = parse_label_value(row[label_col])
        if not sid or label is None:
            skipped += 1
            continue

        if sid in lookup and lookup[sid] != label:
            label_conflicts += 1
        lookup[sid] = label

    diagnostics = {
        "source": source_name,
        "subject_column": sid_col,
        "label_column": label_col,
        "rows_total": int(len(df)),
        "rows_loaded": int(len(lookup)),
        "rows_skipped": int(skipped),
        "label_conflicts": int(label_conflicts),
        "label_distribution": Counter(lookup.values()),
    }
    return lookup, diagnostics


def merge_lookups(primary: Dict[str, int], secondary: Dict[str, int]) -> Dict[str, int]:
    merged = dict(primary)
    for sid, label in secondary.items():
        if sid not in merged:
            merged[sid] = label
    return merged


def serialize_diagnostics(diagnostics: Dict[str, object]) -> Dict[str, object]:
    serialized: Dict[str, object] = {}
    for key, value in diagnostics.items():
        if isinstance(value, Counter):
            serialized[key] = {str(k): int(v) for k, v in value.items()}
        elif isinstance(value, (int, float, str)):
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    fmri_csv = args.fmri_csv.expanduser().resolve()
    smri_csv = args.smri_csv.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not fmri_csv.exists():
        raise FileNotFoundError(f"FMRI CSV not found: {fmri_csv}")
    if not smri_csv.exists():
        raise FileNotFoundError(f"SMRI CSV not found: {smri_csv}")

    fmri_df = pd.read_csv(fmri_csv)
    smri_df = pd.read_csv(smri_csv)

    fmri_lookup, fmri_diag = build_lookup(fmri_df, "fmri")
    smri_lookup, smri_diag = build_lookup(smri_df, "smri")

    # Prefer lookup with more classes/rows as primary
    fmri_score = (len(set(fmri_lookup.values())), len(fmri_lookup))
    smri_score = (len(set(smri_lookup.values())), len(smri_lookup))

    if smri_score > fmri_score:
        primary, secondary = smri_lookup, fmri_lookup
        primary_name = "smri"
    else:
        primary, secondary = fmri_lookup, smri_lookup
        primary_name = "fmri"

    merged = merge_lookups(primary, secondary)

    labels_df = pd.DataFrame(
        {
            "subject_id": sorted(merged.keys()),
            "label": [merged[sid] for sid in sorted(merged.keys())],
        }
    )

    labels_df.to_csv(out_csv, index=False)

    merged_dist = Counter(merged.values())
    summary = {
        "fmri": serialize_diagnostics(fmri_diag),
        "smri": serialize_diagnostics(smri_diag),
        "primary": primary_name,
        "merged_subjects": len(merged),
        "merged_label_distribution": {str(k): int(v) for k, v in merged_dist.items()},
        "output_csv": str(out_csv),
    }

    LOGGER.info("Wrote labels CSV: %s", out_csv)
    LOGGER.info("Merged label distribution: %s", summary["merged_label_distribution"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

