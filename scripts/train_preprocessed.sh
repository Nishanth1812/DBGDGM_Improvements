#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Train MM-DBGDGM from preprocessed split metadata.

Usage:
  ./scripts/train_preprocessed.sh [options] [-- extra_train_local_args]

Options:
  --preprocessed-dir DIR   Directory with train/val/test CSVs (default: /root/mm_dbgdgm_inputs/preprocessed)
  --dataset-root DIR       Dataset root passed to train_local (default: /root/mm_dbgdgm_inputs)
  --output-dir DIR         Training output directory (default: /root/mm_dbgdgm_inputs/runs/<timestamp>)
  --batch-size INT         Batch size override (default: 16)
  --num-workers INT        Worker override (default: 12)
  --num-epochs INT         Epoch override (default: 60)
  --device DEVICE          auto|cuda|cpu (default: cuda)
  --help                   Show this help
EOF
}

PREPROCESSED_DIR="/root/mm_dbgdgm_inputs/preprocessed"
DATASET_ROOT="/root/mm_dbgdgm_inputs"
OUTPUT_DIR=""
BATCH_SIZE="16"
NUM_WORKERS="12"
NUM_EPOCHS="60"
DEVICE="cuda"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preprocessed-dir)
      PREPROCESSED_DIR="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --num-epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="/root/mm_dbgdgm_inputs/runs/$(date +'%Y%m%d_%H%M%S')"
fi

TRAIN_CSV="$PREPROCESSED_DIR/train_metadata.csv"
VAL_CSV="$PREPROCESSED_DIR/val_metadata.csv"
TEST_CSV="$PREPROCESSED_DIR/test_metadata.csv"
BALANCED_TRAIN_CSV="$PREPROCESSED_DIR/train_metadata_balanced.csv"

for required in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required metadata file not found: $required" >&2
    exit 1
  fi
done

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python/python3 not found in PATH" >&2
  exit 1
fi

# Re-balance train split defensively in case source CSV drifted.
"$PYTHON_BIN" - "$TRAIN_CSV" "$BALANCED_TRAIN_CSV" <<'PY'
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
out = Path(sys.argv[2])
df = pd.read_csv(src)
if "label" not in df.columns:
    raise ValueError("train metadata must contain a 'label' column")

counts = df["label"].value_counts().to_dict()
if not counts:
    raise ValueError("train metadata is empty")

min_count = min(counts.values())
parts = []
for label, group in df.groupby("label", sort=True):
    parts.append(group.sample(n=min_count, random_state=42))

balanced = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
out.parent.mkdir(parents=True, exist_ok=True)
balanced.to_csv(out, index=False)
print(f"Balanced training rows: {len(balanced)}")
print("Label distribution:")
print(balanced["label"].value_counts().sort_index().to_string())
PY

mkdir -p "$OUTPUT_DIR"

LAUNCHER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_do_gpu.sh"
if [[ ! -f "$LAUNCHER" ]]; then
  echo "ERROR: train launcher not found: $LAUNCHER" >&2
  exit 1
fi

echo "Starting training with output: $OUTPUT_DIR"

"$LAUNCHER" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-epochs "$NUM_EPOCHS" \
  -- \
  --dataset-root "$DATASET_ROOT" \
  --train-metadata "$BALANCED_TRAIN_CSV" \
  --val-metadata "$VAL_CSV" \
  --test-metadata "$TEST_CSV" \
  "${EXTRA_ARGS[@]}"

