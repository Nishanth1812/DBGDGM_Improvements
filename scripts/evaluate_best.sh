#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Evaluate a saved MM-DBGDGM checkpoint on the prepared test split.

Usage:
  ./scripts/evaluate_best.sh [options] [-- extra_train_local_args]

Options:
  --preprocessed-dir DIR   Directory with train/val/test metadata CSVs (default: /root/mm_dbgdgm_inputs/preprocessed)
  --dataset-root DIR       Dataset root passed to train_local (default: /root/mm_dbgdgm_inputs)
  --checkpoint FILE        Checkpoint to evaluate (default: latest /root/mm_dbgdgm_inputs/runs/*/best_model.pt)
  --output-dir DIR         Evaluation output directory (default: /root/mm_dbgdgm_inputs/eval/<timestamp>)
  --device DEVICE          auto|cuda|cpu (default: cuda)
  --batch-size INT         Batch size for dataloaders (default: 16)
  --num-workers INT        DataLoader workers (default: 12)
  --help                   Show this help

Note:
  This uses train_local.py with --num-epochs 0 and --resume-from CHECKPOINT,
  so it runs the test evaluation without additional training epochs.
EOF
}

PREPROCESSED_DIR="/root/mm_dbgdgm_inputs/preprocessed"
DATASET_ROOT="/root/mm_dbgdgm_inputs"
CHECKPOINT=""
OUTPUT_DIR=""
DEVICE="cuda"
BATCH_SIZE="16"
NUM_WORKERS="12"

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
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
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

TRAIN_CSV="$PREPROCESSED_DIR/train_metadata_balanced.csv"
if [[ ! -f "$TRAIN_CSV" ]]; then
  TRAIN_CSV="$PREPROCESSED_DIR/train_metadata.csv"
fi
VAL_CSV="$PREPROCESSED_DIR/val_metadata.csv"
TEST_CSV="$PREPROCESSED_DIR/test_metadata.csv"

for required in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required metadata file not found: $required" >&2
    exit 1
  fi
done

if [[ -z "$CHECKPOINT" ]]; then
  latest_match=$(ls -1dt /root/mm_dbgdgm_inputs/runs/* 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest_match" && -f "$latest_match/best_model.pt" ]]; then
    CHECKPOINT="$latest_match/best_model.pt"
  fi
fi

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found. Provide --checkpoint /path/to/best_model.pt" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="/root/mm_dbgdgm_inputs/eval/$(date +'%Y%m%d_%H%M%S')"
fi
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python/python3 not found in PATH" >&2
  exit 1
fi

echo "Evaluating checkpoint: $CHECKPOINT"

"$PYTHON_BIN" "$PROJECT_ROOT/train_local.py" \
  --dataset-root "$DATASET_ROOT" \
  --train-metadata "$TRAIN_CSV" \
  --val-metadata "$VAL_CSV" \
  --test-metadata "$TEST_CSV" \
  --resume-from "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-epochs 0 \
  "${EXTRA_ARGS[@]}"

echo "Evaluation outputs saved to: $OUTPUT_DIR"

