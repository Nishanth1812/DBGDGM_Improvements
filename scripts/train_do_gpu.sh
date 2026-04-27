#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
DigitalOcean GPU training launcher for MM-DBGDGM.

Usage:
  ./scripts/train_do_gpu.sh [launcher-options] [-- train_local_args...]

Launcher options:
  --run-name NAME          Prefix for run folders (default: do_mm_dbgdgm)
  --runs-root DIR          Root directory for runs (default: auto-detected scratch path)
  --output-dir DIR         Explicit train_local output dir
  --work-dir DIR           Working directory for extracted/intermediate data
  --config FILE            Config path passed to train_local
  --resume-from FILE       Checkpoint path to resume from
  --device DEVICE          auto|cuda|cpu (default: cuda)
  --batch-size INT         Override batch size
  --num-workers INT        Override dataloader workers
  --num-epochs INT         Override number of epochs
  --seed INT               Override random seed
  --help                   Show this help text

Everything after '--' is passed through to train_local.py unchanged.

Examples:
  ./scripts/train_do_gpu.sh -- \
    --dataset-root /mnt/data/fmri \
    --metadata-file /mnt/data/labels.csv \
    --smri-source-root /mnt/data/smri

  ./scripts/train_do_gpu.sh \
    --resume-from /mnt/scratch/runs/run_001/best_model.pt -- \
    --dataset-root /mnt/data/fmri \
    --metadata-file /mnt/data/labels.csv \
    --smri-source-root /mnt/data/smri
EOF
}

pick_scratch_root() {
  local candidates=(
    "/mnt/scratch"
    "/scratch"
    "/mnt"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" && -w "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_NAME="do_mm_dbgdgm"
RUNS_ROOT=""
OUTPUT_DIR=""
WORK_DIR=""
CONFIG_PATH=""
RESUME_FROM=""
DEVICE="cuda"
BATCH_SIZE=""
NUM_WORKERS=""
NUM_EPOCHS=""
SEED=""

TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --runs-root)
      RUNS_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --resume-from)
      RESUME_FROM="$2"
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
    --num-epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        TRAIN_ARGS+=("$1")
        shift
      done
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$RUNS_ROOT" ]]; then
  if SCRATCH_ROOT="$(pick_scratch_root)"; then
    RUNS_ROOT="${SCRATCH_ROOT}/mm_dbgdgm_runs"
  else
    RUNS_ROOT="${PROJECT_ROOT}/runs"
  fi
fi

mkdir -p "$RUNS_ROOT"

if [[ -n "$RESUME_FROM" && -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(cd "$(dirname "$RESUME_FROM")" && pwd)"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  RUN_STAMP="$(date +"%Y%m%d_%H%M%S")"
  OUTPUT_DIR="${RUNS_ROOT}/${RUN_NAME}_${RUN_STAMP}"
fi

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="${RUNS_ROOT}/work"
fi

if [[ -z "$CONFIG_PATH" ]]; then
  if [[ -f "${PROJECT_ROOT}/MM_DBGDGM/configs/default.yaml" ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/MM_DBGDGM/configs/default.yaml"
  elif [[ -f "${PROJECT_ROOT}/mm_dbgdgm/config.yaml" ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/mm_dbgdgm/config.yaml"
  else
    echo "ERROR: Could not auto-detect config file. Use --config <path>." >&2
    exit 1
  fi
fi

if [[ ! -f "${PROJECT_ROOT}/train_local.py" ]]; then
  echo "ERROR: train_local.py not found under project root: ${PROJECT_ROOT}" >&2
  exit 1
fi

# Linux filesystems are case-sensitive; keep compatibility with imports expecting MM_DBGDGM.
if [[ ! -e "${PROJECT_ROOT}/MM_DBGDGM" && -d "${PROJECT_ROOT}/mm_dbgdgm" ]]; then
  ln -s "mm_dbgdgm" "${PROJECT_ROOT}/MM_DBGDGM"
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$WORK_DIR"

CONSOLE_LOG="${OUTPUT_DIR}/launcher_console.log"
COMMAND_LOG="${OUTPUT_DIR}/launcher_command.txt"
ENV_LOG="${OUTPUT_DIR}/launcher_env.txt"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python/python3 not found in PATH" >&2
  exit 1
fi

if command -v dbgdgm-train-local >/dev/null 2>&1; then
  TRAIN_CMD=(dbgdgm-train-local)
else
  TRAIN_CMD=("${PYTHON_BIN}" "${PROJECT_ROOT}/train_local.py")
fi

TRAIN_CMD+=(
  --config "$CONFIG_PATH"
  --output-dir "$OUTPUT_DIR"
  --work-dir "$WORK_DIR"
  --device "$DEVICE"
)

if [[ -n "$RESUME_FROM" ]]; then
  TRAIN_CMD+=(--resume-from "$RESUME_FROM")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  TRAIN_CMD+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$NUM_WORKERS" ]]; then
  TRAIN_CMD+=(--num-workers "$NUM_WORKERS")
fi
if [[ -n "$NUM_EPOCHS" ]]; then
  TRAIN_CMD+=(--num-epochs "$NUM_EPOCHS")
fi
if [[ -n "$SEED" ]]; then
  TRAIN_CMD+=(--seed "$SEED")
fi
if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${TRAIN_ARGS[@]}")
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"

echo "[$(date -Iseconds)] Project root: ${PROJECT_ROOT}" | tee -a "$CONSOLE_LOG"
echo "[$(date -Iseconds)] Output dir:   ${OUTPUT_DIR}" | tee -a "$CONSOLE_LOG"
echo "[$(date -Iseconds)] Work dir:     ${WORK_DIR}" | tee -a "$CONSOLE_LOG"
echo "[$(date -Iseconds)] Config:       ${CONFIG_PATH}" | tee -a "$CONSOLE_LOG"
echo "[$(date -Iseconds)] Device:       ${DEVICE}" | tee -a "$CONSOLE_LOG"

printf '%q ' "${TRAIN_CMD[@]}" | tee "$COMMAND_LOG"
echo | tee -a "$COMMAND_LOG"

env | sort > "$ENV_LOG"

set +e
"${TRAIN_CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
CMD_EXIT=${PIPESTATUS[0]}
set -e

if [[ $CMD_EXIT -ne 0 ]]; then
  echo "[$(date -Iseconds)] Training failed with exit code ${CMD_EXIT}" | tee -a "$CONSOLE_LOG"
  exit "$CMD_EXIT"
fi

echo "[$(date -Iseconds)] Training completed successfully" | tee -a "$CONSOLE_LOG"

