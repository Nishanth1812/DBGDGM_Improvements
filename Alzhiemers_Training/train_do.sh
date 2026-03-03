#!/usr/bin/env bash
set -euo pipefail

SESSION="dbgdgm_train"
if [[ -z "${TMUX:-}" ]]; then
    tmux new-session -d -s "$SESSION" 2>/dev/null || tmux kill-session -t "$SESSION" && tmux new-session -d -s "$SESSION"
    tmux send-keys -t "$SESSION" "cd $(pwd) && $0 $*" Enter
    echo "Training running in tmux session '$SESSION'"
    echo "  Attach : tmux attach -t $SESSION"
    echo "  Detach : Ctrl+B then D"
    tmux attach -t "$SESSION"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
VOLUME="/mnt/trainingresults"

PYTHON="$VENV_PYTHON"
if [[ ! -f "$VENV_PYTHON" ]]; then
    PYTHON="python3"
    echo "[warn] .venv not found — using system python3. Run setup_droplet.sh first."
fi

DATASET="${DATASET:-oasis}"
CATEGORICAL_DIM="${CATEGORICAL_DIM:-3}"
TRIAL="${TRIAL:-1}"
WINDOW_SIZE="${WINDOW_SIZE:-15}"
WINDOW_STRIDE="${WINDOW_STRIDE:-5}"
GRID_SIZE="${GRID_SIZE:-15}"
GPU="${GPU:-0}"
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume-from)     RESUME_FROM="$2";     shift 2 ;;
        --dataset)         DATASET="$2";         shift 2 ;;
        --categorical-dim) CATEGORICAL_DIM="$2"; shift 2 ;;
        --trial)           TRIAL="$2";           shift 2 ;;
        --gpu)             GPU="$2";             shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SAVE_PATH="$VOLUME/models_${DATASET}_${TRIAL}"
LOG_DIR="$VOLUME/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${DATASET}_trial${TRIAL}_${TIMESTAMP}.log"

PYTHON_ARGS=(
    -u main.py
    --dataset "$DATASET"
    --categorical-dim "$CATEGORICAL_DIM"
    --trial "$TRIAL"
    --gpu "$GPU"
    --window-size "$WINDOW_SIZE"
    --window-stride "$WINDOW_STRIDE"
    --grid-size "$GRID_SIZE"
    --save-path "$SAVE_PATH"
)
[[ -n "$RESUME_FROM" ]] && PYTHON_ARGS+=(--resume-from "$RESUME_FROM")

echo ""
echo "============================================================"
echo "  DBGDGM Training  |  dataset=$DATASET  trial=$TRIAL  gpu=$GPU"
echo "  Checkpoints : $SAVE_PATH"
echo "  Log         : $LOG_FILE"
[[ -n "$RESUME_FROM" ]] && echo "  Resuming    : $RESUME_FROM"
echo "============================================================"
echo ""

cd "$SCRIPT_DIR"
"$PYTHON" "${PYTHON_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE="${PIPESTATUS[0]}"

echo ""
if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "  ✓ Done. Results in $VOLUME"
else
    echo "  ✗ Exited with code $EXIT_CODE — see $LOG_FILE"
    exit "$EXIT_CODE"
fi
