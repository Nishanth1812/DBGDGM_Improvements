#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Download a Google Drive folder directly on the VM into MM-DBGDGM input storage.

Usage:
  ./scripts/download_drive_to_inputs.sh [options]

Options:
  --folder-url URL     Google Drive folder URL
  --folder-id ID       Google Drive folder ID (alternative to --folder-url)
  --dest DIR           Destination directory (default: /root/mm_dbgdgm_inputs)
  --log-file FILE      Log file path (default: <dest>/drive_download_<timestamp>.log)
  --no-install         Fail if gdown is missing (do not auto-install)
  --venv-dir DIR       Python virtualenv for gdown (default: /root/.venvs/gdown)
  --help               Show this help text

Examples:
  ./scripts/download_drive_to_inputs.sh

  ./scripts/download_drive_to_inputs.sh \
    --folder-url "https://drive.google.com/drive/folders/1Qx3jqL_eSxGfWvhxb20M6orxvKsQlQT9?usp=drive_link" \
    --dest /root/mm_dbgdgm_inputs
EOF
}

DEFAULT_FOLDER_URL="https://drive.google.com/drive/folders/1Qx3jqL_eSxGfWvhxb20M6orxvKsQlQT9?usp=drive_link"
FOLDER_URL="$DEFAULT_FOLDER_URL"
FOLDER_ID=""
DEST_DIR="/root/mm_dbgdgm_inputs"
LOG_FILE=""
AUTO_INSTALL=1
VENV_DIR="/root/.venvs/gdown"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder-url)
      FOLDER_URL="$2"
      shift 2
      ;;
    --folder-id)
      FOLDER_ID="$2"
      shift 2
      ;;
    --dest)
      DEST_DIR="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --no-install)
      AUTO_INSTALL=0
      shift
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$DEST_DIR"
if [[ ! -w "$DEST_DIR" ]]; then
  echo "ERROR: Destination is not writable: $DEST_DIR" >&2
  exit 1
fi

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="${DEST_DIR}/drive_download_$(date +"%Y%m%d_%H%M%S").log"
fi

log() {
  local msg="[$(date -Iseconds)] $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

resolve_python_bin() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  return 1
}

PYTHON_BIN="$(resolve_python_bin || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python/python3 not found in PATH" >&2
  exit 1
fi

if [[ -n "$FOLDER_ID" ]]; then
  FOLDER_URL="https://drive.google.com/drive/folders/${FOLDER_ID}"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  if [[ "$AUTO_INSTALL" -eq 1 ]]; then
    log "Creating virtual environment: $VENV_DIR"
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR" 2>>"$LOG_FILE"; then
      log "python -m venv failed. Installing python3-venv via apt."
      apt-get update -y | tee -a "$LOG_FILE"
      apt-get install -y python3-venv | tee -a "$LOG_FILE"
      "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
  else
    log "ERROR: venv not found and --no-install was set: $VENV_DIR"
    exit 1
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -m gdown --help >/dev/null 2>&1; then
  if [[ "$AUTO_INSTALL" -eq 1 ]]; then
    log "gdown not found in venv; installing"
    python -m pip install --upgrade pip gdown | tee -a "$LOG_FILE"
  else
    log "ERROR: gdown is not installed in venv and --no-install was set"
    deactivate || true
    exit 1
  fi
fi

pushd "$DEST_DIR" >/dev/null
log "Starting Google Drive folder download"
log "Folder URL: $FOLDER_URL"
log "Destination: $DEST_DIR"

set +e
python -m gdown --folder --continue "$FOLDER_URL" 2>&1 | tee -a "$LOG_FILE"
CMD_EXIT=${PIPESTATUS[0]}
set -e

popd >/dev/null
deactivate || true

if [[ "$CMD_EXIT" -ne 0 ]]; then
  log "ERROR: Download failed with exit code $CMD_EXIT"
  exit "$CMD_EXIT"
fi

log "Download completed successfully"
log "Files are available under: $DEST_DIR"
