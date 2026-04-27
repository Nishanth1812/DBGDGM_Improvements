#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Extract local and Drive-downloaded ZIP archives into one location on the VM.

Usage:
  ./scripts/extract_inputs.sh [options]

Options:
  --input-dir DIR       Input directory to scan for ZIP files (can be used multiple times)
  --output-dir DIR      Extraction output directory (default: /root/mm_dbgdgm_inputs/extracted)
  --clear-output        Remove output directory before extracting
  --help                Show this help

Defaults:
  Input directories:
    /root/mm_dbgdgm_inputs
    /root/mm_dbgdgm_inputs/ADNI dataset
EOF
}

INPUT_DIRS=()
OUTPUT_DIR="/root/mm_dbgdgm_inputs/extracted"
CLEAR_OUTPUT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIRS+=("$2")
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --clear-output)
      CLEAR_OUTPUT=1
      shift
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

if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
  INPUT_DIRS=(
    "/root/mm_dbgdgm_inputs"
    "/root/mm_dbgdgm_inputs/ADNI dataset"
  )
fi

if [[ "$CLEAR_OUTPUT" -eq 1 ]]; then
  rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

declare -a ZIP_FILES=()
for src_dir in "${INPUT_DIRS[@]}"; do
  if [[ ! -d "$src_dir" ]]; then
    echo "[WARN] Input directory not found, skipping: $src_dir"
    continue
  fi

  while IFS= read -r -d '' zip_file; do
    ZIP_FILES+=("$zip_file")
  done < <(find "$src_dir" -maxdepth 2 -type f -name '*.zip' -print0)
done

if [[ ${#ZIP_FILES[@]} -eq 0 ]]; then
  echo "No ZIP files found in input directories."
  exit 1
fi

echo "Found ${#ZIP_FILES[@]} ZIP file(s)."

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python/python3 not found in PATH" >&2
  exit 1
fi

for zip_path in "${ZIP_FILES[@]}"; do
  zip_name="$(basename "$zip_path")"
  base_name="${zip_name%.zip}"
  safe_name="${base_name// /_}"
  target_dir="$OUTPUT_DIR/$safe_name"
  done_marker="$target_dir/.extract_complete"

  if [[ -f "$done_marker" ]]; then
    echo "[SKIP] Already extracted: $zip_name"
    continue
  fi

  rm -rf "$target_dir"
  mkdir -p "$target_dir"

  echo "[EXTRACT] $zip_name -> $target_dir"
  "$PYTHON_BIN" - "$zip_path" "$target_dir" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(target_dir)
PY

  touch "$done_marker"
done

echo "Extraction completed: $OUTPUT_DIR"

