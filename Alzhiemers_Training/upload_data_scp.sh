#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./Alzhiemers_Training/upload_data_scp.sh <DROPLET_IP> [options]

Uploads the dataset to the DigitalOcean droplet using a single SCP transfer.
This is usually faster than `scp -r` for large directory trees because it avoids
per-file copy overhead.

Options:
  --identity PATH       SSH private key to use
  --source PATH         Local dataset directory
  --remote-repo PATH    Remote repo path
  --remote-user USER    Remote SSH user (default: root)
  --keep-archive        Keep the local tar archive after upload
  -h, --help            Show this help message
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

DROPLET_IP="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE="$SCRIPT_DIR/data"
SOURCE_PATH="$DEFAULT_SOURCE"
REMOTE_REPO="/root/DBGDGM_Improvements"
REMOTE_USER="root"
IDENTITY_PATH=""
KEEP_ARCHIVE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --identity)
            IDENTITY_PATH="${2:-}"
            shift 2
            ;;
        --source)
            SOURCE_PATH="${2:-}"
            shift 2
            ;;
        --remote-repo)
            REMOTE_REPO="${2:-}"
            shift 2
            ;;
        --remote-user)
            REMOTE_USER="${2:-}"
            shift 2
            ;;
        --keep-archive)
            KEEP_ARCHIVE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$DROPLET_IP" ]]; then
    echo "Droplet IP is required." >&2
    usage
    exit 1
fi

if [[ ! -d "$SOURCE_PATH" ]]; then
    echo "Source directory does not exist: $SOURCE_PATH" >&2
    exit 1
fi

if [[ -n "$IDENTITY_PATH" && ! -f "$IDENTITY_PATH" ]]; then
    echo "Identity file does not exist: $IDENTITY_PATH" >&2
    exit 1
fi

SOURCE_PATH="$(cd "$SOURCE_PATH" && pwd)"
SOURCE_PARENT="$(dirname "$SOURCE_PATH")"
SOURCE_NAME="$(basename "$SOURCE_PATH")"
ARCHIVE_PATH="${TMPDIR:-/tmp}/${SOURCE_NAME}_$(date +%Y%m%d_%H%M%S).tar"
REMOTE_BASE="${REMOTE_USER}@${DROPLET_IP}"
REMOTE_PARENT="${REMOTE_REPO}/Alzhiemers_Training"
REMOTE_ARCHIVE="/tmp/${SOURCE_NAME}.tar"

SSH_ARGS=(
    -o Compression=no
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=6
)

SCP_ARGS=(
    -o Compression=no
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=6
)

if [[ -n "$IDENTITY_PATH" ]]; then
    SSH_ARGS+=(-i "$IDENTITY_PATH")
    SCP_ARGS+=(-i "$IDENTITY_PATH")
fi

cleanup() {
    if [[ "$KEEP_ARCHIVE" != "true" && -f "$ARCHIVE_PATH" ]]; then
        rm -f "$ARCHIVE_PATH"
    fi
}

trap cleanup EXIT

echo "Creating archive: $ARCHIVE_PATH"
tar -cf "$ARCHIVE_PATH" -C "$SOURCE_PARENT" "$SOURCE_NAME"

echo "Uploading archive with SCP to ${REMOTE_BASE}:${REMOTE_ARCHIVE}"
scp "${SCP_ARGS[@]}" "$ARCHIVE_PATH" "${REMOTE_BASE}:${REMOTE_ARCHIVE}"

echo "Extracting archive on remote host into ${REMOTE_PARENT}"
ssh "${SSH_ARGS[@]}" "$REMOTE_BASE" \
    "mkdir -p '${REMOTE_PARENT}' && tar -xf '${REMOTE_ARCHIVE}' -C '${REMOTE_PARENT}' && rm -f '${REMOTE_ARCHIVE}'"

echo "Upload complete: ${REMOTE_PARENT}/${SOURCE_NAME}"

if [[ "$KEEP_ARCHIVE" == "true" ]]; then
    trap - EXIT
    echo "Local archive kept at: $ARCHIVE_PATH"
fi