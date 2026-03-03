#!/usr/bin/env bash
# =============================================================================
# sync_results.sh — Run LOCALLY to pull results from DigitalOcean Droplet
# =============================================================================
# Usage:
#   chmod +x sync_results.sh
#   ./Alzhiemers_Training/sync_results.sh <DROPLET_IP> [--all]
#
#   # Sync only checkpoints (default)
#   ./Alzhiemers_Training/sync_results.sh 123.456.78.90
#
#   # Sync checkpoints + logs
#   ./Alzhiemers_Training/sync_results.sh 123.456.78.90 --all
# =============================================================================
set -euo pipefail

DROPLET_IP="${1:-}"
SYNC_ALL=false

if [[ -z "$DROPLET_IP" ]]; then
    echo "Usage: ./sync_results.sh <DROPLET_IP> [--all]"
    echo ""
    echo "  <DROPLET_IP>  The public IP of your DigitalOcean GPU droplet"
    echo "  --all         Also sync training logs (logs/)"
    exit 1
fi

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) SYNC_ALL=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_RESULTS="$REPO_DIR/local_results"
REMOTE_USER="root"
REMOTE_REPO="/root/DBGDGM_Improvements"

mkdir -p "$LOCAL_RESULTS"

echo ""
echo "============================================================"
echo "  Syncing results from DigitalOcean Droplet"
echo "  Droplet IP  : $DROPLET_IP"
echo "  Local target: $LOCAL_RESULTS"
echo "============================================================"
echo ""

# ── Sync model checkpoints ────────────────────────────────────────────────────
echo "[1] Syncing model checkpoints (models_*/)..."
rsync -avz --progress \
    --include="models_*/" \
    --include="models_*/*.pt" \
    --include="models_*/*.npy" \
    --exclude="*" \
    "${REMOTE_USER}@${DROPLET_IP}:${REMOTE_REPO}/Alzhiemers_Training/" \
    "$LOCAL_RESULTS/checkpoints/"
echo "  ✓ Checkpoints synced to $LOCAL_RESULTS/checkpoints/"

# ── Sync logs (optional) ──────────────────────────────────────────────────────
if [[ "$SYNC_ALL" == "true" ]]; then
    echo ""
    echo "[2] Syncing training logs (logs/)..."
    rsync -avz --progress \
        "${REMOTE_USER}@${DROPLET_IP}:${REMOTE_REPO}/logs/" \
        "$LOCAL_RESULTS/logs/"
    echo "  ✓ Logs synced to $LOCAL_RESULTS/logs/"
fi

echo ""
echo "============================================================"
echo "  ✓ Sync complete!"
echo "  Local results: $LOCAL_RESULTS"
echo "============================================================"
echo ""
