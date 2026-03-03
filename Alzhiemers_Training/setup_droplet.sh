#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Setting up DigitalOcean GPU Droplet..."

apt-get update -q
apt-get install -y -q python3-pip python3-venv libgl1 libglib2.0-0 git rsync tmux

python3 -m venv "$REPO_DIR/.venv"
source "$REPO_DIR/.venv/bin/activate"
pip install --upgrade pip --quiet

pip install torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --quiet

pip install nilearn networkx "numpy>=1.26" scipy scikit-learn opencv-python-headless --quiet

TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-scatter torch-sparse torch-geometric \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html" \
    --quiet

VOLUME="/mnt/trainingresults"
if [[ ! -d "$VOLUME" ]]; then
    echo "WARNING: Volume not mounted at $VOLUME. Check DigitalOcean volume attachment."
else
    mkdir -p "$VOLUME/models" "$VOLUME/cache/oasis" "$VOLUME/logs"
    ln -sfn "$VOLUME/cache/oasis" "$SCRIPT_DIR/data/oasis"
    echo "Volume directories ready at $VOLUME"
fi

python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}  ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)')
    print(f'  CUDA: {torch.version.cuda}  ✓')
else:
    print('  WARNING: No GPU found.')
"

echo "Setup complete. Run: source .venv/bin/activate && ./Alzhiemers_Training/train_do.sh"
