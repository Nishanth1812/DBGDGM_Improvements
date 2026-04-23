# MM-DBGDGM Automated Pipeline

This repository contains an end-to-end automated pipeline for training Multimodal Dynamic Brain Graph Deep Generative Models (MM-DBGDGM), optimized for high-performance VMs (e.g., AMD MI300X).

## Features
- **Automated Data Ingestion**: Downloads matched fMRI/sMRI datasets directly from Google Drive.
- **Smart Preprocessing**: Parallel extraction and Subject-ID-based matching of modalities.
- **MI300X Optimized**: Pre-configured for ROCm, 20 vCPUs, and high VRAM utilization.
- **Progress Tracking**: Real-time terminal progress bars using `tqdm`.
- **Best Model Storage**: Automatically keeps only the highest-performing model and exports inference samples.

## Setup

### 1. Create Virtual Environment
We recommend using a dedicated virtual environment.

**Linux/Ubuntu (MI300X VM):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. ROCm PyTorch (For MI300X)
If your VM uses AMD GPUs, ensure you have the ROCm-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

## Running the Pipeline

To run the full end-to-end pipeline (Download -> Preprocess -> Train -> Evaluate):

```bash
python pipeline.py
```

### Advanced Options
- `--skip-download`: Use if data is already in `data/raw/`.
- `--skip-preprocess`: Use if data is already in `data/processed/`.

## Project Structure
- `pipeline.py`: Main orchestration script.
- `train_local.py`: Core training entrypoint with hardware optimizations.
- `MM_DBGDGM/`: Core library containing models, datasets, and the trainer.
- `best_model/`: Directory where the final model and inference samples are saved.
- `results/`: Detailed training logs and metrics.

## Decluttering
All legacy cloud-specific (Modal) and Kaggle-specific scripts have been removed to keep the workspace focused on the VM pipeline.