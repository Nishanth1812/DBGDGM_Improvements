# MM-DBGDGM Improvements

This repository contains a cleaned-up training and preprocessing workflow for the MM-DBGDGM project. The codebase is organized around two main entrypoints:

- `pipeline.py` for the full end-to-end workflow
- `train_local.py` for standalone local training

## What the pipeline does

1. Downloads the raw dataset from Google Drive
2. Extracts and organizes the archive contents
3. Matches fMRI and sMRI subjects
4. Builds `labels.csv` and lightweight sMRI proxy features
5. Launches training and archives the best checkpoint

## Installation

Create a virtual environment and install the project dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer editable installs, the project metadata is defined in `pyproject.toml`.

## Running

Run the full pipeline:

```bash
python pipeline.py
```

Or use the shared entrypoint:

```bash
python main.py
```

Run local training directly:

```bash
python train_local.py --help
```

## DigitalOcean GPU VM launcher

Use `scripts/train_do_gpu.sh` on the VM to run long trainings with structured logs, run directories, and resume support.

```bash
chmod +x scripts/train_do_gpu.sh
./scripts/train_do_gpu.sh --help
```

Example launch (data paths are placeholders for now):

```bash
./scripts/train_do_gpu.sh -- \
  --dataset-root /mnt/data/fmri \
  --metadata-file /mnt/data/labels.csv \
  --smri-source-root /mnt/data/smri
```

Example resume from checkpoint:

```bash
./scripts/train_do_gpu.sh \
  --resume-from /mnt/scratch/mm_dbgdgm_runs/do_mm_dbgdgm_YYYYMMDD_HHMMSS/best_model.pt -- \
  --dataset-root /mnt/data/fmri \
  --metadata-file /mnt/data/labels.csv \
  --smri-source-root /mnt/data/smri
```

The launcher writes:

- `launcher_console.log` (combined stdout/stderr)
- `launcher_command.txt` (exact command)
- `launcher_env.txt` (captured environment)

under the selected `--output-dir` (or auto-created run directory).

## Upload local dataset ZIPs to VM

Use `upload_vm_datasets.ps1` from your Windows machine to upload both local ZIP files to the VM.

```powershell
.\upload_vm_datasets.ps1 -HostName 129.212.183.140
```

Optional explicit override:

```powershell
.\upload_vm_datasets.ps1 `
  -HostName 129.212.183.140 `
  -FmriZipPath "C:\Users\Devab\Downloads\FMRI_DOWNLOAD_dataset.zip" `
  -SmriZipPath "C:\Users\Devab\Downloads\SMRI DOWNLOAD_dataset.zip" `
  -RemoteBaseDir "/root/mm_dbgdgm_inputs"
```

## Download Google Drive data directly on VM

Use `scripts/download_drive_to_inputs.sh` on the VM to download your Drive folder into the same data directory (`/root/mm_dbgdgm_inputs`).

```bash
chmod +x scripts/download_drive_to_inputs.sh
./scripts/download_drive_to_inputs.sh
```

Explicit URL and destination:

```bash
./scripts/download_drive_to_inputs.sh \
  --folder-url "https://drive.google.com/drive/folders/1Qx3jqL_eSxGfWvhxb20M6orxvKsQlQT9?usp=drive_link" \
  --dest /root/mm_dbgdgm_inputs
```

## Modular VM pipeline (extract -> preprocess -> train -> evaluate)

Based on your VM layout:
- Drive downloads under `/root/mm_dbgdgm_inputs/ADNI dataset`
- Uploaded ZIPs under `/root/mm_dbgdgm_inputs`

Use these scripts in order.

1) Extract all ZIPs into one location:

```bash
chmod +x scripts/extract_inputs.sh
./scripts/extract_inputs.sh \
  --input-dir "/root/mm_dbgdgm_inputs" \
  --input-dir "/root/mm_dbgdgm_inputs/ADNI dataset" \
  --output-dir "/root/mm_dbgdgm_inputs/extracted"
```

2) Preprocess and create balanced train/val/test splits, with one random inference sample held out:

```bash
python3 scripts/preprocess_balanced_splits.py \
  --extracted-root "/root/mm_dbgdgm_inputs/extracted" \
  --output-root "/root/mm_dbgdgm_inputs/preprocessed" \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --inference-count 1 \
  --seed 42
```

3) Train from preprocessed metadata:

```bash
chmod +x scripts/train_preprocessed.sh
./scripts/train_preprocessed.sh \
  --preprocessed-dir "/root/mm_dbgdgm_inputs/preprocessed" \
  --dataset-root "/root/mm_dbgdgm_inputs" \
  --device cuda
```

4) Evaluate best checkpoint on the prepared test split:

```bash
chmod +x scripts/evaluate_best.sh
./scripts/evaluate_best.sh \
  --preprocessed-dir "/root/mm_dbgdgm_inputs/preprocessed" \
  --dataset-root "/root/mm_dbgdgm_inputs" \
  --device cuda
```

## Repository Layout

- `pipeline.py`: end-to-end orchestration
- `train_local.py`: local training entrypoint and evaluation flow
- `scripts/train_do_gpu.sh`: DigitalOcean VM launcher for training runs
- `MM_DBGDGM/`: package with models, data loaders, losses, and trainer code
- `full_pipeline_demo.ipynb`: notebook walkthrough of the workflow

## Notes

- The repository keeps generated outputs such as checkpoints and results outside the source package.
- The workspace is focused on the VM/local training flow rather than legacy cloud-specific scripts.
