# DBGDGM_Improvements

## Kaggle Training

For the end-to-end Kaggle path from raw preprocessing outputs to final `fmri.npy`, `smri.npy`, and `labels.npy`, use `KAGGLE_README.md`.

## JPG sMRI Dataset Prep

If the root `Data/` folder already contains the four Alzheimer class folders with JPG slices, use the packager script to rebuild them into subject-level folders and a label manifest:

```bash
python prepare_smri_jpg_dataset.py --input-root Data --output-root prepared_smri_dataset --create-zip --overwrite
```

The script writes `labels.csv` and `dataset_summary.json` into the output root. The prepared folder, or the zip created from it, can then be used with the existing Modal training flow or with the local trainer in `train_local.py`.

## Local Prepare And Upload

To build the prepared JPG dataset, generate the Modal-compatible sample cache, and upload both into the persistent `mm-dbgdgm-data` volume in one step:

```bash
python prepare_and_upload_modal_inputs.py --raw-data-root Data --prepared-smri-root prepared_smri_dataset --artifacts-root modal_training_artifacts --overwrite
```

This writes the cache file to `modal_training_artifacts/cache/` locally, then uploads `prepared_smri_dataset.zip` to `adni_drive_pipeline/prepared_smri_dataset.zip` and the cache to `adni_drive_pipeline/cache` inside the Modal volume.
The prepared dataset is uploaded as a single zip file, which keeps the volume small and avoids storing tens of thousands of JPGs directly in the volume.

## Local DICOM Bundle Prep

If a Google Drive download arrived as a mix of deep folders and nested ZIP files, bundle it into one uploadable ZIP first:

```powershell
python prepare_and_upload_dicom_bundle.py `
  --input-root "C:\Users\Devab\Downloads\DGBDGM Fmri-005" `
  --input-root "C:\Users\Devab\Downloads\DGBDGM Fmri1-001" `
  --input-root "C:\Users\Devab\Downloads\ADNI dataset -20260414T111300Z-3-008" `
  --input-root "C:\Users\Devab\Downloads\ADNI dataset -20260414T111300Z-3-004" `
  --input-root "C:\Users\Devab\Downloads\ADNI dataset -20260414T111300Z-3-003" `
  --input-root "C:\Users\Devab\Downloads\ADNI dataset -20260414T111300Z-3-002" `
  --input-root "C:\Users\Devab\Downloads\ADNI dataset -20260414T111300Z-3-007" `
  --input-root "C:\Users\Devab\Downloads\DGBDGM Fmri (1)-006" `
  --output-root modal_training_artifacts/dicom_bundle --overwrite
```

The script preserves the source hierarchy, recurses into nested ZIP archives, and writes one bundle at `modal_training_artifacts/dicom_bundle/prepared_dicom_bundle.zip` plus a `bundle_summary.json` manifest.
If you leave upload enabled, it will push that ZIP into the Modal volume cache under `adni_drive_pipeline/raw_zips/`, which is the folder the existing real-data training flow already consumes.

For a single pass with visible progress logs, add `--progress-every 1000` and omit `--skip-upload`.

After that upload completes, training will prefer the uploaded bundle automatically when `MM_DBGDGM/configs/default_config.yaml` contains `uploaded_bundle_name: "prepared_dicom_bundle.zip"`.

Use this to train from the uploaded bundle without downloading from Drive again:

```powershell
modal run modal_train.py --config MM_DBGDGM/configs/default_config.yaml --use-mock-data false --seed-bundle-path "C:\Users\Devab\Downloads\dicom_bundle\prepared_dicom_bundle.zip" --uploaded-bundle-name prepared_dicom_bundle.zip
```

## DigitalOcean AMD GPU Training

Use this on a DigitalOcean AMD GPU droplet with your local data.

### 1. Install ROCm PyTorch

Install the PyTorch ROCm wheels that match the droplet image, then install the remaining packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements-digitalocean.txt
```

If your droplet image uses a different ROCm release, replace the index URL with the matching one from the PyTorch install docs.

### 2. Prepare the local SMRI dataset

If your SMRI data is in the four class folders already, rebuild it into the subject-level layout used by the local trainer:

```bash
python prepare_smri_jpg_dataset.py --input-root Data --output-root prepared_smri_dataset --create-zip --overwrite
```

The generated `labels.csv` now includes both `prepared_folder` and `smri_path`, so the local trainer can resolve the prepared JPG folder directly.

### 3. Train locally

If you have a preprocessed multimodal root for fMRI and sMRI, point the trainer at it and let the loader use the prepared SMRI manifest:

```bash
python train_local.py \
  --config MM_DBGDGM/configs/default.yaml \
  --dataset-root /path/to/preprocessed_data \
  --metadata-file prepared_smri_dataset/labels.csv \
  --smri-source-root prepared_smri_dataset \
  --output-dir local_results/run_01
```

If you already have separate train/val/test CSVs, pass `--train-metadata`, `--val-metadata`, and `--test-metadata` instead of `--metadata-file`.

### 4. Train from raw VM uploads

If you want to move the DICOM bundle and SMRI zip to the VM first, use the parallel SCP helper from Windows:

```powershell
.\upload_vm_inputs.ps1 -Host 134.199.200.241 -DicomBundlePath .\modal_training_artifacts\dicom_bundle\prepared_dicom_bundle.zip -SmriZipPath "C:\Users\Devab\Downloads\SMRI.zip"
```

That copies both archives at the same time and places them under `/root/mm_dbgdgm_inputs` on the VM.

Then run the local trainer in raw-zip mode on the VM:

```bash
python train_local.py \
  --config MM_DBGDGM/configs/default.yaml \
  --dicom-bundle-zip /root/mm_dbgdgm_inputs/prepared_dicom_bundle.zip \
  --smri-zip /root/mm_dbgdgm_inputs/SMRI.zip \
  --output-dir local_results/run_01
```

The loader will extract the two archives, rebuild the SMRI subject-level dataset if needed, and decode raw DICOM series directly with the 20-worker data loader settings from `MM_DBGDGM/configs/default.yaml`.

### 5. What the loader expects

The local trainer looks for these fields in the manifest when they are available:

- `subject_id`
- `timepoint` for exact fMRI pairing
- `label`
- `fmri_path` for explicit fMRI files
- `smri_path` or `prepared_folder` for the prepared local SMRI folder

If `timepoint` or `fmri_path` is missing, the loader will fall back to the first matching fMRI sample for that subject under `dataset_root/fmri`.
If your manifest uses different path column names, set `fmri_path_column` and `smri_path_column` in `MM_DBGDGM/configs/default.yaml`.

### 6. Resume later

You can resume with:

```bash
python train_local.py --resume-from local_results/run_01/best_loss.pt
```

The script also writes `run_summary.json`, `history.json`, and `final.pt` into the chosen output directory.