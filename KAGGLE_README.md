# MM-DBGDGM Kaggle Training Guide

Use `kaggle_mmdbgdgm.py` as the Kaggle entry point. The raw-dataset-to-training path has two distinct stages:

1. preprocess raw imaging data into per-subject files
2. package those per-subject files into the final `fmri.npy`, `smri.npy`, and `labels.npy` arrays required by `kaggle_mmdbgdgm.py`

This guide covers both stages.

## What the trainer actually needs

Before you can train on Kaggle, you need these three files:

1. `fmri.npy`
2. `smri.npy`
3. `labels.npy`

Accepted shapes:

1. `fmri.npy`: `[N, 200, 50]` for pre-windowed DBGDGM samples
2. `fmri.npy`: `[N, 200, T]` only if you plan to pass `--subject-timeseries`
3. `smri.npy`: `[N, 5]`
4. `labels.npy`: `[N]`

If you pass pre-windowed fMRI data, the three arrays must already have the same sample count. In practice that means repeating each subject's sMRI feature vector and label once for every fMRI window.

## Important constraints

1. The current Kaggle trainer expects exactly `5` sMRI features per sample.
2. The OASIS Kaggle preprocessing notebooks in this repo generate sMRI features only. They do not produce the multimodal `fmri.npy` required by the current Kaggle trainer.
3. The ADNI Kaggle preprocessing notebooks generate both fMRI windows and sMRI features, but they still do not automatically create the final aggregate `fmri.npy`, `smri.npy`, and `labels.npy` files.
4. The preprocessing notebooks also do not build your label file for you. You need a CSV containing `subject_id`, `label`, and optionally `timepoint`.

## Recommended end-to-end path

If you do not already have the training arrays, use this order:

1. Preprocess ADNI raw data with one of the notebooks in `Preprocessing_Kaggle/notebooks/`.
2. Create a label CSV for the processed subjects.
3. Run `prepare_kaggle_arrays.py` to build the final three `.npy` files.
4. Upload those `.npy` files to a Kaggle notebook or keep them in `/kaggle/working/`.
5. Run `kaggle_mmdbgdgm.py`.

## Step 1: Preprocess the raw dataset

### ADNI

Use one of these notebooks:

1. `Preprocessing_Kaggle/notebooks/adni_preprocessing_gpu.ipynb`
2. `Preprocessing_Kaggle/notebooks/adni_preprocessing_kaggle.ipynb`

The GPU notebook is the better default on Kaggle.

What it produces per processed subject and timepoint:

1. `fmri/<subject_id>/<timepoint>/fmri_windows_dbgdgm.npy` in the GPU notebook
2. `fmri/<subject_id>/<timepoint>/windows.npy` in the older non-GPU notebook
3. `smri/<subject_id>/<timepoint>/features.npy`

Those files are intermediate preprocessing outputs. They are not yet the final arrays that the Kaggle trainer consumes.

### OASIS

Use one of these notebooks only if you specifically want OASIS sMRI preprocessing:

1. `Preprocessing_Kaggle/notebooks/oasis_preprocessing_gpu.ipynb`
2. `Preprocessing_Kaggle/notebooks/oasis_preprocessing_kaggle.ipynb`

Current limitation:

1. these OASIS notebooks produce `smri/<subject_id>/features.npy`
2. they do not produce multimodal fMRI inputs for `kaggle_mmdbgdgm.py`
3. they currently output `6` sMRI features, while the Kaggle trainer currently expects `5`

So OASIS preprocessing alone is not enough to run the current multimodal Kaggle trainer in this repo.

## Step 2: Create the labels CSV

You need a CSV with at least these columns:

1. `subject_id`
2. `label`

Optional column:

1. `timepoint`

Use `timepoint` for ADNI if the same subject has multiple processed timepoints.

Example:

```csv
subject_id,timepoint,label
ADNI_001,BL,0
ADNI_002,M06,1
ADNI_003,M12,2
ADNI_004,BL,3
```

Label mapping used by the MM-DBGDGM package:

1. `0 = CN`
2. `1 = eMCI`
3. `2 = lMCI`
4. `3 = AD`

If your preprocessing output folders do not use a meaningful timepoint, leave the `timepoint` column blank or omit it.

## Step 3: Build the final `.npy` training arrays

Use `prepare_kaggle_arrays.py` from the repo root.

Example for ADNI outputs:

```bash
python prepare_kaggle_arrays.py \
  --dataset-root /kaggle/working/preprocessed_adni_gpu \
  --labels-csv /kaggle/input/your-labels/labels.csv \
  --output-dir /kaggle/working/kaggle_arrays \
  --smri-feature-count 5
```

What the script does:

1. reads your label CSV
2. finds each subject's preprocessed fMRI and sMRI files
3. concatenates all fMRI windows into one `fmri.npy`
4. repeats the matching sMRI vector once per fMRI window to build `smri.npy`
5. repeats the matching label once per fMRI window to build `labels.npy`
6. writes `build_summary.json` and `samples_manifest.csv`

The script supports these fMRI filenames automatically:

1. `fmri_windows_dbgdgm.npy`
2. `timeseries_windows.npy`
3. `windows.npy`

Output files written by `prepare_kaggle_arrays.py`:

1. `fmri.npy`
2. `smri.npy`
3. `labels.npy`
4. `build_summary.json`
5. `samples_manifest.csv`

## Step 4: Run training on Kaggle

### 1. Enable GPU

Enable a GPU accelerator in the Kaggle notebook settings.

### 2. Install dependencies

If the repo is attached as a Kaggle dataset:

```bash
!pip install -r /kaggle/input/dbgdgm-improvements/requirements.txt
```

### 3. Run the trainer

```bash
!python /kaggle/input/dbgdgm-improvements/kaggle_mmdbgdgm.py \
  --fmri /kaggle/working/kaggle_arrays/fmri.npy \
  --smri /kaggle/working/kaggle_arrays/smri.npy \
  --labels /kaggle/working/kaggle_arrays/labels.npy \
  --output-dir /kaggle/working/kaggle_outputs
```

## What `kaggle_mmdbgdgm.py` does

1. validates the input arrays
2. splits them into train, validation, and test sets
3. trains the MM-DBGDGM model
4. evaluates the test split
5. saves training outputs and test metrics

## Output files to expect after training

Under `/kaggle/working/kaggle_outputs` you should expect files like:

1. `kaggle_config.json`
2. `test_metrics.json`
3. `best_loss.pt`
4. `best_acc.pt`
5. `training_history.json`
6. periodic `epoch_*.pt` checkpoints

## Troubleshooting

### I only have raw ADNI data

Run an ADNI preprocessing notebook first, then create the label CSV, then run `prepare_kaggle_arrays.py`.

### I only have raw OASIS data

That is not enough for the current Kaggle multimodal trainer in this repo. The OASIS notebooks currently create only sMRI features, and they output `6` features instead of the `5` expected by `kaggle_mmdbgdgm.py`.

### I already have preprocessed ADNI folders but not `fmri.npy`, `smri.npy`, or `labels.npy`

Use `prepare_kaggle_arrays.py`. That is exactly what it is for.

### Some subjects were skipped

Check `build_summary.json` and `samples_manifest.csv`. Common causes are:

1. missing `features.npy`
2. missing fMRI windows file
3. mismatched `timepoint`
4. unexpected sMRI feature count

## Practical notes

1. Save generated arrays and training outputs under `/kaggle/working/` so they are downloadable.
2. Keep fMRI aligned with DBGDGM expectations: `200` ROIs and window length `50`.
3. The Kaggle preprocessing notebooks produce intermediate per-subject artifacts. They do not finish the final packaging step automatically.