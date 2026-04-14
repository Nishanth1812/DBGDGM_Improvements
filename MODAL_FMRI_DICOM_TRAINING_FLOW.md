# Modal Training Guide: ZIP DICOM Ingestion, fMRI-only Handling, and Full Model Flow

## 1. What This Setup Now Does

The Modal pipeline now supports direct training from a Google Drive folder that contains multiple ZIP files.

For each training run in real-data mode:
1. Download all ZIP files from the shared Drive folder URL.
2. Extract all ZIP files into the Modal data volume.
3. Discover DICOM series folders.
4. Reserve a deterministic 2% inference holdout (configurable) and save it separately.
5. Build model-ready samples from the remaining series.
6. Train/validate/test in the same run.

The processing is cached in Modal volume storage so repeated runs reuse preprocessed samples instead of redoing download and extraction.

## 2. DICOM Handling (Including Filename/Folder Clues)

### 2.1 DICOM Discovery

A file is treated as DICOM when:
- extension is `.dcm`, or
- extension is empty (common in ADNI exports like `I240811`).

### 2.2 Series Selection Using Clues

The loader can prioritize likely fMRI series by folder/file-name clues such as:
- `fmri`
- `bold`
- `rest`
- `resting_state`
- `rsfmri`
- `ep2d`
- `epi`

Config controls:
- `use_filename_clues: true` -> prefer clue-matched series if available.
- `strict_fmri_clues: false` -> fallback to all series if no clues are found.
- if `strict_fmri_clues: true` and no clue match is found, run fails fast.

### 2.3 Robust Slice/Frame Ordering

Inside each DICOM series, files are ordered by a robust key:
1. `TemporalPositionIdentifier`
2. `AcquisitionNumber`
3. `InstanceNumber`
4. numeric hint parsed from filename (for names like `I240811`)
5. filename lexicographic fallback

This reduces ordering errors when metadata is incomplete but filenames still carry sequence hints.

## 3. fMRI-only Dataset: How sMRI Is Handled

Your downloaded dataset contains only fMRI DICOMs. The model still expects both `fmri` and `smri` tensors, so the pipeline creates **proxy sMRI features** from DICOM intensity + header metadata + clue tokens.

### 3.1 Proxy sMRI Source Features

The pipeline derives a feature pool from:
- intensity statistics across ordered DICOM images
- DICOM metadata summaries (`TR`, `TE`, `SliceThickness`, `Rows`, `Columns`, `PixelSpacing`)
- clue flags from path/filename (`rest`, `bold`, `epi`, `fmri`, etc.)

That pool is resampled to `n_smri_features` (default `5`) so it fits the existing model interface.

### 3.2 Important Note

This is a practical compatibility strategy for fMRI-only training, not a true structural MRI substitute.

If true sMRI becomes available later, replace proxy feature generation with real structural features and keep the rest of the training stack unchanged.

## 4. Full Training Flow (What Trains First, Next, and How)

## 4.1 Model Initialization

The full MM-DBGDGM model is built once and contains:
1. fMRI encoder (DBGDGM-style)
2. sMRI encoder (GAT/MLP)
3. cross-modal fusion module
4. VAE module
5. classification head
6. neurodegeneration head
7. survival head

## 4.2 Is It Sequential Training?

No staged pretraining is used here.

All modules train **end-to-end jointly** from the first optimizer step. In each batch, gradients flow through the entire network and all trainable components are updated together.

## 4.3 Per-Batch Forward Pass

For each batch:
1. Encode fMRI -> `z_fmri`
2. Encode sMRI (proxy in fMRI-only mode) -> `z_smri`
3. Fuse modalities -> `z_fused` 
4. VAE -> `mu`, `logvar`, latent sample `z`, reconstructions
5. Classification head -> stage logits/prediction
6. Degeneration head -> regression/localization outputs
7. Survival head -> Weibull parameters

## 4.4 Losses Used Together

Total loss combines:
- classification loss
- KL divergence (with annealing)
- cross-modal alignment loss
- reconstruction losses (fMRI and sMRI)
- biomarker regression loss
- survival loss

All are optimized in the same backward pass.

## 5. Prediction Outputs and Final Run Outputs

## 5.1 Batch/Inference-Level Prediction Outputs

From model forward/inference, key outputs include:
- `logits`, `predictions`
- `stage_probabilities` (softmax)
- latent variables (`z`, `mu`, `logvar`)
- degeneration outputs:
  - `atrophy_localization`
  - `hippocampal_volume`
  - `cortical_thinning_rate`
  - `dmn_connectivity`
  - `nss`
- survival outputs:
  - Weibull `shape`, `scale`
  - expected time-to-event estimates

## 5.2 Run Artifacts Saved by Modal Training

The run saves:
- final model checkpoint (`*_final.pt`)
- training history JSON (`*_history.json`)
- best-checkpoint test report (`test_results.json`)
- inference holdout manifest (`/checkpoints/<output_prefix>/inference_holdout_manifest.json`)
- copied holdout series root (`/data/adni_drive_pipeline/inference_holdout/<cache_key>/...`)

`test_results.json` includes:
- overall accuracy
- macro and weighted F1
- confusion matrix
- per-class accuracy
- classification report
- predictions, probabilities, targets

## 5.3 Return Summary From `train.remote(...)`

The returned result dict includes:
- status and artifact paths
- best validation loss/accuracy
- test accuracy and F1 metrics
- per-class accuracy and confusion matrix

## 6. Config Knobs You Can Tune

In `MM_DBGDGM/configs/default_config.yaml` under `data`:
- `mock_data`
- `drive_folder_url`
- `labels_csv`
- `max_series`
- `max_dicoms_per_series`
- `inference_holdout_ratio` (default `0.02`)
- `use_filename_clues`
- `strict_fmri_clues`
- `seed`

## 7. Command Examples

Use default config (already points to your Drive folder):

```bash
modal run modal_train.py --config MM_DBGDGM/configs/default_config.yaml --use-mock-data false
```

Override the Drive URL explicitly:

```bash
modal run modal_train.py --config MM_DBGDGM/configs/default_config.yaml --use-mock-data false --drive-folder-url "https://drive.google.com/drive/folders/1Qx3jqL_eSxGfWvhxb20M6orxvKsQlQT9?usp=sharing"
```

## 8. Practical Recommendation

For medically meaningful classification targets, provide a `labels_csv` mapping subjects to diagnosis labels. Without it, the fallback label assignment is deterministic but synthetic and intended only to keep full pipeline execution possible.

## 9. Best Model Saving and Single-Sample Inference

### 9.1 Is the best model saved?

Yes. During training, checkpoints are saved under the checkpoint volume path:

- `/checkpoints/<output_prefix>/best_loss.pt`
- `/checkpoints/<output_prefix>/best_acc.pt`

At the end of training, the test pass is run on `best_loss.pt` when available.

### 9.2 Can you run the best model on one sample?

Yes. `modal_train.py` now supports `predict_only` mode.

You can do either:
1. give an explicit series folder path (`sample_series_path`), or
2. let the code auto-pick a series from the saved 2% holdout (`use_saved_holdout=true`, default), or
3. fallback to Drive dataset scan (`sample_index`) when no holdout is available.

Example (auto-pick sample index 0 from Drive data):

```bash
modal run modal_train.py \
  --config MM_DBGDGM/configs/default_config.yaml \
  --predict-only true \
  --checkpoint-name best_loss.pt \
  --use-saved-holdout true \
  --sample-index 0
```

Example (explicit DICOM series folder path in Modal volume):

```bash
modal run modal_train.py \
  --config MM_DBGDGM/configs/default_config.yaml \
  --predict-only true \
  --checkpoint-name best_loss.pt \
  --sample-series-path "/data/adni_drive_pipeline/extracted/<zip_name>/ADNI/<subject>/.../<series_folder>"
```

### 9.3 Output from single-sample prediction

Returned JSON includes:
- `predicted_class_id`
- `predicted_stage` (`CN`, `eMCI`, `lMCI`, `AD`)
- `stage_probabilities`
- `healthy_vs_ad_summary`
- `ad_probability`
- `cn_probability`

Interpretation:
- `CN` => Healthy/Control-like
- `AD` => Alzheimer-like
- `eMCI`/`lMCI` => intermediate MCI states (not strictly healthy, not full AD)

### 9.4 Run inference on total unseen data

Yes. Use `predict_dataset` mode to score all detected unseen series.

If you already trained with real data, you can run inference directly on the saved 2% holdout subset:

```bash
modal run modal_train.py \
  --config MM_DBGDGM/configs/default_config.yaml \
  --predict-dataset true \
  --checkpoint-name best_loss.pt \
  --use-saved-holdout true \
  --unseen-output-file holdout_predictions.json
```

Unseen data from a separate Drive folder:

```bash
modal run modal_train.py \
  --config MM_DBGDGM/configs/default_config.yaml \
  --predict-dataset true \
  --checkpoint-name best_loss.pt \
  --drive-folder-url "https://drive.google.com/drive/folders/<UNSEEN_FOLDER_ID>"
```

Unseen data from an already-extracted folder path:

```bash
modal run modal_train.py \
  --config MM_DBGDGM/configs/default_config.yaml \
  --predict-dataset true \
  --checkpoint-name best_loss.pt \
  --unseen-series-root "/data/unseen_inference_pipeline/extracted"
```

Optional controls:
- `--max-infer-series 0` means no limit (all series).
- `--unseen-output-file unseen_predictions.json` sets output JSON name.

### 9.5 Output from unseen-dataset inference

A JSON is saved under:
- `/checkpoints/<output_prefix>/<unseen_output_file>`

It includes:
- series-level predictions for all processed unseen samples
- probabilities for each class (`CN`, `eMCI`, `lMCI`, `AD`)
- stage counts summary
- healthy-vs-AD summary counts
- skipped count for invalid/unreadable series
- source metadata (`source_mode`, and holdout manifest path when used)
