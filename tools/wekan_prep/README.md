# WeKan Data Prep (MM-DBGDGM)

This tool pairs fMRI and sMRI subject folders, copies them into a clean dataset
layout, and writes `labels.csv` plus lightweight proxy features so the current
MM-DBGDGM pipeline can run without extra preprocessing.

## Output layout

```
<output_root>/
  fmri/<subject_id>/...
  smri/<subject_id>/...
  labels.csv
```

The script writes proxy files by default:
- `fmri/<subject_id>/fmri.npy` -> shape `(90, 200)`
- `smri/<subject_id>/features.npy` -> shape `(90, 4)`

These are **proxy features** derived from DICOM intensity statistics. Replace
with real ROI-level preprocessing if you have it.

## Example (Windows)

```powershell
python tools\wekan_prep\prepare_wekan_data.py `
  --input-root "C:\WeKan Training Data" `
  --fmri-root "C:\WeKan Training Data\fMRI" `
  --smri-root "C:\WeKan Training Data\sMRI"
```

## Optional flags

- `--label-map` : JSON string or JSON file path for label mapping.
- `--skip-fmri-npy` : Do not generate `fmri.npy`.
- `--skip-smri-features` : Do not generate `features.npy`.
- `--dry-run` : Scan and print pairs, no files written.
- `--accept-any-files` : Treat any files as slices if DICOM headers are missing.

## Next step

Use the generated `labels.csv` with `train_local.py`, for example:

```powershell
python train_local.py `
  --dataset-root "C:\WeKan Training Data\mm_dbgdgm_prepared" `
  --metadata-file "C:\WeKan Training Data\mm_dbgdgm_prepared\labels.csv"
```

