# MM-DBGDGM Kaggle Usage

The Kaggle entry point is [kaggle_mmdbgdgm.py](h:/Personal/Internships/WeKan/DBGDGM_Improvements/kaggle_mmdbgdgm.py). It now wraps the canonical package implementation instead of maintaining a second model and training loop.

## Accepted fMRI inputs

Use one of these two formats:

1. Pre-windowed DBGDGM arrays: `[N, 200, 50]`
2. Subject-level ROI time series: `[N, 200, T]` with `T >= 50`

When subject-level time series are provided, the script converts them into DBGDGM sliding windows with window size `50` and configurable step size, then repeats the matching sMRI feature vector and label for each generated window.

## Required companion arrays

- `smri.npy`: `[N, 5]`
- `labels.npy`: `[N]`

If you pass pre-windowed fMRI arrays, all three arrays must already have matching sample counts.

## Run

```bash
python kaggle_mmdbgdgm.py --fmri fmri.npy --smri smri.npy --labels labels.npy --output-dir kaggle_outputs
```

For subject-level fMRI time series:

```bash
python kaggle_mmdbgdgm.py --fmri fmri_subjects.npy --smri smri.npy --labels labels.npy --subject-timeseries --window-step 1
```

## What the script does

1. Validates the DBGDGM input contract.
2. Converts subject-level fMRI time series into `[N_windows, 200, 50]` windows when needed.
3. Splits data into train, validation, and test subsets.
4. Trains the canonical MM-DBGDGM model with the package trainer and loss.
5. Writes `kaggle_config.json` and `test_metrics.json` to the output directory.

## Notes

- The fMRI training path is now tied directly to the DBGDGM window format and package model.
- The previous dummy-data Kaggle demo and duplicate cheat-sheet documentation were removed.
- Use the main package for local training and inference code; the Kaggle script is only a thin dataset-loading wrapper.