# Kaggle-Compatible Preprocessing Pipeline

Standalone Jupyter notebooks for OASIS and ADNI preprocessing that run directly on Kaggle.

## Quick Start

### 1. Upload to Kaggle

- Go to [Kaggle](https://www.kaggle.com/)
- Create a new notebook
- Copy and paste the notebook content
- Upload OASIS or ADNI dataset as input
- Run cells sequentially

### 2. OASIS Preprocessing on Kaggle

**Notebook**: `oasis_preprocessing_kaggle.ipynb`

#### Input Files
- OASIS-1 dataset with structure: `OAS1_XXXX/OAS1_XXXX_MR1/` containing `.nii` or `.nii.gz` files

#### Output
- `preprocessed_oasis/smri/` - Structural features (brain volume, tissue volumes, etc.)
- `preprocessed_oasis/metadata/` - Subject summary CSV

#### Features Extracted
- Brain volume (cm³)
- Gray matter volume (cm³)
- White matter volume (cm³)
- CSF volume (cm³)
- GM/WM ratio

#### Steps
1. Load NIfTI images
2. Normalize intensities (0-1)
3. Skull strip using intensity thresholding + morphology
4. Segment tissues (GM, WM, CSF) using intensity ranges
5. Extract volumetric features
6. Save as [1 × 5] feature vectors

**Runtime**: ~2-5 mins per subject on Kaggle GPU

### 3. ADNI Preprocessing on Kaggle

**Notebook**: `adni_preprocessing_kaggle.ipynb`

#### Input Files
- ADNI dataset with structure: `ADNI_XXX/TIMEPOINT_X/` containing subdirectories with DICOM or NIfTI files
- Optional: T1-weighted (sMRI)
- Optional: rsfMRI (fMRI)

#### Output
- `preprocessed_adni/smri/` - Structural features
- `preprocessed_adni/fmri/` - fMRI timeseries windows
- `preprocessed_adni/metadata/` - Processing summary

#### fMRI Processing
- Input: 4D DICOM series [H, W, D, T]
- Parcellation: ~100 ROIs (simplified from Schaefer-200 for memory)
- Temporal filtering: 0.01-0.1 Hz bandpass
- Output: Sliding windows [N_windows × 50 TRs]

#### sMRI Processing
- Same as OASIS: volumetric features

#### Steps
1. Discover subjects and timepoints
2. Extract DICOM or NIfTI files
3. For T1: skull strip → segmentation → feature extraction
4. For fMRI: temporal filtering → parcellation → sliding windows
5. Save outputs

**Runtime**: ~5-15 mins per subject (depending on data size)

## Installation Requirements

Both notebooks automatically install:
```
nibabel
nilearn
pydicom
scikit-image
scipy
numpy
pandas
```

## File Locations (Kaggle)

```
/kaggle/input/         ← Your uploaded datasets
/kaggle/working/       ← Outputs are saved here
```

Change these paths in the notebooks if using different locations.

## Advanced: Customize Preprocessing

### Modify fMRI Parameters

```python
# In the fMRI preprocessing cell, change:
high_pass = 0.01        # Hz
low_pass = 0.1          # Hz
window_size = 50        # TRs
n_rois = 100            # Number of ROIs
```

### Modify sMRI Parameters

```python
# In skull stripping:
threshold = np.percentile(t1_image, 60)  # Adjust percentile

# In tissue segmentation:
p30, p50, p85 = np.percentile(brain_voxels, [30, 50, 85])  # Adjust thresholds
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce `n_rois` in fMRI preprocessing or process fewer subjects per run

### Issue: No DICOM files found

**Solution**: Check dataset structure. DICOM files should be in subdirectories of modality folders

### Issue: Shape mismatch errors

**Solution**: Ensure all subjects have valid 3D or 4D data. Use `print(image.shape)` to debug

### Issue: NaN or Inf in output

**Solution**: Check input normalization. Ensure all images are properly loaded before preprocessing.

## Output Format for DBGDGM Model

### fMRI Input
```python
windows = np.load('preprocessed_adni/fmri/ADNI_XXX/TIMEPOINT_X/windows.npy')
# Shape: [N_windows, 50]
# Each row is a 50-TR sliding window from a single ROI
```

### sMRI Input
```python
features = np.load('preprocessed_adni/smri/ADNI_XXX/TIMEPOINT_X/features.npy')
# Shape: [1, 5]
# [brain_volume, gm_volume, wm_volume, csf_volume, gm_wm_ratio]
```

### Loading for DBGDGM

```python
# Combine fMRI and sMRI
fmri_data = np.load('path/to/fmri/windows.npy')  # [W, 50]
smri_data = np.load('path/to/smri/features.npy')  # [1, 5]

# fMRI shape: [N_ROI, T] after reshaping
# sMRI shape: [1, N_features] (keep as is)

# Feed to DBGDGM encoder
# Output will be latent representations for classification/generation
```

## Notes

- **Data Leakage Prevention**: Only first timepoint/session per subject is used
- **Reproducibility**: Set random seeds if needed
- **Memory**: For large datasets, consider processing in batches
- **Quality Control**: Check sample outputs before batch processing

## Known Limitations

- Simplified tissue segmentation (not using specialized FSL/SPM)
- Simplified parcellation (not full Schaefer-200 atlas)
- No cortical thickness (requires FreeSurfer)
- No advanced motion correction (MCFLIRT/SPM)

For production use, consider the full `Preprocessing/` pipeline.
