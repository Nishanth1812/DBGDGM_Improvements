# Comprehensive Guide: DBGDGM Preprocessing Pipeline

## Overview

Two complete preprocessing pipelines for OASIS and ADNI datasets designed for the Multimodal Deep Brain Generative Dynamic Graph Model (DBGDGM):

1. **Preprocessing/** - Full-featured Python package for local/research environments
2. **Preprocessing_Kaggle/** - Kaggle-optimized Jupyter notebooks for cloud execution

## Comparison

| Feature | Preprocessing/ | Preprocessing_Kaggle/ |
|---------|---|---|
| **Execution** | Command-line / Python script | Kaggle notebooks |
| **Setup** | Virtual environment + pip install | Auto (Kaggle) |
| **DICOM handling** | Full `pydicom` support | Basic DICOM loading |
| **Preprocessing** | Advanced options | Simplified but effective |
| **Scalability** | Local/HPC | Kaggle (GPU/TPU available) |
| **Documentation** | Inline + config file | Cell-by-cell comments |
| **Data leakage prevention** | Built-in grouping | Manual review needed |
| **Quality control** | Comprehensive QC checks | Basic validation |

## When to Use Each Version

### Use Preprocessing/ (Local)

✓ Processing complete dataset locally  
✓ Need advanced preprocessing options  
✓ Want to integrate with other tools  
✓ Require batch processing scripts  
✓ Have GPU/HPC cluster access  

### Use Preprocessing_Kaggle/ (Cloud)

✓ Testing data quickly  
✓ Limited local compute  
✓ Prefer cloud storage (Kaggle datasets)  
✓ Interactive exploration  
✓ Reproducible notebooks for publication  

## Architecture

### fMRI Preprocessing Pipeline

```
DICOM/NIfTI 4D Image [H, W, D, T]
        ↓
   Despike (optional)
        ↓
   Brain Mask
        ↓
   Temporal Filtering (0.01-0.1 Hz)
        ↓
   Parcellation (Schaefer-200 or simplified)
        ↓
   ROI Timeseries [N_ROI, T]
        ↓
   Motion Scrubbing (optional)
        ↓
   Standardization
        ↓
   Sliding Windows [N_ROI*N_windows, window_size]
```

**For DBGDGM**:
- Input to encoder: `[N_ROI*N_windows, 50]`
- Output: Latent representations for fusion module

### sMRI Preprocessing Pipeline

```
DICOM/NIfTI 3D T1 Image [H, W, D]
        ↓
   Intensity Normalization (0-1)
        ↓
   Skull Stripping (Binary Mask)
        ↓
   Tissue Segmentation
        ├── Gray Matter (GM)
        ├── White Matter (WM)
        └── Cerebrospinal Fluid (CSF)
        ↓
   Feature Extraction
        ├── Brain Volume
        ├── GM/WM/CSF Volumes
        ├── GM/WM Ratio
        ├── Regional Volumes
        └── Cortical Thickness (optional)
        ↓
   Feature Vector [1, N_features]
```

**For DBGDGM**:
- Input to fusion module: `[1, N_features]` (concatenated with fMRI Z)
- ~200-250 features total

## Dataset-Specific Details

### OASIS-1 (Open Access Series of Imaging Studies)

#### Dataset Characteristics
- **Subjects**: 416 (221 normal, 195 with dementia)
- **Modality**: Structural MRI (T1-weighted)
- **Format**: 2D axial slices or 3D NIfTI
- **Scans per subject**: 1-2 sessions
- **Brain coverage**: Full brain

#### Processing Strategy
1. **2D vs 3D**: Can process as 2D slice series or reconstruct 3D
   - Default: Reconstruct 3D → process as volume
   - Alternative: Process individual 2D slices

2. **Data Leakage Prevention**
   - Only use **first session** per subject
   - OASIS has multiple sessions per person
   - Grouping by true subject ID avoids test/train leakage

3. **Expected Features** (~5-250 per subject)
   - Basic: Brain/tissue volumes, ratios
   - Extended: Regional volumes (if atlas registration done)

#### OASIS Example
```
OAS1_0001/
├── OAS1_0001_MR1/  ← Use this
│   ├── anat/
│   │   └── OAS1_0001_MR1_mpr.img
│   │   └── OAS1_0001_MR1_mpr.hdr
│   └── ...
├── OAS1_0001_MR2/  ← Skip (avoid leakage)
│   └── ...
```

### ADNI (Alzheimer's Disease Neuroimaging Initiative)

#### Dataset Characteristics
- **Subjects**: 1,737
- **Modalities**: T1 (structural), rsfMRI (functional), DWI, PET
- **Format**: DICOM
- **Timepoints**: Up to 11 visits per subject
- **Key feature**: Longitudinal with cognitive follow-ups

#### Processing Strategy

1. **Series Selection**
   - T1-weighted: MPRAGE, 3D SPGR, or T1w sequences
   - rsfMRI: REST series (eyes closed resting state)
   - Auto-detect from DICOM SeriesDescription

2. **Data Leakage Prevention**
   - Baseline (BL) only, OR
   - One random timepoint per subject, OR
   - Stratified sampling across time

3. **DICOM Handling**
   - Extract SeriesDescription to identify modality
   - Load pixel arrays, sort by SliceLocation
   - Apply anonymization if needed

4. **Expected Outputs**
   - fMRI: N_windows × 50 TR windows (~100+ subjects can generate thousands of windows)
   - sMRI: Volumetric + regional features

#### ADNI Example
```
ADNI_001/
├── 20070101_BASELINE/
│   ├── T1w_MRI/
│   │   └── ADNI_001_S_0005_MR_MPRAGE_br_raw_*.dcm
│   ├── rsfMRI/
│   │   └── ADNI_001_S_0005_MR_EPI_rsfMRI_br_*.dcm
│   └── ...
├── 20080101_M12/  ← Second timepoint
│   └── ...
```

## Advanced Configuration

### Preprocessing/config/preprocessing_config.yaml

```yaml
fmri:
  # Adjust these for different protocols
  high_pass_filter: 0.01        # Hz (↑ removes more drift)
  low_pass_filter: 0.1          # Hz (↑ removes more noise)
  smoothing_fwhm: 5.0           # mm (↑ adds blur)
  sliding_window_size: 50       # TRs (↑ longer context)
  
smri:
  gm_threshold: 0.5             # Tissue probability cutoff
  wm_threshold: 0.5
  csf_threshold: 0.5
  compute_cortical_thickness: false  # Requires FreeSurfer
```

### Kaggle Notebooks: Inline Customization

```python
# In notebook cells, modify directly:
high_pass = 0.01
low_pass = 0.1
window_size = 50
```

## Quality Control (QC)

### fMRI QC Checks
- ✓ Minimum timepoints (300+)
- ✓ No NaN/Inf values
- ✓ Valid motion parameters
- ✓ Brain mask > 10000 voxels

### sMRI QC Checks
- ✓ Brain volume 1000-1500 cm³
- ✓ Valid tissue maps (sum ≈ 1.0)
- ✓ No negative values
- ✓ No NaN/Inf

### Metadata Output
Each subject generates `metadata.json`:
```json
{
  "subject_id": "ADNI_001",
  "preprocessing": {
    "despike": "applied",
    "temporal_filtering": "0.01-0.1 Hz",
    "n_rois": 200,
    "n_timepoints_original": 500,
    "n_timepoints_final": 450,
    "qc_pass": true
  }
}
```

## Integration with DBGDGM

### Data Loading

```python
# fMRI preprocessing output
fmri_windows = np.load('path/to/fmri_windows.npy')  # [N_windows, 50]

# sMRI preprocessing output
smri_features = np.load('path/to/smri_features.npy')  # [1, N_features]

# DBGDGM expects:
# fMRI encoder input: [batch, 200, 50]
# sMRI encoder input: [batch, N_features]
# After fusion: [batch, latent_dim]
```

### Example DBGDGM Usage

```python
from Alzhiemers_Training.src.model import DBGDGM

model = DBGDGM(
    n_roi=200,
    n_fmri_features=50,
    n_smri_features=250,
    latent_dim=32
)

# fMRI data
fmri_batch = torch.randn(batch_size, 200, 50)

# sMRI data
smri_batch = torch.randn(batch_size, 250)

# Forward pass
x_recon, z, mu, logvar = model(fmri_batch, smri_batch)
```

## Performance Benchmarks

### Processing Time

| Dataset | N_subjects | CPU Time | GPU Time | Output Size |
|---|---|---|---|---|
| OASIS-1 | 416 | ~30 mins | ~15 mins | ~100 MB |
| ADNI (BL) | 1100 | ~2-3 hrs | ~45 mins | ~500 MB |

### Memory Usage

- Single subject: ~100-500 MB
- Batch (100 subjects): ~5-10 GB
- Full dataset cached: ~50-100 GB

## Troubleshooting

### Common Issues

1. **DICOM not found**
   - Check directory structure
   - Use `!find /path -name "*.dcm" | head` to verify

2. **Memory error**
   - Reduce batch size
   - Process one subject at a time
   - Use memory mapping for large files

3. **Registration/Atlas misalignment**
   - Check affine matrices consistent
   - Ensure input/output spaces match (MNI152)

4. **Data quality issues**
   - Review sample preprocessed outputs
   - Check QC report for failing subjects
   - Visualize with nibabel/nilearn

## Next Steps

1. **Run Preprocessing**
   - Use local version for full dataset
   - Use Kaggle version for testing

2. **Train DBGDGM**
   - Prepare data loaders in PyTorch format
   - Follow Alzhiemers_Training training scripts

3. **Model Analysis**
   - Check results_analysis.md for evaluation metrics
   - Analyze learned latent representations

## References

- **OASIS**: Marcus et al. (2007) DOI: 10.1162/jocn.2007.19.9.1498
- **ADNI**: Mueller et al. (2005) DOI: 10.1016/j.jalz.2005.03.003
- **DBGDGM**: See Alzhiemers_Training/README
- **Schaefer Atlas**: Schaefer et al. (2018) DOI: 10.1093/cercor/bhx179

---

**Questions?** Check individual README files in Preprocessing/ and Preprocessing_Kaggle/
