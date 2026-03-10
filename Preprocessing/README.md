# DBGDGM Preprocessing Pipeline

Comprehensive preprocessing pipeline for OASIS and ADNI datasets designed for the Multimodal Deep Brain Generative Dynamic Graph Model (DBGDGM).

## Features

### fMRI Preprocessing
- Motion correction
- Skull stripping
- Registration to MNI152 space
- Temporal filtering (0.01-0.1 Hz bandpass)
- Parcellation with Schaefer-200 atlas
- Sliding window extraction (50 TRs, 1 TR step)
- Output: `[N_ROI × T]` timeseries ready for DBGDGM

### sMRI Preprocessing  
- Intensity normalization
- Skull stripping
- Tissue segmentation (GM, WM, CSF)
- Brain volume calculation
- Regional volume extraction
- Cortical thickness estimation
- Output: `[1 × N_features]` structural features

### Dataset Support
- **OASIS-1**: 2D T1-weighted images with 3D reconstruction
- **ADNI**: DICOM format with automatic series selection
- Subject grouping to prevent data leakage
- BIDS-compatible output organization

## Directory Structure

```
Preprocessing/
├── config/
│   └── preprocessing_config.yaml        # Configuration file
├── src/
│   ├── fmri/
│   │   └── fmri_preprocessing.py       # fMRI preprocessing module
│   ├── smri/
│   │   └── smri_preprocessing.py       # sMRI preprocessing module
│   ├── utils/
│   │   ├── dicom_utils.py              # DICOM loading utilities
│   │   └── preprocessing_utils.py      # Common preprocessing functions
│   ├── oasis_processor.py              # OASIS dataset processor
│   └── adni_processor.py               # ADNI dataset processor
├── main.py                              # Main orchestrator script
├── requirements.txt                     # Dependencies
└── README.md                            # This file
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Process both datasets
python main.py --oasis-dir /path/to/oasis --adni-dir /path/to/adni --output-dir ./output

# Process only OASIS
python main.py --dataset oasis --oasis-dir /path/to/oasis --output-dir ./output

# Process only ADNI
python main.py --dataset adni --adni-dir /path/to/adni --output-dir ./output

# With verbose output
python main.py --dataset both --oasis-dir /path/to/oasis --adni-dir /path/to/adni --verbose
```

### Python Script

```python
from src.oasis_processor import OASISDatasetProcessor
from src.adni_processor import ADNIDatasetProcessor

# Process OASIS
oasis_proc = OASISDatasetProcessor(
    data_dir='/path/to/oasis',
    output_dir='./preprocessed/oasis'
)
oasis_proc.discover_subjects()
oasis_summary = oasis_proc.process_all_subjects()

# Process ADNI
adni_proc = ADNIDatasetProcessor(
    data_dir='/path/to/adni',
    output_dir='./preprocessed/adni'
)
adni_proc.discover_subjects()
adni_summary = adni_proc.process_all_subjects()
```

## Output Structure

```
preprocessed_data/
├── OASIS/
│   ├── smri/
│   │   ├── OAS1_XXXX/
│   │   │   ├── MR1/
│   │   │   │   ├── features.npy          [1 × N_features]
│   │   │   │   └── metadata.json
│   ├── metadata/
│   │   └── subjects_summary.csv
│
├── ADNI/
│   ├── fmri/
│   │   ├── ADNI_XXX/
│   │   │   ├── TIMEPOINT_X/
│   │   │   │   ├── timeseries_windows.npy [N_windows × 50]
│   │   │   │   └── metadata.json
│   ├── smri/
│   │   ├── ADNI_XXX/
│   │   │   ├── TIMEPOINT_X/
│   │   │   │   ├── features.npy           [1 × N_features]
│   │   │   │   └── metadata.json
│   ├── metadata/
│   │   └── subjects_summary.csv
│
└── preprocessing_results.json
```

## Configuration

Edit `config/preprocessing_config.yaml` to customize preprocessing:

```yaml
fmri:
  smoothing_fwhm: 5.0              # Smoothing kernel (mm)
  high_pass_filter: 0.01           # High-pass cutoff (Hz)
  low_pass_filter: 0.1             # Low-pass cutoff (Hz)
  sliding_window_size: 50          # Window size in TRs
  parcellation_atlas: schaefer_200 # ROI atlas

smri:
  output_resolution: 1             # Output resolution (mm)
  compute_cortical_thickness: false # Requires FreeSurfer
  gm_threshold: 0.5                # Tissue classification threshold
```

## Important Notes

### Data Leakage Prevention
- **OASIS**: Only first session per subject used (OASIS has multiple scans per subject)
- **ADNI**: Only first timepoint per subject used by default
- This ensures no subject appears multiple times in the dataset

### fMRI Output Format
- Time series windows are `[N_ROI*N_windows, window_size]`
- Ready for direct input to DBGDGM encoder
- 200 ROIs from Schaefer atlas
- 50 TR windows with 1 TR step

### sMRI Output Format
- Features are `[1, N_features]` (single row vector)
- Includes brain volume, tissue volumes, regional features
- Can be concatenated with fMRI features in DBGDGM fusion module

## Quality Control

Each subject's preprocessing includes QC checks:
- Valid timepoint count (minimum 300 TRs for fMRI)
- Brain volume range (1000-1500 cm³)
- No NaN/Inf values
- Proper tensor shapes

Subjects failing QC are flagged in output metadata.

## Dependencies

- **nibabel**: NIfTI image I/O
- **nilearn**: neuroimaging data preprocessing
- **pydicom**: DICOM file handling  
- **scipy**: Image processing and filtering
- **scikit-learn**: Machine learning utilities
- **numpy, pandas**: Data manipulation
- **PyYAML**: Configuration parsing

## Advanced Usage

### Custom fMRI Preprocessing

```python
from src.fmri.fmri_preprocessing import fMRIPreprocessor, fMRIConfig

config = fMRIConfig(
    smoothing_fwhm=6.0,
    high_pass_filter=0.02,
    low_pass_filter=0.08,
    sliding_window_size=60
)

preprocessor = fMRIPreprocessor(config)
windows, metadata = preprocessor.preprocess(fmri_image, affine, motion_params)
```

### Custom sMRI Preprocessing

```python
from src.smri.smri_preprocessing import sMRIPreprocessor, sMRIConfig

config = sMRIConfig(
    compute_cortical_thickness=True,
    gm_threshold=0.6
)

preprocessor = sMRIPreprocessor(config)
features,metadata = preprocessor.preprocess(t1_image, affine, brain_mask)
```

## References

- **OASIS-1**: Marcus et al. (2007). "The Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented and Demented Older Adults"
- **ADNI**: Mueller et al. (2005). "The Alzheimer's Disease Neuroimaging Initiative"
- **Schaefer Atlas**: Schaefer et al. (2018). "Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI"

