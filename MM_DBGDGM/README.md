# MM-DBGDGM Model & Training Pipeline

Complete implementation of Multimodal Dynamic Brain Graph Deep Generative Model for Alzheimer's Disease classification,combining fMRI and sMRI data.

## Project Structure

```
MM_DBGDGM/
├── models/                  # Model components
│   ├── dbgdgm_encoder.py   # fMRI processor with DBGDGM
│   ├── smri_encoder.py     # sMRI encoder (GAT or MLP)
│   ├── fusion_module.py    # Cross-modal fusion (attention-based)
│   ├── vae.py              # VAE encoder + classifier + decoder
│   ├── mm_dbgdgm.py        # Complete model
│   └── __init__.py
│
├── data/                    # Data loading
│   ├── dataset.py          # MultimodalBrainDataset class
│   └── __init__.py
│
├── training/               # Training pipeline
│   ├── losses.py           # Loss functions
│   ├── trainer.py          # Trainer class
│   └── __init__.py
│
├── configs/                # Configuration files
│   └── default.yaml        # Default configuration
│
├── train.py                # Main training script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
cd MM_DBGDGM
pip install -r requirements.txt
```

## Quick Start

### 1. Create Configuration

```bash
python train.py --create-config --config configs/custom.yaml
```

This creates a default configuration file. Edit it with your dataset paths:

```yaml
data:
  dataset_root: '/path/to/preprocessed_data'  # fmri/ and smri/ subdirs
  train_metadata: '/path/to/train_metadata.csv'
  val_metadata: '/path/to/val_metadata.csv'
  test_metadata: '/path/to/test_metadata.csv'
```

### 2. Prepare Metadata CSV Files

Required columns: `subject_id`, `timepoint`, `label`

```csv
subject_id,timepoint,label
ADNI_001,2024_01_15,0
ADNI_002,2024_02_20,1
OASIS_001,2024_03_10,2
...
```

Where labels map to: 0=CN (Control), 1=eMCI, 2=lMCI, 3=AD

### 3. Train Model

```bash
python train.py --config configs/custom.yaml
```

## Model Architecture

### Components

1. **fMRI Encoder (DBGDGM)**
   - Input: [batch, 200 ROIs, 50 TRs]
   - GRU for temporal processing per ROI
   - Graph Attention for ROI connectivity
   - Output: [batch, 256] latent code

2. **sMRI Encoder**
   - Input: [batch, N_features] (5 for ADNI, 6 for OASIS)
   - Graph Attention Network (2 layers) OR simple MLP
   - Output: [batch, 256] latent code

3. **Cross-Modal Fusion**
   - Bidirectional cross-attention (2 iterations)
   - Learned fusion gates
   - Output: [batch, 256] fused latent code

4. **VAE Module**
   - Encoder: z_fused → μ, logvar
   - Classifier: μ → [batch, 4] logits
   - Decoder: z → fMRI/sMRI reconstructions

### Loss Function

```
L_total = L_class + β·L_KL + λ·L_align + λ·L_recon

where:
  L_class = CrossEntropy (classification)
  β·L_KL = KL divergence (0→1 annealing over 20 epochs)
  λ·L_align = -mean(cosine_similarity(z_fmri, z_smri))
  λ·L_recon = MSE(fMRI_recon, fMRI_orig) + MSE(sMRI_recon, sMRI_orig)
```

Default weights:
- λ_kl = 0.1
- λ_align = 0.1
- λ_recon = 0.1 (fMRI: 2.0, sMRI: 1.0)

## Training Pipeline

### Data Loading

```python
from data import create_dataloaders

dataloaders = create_dataloaders(
    dataset_root='/path/to/preprocessed_data',
    train_metadata='train.csv',
    val_metadata='val.csv',
    batch_size=32,
    num_workers=4
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']
```

### Create Model

```python
from models import MM_DBGDGM

model = MM_DBGDGM(
    n_roi=200,
    seq_len=50,
    n_smri_features=5,
    latent_dim=256,
    num_classes=4,
    use_gat_encoder=True,     # Graph Attention for sMRI
    use_attention_fusion=True, # Bidirectional fusion
)
```

### Training

```python
from training import Trainer, MM_DBGDGM_Loss
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = MM_DBGDGM_Loss(num_classes=4)
trainer = Trainer(model, criterion, device)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    patience=10
)
```

### Inference

```python
# Classification
predictions, probabilities = model.predict(fmri, smri)
# predictions: [batch] class IDs
# probabilities: [batch, 4] softmax

# Extract latent representations
latents = model.get_latent(fmri, smri)
# z_fmri, z_smri, z_fused, z

# Get attention weights
attentions = model.get_attention_weights(fmri, smri)
# fmri_attn, smri_attn, fusion_attn
```

## Outputs

### Checkpoints

Saved in `experiments/{timestamp}/`:
- `best_loss.pt` - Best validation loss
- `best_acc.pt` - Best validation accuracy
- `epoch_N.pt` - Periodic checkpoints
- `config.yaml` - Training configuration

### Logs

- `training.log` - Detailed training logs
- `experiments/{timestamp}/training_history.json` - Loss/accuracy history

### Metrics Tracked

**Training:**
- train_classification, train_kl, train_alignment
- train_fmri_recon, train_smri_recon, train_accuracy

**Validation:**
- val_classification, val_kl, val_alignment
- val_fmri_recon, val_smri_recon, val_accuracy

## Configuration Parameters

### Model Parameters

```yaml
n_roi: 200              # Schaefer-200 atlas
seq_len: 50             # Temporal window (fixed for DBGDGM)
n_smri_features: 5      # ADNI=5, OASIS=6
latent_dim: 256         # Latent representation dimension
num_classes: 4          # CN, eMCI, lMCI, AD
use_gat_encoder: true   # Graph Attention for sMRI
use_attention_fusion: true  # Bidirectional attention fusion
dropout: 0.1            # Dropout rate
```

### Training Parameters

```yaml
batch_size: 32          # Batch size
num_workers: 4          # Data loading workers
num_epochs: 50          # Maximum epochs
learning_rate: 0.001    # Initial learning rate
weight_decay: 1e-5      # L2 regularization
patience: 10            # Early stopping patience
annealing_epochs: 20    # KL annealing duration
```

### Loss Weights

```yaml
lambda_kl: 0.1          # KL divergence weight
lambda_align: 0.1       # Cross-modal alignment weight
lambda_recon: 0.1       # Reconstruction weight
```

## Advanced Usage

### Using Different Encoders

```python
# sMRI encoder options
model = MM_DBGDGM(
    ...
    use_gat_encoder=True,      # Graph Attention (recommended)
    # use_gat_encoder=False,   # Simple MLP (faster)
)

# Fusion options
model = MM_DBGDGM(
    ...
    use_attention_fusion=True,  # Bidirectional attention
    # use_attention_fusion=False,  # Simple concatenation (faster)
)
```

### Loading Checkpoints

```python
from training import Trainer

trainer = Trainer(model, criterion, device)
trainer.load_checkpoint('experiments/.../best_loss.pt', optimizer)
```

### Visualization

```python
import matplotlib.pyplot as plt
import json

# Load history
with open('experiments/{timestamp}/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history['train_total'], label='Train')
plt.plot(history['val_total'], label='Val')
plt.ylabel('Total Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['train_accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Val')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history['train_kl'], label='Train KL')
plt.plot(history['val_kl'], label='Val KL')
plt.ylabel('KL Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
```

## Performance Benchmarks

### Hardware Requirements

- **Minimum:** 1 GPU with 8GB VRAM, CPU 8-cores
- **Recommended:** 1 GPU with 16GB+ VRAM, CPU 16-cores, 32GB RAM

### Training Time (per epoch)

| Hardware | Batch=32 | Batch=64 |
|----------|----------|----------|
| GPU T4 (15GB) | 5-8 min | 8-12 min |
| GPU V100 (32GB) | 3-5 min | 4-7 min |
| GPU A100 (80GB) | 2-3 min | 2-4 min |

### Model Size

- **Parameters:** ~15-20M depending on config
- **Checkpoint size:** ~60-80 MB

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
batch_size: 16  # or 8

# Reduce model size
latent_dim: 128  # instead of 256

# Reduce number of workers
num_workers: 0  # or 2
```

### Poor Convergence

- Increase `annealing_epochs` (20 → 30)
- Reduce initial learning rate (0.001 → 0.0005)
- Increase patience (10 → 15)
- Check data normalization

### Data Loading Issues

```python
# Verify dataset structure
dataset_root/
├── fmri/
│   └── subject_id/timepoint/fmri_windows_dbgdgm.npy
└── smri/
    └── subject_id/timepoint/features.npy
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{DBGDGM2023,
  title={Dynamic Brain Graph Deep Generative Model for AD Classification},
  author={Your Authors},
  journal={Your Journal},
  year={2023}
}
```

## License

Open source - modify as needed for research purposes.

## Contributing

Contributions welcome! Areas for improvement:
- Multi-GPU training support
- Attention visualization tools
- Additional encoder variants
- Performance optimization

## Support

For issues or questions:
1. Check troubleshooting section
2. Review documentation
3. Verify data format and metadata
4. Check configuration parameters
