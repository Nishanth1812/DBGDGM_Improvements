# MM-DBGDGM ? Multimodal Deep Brain Generative Dynamic Graph Model

A dual-stream deep learning system for early preclinical Alzheimer's Disease detection using paired fMRI and sMRI neuroimaging data.

## Architecture

```
fMRI stream (DBGDGM)          sMRI stream (structural)
       ?                            ?
       ?                            ?
  T-GNN (shared GAT           GAT (static graph)
  + BiLSTM)                   attention pooling
       ?                            ?
       ????????? cross-attention ?????
                 (bidirectional)
                      ?
                      ?
               ?-VAE (latent dim=128)
               ?, log ??
                      ?
                      ?
              MLP classifier ? 4 classes
              (CN, MCI, lMCI, AD)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic ADNI data
python data/synthetic/generate_synthetic_adni.py --seed 42 --output_dir data/synthetic/subjects

# Run full pipeline
python run_experiment.py --data_dir data/synthetic/subjects \
                          --manifest data/synthetic/subjects/manifest.csv \
                          --epochs 100 --beta 2.0 --seed 42 \
                          --output_dir results/
```

## Project Structure

```
mm_dbgdgm/
??? config.yaml                    # All hyperparameters
??? data/
?   ??? synthetic/
?   ?   ??? generate_synthetic_adni.py
?   ?   ??? subjects/              # 25 generated .npz files
?   ??? loaders.py                 # Dataset + stratified K-fold
??? preprocessing/
?   ??? fmri_pipeline.py           # Sliding-window dynamic graphs
?   ??? smri_pipeline.py           # AAL90 static structural graph
??? models/
?   ??? fmri_encoder.py            # T-GNN: shared GAT + BiLSTM
?   ??? smri_encoder.py            # Static GAT + attention pool
?   ??? fusion.py                  # Bidirectional cross-attention
?   ??? vae.py                     # ?-VAE (?, log??, reparam)
?   ??? classifier.py              # 3-layer MLP head
?   ??? mm_dbgdgm.py               # Full model wiring
??? training/
?   ??? losses.py                  # CE + annealed KL
?   ??? trainer.py                # Training loop + early stopping
?   ??? evaluate.py               # AUC, F1, accuracy, calibration
??? inference/
?   ??? predict.py                 # Single-subject inference
??? visualisation/
?   ??? attention_maps.py          # Cross-attention bar charts
?   ??? latent_space.py            # t-SNE/UMAP
?   ??? uncertainty_plots.py      # ?? distributions, reliability
??? notebooks/
?   ??? full_pipeline_demo.ipynb  # End-to-end demo
??? run_experiment.py            # CLI entry point
```

## Model Details

- **fMRI encoder**: 2-layer GAT (8 heads, 32 dim) with shared weights across K temporal windows ? BiLSTM ? 512-dim embedding
- **sMRI encoder**: 2-layer GAT on AAL90 anatomical graph ? attention-weighted pooling ? 512-dim embedding
- **Cross-attention**: bidirectional QKV attention (8 heads) ? concat ? MLP ? 512-dim fused
- **?-VAE**: 512 ? 128 dims, outputs ? and log ??, reparameterised sample at training
- **Classifier**: 128 ? 256 ? 128 ? 4 with BatchNorm, ReLU, Dropout(0.4)
- **Uncertainty**: ?? = exp(log ??).mean(dim=-1) per subject

## Key Features

- KL annealing: ?_VAE linearly increases from 0?0.1 over first 20 epochs
- Stratified 5-fold CV with weighted sampling for class imbalance
- Early stopping on validation AUC (patience=15)
- Calibration: reliability diagram + uncertainty-based accuracy split
- Interpretability: cross-attention weights stored for visualisation

## Synthetic Data

25 subjects: 7 CN, 6 MCI, 6 lMCI, 6 AD

- **fMRI**: 90 AAL ROIs ? 200 timepoints; group differences in DMN coherence (CN=1.0, AD=0.45)
- **sMRI**: 90 regions ? 4 features; hippocampal/entorhinal atrophy (CN=none, AD=38%)

## Results (on synthetic data)

Expected output after running `run_experiment.py`:

- `metrics_summary.json`: accuracy, AUC, F1 per fold + mean ? std
- `figures/attention_maps.png`: top-15 regions by cross-attention weight
- `figures/latent_space_tsne.png`: t-SNE coloured by diagnosis
- `figures/uncertainty_boxplot.png`: ?? per class
- `predictions.csv`: per-subject predictions + uncertainty