# MM-DBGDGM Architecture
**Multimodal Deep Brain Generative Dynamic Graph Model**

---

## Pipeline Diagram

```
┌─────────────────────────┐       ┌─────────────────────────┐
│        fMRI Input       │       │        sMRI Input       │
│  [N_subj × N_ROI × T]  │       │ [N_subj × N_region × F] │
└────────────┬────────────┘       └────────────┬────────────┘
             │                                 │
┌────────────▼────────────┐       ┌────────────▼────────────┐
│         DBGDGM          │       │   Structural Graph      │
│  Dynamic Brain Graph    │       │     Encoder GAT ×2      │
│  Deep Generative Model  │       │   Trained on            │
│  Pretrained: HCP + UKB  │       │   ADNI + OASIS          │
│  Fine-tuned: ADNI       │       │                         │
└────────────┬────────────┘       └────────────┬────────────┘
             │ z_fmri                           │ z_smri
             └──────────────────┬──────────────┘
                                │
             ┌──────────────────▼──────────────┐
             │         Cross-Modal Fusion      │
             │    Bidirectional Cross-Attention │
             └──────────────────┬──────────────┘
                                │ z_fused
             ┌──────────────────▼──────────────┐
             │            VAE Encoder          │
             │     q(z|z_fused) → μ, logvar    │
             └──────────┬────────────┬─────────┘
                        │            │
             ┌──────────▼──┐   ┌─────▼──────────────┐
             │  Classifier │   │ Generative Decoder  │
             │   MLP Head  │   │  p(x|z)  MSE recon  │
             │ CN/eMCI/    │   └─────────────────────┘
             │ lMCI/AD     │
             └─────────────┘
```

---

## Components

| Block | Component | Key Layers |
|---|---|---|
| 1. fMRI Input | `[N_subj × N_ROI × T]` | Schaefer-200, sliding window 50 TRs |
| 1. sMRI Input | `[N_subj × N_region × F]` | FreeSurfer volumes + cortical thickness |
| 2. DBGDGM | Pretrained HCP+UKB → fine-tuned ADNI | Dynamic brain graph encoder with generative latent space → z_fmri |
| 2. Structural Graph Encoder | Trained on ADNI + OASIS | Anatomical graph → GAT ×2 → Pool MLP → z_smri |
| 3. Cross-Modal Fusion | Bidirectional cross-attention | fMRI↔sMRI MultiheadAttn (4 heads) → Fusion MLP → z_fused |
| 4. VAE Latent Space | β-VAE with KL annealing | fc_mu + fc_logvar → reparameterize → z |
| 5. Classifier | Uses μ at inference | Linear → ReLU → Dropout → Linear → CN/eMCI/lMCI/AD |
| 5. Generative Decoder | Auxiliary, uses z | Reconstruct fMRI + sMRI → MSE loss |

---

## Training Loss

$$L_{\text{total}} = L_{\text{class}} + \beta \cdot L_{\text{KL}} + \lambda \cdot L_{\text{align}} + \lambda \cdot L_{\text{recon}}$$

| Term | Description |
|---|---|
| `L_class` | CrossEntropy (CN / eMCI / lMCI / AD) |
| `β·L_KL` | KL divergence, β annealed 0→1 over 20 epochs |
| `λ·L_align` | Cosine similarity between z_fmri and z_smri |
| `λ·L_recon` | MSE reconstruction of fMRI + sMRI |
