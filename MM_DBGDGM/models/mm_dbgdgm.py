"""
Complete Multimodal DBGDGM Model
Combines all components: fMRI encoder, sMRI encoder, fusion, and VAE.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

try:
    from .dbgdgm_encoder import DBGDGMfMRIEncoder
    from .smri_encoder import StructuralGraphEncoder, SimplesMRIEncoder
    from .fusion_module import BidirectionalCrossModalFusion, SimpleFusion
    from .vae import CompleteVAE
    from .classifier import ClassificationHead
    from .degeneration_head import NeurodegenerationHead
    from .survival_head import WeibullSurvivalHead
except ImportError:
    from models.dbgdgm_encoder import DBGDGMfMRIEncoder
    from models.smri_encoder import StructuralGraphEncoder, SimplesMRIEncoder
    from models.fusion_module import BidirectionalCrossModalFusion, SimpleFusion
    from models.vae import CompleteVAE
    from models.classifier import ClassificationHead
    from models.degeneration_head import NeurodegenerationHead
    from models.survival_head import WeibullSurvivalHead


class MM_DBGDGM(nn.Module):
    """
    Complete Multimodal Dynamic Brain Graph Deep Generative Model.
    
    Pipeline:
    1. fMRI → DBGDGM encoder → z_fmri
    2. sMRI → Structural encoder → z_smri
    3. z_fmri + z_smri → Cross-modal fusion → z_fused
    4. z_fused → VAE → z, logits, reconstructions
    """
    
    def __init__(
        self,
        # fMRI params
        n_roi: int = 200,
        seq_len: int = 50,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        
        # sMRI params
        n_smri_features: int = 5,
        use_gat_encoder: bool = True,  # If False, use simple encoder
        
        # Common params
        latent_dim: int = 256,
        num_classes: int = 4,  # CN, eMCI, lMCI, AD
        
        # Fusion params
        use_attention_fusion: bool = True,  # If False, use simple concatenation
        num_fusion_heads: int = 4,
        num_fusion_iterations: int = 2,
        
        # Dropout
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_roi = n_roi
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        print("Building MM-DBGDGM model...")
        
        # 1. fMRI Encoder
        print(f"  → fMRI encoder: DBGDGM ({n_roi} ROIs, {seq_len} TRs)")
        self.fmri_encoder = DBGDGMfMRIEncoder(
            n_roi=n_roi,
            seq_len=seq_len,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            latent_dim=latent_dim,
            num_heads=num_fusion_heads
        )
        
        # 2. sMRI Encoder
        if use_gat_encoder:
            print(f"  → sMRI encoder: Graph Attention Network ({n_smri_features} features)")
            self.smri_encoder = StructuralGraphEncoder(
                n_features=n_smri_features,
                hidden_dim=128,
                latent_dim=latent_dim,
                num_gat_layers=2,
                num_heads=num_fusion_heads,
                dropout=dropout
            )
        else:
            print(f"  → sMRI encoder: Simple MLP ({n_smri_features} features)")
            self.smri_encoder = SimplesMRIEncoder(
                n_features=n_smri_features,
                hidden_dims=[128, 256, 512],
                latent_dim=latent_dim,
                dropout=dropout
            )
        
        # 3. Cross-Modal Fusion
        if use_attention_fusion:
            print(f"  → Fusion: Bidirectional Cross-Attention ({num_fusion_iterations} iterations)")
            self.fusion = BidirectionalCrossModalFusion(
                latent_dim=latent_dim,
                num_heads=num_fusion_heads,
                hidden_dim=512,
                dropout=dropout,
                num_iterations=num_fusion_iterations
            )
        else:
            print(f"  → Fusion: Simple concatenation")
            self.fusion = SimpleFusion(latent_dim=latent_dim)
        
        # 4. VAE Module
        print(f"  → VAE: Latent dim {latent_dim}")
        self.vae = CompleteVAE(
            latent_dim=latent_dim,
            n_roi=n_roi,
            n_time=seq_len,
            n_smri_features=n_smri_features,
            hidden_dims_encoder=[512, 256],
            hidden_dims_decoder=[512, 1024],
            dropout=dropout
        )
        
        # 5. Classification Head
        print(f"  → Classifier: {num_classes} classes")
        self.classifier = ClassificationHead(
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_dims=[512, 256],
            dropout=dropout
        )
        
        # 6. Neurodegeneration Forecasting Head
        print(f"  → Neurodegeneration Head: Localization & Regression")
        self.degeneration_head = NeurodegenerationHead(
            latent_dim=latent_dim,
            n_roi=n_roi,
            dropout=dropout
        )
        
        # 7. Survival Head (Weibull AFT)
        print(f"  → Survival Head: Weibull AFT model (3 transition events)")
        self.survival_head = WeibullSurvivalHead(
            latent_dim=latent_dim,
            num_events=3,
            hidden_dims=[256, 128],
            dropout=dropout
        )
        
        print("✓ MM-DBGDGM model built successfully!")
    
    def forward(
        self,
        fmri: torch.Tensor,
        smri: torch.Tensor,
        return_all: bool = False
    ) -> Dict:
        """
        Args:
            fmri: [batch_size, n_roi=200, seq_len=50]
            smri: [batch_size, n_smri_features]
            return_all: if True, return all intermediate representations
        
        Returns:
            outputs dict:
                - logits: [batch_size, num_classes]
                - predictions: [batch_size] - class predictions
                - z: [batch_size, latent_dim]
                - z_fmri: [batch_size, latent_dim] (if return_all)
                - z_smri: [batch_size, latent_dim] (if return_all)
                - z_fused: [batch_size, latent_dim] (if return_all)
                - mu: [batch_size, latent_dim] (if return_all)
                - logvar: [batch_size, latent_dim] (if return_all)
                - fmri_recon: [batch_size, n_roi, seq_len] (if return_all)
                - smri_recon: [batch_size, n_smri_features] (if return_all)
                - fmri_attn: dict (if return_all)
                - smri_attn: dict (if return_all)
                - fusion_attn: dict (if return_all)
                - degeneration: dict of regression and localization targets
                - survival: dict of 'shape' and 'scale' params
        """
        # 1. Encode modalities
        z_fmri, fmri_attn = self.fmri_encoder(fmri)
        z_smri, smri_attn = self.smri_encoder(smri)
        
        # 2. Fuse modalities
        z_fused, fusion_attn = self.fusion(z_fmri, z_smri)
        
        # 3. VAE encoder (fusion -> mu, logvar, z, recon if return_all)
        vae_outputs = self.vae(z_fused, return_all=return_all)
        
        # 4. Disease Classification
        classifier_input = vae_outputs['z'] if self.training else vae_outputs['mu']
        logits = self.classifier(classifier_input)
        
        # 5. Neurodegeneration Forecasting
        degeneration_outputs = self.degeneration_head(z_fused)
        
        # 6. Survival Analysis Prediction
        survival_outputs = self.survival_head(vae_outputs['mu'])
        
        # Prepare output
        outputs = {
            'logits': logits,
            'predictions': torch.argmax(logits, dim=1),
            'z': vae_outputs['z'],
            'mu': vae_outputs['mu'],
            'logvar': vae_outputs['logvar'],
            'degeneration': degeneration_outputs,
            'survival': survival_outputs
        }
        
        if return_all:
            outputs.update({
                'z_fmri': z_fmri,
                'z_smri': z_smri,
                'z_fused': z_fused,
                'fmri_recon': vae_outputs.get('fmri_recon'),
                'smri_recon': vae_outputs.get('smri_recon'),
                'fmri_attn': fmri_attn,
                'smri_attn': smri_attn,
                'fusion_attn': fusion_attn
            })
        
        return outputs
    
    def predict(
        self,
        fmri: torch.Tensor,
        smri: torch.Tensor,
        uncertainty_threshold: float = 0.5
    ) -> Dict:
        """
        Deep clinical reporting inference method.
        
        Args:
            fmri: [batch_size, n_roi, seq_len]
            smri: [batch_size, n_smri_features]
            uncertainty_threshold: threshold sum(logvar) to flag uncertainty
        
        Returns:
            dict containing comprehensive subject predictions.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(fmri, smri, return_all=False)
            
            predictions = outputs['predictions']
            probabilities = torch.softmax(outputs['logits'], dim=1)
            
            # Compute uncertainty based on VAE posterior variance
            variance = torch.exp(outputs['logvar'])
            mean_variance = torch.mean(variance, dim=1)
            uncertainty_flags = (mean_variance > uncertainty_threshold)
            
            # Extract degeneration predictions
            deg = outputs['degeneration']
            
            # Extract survival parameters
            surv = outputs['survival']
            shape, scale = surv['shape'], surv['scale']
            expected_time = self.survival_head.expected_time(shape, scale)
            
            clinical_reports = []
            batch_size = fmri.size(0)
            
            for i in range(batch_size):
                report = {
                    'current_stage_prediction': predictions[i].item(),
                    'stage_probabilities': probabilities[i].cpu().numpy(),
                    'atrophy_localization_scores': deg['atrophy_localization'][i].cpu().numpy(),
                    'hippocampal_volume': deg['hippocampal_volume'][i].item(),
                    'cortical_thinning_rate': deg['cortical_thinning_rate'][i].item(),
                    'dmn_connectivity': deg['dmn_connectivity'][i].item(),
                    'nss_score': deg['nss'][i].item(),
                    'survival_shape': shape[i].cpu().numpy(),
                    'survival_scale': scale[i].cpu().numpy(),
                    'expected_time_to_events': expected_time[i].cpu().numpy(),
                    'uncertainty_flag': uncertainty_flags[i].item(),
                    'mean_posterior_variance': mean_variance[i].item()
                }
                clinical_reports.append(report)
        
        if batch_size == 1:
            return clinical_reports[0]
        return clinical_reports
    
    def get_latent(
        self,
        fmri: torch.Tensor,
        smri: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract latent representations (for visualization, etc.)
        
        Args:
            fmri: [batch_size, n_roi, seq_len]
            smri: [batch_size, n_smri_features]
        
        Returns:
            latent dict:
                - z_fmri: [batch_size, latent_dim]
                - z_smri: [batch_size, latent_dim]
                - z_fused: [batch_size, latent_dim]
                - z: [batch_size, latent_dim]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(fmri, smri, return_all=True)
            return {
                'z_fmri': outputs['z_fmri'],
                'z_smri': outputs['z_smri'],
                'z_fused': outputs['z_fused'],
                'z': outputs['z']
            }
    
    def get_attention_weights(
        self,
        fmri: torch.Tensor,
        smri: torch.Tensor
    ) -> Dict:
        """
        Extract attention weights for interpretability.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(fmri, smri, return_all=True)
            return {
                'fmri_attn': outputs['fmri_attn'],
                'smri_attn': outputs['smri_attn'],
                'fusion_attn': outputs['fusion_attn']
            }
