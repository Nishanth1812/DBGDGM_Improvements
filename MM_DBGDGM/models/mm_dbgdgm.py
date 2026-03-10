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
except ImportError:
    from models.dbgdgm_encoder import DBGDGMfMRIEncoder
    from models.smri_encoder import StructuralGraphEncoder, SimplesMRIEncoder
    from models.fusion_module import BidirectionalCrossModalFusion, SimpleFusion
    from models.vae import CompleteVAE


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
        print(f"  → VAE: {num_classes} classes")
        self.vae = CompleteVAE(
            latent_dim=latent_dim,
            num_classes=num_classes,
            n_roi=n_roi,
            n_time=seq_len,
            n_smri_features=n_smri_features,
            hidden_dims_encoder=[512, 256],
            hidden_dims_classifier=[512, 256],
            hidden_dims_decoder=[512, 1024],
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
        """
        # 1. Encode modalities
        z_fmri, fmri_attn = self.fmri_encoder(fmri)
        z_smri, smri_attn = self.smri_encoder(smri)
        
        # 2. Fuse modalities
        z_fused, fusion_attn = self.fusion(z_fmri, z_smri)
        
        # 3. VAE encoder + classifier + decoder
        vae_outputs = self.vae(z_fused, return_all=return_all)
        
        # Prepare output
        outputs = {
            'logits': vae_outputs['logits'],
            'predictions': torch.argmax(vae_outputs['logits'], dim=1),
            'z': vae_outputs['z']
        }
        
        if return_all:
            outputs.update({
                'z_fmri': z_fmri,
                'z_smri': z_smri,
                'z_fused': z_fused,
                'mu': vae_outputs['mu'],
                'logvar': vae_outputs['logvar'],
                'fmri_recon': vae_outputs['fmri_recon'],
                'smri_recon': vae_outputs['smri_recon'],
                'fmri_attn': fmri_attn,
                'smri_attn': smri_attn,
                'fusion_attn': fusion_attn
            })
        
        return outputs
    
    def predict(
        self,
        fmri: torch.Tensor,
        smri: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference method: returns class predictions and probabilities.
        
        Args:
            fmri: [batch_size, n_roi, seq_len]
            smri: [batch_size, n_smri_features]
        
        Returns:
            predictions: [batch_size] - class IDs
            probabilities: [batch_size, num_classes] - softmax probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(fmri, smri, return_all=False)
            predictions = outputs['predictions']
            probabilities = torch.softmax(outputs['logits'], dim=1)
        
        return predictions, probabilities
    
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
