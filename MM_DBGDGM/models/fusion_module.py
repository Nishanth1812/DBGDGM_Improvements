"""
Cross-Modal Fusion Module - Bidirectional Cross-Attention
Fuses fMRI and sMRI latent representations through mutual attention mechanisms.

Architecture:
- Bidirectional cross-attention between modalities
- Learned modality-specific transformations
- Fusion gate for weighted combination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


class ModuleSpecificTransform(nn.Module):
    """
    Modality-specific transformation before fusion.
    Learns to project modality-specific features to a shared space.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.transform = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, latent_dim]
        
        Returns:
            transformed: [batch_size, latent_dim]
        """
        out = self.transform(x)
        out = self.norm(out + x)  # Residual connection
        return out


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion mechanism.
    One modality queries, the other provides keys and values.
    """
    
    def __init__(self, latent_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        assert latent_dim % num_heads == 0, f"{latent_dim} must be divisible by {num_heads}"
        
        self.head_dim = latent_dim // num_heads
        
        # Linear projections
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, latent_dim] - queries from one modality
            key: [batch_size, latent_dim] - context from other modality
        
        Returns:
            attended: [batch_size, latent_dim] - attended features
            attention_weights: [batch_size, num_heads, 1, 1] - attention visualization
        """
        batch_size = query.shape[0]
        
        # Project
        Q = self.query(query)  # [batch, latent_dim]
        K = self.key(key)      # [batch, latent_dim]
        V = self.value(key)    # [batch, latent_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention_weights = attention.clone()
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, self.latent_dim)
        out = self.out_proj(out)
        
        # Residual connection and normalization
        out = self.norm(out + query)
        
        return out, attention_weights


class BidirectionalCrossModalFusion(nn.Module):
    """
    Complete cross-modal fusion using bidirectional cross-attention.
    
    Process:
    1. fMRI → sMRI attention (what does fMRI learn from structure?)
    2. sMRI → fMRI attention (what does structure learn from dynamics?)
    3. Fusion gate to combine all information
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        num_iterations: int = 2
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_iterations = num_iterations
        
        # Modality-specific transforms
        self.fmri_transform = ModuleSpecificTransform(latent_dim, hidden_dim)
        self.smri_transform = ModuleSpecificTransform(latent_dim, hidden_dim)
        
        # Cross-attention layers
        self.fmri_to_smri_attn = CrossAttentionFusion(latent_dim, num_heads, dropout)
        self.smri_to_fmri_attn = CrossAttentionFusion(latent_dim, num_heads, dropout)
        
        # Fusion gates
        self.fmri_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        self.smri_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        
        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(
        self,
        z_fmri: torch.Tensor,
        z_smri: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            z_fmri: [batch_size, latent_dim] - fMRI latent representation
            z_smri: [batch_size, latent_dim] - sMRI latent representation
        
        Returns:
            z_fused: [batch_size, latent_dim] - fused representation
            attention_dict: attention weights for visualization
        """
        attention_dict = {}
        
        # Initial transformation
        z_fmri_t = self.fmri_transform(z_fmri)
        z_smri_t = self.smri_transform(z_smri)
        
        # Bidirectional cross-attention (iterative refinement)
        for iteration in range(self.num_iterations):
            # fMRI attends to sMRI
            z_fmri_attn, attn_fmri_to_smri = self.fmri_to_smri_attn(z_fmri_t, z_smri_t)
            attention_dict[f'iter_{iteration}_fmri_to_smri'] = attn_fmri_to_smri
            
            # sMRI attends to fMRI
            z_smri_attn, attn_smri_to_fmri = self.smri_to_fmri_attn(z_smri_t, z_fmri_t)
            attention_dict[f'iter_{iteration}_smri_to_fmri'] = attn_smri_to_fmri
            
            # Gated combination
            fmri_gate_input = torch.cat([z_fmri_t, z_fmri_attn], dim=1)
            fmri_gate = self.fmri_gate(fmri_gate_input)
            z_fmri_t = fmri_gate * z_fmri_attn + (1 - fmri_gate) * z_fmri_t
            
            smri_gate_input = torch.cat([z_smri_t, z_smri_attn], dim=1)
            smri_gate = self.smri_gate(smri_gate_input)
            z_smri_t = smri_gate * z_smri_attn + (1 - smri_gate) * z_smri_t
            
            attention_dict[f'iter_{iteration}_fmri_gate'] = fmri_gate
            attention_dict[f'iter_{iteration}_smri_gate'] = smri_gate
        
        # Final fusion
        z_fused = torch.cat([z_fmri_t, z_smri_t], dim=1)
        z_fused = self.fusion_proj(z_fused)
        
        attention_dict['z_fused_shape'] = z_fused.shape
        
        return z_fused, attention_dict


class SimpleFusion(nn.Module):
    """
    Simple fusion without attention (baseline).
    Just concatenates and projects.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(
        self,
        z_fmri: torch.Tensor,
        z_smri: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            z_fmri: [batch_size, latent_dim]
            z_smri: [batch_size, latent_dim]
        
        Returns:
            z_fused: [batch_size, latent_dim]
            attention_dict: empty (no attention)
        """
        z_concat = torch.cat([z_fmri, z_smri], dim=1)
        z_fused = self.fusion(z_concat)
        
        return z_fused, {'z_fused_shape': z_fused.shape}
