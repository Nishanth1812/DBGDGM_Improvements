"""
sMRI Structural Encoder - GAT-based Anatomical Feature Encoding
Processes structural MRI features and regional anatomy to extract
meaningful structural representations.

Architecture:
- Feature normalization and enrichment
- Graph Attention Networks (GAT) for anatomical connectivity
- Multi-layer graph convolutions
- Output latent representation for fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class GraphAttentionLayer(nn.Module):
    """Single Graph Attention layer."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        assert out_features % num_heads == 0, f"{out_features} must be divisible by {num_heads}"
        
        # Linear transformation for Q, K, V
        self.linear = nn.Linear(in_features, out_features)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=out_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_nodes, in_features]
            adj: [batch_size, num_nodes, num_nodes] - adjacency matrix (optional)
        
        Returns:
            out: [batch_size, num_nodes, out_features]
        """
        # Linear transformation
        x_transformed = self.linear(x)  # [batch, num_nodes, out_features]
        
        # Multi-head attention
        attn_out, _ = self.attention(
            query=x_transformed,
            key=x_transformed,
            value=x_transformed,
            key_padding_mask=None
        )
        
        # Residual connection and normalization
        out = self.norm(x_transformed + self.dropout(attn_out))
        
        return out


class StructuralGraphEncoder(nn.Module):
    """
    GAT-based encoder for structural MRI features.
    
    Creates a virtual graph where each region/feature is a node,
    and learns connectivity through attention mechanisms.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(n_features)
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=hidden_dim if i > 0 else hidden_dim,
                out_features=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for i in range(num_gat_layers)
        ])
        
        self.num_gat_layers = num_gat_layers
        
        # Pooling and projection
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # FC layers for latent projection
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, smri_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            smri_features: [batch_size, n_features]
                          After preprocessing: typically [batch, 5] for ADNI or [batch, 6] for OASIS
        
        Returns:
            z_smri: [batch_size, latent_dim]
            features_dict: debug/logging information
        """
        batch_size = smri_features.shape[0]
        features = {}
        
        # Normalize input
        x = self.input_norm(smri_features)  # [batch, n_features]
        
        # Project to hidden dimension
        x = self.input_proj(x)  # [batch, hidden_dim]
        
        # Add spatial context: create virtual graph
        # For each sample, create a virtual graph structure
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Expand to create a graph-like structure
        # Each feature becomes a node that attends to all others
        x_nodes = x.expand(-1, self.n_features, -1)  # [batch, n_features, hidden_dim]
        features['x_nodes_shape'] = x_nodes.shape
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x_nodes = gat_layer(x_nodes, adj=None)
            features[f'gat_layer_{i}_shape'] = x_nodes.shape
        
        # Global pooling: aggregate across nodes
        # [batch, hidden_dim, n_features] → [batch, hidden_dim, 1] → [batch, hidden_dim]
        x_pooled = self.global_pool(x_nodes.transpose(1, 2)).squeeze(-1)
        features['x_pooled_shape'] = x_pooled.shape
        
        # Project to latent dimension
        z_smri = self.fc_layers(x_pooled)  # [batch, latent_dim]
        features['z_smri_shape'] = z_smri.shape
        
        return z_smri, features


class SimplesMRIEncoder(nn.Module):
    """
    Simpler alternative sMRI encoder for faster training.
    Direct feature processing without graph attention.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: list = [128, 256, 512],
        latent_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(n_features)
        
        # Build FC layers
        layers = []
        prev_dim = n_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final projection to latent
        layers.extend([
            nn.Linear(prev_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        ])
        
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, smri_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            smri_features: [batch_size, n_features]
        
        Returns:
            z_smri: [batch_size, latent_dim]
            features_dict: debug/logging information
        """
        features = {}
        
        # Normalize input
        x = self.input_norm(smri_features)
        
        # Pass through FC layers
        z_smri = self.fc_layers(x)
        
        features['z_smri_shape'] = z_smri.shape
        
        return z_smri, features
