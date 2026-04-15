"""
fMRI Encoder - Dynamic Brain Graph Deep Generative Model (DBGDGM)
Adapted from: Spasov et al. (github.com/simeon-spasov/dynamic-brain-graph-deep-generative-model)

Processes fMRI timeseries in DBGDGM format [N_samples, N_ROI=200, T=50]
and outputs latent representation suitable for generative modeling.

Architecture:
- Temporal GRU encoders for each ROI
- ROI connectivity graph learning
- Dynamic graph attention mechanisms
- Outputs latent code z_fmri for fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class TemporalGRUEncoder(nn.Module):
    """GRU encoder for individual ROI timeseries."""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.hidden_size_bi = hidden_size * 2
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len] - single ROI timeseries
        
        Returns:
            output: [batch_size, seq_len, hidden_size*2] - temporal features
            hidden: [batch_size, hidden_size*2] - final hidden state
        """
        output, hidden = self.gru(x.unsqueeze(-1))  # Add feature dim

        # GRU hidden shape: [num_layers * num_directions, batch, hidden_size].
        # For bidirectional GRU, concatenate forward/backward states from last layer.
        if self.gru.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        return output, hidden


class ROIGraphAttention(nn.Module):
    """Graph attention mechanism for ROI connectivity."""
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, f"{feature_dim} must be divisible by {num_heads}"
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_features: [batch_size, n_roi, feature_dim]
        
        Returns:
            attended_features: [batch_size, n_roi, feature_dim]
        """
        batch_size, n_roi, feature_dim = roi_features.shape
        
        Q = self.query(roi_features)  # [batch, n_roi, feature_dim]
        K = self.key(roi_features)
        V = self.value(roi_features)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, n_roi, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_roi, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_roi, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, n_roi, feature_dim)
        out = self.fc_out(out)
        
        return out


class DBGDGMfMRIEncoder(nn.Module):
    """
    Complete fMRI encoder for DBGDGM.
    
    Input: [batch_size, n_roi=200, seq_len=50]
    Output: latent representation for fusion module
    """
    
    def __init__(
        self,
        n_roi: int = 200,
        seq_len: int = 50,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        latent_dim: int = 256,
        num_heads: int = 4
    ):
        super().__init__()
        self.n_roi = n_roi
        self.seq_len = seq_len
        self.gru_hidden = gru_hidden
        self.latent_dim = latent_dim
        
        # Temporal encoder for each ROI
        self.temporal_encoder = TemporalGRUEncoder(
            input_size=1,
            hidden_size=gru_hidden,
            num_layers=gru_layers
        )
        self.gru_output_size = gru_hidden * 2  # bidirectional
        
        # ROI graph attention
        self.graph_attention = ROIGraphAttention(
            feature_dim=self.gru_output_size,
            num_heads=num_heads
        )
        
        # Fusion layers
        self.bn1 = nn.BatchNorm1d(self.gru_output_size)
        
        # MLPs for temporal and spatial features
        self.temporal_fc = nn.Sequential(
            nn.Linear(self.gru_output_size, self.gru_output_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.gru_output_size, self.latent_dim // 2)
        )
        
        self.spatial_fc = nn.Sequential(
            nn.Linear(self.gru_output_size * n_roi, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.latent_dim // 2)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
    
    def forward(self, fmri: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            fmri: [batch_size, n_roi, seq_len] - DBGDGM format
        
        Returns:
            z_fmri: [batch_size, latent_dim] - latent representation
            features_dict: debug/logging information
        """
        batch_size = fmri.shape[0]
        features = {}
        
        # Process each ROI through temporal encoder
        roi_temporal_features = []
        for roi_idx in range(self.n_roi):
            roi_timeseries = fmri[:, roi_idx, :]  # [batch, seq_len]
            _, hidden = self.temporal_encoder(roi_timeseries)  # [batch, gru_hidden*2]
            roi_temporal_features.append(hidden)
        
        # Stack ROI features: [batch, n_roi, gru_hidden*2]
        roi_features = torch.stack(roi_temporal_features, dim=1)
        features['roi_temporal_shape'] = roi_features.shape
        
        # Apply graph attention (ROI connectivity)
        roi_attended = self.graph_attention(roi_features)  # [batch, n_roi, gru_hidden*2]
        features['roi_attended_shape'] = roi_attended.shape
        
        # Normalize
        roi_attended_flat = roi_attended.view(batch_size, -1)
        roi_attended_pooled = roi_attended.mean(dim=1)  # [batch, gru_hidden*2]
        roi_attended_pooled = self.bn1(roi_attended_pooled)
        
        # Temporal feature extraction
        temporal_feat = self.temporal_fc(roi_attended_pooled)  # [batch, latent_dim//2]
        
        # Spatial feature extraction
        spatial_feat = self.spatial_fc(roi_attended_flat)  # [batch, latent_dim//2]
        
        # Combine features
        z_fmri = torch.cat([temporal_feat, spatial_feat], dim=1)  # [batch, latent_dim]
        z_fmri = self.output_proj(z_fmri)
        
        features['z_fmri_shape'] = z_fmri.shape
        
        return z_fmri, features
