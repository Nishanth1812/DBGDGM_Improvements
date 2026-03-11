"""
Variational Autoencoder (VAE) Module
Encodes fused multimodal representation into latent code z with:
- Classification head for disease prediction
- Generative decoder for reconstruction
- KL divergence for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class VAEEncoder(nn.Module):
    """
    Encodes fused representation into latent Gaussian distribution.
    Maps z_fused → μ and log-variance parameters.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [512, 256],
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for mean and log-variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, z_fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_fused: [batch_size, latent_dim]
        
        Returns:
            mu: [batch_size, latent_dim] - mean of z distribution
            logvar: [batch_size, latent_dim] - log-variance of z distribution
        """
        x = self.encoder(z_fused)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class ClassificationHead(nn.Module):
    """
    Classification head for disease diagnosis.
    Uses μ (mean of latent distribution) at inference time.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_classes: int = 4,  # CN, eMCI, lMCI, AD
        hidden_dims: list = [512, 256],
        dropout: float = 0.2
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Build classifier layers
        layers = []
        prev_dim = latent_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.append(nn.Linear(prev_dim, hidden_dim))
            else:
                layers.extend([
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(prev_dim, hidden_dim)
                ])
            
            if i < len(hidden_dims) - 1:
                layers.extend([
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout)
                ])
            
            prev_dim = hidden_dim
        
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim, num_classes)
        ])
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        logits = self.classifier(z)
        return logits


class GenerativeDecoder(nn.Module):
    """
    Generative decoder for reconstruction.
    Reconstructs both fMRI and sMRI modalities from latent code z.
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_roi: int = 200,
        n_time: int = 50,
        n_smri_features: int = 5,
        hidden_dims: list = [512, 1024],
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_roi = n_roi
        self.n_time = n_time
        self.n_smri_features = n_smri_features
        
        # Build decoder backbone
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.decoder_backbone = nn.Sequential(*layers)
        
        # fMRI reconstruction head
        # Reconstruct [n_roi, n_time] = [200, 50]
        self.fmri_head = nn.Sequential(
            nn.Linear(prev_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, n_roi * n_time),
            nn.Tanh()  # Bounded output [-1, 1]
        )
        
        # sMRI reconstruction head
        self.smri_head = nn.Sequential(
            nn.Linear(prev_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_smri_features)
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [batch_size, latent_dim]
        
        Returns:
            fmri_recon: [batch_size, n_roi, n_time]
            smri_recon: [batch_size, n_smri_features]
        """
        x = self.decoder_backbone(z)
        
        # Reconstruct fMRI
        fmri_recon = self.fmri_head(x)
        fmri_recon = fmri_recon.view(-1, self.n_roi, self.n_time)
        
        # Reconstruct sMRI
        smri_recon = self.smri_head(x)
        
        return fmri_recon, smri_recon


class CompleteVAE(nn.Module):
    """
    Complete VAE module combining:
    - Encoder (z_fused → μ, logvar)
    - Classifier head
    - Generative decoder
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_classes: int = 4,
        n_roi: int = 200,
        n_time: int = 50,
        n_smri_features: int = 5,
        hidden_dims_encoder: list = [512, 256],
        hidden_dims_classifier: list = [512, 256],
        hidden_dims_decoder: list = [512, 1024],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # VAE Encoder
        self.encoder = VAEEncoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims_encoder,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims_classifier,
            dropout=dropout
        )
        
        # Generative decoder
        self.decoder = GenerativeDecoder(
            latent_dim=latent_dim,
            n_roi=n_roi,
            n_time=n_time,
            n_smri_features=n_smri_features,
            hidden_dims=hidden_dims_decoder,
            dropout=dropout
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(μ, σ²).
        
        Args:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        
        Returns:
            z: [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z
    
    def forward(
        self,
        z_fused: torch.Tensor,
        return_all: bool = False
    ) -> Dict:
        """
        Args:
            z_fused: [batch_size, latent_dim] - fused representation
            return_all: if True, return all intermediate outputs
        
        Returns:
            outputs dict:
                - mu: [batch_size, latent_dim]
                - logvar: [batch_size, latent_dim]
                - z: [batch_size, latent_dim]
                - logits: [batch_size, num_classes]
                - fmri_recon: [batch_size, n_roi, n_time]
                - smri_recon: [batch_size, n_smri_features]
        """
        # Encode to latent distribution
        mu, logvar = self.encoder(z_fused)
        
        # Sample z from posterior
        z = self.reparameterize(mu, logvar)

        # Use stochastic samples during training and deterministic means during evaluation.
        classifier_input = z if self.training else mu
        logits = self.classifier(classifier_input)
        
        # Reconstruction
        fmri_recon, smri_recon = self.decoder(z)
        
        outputs = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'logits': logits,
            'fmri_recon': fmri_recon,
            'smri_recon': smri_recon
        }
        
        if not return_all:
            # At inference, only return predictions
            outputs = {k: v for k, v in outputs.items() if k in ['logits', 'z']}
        
        return outputs
    
    def classify(self, z_fused: torch.Tensor) -> torch.Tensor:
        """
        Inference-only method: classify without reconstruction.
        Uses μ for deterministic predictions.
        """
        mu, _ = self.encoder(z_fused)
        logits = self.classifier(mu)
        return logits
