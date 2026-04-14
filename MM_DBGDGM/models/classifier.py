"""
Classification Head
Extracted from VAE module for independent downstream disease staging.
"""

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Classification head for disease diagnosis.
    Uses latent representation (e.g. μ from VAE or z_fused) to predict status.
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
