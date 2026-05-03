import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Maps latent mu -> 4-class logits.

    Architecture:
        Linear(128 -> 256) -> BatchNorm -> ReLU -> Dropout(0.4)
        Linear(256 -> 128) -> BatchNorm -> ReLU -> Dropout(0.4)
        Linear(128 -> 4)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # LayerNorm works with any batch size (incl. 1)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, mu: torch.Tensor):
        return self.net(mu)