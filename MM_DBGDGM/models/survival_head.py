"""
Survival Analysis Head
Answers:
C. When is deterioration predicted to occur? (Temporal Forecasting / Time-to-Event Prediction)

Uses a Weibull-AFT (Accelerated Failure Time) model parameterization on top of latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class WeibullSurvivalHead(nn.Module):
    """
    Weibull Accelerated Failure Time (AFT) model for survival analysis.
    Predicts the shape (k) and scale (lambda) parameters of the Weibull distribution
    for multiple conversion events:
    0: CN -> eMCI
    1: eMCI -> lMCI
    2: lMCI -> AD
    """
    
    def __init__(self, latent_dim: int, num_events: int = 3, hidden_dims: list = [256, 128], dropout: float = 0.2):
        super().__init__()
        self.num_events = num_events
        
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
            
        self.shared_mlp = nn.Sequential(*layers)
        
        # Weibull distribution parameters must be strictly positive.
        # Shape parameter (k): controls how hazard rate changes over time
        # Scale parameter (lambda): controls the overall scale
        # We output parameters for each of the `num_events`
        self.fc_shape = nn.Linear(prev_dim, num_events)
        self.fc_scale = nn.Linear(prev_dim, num_events)
        
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [batch_size, latent_dim] - e.g., the posterior mean `mu`
            
        Returns:
            dict with 'shape' and 'scale' parameters for the Weibull distribution.
            Each tensor is of shape [batch_size, num_events]
        """
        features = self.shared_mlp(z)
        
        # Softplus ensures parameters are strictly positive
        # Add epsilon for numerical stability
        shape = F.softplus(self.fc_shape(features)) + 1e-6
        scale = F.softplus(self.fc_scale(features)) + 1e-6
        
        return {
            'shape': shape,
            'scale': scale
        }
    
    def expected_time(self, shape: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Calculates the expected time to event (mean of Weibull distribution).
        E[T] = scale * Gamma(1 + 1/shape)
        
        Args:
            shape: [batch_size, num_events]
            scale: [batch_size, num_events]
            
        Returns:
            expected_time: [batch_size, num_events]
        """
        # torch.lgamma computes natural logarithm of absolute value of Gamma function
        # E[X] = scale * exp(lgamma(1 + 1/shape))
        gamma_arg = 1.0 + (1.0 / shape)
        expected_t = scale * torch.exp(torch.lgamma(gamma_arg))
        return expected_t
    
    def survival_function(self, t: torch.Tensor, shape: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Calculates the survival probability S(t) = P(T > t)
        S(t) = exp(-(t/scale)^shape)
        
        Args:
            t: [batch_size, 1] or scalar - time points
            shape: [batch_size, num_events]
            scale: [batch_size, num_events]
            
        Returns:
            survival_prob: [batch_size, num_events]
        """
        # (t/scale)^shape
        hazard = torch.pow(t / scale, shape)
        return torch.exp(-hazard)
    
    def hazard_function(self, t: torch.Tensor, shape: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Calculates the hazard rate h(t)
        h(t) = (shape/scale) * (t/scale)^(shape-1)
        """
        # Add epsilon to prevent 0^(shape-1) when shape < 1
        t = t + 1e-6
        return (shape / scale) * torch.pow(t / scale, shape - 1)
