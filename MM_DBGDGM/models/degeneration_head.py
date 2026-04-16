"""
Neurodegeneration Forecasting Module
Answers:
A. Where is deterioration happening? (Regional Atrophy Localization)
B. How much deterioration is predicted? (Quantitative Biomarker Regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class RegionalAtrophyLocalization(nn.Module):
    """
    Outputs a per-region deterioration score over all brain ROIs.
    Indicates which regions are structurally or functionally degrading.
    """
    
    def __init__(self, latent_dim: int, n_roi: int = 200, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.n_roi = n_roi
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_roi)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim] (e.g., z_fused)
            
        Returns:
            scores: [batch_size, n_roi] probability distribution of deterioration
        """
        logits = self.network(z)
        # Normalize to probability distribution for interpretability (sum = 1)
        # so clinicians can rank top-k regions.
        scores = F.softmax(logits, dim=1)
        return scores


class BiomarkerRegressionHead(nn.Module):
    """
    Quantitative Biomarker Regression.
    Predicts:
    - Future hippocampal volume
    - Rate of cortical thinning per region (averaged scalar for simplicity, or per-region)
    - Functional connectivity strength of the DMN
    - Neurodegeneration Severity Score (NSS) [0-100]
    """
    
    def __init__(self, latent_dim: int, hidden_dims: list = [256, 128], dropout: float = 0.2):
        super().__init__()
        
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
        
        # 4 independent continuous targets
        self.fc_hippo_vol = nn.Linear(prev_dim, 1)
        self.fc_cortical_thinning = nn.Linear(prev_dim, 1)
        self.fc_dmn_conn = nn.Linear(prev_dim, 1)
        self.fc_nss = nn.Linear(prev_dim, 1)
        
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [batch_size, latent_dim]
            
        Returns:
            dict of regression targets
        """
        features = self.shared_mlp(z)
        
        hippo_vol = self.fc_hippo_vol(features).squeeze(1)
        cortical_thinning = self.fc_cortical_thinning(features).squeeze(1)
        dmn_conn = self.fc_dmn_conn(features).squeeze(1)
        # NSS is bounded between 0 and 100
        nss = torch.sigmoid(self.fc_nss(features).squeeze(1)) * 100.0
        
        return {
            'hippocampal_volume': hippo_vol,
            'cortical_thinning_rate': cortical_thinning,
            'dmn_connectivity': dmn_conn,
            'nss': nss
        }


class NeurodegenerationHead(nn.Module):
    """
    Combines Atrophy Localization and Biomarker Regression.
    """
    
    def __init__(self, latent_dim: int = 256, n_roi: int = 200, dropout: float = 0.2):
        super().__init__()
        
        self.localization = RegionalAtrophyLocalization(latent_dim, n_roi, dropout=dropout)
        self.regression = BiomarkerRegressionHead(latent_dim, dropout=dropout)
        
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [batch_size, latent_dim] - typically z_fused
            
        Returns:
            Dictionary containing localization scores and regression targets
        """
        loc_scores = self.localization(z)
        reg_targets = self.regression(z)
        
        outputs = {'atrophy_localization': loc_scores}
        outputs.update(reg_targets)
        
        return outputs
