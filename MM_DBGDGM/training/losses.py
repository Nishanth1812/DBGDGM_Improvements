"""
Loss functions for MM-DBGDGM training.

Total Loss:
L_total = L_classification + β·L_KL + λ·L_align + λ·L_recon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for disease classification."""
    
    def __init__(self, num_classes: int = 4, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size] - class labels
        
        Returns:
            loss: scalar
        """
        return self.criterion(logits, targets)


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss for VAE regularization.
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        
        Returns:
            kl_loss: scalar
        """
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return kl_loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for generative decoding.
    Separately measures fMRI and sMRI reconstruction error.
    """
    
    def __init__(self, fmri_weight: float = 1.0, smri_weight: float = 1.0):
        super().__init__()
        self.fmri_weight = fmri_weight
        self.smri_weight = smri_weight
    
    def forward(
        self,
        fmri_recon: torch.Tensor,
        fmri_orig: torch.Tensor,
        smri_recon: torch.Tensor,
        smri_orig: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fmri_recon: [batch_size, 200, 50]
            fmri_orig: [batch_size, 200, 50]
            smri_recon: [batch_size, n_features]
            smri_orig: [batch_size, n_features]
        
        Returns:
            losses dict:
                - fmri_recon: scalar
                - smri_recon: scalar
                - total_recon: scalar
        """
        # fMRI reconstruction (MSE)
        fmri_loss = F.mse_loss(fmri_recon, fmri_orig, reduction='mean')
        
        # sMRI reconstruction (MSE)
        smri_loss = F.mse_loss(smri_recon, smri_orig, reduction='mean')
        
        # Total reconstruction loss
        total_recon = self.fmri_weight * fmri_loss + self.smri_weight * smri_loss
        
        return {
            'fmri_recon': fmri_loss,
            'smri_recon': smri_loss,
            'total_recon': total_recon
        }


class AlignmentLoss(nn.Module):
    """
    Cross-modal alignment loss.
    Encourages z_fmri and z_smri to be similar in latent space.
    Using cosine similarity: L_align = -cos(z_fmri, z_smri)
    """
    
    def forward(self, z_fmri: torch.Tensor, z_smri: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_fmri: [batch_size, latent_dim]
            z_smri: [batch_size, latent_dim]
        
        Returns:
            alignment_loss: scalar (higher = more aligned)
        """
        # Normalize features
        z_fmri_norm = F.normalize(z_fmri, p=2, dim=1)
        z_smri_norm = F.normalize(z_smri, p=2, dim=1)
        
        # Cosine similarity
        cosine_sim = torch.sum(z_fmri_norm * z_smri_norm, dim=1)  # [batch_size]
        
        # Alignment loss: we want to maximize similarity, so minimize negative similarity
        alignment_loss = -torch.mean(cosine_sim)
        
        return alignment_loss

class BiomarkerRegressionLoss(nn.Module):
    """
    Computes MSE loss for continuous neurodegeneration targets.
    """
    def __init__(self):
        super().__init__()
        # Can also use HuberLoss for robustness to outliers
        self.criterion = nn.HuberLoss(reduction='mean')
        
    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        keys = ['hippocampal_volume', 'cortical_thinning_rate', 'dmn_connectivity', 'nss']
        for k in keys:
            if k in targets:
                loss += self.criterion(preds[k], targets[k])
        return loss / max(1, len(targets))

class WeibullSurvivalLoss(nn.Module):
    """
    Negative Log-Likelihood for Weibull AFT model handling right-censored data.
    """
    def forward(self, shape: torch.Tensor, scale: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shape: [batch_size, num_events]
            scale: [batch_size, num_events]
            times: [batch_size, num_events] - observed or censored time
            events: [batch_size, num_events] - 1 if event occurred, 0 if censored
        """
        # Ensure time > 0
        t = times + 1e-6
        
        # log(S(t)) = -(t/scale)^shape
        log_surv = -torch.pow(t / scale, shape)
        
        # log(h(t)) = log(shape) - log(scale) + (shape-1)*log(t/scale)
        log_haz = torch.log(shape) - torch.log(scale) + (shape - 1.0) * torch.log(t / scale)
        
        # LogLikelihood = event * log(h(t)) + log(S(t))
        ll = events * log_haz + log_surv
        
        return -torch.mean(ll)



class MM_DBGDGM_Loss(nn.Module):
    """
    Complete loss function for MM-DBGDGM.
    
    L_total = L_class + β·L_KL + λ·L_align + λ·L_recon + λ·L_regression + λ·L_survival
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        lambda_kl: float = 0.1,
        lambda_align: float = 0.1,
        lambda_recon: float = 0.1,
        lambda_regression: float = 0.1,
        lambda_survival: float = 0.1,
        fmri_recon_weight: float = 2.0,
        smri_recon_weight: float = 1.0,
        class_weights: Optional[Dict[int, float]] = None
    ):
        super().__init__()
        
        self.lambda_kl = lambda_kl
        self.lambda_align = lambda_align
        self.lambda_recon = lambda_recon
        self.lambda_regression = lambda_regression
        self.lambda_survival = lambda_survival
        
        # Component losses
        if class_weights is not None:
            weights = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)])
        else:
            weights = None
        
        self.classification = ClassificationLoss(num_classes, weight=weights)
        self.kl_divergence = KLDivergenceLoss()
        self.reconstruction = ReconstructionLoss(
            fmri_weight=fmri_recon_weight,
            smri_weight=smri_recon_weight
        )
        self.alignment = AlignmentLoss()
        self.regression = BiomarkerRegressionLoss()
        self.survival = WeibullSurvivalLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        fmri_recon: torch.Tensor,
        fmri_orig: torch.Tensor,
        smri_recon: torch.Tensor,
        smri_orig: torch.Tensor,
        z_fmri: torch.Tensor,
        z_smri: torch.Tensor,
        degeneration_preds: Dict[str, torch.Tensor],
        regression_targets: Dict[str, torch.Tensor],
        survival_shape: torch.Tensor,
        survival_scale: torch.Tensor,
        survival_times: torch.Tensor,
        survival_events: torch.Tensor,
        beta_annealing: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
            fmri_recon: [batch_size, 200, 50]
            fmri_orig: [batch_size, 200, 50]
            smri_recon: [batch_size, n_features]
            smri_orig: [batch_size, n_features]
            z_fmri: [batch_size, latent_dim]
            z_smri: [batch_size, latent_dim]
            beta_annealing: KL annealing factor (0→1 during training)
        
        Returns:
            losses dict:
                - classification: scalar
                - kl: scalar
                - recon: scalar
                - alignment: scalar
                - regression: scalar
                - survival: scalar
                - total: scalar
        """
        # Classification loss
        loss_class = self.classification(logits, targets)
        
        # KL divergence loss (with annealing)
        loss_kl = self.kl_divergence(mu, logvar)
        
        # Reconstruction losses
        recon_losses = self.reconstruction(fmri_recon, fmri_orig, smri_recon, smri_orig)
        loss_recon = recon_losses['total_recon']
        
        # Alignment loss
        loss_align = self.alignment(z_fmri, z_smri)
        
        # Regression loss
        loss_reg = self.regression(degeneration_preds, regression_targets)
        
        # Survival loss
        loss_surv = self.survival(survival_shape, survival_scale, survival_times, survival_events)
        
        # Total loss
        total_loss = (
            loss_class +
            self.lambda_kl * beta_annealing * loss_kl +
            self.lambda_align * loss_align +
            self.lambda_recon * loss_recon +
            self.lambda_regression * loss_reg +
            self.lambda_survival * loss_surv
        )
        
        return {
            'classification': loss_class,
            'kl': loss_kl,
            'fmri_recon': recon_losses['fmri_recon'],
            'smri_recon': recon_losses['smri_recon'],
            'alignment': loss_align,
            'regression': loss_reg,
            'survival': loss_surv,
            'total': total_loss,
            'beta_annealing': torch.tensor(beta_annealing, device=logits.device)
        }
