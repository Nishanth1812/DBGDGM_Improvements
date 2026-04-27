import torch
import torch.nn as nn


class BetaVAE(nn.Module):
    """
    Maps z_fused -> probabilistic latent code.

    Encoder:
        mu_head: Linear(512 -> latent_dim=128)
        logvar_head: Linear(512 -> latent_dim=128)

    Reparameterisation: z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I)

    At inference: return mu (point estimate) + exp(logvar) as uncertainty ??

    ELBO loss method computes reconstruction + KL.
    """

    def __init__(self, in_dim: int = 512, latent_dim: int = 128):
        super().__init__()
        self.mu_head = nn.Linear(in_dim, latent_dim)
        self.logvar_head = nn.Linear(in_dim, latent_dim)

    def forward(self, z_fused: torch.Tensor, return_sample: bool = True):
        """
        Args:
            z_fused: (B, 512) fused embedding
            return_sample: if True, return reparameterised sample; else return mu

        Returns:
            mu: (B, 128)
            logvar: (B, 128)
            z_sample: (B, 128) or mu (if return_sample=False)
            uncertainty: (B,) scalar per subject = exp(logvar).mean(dim=-1)
        """
        mu = self.mu_head(z_fused)
        logvar = self.logvar_head(z_fused)

        if return_sample:
            eps = torch.randn_like(mu)
            z_sample = mu + eps * torch.exp(0.5 * logvar)
        else:
            z_sample = mu

        uncertainty = torch.exp(logvar).mean(dim=-1)  # (B,)
        return mu, logvar, z_sample, uncertainty

    def elbo_loss(self, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
        """
        Compute KL divergence term for ?-VAE loss.

        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

        Args:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            beta: scaling factor

        Returns:
            beta * KL loss (scalar)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
        return beta * kl.mean()