import torch
import torch.nn as nn


def combined_loss(logits, labels, mu, logvar, beta=1.0, lambda_vae=0.1, current_epoch=0, warmup_epochs=20):
    """
    L_total = L_CE(logits, labels) + lambda_vae * beta * KL_loss(mu, logvar)

    KL annealing: lambda_vae starts at 0 and linearly increases to 0.1 over first warmup_epochs.
    """
    ce_loss = nn.CrossEntropyLoss()(logits, labels)

    # KL annealing
    if current_epoch < warmup_epochs:
        anneal_factor = lambda_vae * current_epoch / warmup_epochs
    else:
        anneal_factor = lambda_vae

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1).mean()
    vae_loss = anneal_factor * beta * kl

    total = ce_loss + vae_loss
    return total, ce_loss, kl