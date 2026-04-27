import torch
import torch.nn as nn

from .fmri_encoder import DynamicGraphEncoder
from .smri_encoder import StructuralGraphEncoder
from .fusion import CrossModalFusion
from .vae import BetaVAE
from .classifier import ClassificationHead


class MM_DBGDGM(nn.Module):
    """
    Full multimodal model: fMRI encoder + sMRI encoder + cross-attention fusion + ?-VAE + classifier.

    forward(fmri_graphs, smri_graph):
        z_f = fmri_encoder(fmri_graphs)
        z_s = smri_encoder(smri_graph)
        z_fused = fusion(z_f, z_s)
        mu, logvar, z_sample = vae(z_fused)
        logits = classifier(mu)

        return {
            'logits': logits,
            'mu': mu,
            'logvar': logvar,
            'uncertainty': exp(logvar).mean(dim=-1),
            'attention_fs': fusion.attn_fs,
            'attention_sf': fusion.attn_sf
        }
    """

    def __init__(
        self,
        fmri_in_channels: int = 1,
        smri_in_channels: int = 4,
        gat_hidden_dim: int = 32,
        gat_heads: int = 8,
        lstm_hidden: int = 256,
        latent_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.fmri_encoder = DynamicGraphEncoder(
            in_channels=fmri_in_channels,
            hidden_dim=gat_hidden_dim,
            heads=gat_heads,
            lstm_hidden=lstm_hidden,
        )
        self.smri_encoder = StructuralGraphEncoder(
            in_channels=smri_in_channels,
            hidden_dim=gat_hidden_dim,
            heads=gat_heads,
            out_dim=512,
        )
        self.fusion = CrossModalFusion(dim=512, num_heads=8)
        self.vae = BetaVAE(in_dim=512, latent_dim=latent_dim)
        self.classifier = ClassificationHead(
            latent_dim=latent_dim,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, fmri_graphs, smri_graph, return_sample: bool = True):
        if isinstance(smri_graph, list):
            b_size = len(smri_graph)
            z_f_list = []
            z_s_list = []
            for b in range(b_size):
                z_f_list.append(self.fmri_encoder(fmri_graphs[b]))
                z_s_list.append(self.smri_encoder(smri_graph[b]))
            z_f = torch.cat(z_f_list, dim=0)
            z_s = torch.cat(z_s_list, dim=0)
        else:
            z_f = self.fmri_encoder(fmri_graphs)
            z_s = self.smri_encoder(smri_graph)
        z_fused = self.fusion(z_f, z_s)

        mu, logvar, z_sample, uncertainty = self.vae(z_fused, return_sample=return_sample)
        logits = self.classifier(mu)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "uncertainty": uncertainty,
            "attention_fs": self.fusion.attn_fs,
            "attention_sf": self.fusion.attn_sf,
        }