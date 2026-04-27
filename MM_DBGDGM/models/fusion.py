import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    Bidirectional cross-attention between fMRI and sMRI embeddings.

    z_f in R^512, z_s in R^512 -> z_fused in R^512

    Stores attention weights as self.attn_fs and self.attn_sf for interpretability.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_Q_f = nn.Linear(dim, dim)
        self.W_K_f = nn.Linear(dim, dim)
        self.W_V_f = nn.Linear(dim, dim)

        self.W_Q_s = nn.Linear(dim, dim)
        self.W_K_s = nn.Linear(dim, dim)
        self.W_V_s = nn.Linear(dim, dim)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim),
        )

        self.attn_fs = None
        self.attn_sf = None

    def forward(self, z_f: torch.Tensor, z_s: torch.Tensor):
        """
        Args:
            z_f: (B, 512) or (512,) fMRI embedding
            z_s: (B, 512) or (512,) sMRI embedding

        Returns:
            z_fused: (B, 512) or (512,)
        """
        if z_f.dim() == 1:
            z_f = z_f.unsqueeze(0)
            z_s = z_s.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B = z_f.size(0)

        # Functional -> Structural cross-attention
        Q_fs = self.W_Q_f(z_f).view(B, self.num_heads, self.head_dim)
        K_fs = self.W_K_s(z_s).view(B, self.num_heads, self.head_dim)
        V_fs = self.W_V_s(z_s).view(B, self.num_heads, self.head_dim)

        attn_fs = torch.softmax(Q_fs @ K_fs.transpose(-2, -1) * self.scale, dim=-1)
        self.attn_fs = attn_fs  # store for visualisation
        c_fs = (attn_fs @ V_fs).view(B, -1)

        # Structural -> Functional cross-attention
        Q_sf = self.W_Q_s(z_s).view(B, self.num_heads, self.head_dim)
        K_sf = self.W_K_f(z_f).view(B, self.num_heads, self.head_dim)
        V_sf = self.W_V_f(z_f).view(B, self.num_heads, self.head_dim)

        attn_sf = torch.softmax(Q_sf @ K_sf.transpose(-2, -1) * self.scale, dim=-1)
        self.attn_sf = attn_sf
        c_sf = (attn_sf @ V_sf).view(B, -1)

        # Concatenate and project
        z_fused = self.mlp(torch.cat([c_fs, c_sf], dim=-1))

        if squeeze_output:
            z_fused = z_fused.squeeze(0)

        return z_fused