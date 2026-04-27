import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class StructuralGraphEncoder(nn.Module):
    """
    Processes the static structural brain graph.

    Architecture:
    - 2-layer GAT (8 heads, 32 hidden dim) on static structural graph
    - Attention-weighted global pooling (learns to up-weight AD-sensitive nodes)
    - Linear projection -> z_s ? R^512
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 32, heads: int = 8, out_dim: int = 512):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.gat_out_dim = hidden_dim * heads

        self.pool_attn = nn.Linear(self.gat_out_dim, 1)
        self.proj = nn.Linear(self.gat_out_dim, out_dim)

    def forward(self, graph):
        """
        Args:
            graph: PyG Data object with x (N_roi, F) and edge_index

        Returns:
            z_s: (512,) structural embedding
        """
        x, edge_index = graph.x, graph.edge_index

        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        x = torch.relu(x)

        # Attention-weighted pooling
        attn_weights = torch.softmax(self.pool_attn(x), dim=0)
        pooled = (attn_weights * x).sum(dim=0)

        z_s = self.proj(pooled).unsqueeze(0)
        return z_s