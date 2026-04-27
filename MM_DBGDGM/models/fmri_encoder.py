import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class DynamicGraphEncoder(nn.Module):
    """
    Processes a sequence of K temporal graph snapshots.
    Uses shared GAT weights across time steps, then BiLSTM.

    Architecture:
    - 2-layer GAT (8 heads, 32 hidden dim per head) applied to each snapshot
      with SHARED weights across time steps
    - Global mean pooling -> graph-level embedding per snapshot: e_k ? R^256
    - BiLSTM (hidden_dim=256) over sequence [e_1, ..., e_K]
    - Output: z_f ? R^512 (concatenated final fwd+bwd hidden states)
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 32, heads: int = 8, lstm_hidden: int = 256):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.out_dim = hidden_dim * heads  # 256

        self.lstm = nn.LSTM(
            input_size=self.out_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_hidden = lstm_hidden

    def forward(self, graphs):
        """
        Args:
            graphs: list of K PyG Data objects (each with x and edge_index)

        Returns:
            z_f: (512,) concatenated BiLSTM final hidden state
        """
        embeddings = []
        for g in graphs:
            x, edge_index = g.x, g.edge_index

            x = self.gat1(x, edge_index)
            x = torch.relu(x)
            x = self.gat2(x, edge_index)
            x = torch.relu(x)

            # Global mean pooling
            if g.batch is None:
                g_batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                g_batch = g.batch
            pooled = global_mean_pool(x, g_batch)
            embeddings.append(pooled)

        # (K, 256)
        seq = torch.stack(embeddings, dim=1)  # (K, 256)

        lstm_out, (h_n, c_n) = self.lstm(seq)
        # h_n has shape (num_directions, batch, hidden_size) = (2, 1, 256)
        h_fwd = h_n[0]  # (1, 256)
        h_bwd = h_n[1]  # (1, 256)
        z_f = torch.cat([h_fwd, h_bwd], dim=-1)  # (1, 512)
        return z_f