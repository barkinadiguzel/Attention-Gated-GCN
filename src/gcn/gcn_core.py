import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_layers import GraphAttentionLayer
from .gated_skip import GatedSkipConnection

class AttentionGatedGCNLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attention = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
        self.gated_skip = GatedSkipConnection(hidden_dim)

    def forward(self, H, A):
        H_new = self.attention(H, A)
        H_out = self.gated_skip(H_new, H)
        return H_out

class AttentionGatedGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=5, num_heads=4, readout_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([AttentionGatedGCNLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.readout = nn.Linear(hidden_dim, readout_dim)
        self.predictor = nn.Linear(readout_dim, 1)  # regression

    def forward(self, X, A):
        H = F.relu(self.input_layer(X))
        for layer in self.layers:
            H = layer(H, A)
        graph_feat = torch.sum(F.relu(self.readout(H)), dim=0, keepdim=True)
        y_pred = self.predictor(graph_feat)
        return y_pred, H, graph_feat
