import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedSkipConnection(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Uz1 = nn.Linear(hidden_dim, hidden_dim)
        self.Uz2 = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, H_new, H_prev):
        z = self.sigmoid(self.Uz1(H_new) + self.Uz2(H_prev))
        return z * H_new + (1 - z) * H_prev
