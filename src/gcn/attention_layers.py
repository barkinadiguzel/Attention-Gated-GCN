import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.W = nn.ParameterList([nn.Parameter(torch.randn(in_dim, out_dim)) for _ in range(num_heads)])
        self.C = nn.ParameterList([nn.Parameter(torch.randn(out_dim, out_dim)) for _ in range(num_heads)])

    def forward(self, H, A):
        N = H.size(0)
        out = 0
        for k in range(self.num_heads):
            H_k = H @ self.W[k]
            alpha = torch.sigmoid(H_k @ self.C[k] @ H_k.T)  # pairwise interaction strength
            alpha = alpha * A  # sadece komşular
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)  # normalize
            out += torch.matmul(alpha, H_k)
        out = out / self.num_heads
        return F.relu(out)
