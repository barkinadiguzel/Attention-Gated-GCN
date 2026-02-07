import torch

def aggregate_messages(H, A, W):
    return torch.matmul(A, H) @ W
