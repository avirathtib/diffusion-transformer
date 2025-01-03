import torch
import torch.nn as nn
import numpy as np

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim, min_frequency=1.0, max_frequency=1000.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(np.log(min_frequency), np.log(max_frequency), embedding_dim // 2))
        self.register_buffer("freqs", freqs * 2.0 * torch.pi)

    def forward(self, x):
        embeddings = torch.cat(
            [torch.sin(self.freqs * x), torch.cos(self.freqs * x)], dim=-1
        )
        return embeddings
