import torch
import torch.nn as nn
from einops import rearrange

class MLPConv(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * mlp_multiplier, kernel_size=1),
            nn.Conv2d(embed_dim * mlp_multiplier, embed_dim * mlp_multiplier, kernel_size=3, padding=1, groups=embed_dim * mlp_multiplier),
            nn.GELU(),
            nn.Conv2d(embed_dim * mlp_multiplier, embed_dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h, w = int(x.size(1) ** 0.5), int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.conv(x)
        return rearrange(x, "b c h w -> b (h w) c")
