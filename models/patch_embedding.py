import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_height, image_width, image_channels, patch_height, patch_width, embedding_dim):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.embedding_dim = embedding_dim

        patch_dim = self.patch_height * self.patch_width * self.image_channels

        self.patchify_and_embed = nn.Sequential(
            nn.Conv2d(self.image_channels, patch_dim, kernel_size=self.patch_height, stride=self.patch_width),
            Rearrange("bs d h w -> bs (h w) d"),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )
        self.pos_embed = nn.Embedding(
            self.image_height * self.image_width // (self.patch_height * self.patch_width), self.embedding_dim
        )
        self.register_buffer("precomputed_pos_enc", torch.arange(0, ((self.image_height * self.image_width) /
                                                                     (self.patch_width * self.patch_height))).long())

    def forward(self, x):
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[: x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)
        return x
