import torch.nn as nn
from models.sinusoidal_embedding import SinusoidalEmbedding
from models.patch_embedding import PatchEmbedding
from models.attention import SelfAttention, CrossAttention
from models.mlp_conv import MLPConv
from einops import rearrange
import torch

class Denoiser(nn.Module):
    def __init__(self, image_size, noise_embed_dims, patch_size, embed_dim, dropout, n_layers=10, text_emb_size=768, mlp_multiplier=8, n_channels=4):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        
        # Noise and label handling
        self.noise_handling = nn.Sequential(
            SinusoidalEmbedding(noise_embed_dims),
            nn.Linear(noise_embed_dims, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.patch_embedding_instance = PatchEmbedding(image_size, image_size, n_channels, patch_size, patch_size, embed_dim)
        self.label_embedding = nn.Linear(text_emb_size, embed_dim)
        
        # Repeated layers with normalization
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn_norm": nn.LayerNorm(embed_dim),
                "self_attention": SelfAttention(embed_dim, n_heads=10, dropout=dropout),
                "cross_attn_norm": nn.LayerNorm(embed_dim),
                "cross_attention": CrossAttention(embed_dim, n_heads=10, dropout=dropout),
                "mlp_norm": nn.LayerNorm(embed_dim),
                "mlp_conv": MLPConv(embed_dim, mlp_multiplier, dropout)
            }) for _ in range(10)
        ])
        
        # Final output
        self.final_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, patch_size * patch_size * n_channels)
        
    def forward(self, x, noise, text_label):
        # Patch and embeddings
        patched_x = self.patch_embedding_instance(x)
        label_embedding = self.label_embedding(text_label).unsqueeze(1)
        noise_embedding = self.noise_handling(noise).unsqueeze(1)
        noise_and_label_embedding = torch.cat([label_embedding, noise_embedding], dim=1)
        
        # Pass through repeated layers with normalization
        out_x = patched_x
        for layer in self.layers:
            # Self attention with normalization
            normed_x = layer["self_attn_norm"](out_x)
            out_x = layer["self_attention"](normed_x) + out_x
            
            # Cross attention with normalization
            normed_x = layer["cross_attn_norm"](out_x)
            out_x = layer["cross_attention"](normed_x, noise_and_label_embedding) + out_x
            
            # MLP with normalization
            normed_x = layer["mlp_norm"](out_x)
            out_x = layer["mlp_conv"](normed_x) + out_x
        
        # Final output
        out_x = self.final_norm(out_x)
        out_x = self.out(out_x)
        out = rearrange(out_x, 'b (h w) (c ps1 ps2) -> b c (h ps1) (w ps2)',
                       ps1=self.patch_size, ps2=self.patch_size,
                       h=int(self.image_size // self.patch_size), 
                       w=int(self.image_size // self.patch_size))
        return out