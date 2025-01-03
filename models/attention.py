import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.to_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q, k, v = self.to_query(x), self.to_key(x), self.to_value(x)
        q, k, v = map(
            lambda t: t.view(t.shape[0], t.shape[1], self.n_heads, self.head_dim).transpose(1, 2),
            (q, k, v)
        )
        attention_weights = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5), dim=-1)
        attention_weights = self.to_dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, v).transpose(1, 2).reshape(x.shape[0], x.shape[1], self.embed_dim)
        return self.to_out(attention_output)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.to_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, y):
        q, k, v = self.to_query(x), self.to_key(y), self.to_value(y)
        q, k, v = map(
            lambda t: t.view(t.shape[0], t.shape[1], self.n_heads, self.head_dim).transpose(1, 2),
            (q, k, v)
        )
        attention_weights = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5), dim=-1)
        attention_weights = self.to_dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, v).transpose(1, 2).reshape(x.shape[0], x.shape[1], self.embed_dim)
        return self.to_out(attention_output)
