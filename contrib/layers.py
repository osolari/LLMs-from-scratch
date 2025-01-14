import torch
from torch import nn
from torch.nn import GELU

from contrib.attention import MultiHeadSelfAttention


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (std + self.eps)

        return self.scale * normalized + self.shift


class FeedForward(nn.Module):

    def __init__(self, emb_dim, *args, **kwargs):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):

    def __init__(self, emb_dim, context_length, n_heads, qkv_bias, drop_rate, **kwargs):
        super().__init__()

        self.att = MultiHeadSelfAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            num_heads=n_heads,
            qkv_bias=qkv_bias,
            dropout=drop_rate,
        )

        self.ff = FeedForward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut

        return x
