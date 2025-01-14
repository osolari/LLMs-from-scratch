import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out):

        super().__init__()

        self.W_q = nn.Parameter(torch.rand(d_in, d_out))
        self.W_k = nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = nn.Parameter(torch.rand(d_in, d_out))

        self.d_out = d_out
        self.d_in = d_in

    def forward(self, x):

        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        attention_scores = q @ k.T
        attention_weights = torch.softmax(attention_scores / self.d_out**0.5, dim=-1)
        context_vec = attention_weights @ v
        return context_vec


class SelfAttentionLinear(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.d_out = d_out
        self.d_in = d_in

    def forward(self, x):
        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)

        attn_scores = q @ k.T  # omega
        attn_weights = torch.softmax(attn_scores / (self.d_out**0.5), dim=-1)

        context_vec = attn_weights @ v
        return context_vec


class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.d_out = d_out
        self.d_in = d_in
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):

        b, num_tokens, d_in = x.shape

        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)

        attn_scores = q @ k.transpose(1, 2)
        attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ v

        return context_vec


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        self.head = nn.ModuleList(
            [
                CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):

        return torch.cat([hea(x) for hea in self.head], dim=-1)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(d_out, d_out)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):

        b, num_tokens, d_in = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = q @ k.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ v).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
