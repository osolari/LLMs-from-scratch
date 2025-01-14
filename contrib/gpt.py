import torch
from torch import nn

from contrib.layers import LayerNorm, TransformerBlock


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        emb_dim,
        n_heads,
        qkv_bias,
        n_layers,
        drop_rate,
        *args,
        **kwargs
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)

        self.dropout_emb = nn.Dropout(drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(emb_dim, context_length, n_heads, qkv_bias, drop_rate)
                for _ in range(n_layers)
            ]
        )

        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(
            emb_dim,
            vocab_size,
        )

    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape
        token_embedding = self.token_embedding(in_idx)
        position_embedding = self.position_embedding(
            torch.arange(seq_len, device=token_embedding.device)
        )

        x = token_embedding + position_embedding
        x = self.dropout_emb(x)

        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

    def generate_text_simple(self, idx, max_new_tokens, context_size):
        self.eval()
        # idx is (batch, n_tokens) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = self(idx_cond)

            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx
