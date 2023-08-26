import math

from attention.base import Attention
from embeddings.alibi import get_alibi_biases, get_slopes
from embeddings.rotary_embeddings import RotaryPositionalEmbeddings
from torch.nn import functional as F
import torch


class SelfAttention(Attention):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, x):
        q, k, v, B, T, C = super().forward(x)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Compute attention scores
            # Apply RoPE to queries and keys if needed
            if self.pos_enc_type == "rope":
                self.rope = RotaryPositionalEmbeddings(d=C // self.n_head)
                k, q = self.apply_rotary_embeddings(k, q)

            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # If using Alibi, create and add biases
            if self.pos_enc_type == "alibi":
                att = self.apply_alibi_embeddings(T, att)

            # Apply mask
            if self.model_type == "GPT":
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

            # Apply softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Multiply by values
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        return self.resid_dropout(self.c_proj(y))

    def apply_alibi_embeddings(self, T, att):
        self.register_buffer('slopes', get_slopes(self.n_head))
        if self.alibi_biases is None or self.alibi_biases.shape[1] < T:
            self.alibi_biases = get_alibi_biases(att.size(-1), self.bias[:, :, :T, :T][:, :, 0, 0])
        att += self.alibi_biases[:T, :T, None, :]
        return att
