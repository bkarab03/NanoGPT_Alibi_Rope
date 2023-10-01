import torch
from torch import nn as nn
import math
import torch.nn.functional as F

class AttentionHPROJ(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.model_type = config.model_type
        self.pos_enc_type = config.pos_enc_type
        self.rope = None
        self.alibi_biases = None
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape and transpose to bring "Heads" before "Time"
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Calculates the attention scores (logits) by taking the dot product of queries (q) and keys (k)
        # bhnk, bhmk -> bhnm
        # b = batch size, h = number of heads, n = sequence length, k/m = head dimension
        logits = torch.einsum("bhnk,bhmk->bhnm", q, k)

        # Apply mask
        if self.model_type == "GPT":
            logits = logits.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Apply softmax to the logits to get the attention weights
        weights = torch.softmax(logits, dim=-1)
        weights = self.attn_dropout(weights)

        # Calculate the output (O) by using the attention weights to weight the values (v)
        # bhnm, bhmv -> bhnv
        # b = batch size, h = number of heads, n = sequence length, m = key sequence length, v = value dimension
        O = torch.einsum("bhnm,bhmv->bhnv", weights, v)

        # Project the output (O) back to the original embedding space (d_model) using the output projection weight (W_o)
        # bhnv, hdv -> bnd
        # b = batch size, h = number of heads, n = sequence length, d = original embedding size, v = value dimension
        Y = torch.einsum("bhnv,hdv->bnd", O, self.c_proj.weight)

        return Y
