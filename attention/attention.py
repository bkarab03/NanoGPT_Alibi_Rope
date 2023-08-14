import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from embedding import RotaryPositionalEmbeddings, get_slopes, get_alibi_biases


class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.pos_enc_type = config.pos_enc_type
        self.rope = None
        self.alibi_biases = None

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attentionFalse')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):

        B, T, C = x.size()

        q, k, v = self.get_qkv(x)

        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        return q, k, v, B, T, C, self.attention(q, k, v, B, T, C)

    def get_qkv(self, x):
        return self.c_attn(x).split(self.config.n_embd, dim=2)
