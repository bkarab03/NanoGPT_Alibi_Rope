import math

from einops import rearrange

from attention.attention import Attention, register_attention
from embedding import RotaryPositionalEmbeddings, get_alibi_biases, get_slopes
from torch.nn import functional as F
import torch
import torch.nn as nn


def get_decay_matrix(dim, gamma):
    d = torch.ones(dim)
    d = torch.tril(d)

    for index, head in enumerate(d):
        g = gamma[index]
        for idx, x in enumerate(torch.tril(head)):
            for idy, y in enumerate(x):
                if idx >= idy:
                    head[idx][idy] = g ** (idx-idy)
    return d


class ChunkwiseRetention(nn.Module):

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
        self.gn = nn.GroupNorm(1, config.n_head)
        self.chunk_decay = 1.0 - 2.0**(-5 - torch.arange(0, self.n_head))
        self.decay_mask = get_decay_matrix((self.n_head, self.n_embd, self.n_embd), self.chunk_decay)

        self.dropout = config.dropout
        self.chunk_size = self.n_embd//self.n_head
    def forward(self, x, past_kv):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        retention = (q @ k.transpose(-2, -1))
        inner_decay = self.n_head*self.chunk_size
        past_kv = past_kv.to(retention.device)

        # Apply mask
        retention = retention * self.decay_mask.to(retention.device)
        inner_retention = retention @ v

        cross_retention = (q @ past_kv) * inner_decay

        # Combine both
        retention = inner_retention + cross_retention

        current_kv = self.chunk_decay.to(retention.device).view(self.n_head, 1, 1) * past_kv + (k.transpose(-1, -2) @ v)
        output = self.gn(retention.transpose(-1,-2))
        output = rearrange(output, 'b c h t -> b t (c h)')
        return output, current_kv.mean(dim=0)

@register_attention('GatedMultiScaleRetention')
class GatedMultiScaleRetention(Attention):
    def __init__(self, config):
        self.config=config
        self.chunk_size = config.n_embd//config.n_head
        super().__init__(config)
        self.wg = nn.Linear(config.n_embd,  config.n_embd, bias = False)
        self.act = nn.SiLU()
        self.cwr= ChunkwiseRetention(config)
        self.wo = nn.Linear(config.n_embd,  config.n_embd, bias = False)
        self.past = torch.zeros(config.n_head, self.chunk_size, self.chunk_size)

    def forward(self, x):

        wgx = self.wg(x)
        wgx = self.act(wgx)
        y, past = self.cwr(wgx, self.past)
        self.past = past.detach()
        y = wgx * y
        return self.wo(y)




