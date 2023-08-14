from einops import rearrange
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class ChunkwiseRetention(nn.Module):
    def __init__(self, chunk_size, num_head, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed,  chunk_size * num_head, bias = False)
        self.query = nn.Linear(n_embed,  chunk_size * num_head, bias = False)
        self.value = nn.Linear(n_embed,  chunk_size * num_head, bias = False)
        self.gamma = 1.0-2.0**(-5-torch.arange(0,num_head))
        self.decay_mask = get_decay_matrix((num_head, block_size, block_size), self.gamma)
        self.chunk_decay = self.gamma
        self.gn = nn.GroupNorm(1, num_head)
        self.num_head = num_head
        self.chunk_size = chunk_size

    def forward(self, x, past_kv):

        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = rearrange(k, ('b t (h c) -> b h t c'), t=T, h=self.num_head, c =self.chunk_size)
        q = rearrange(q, ('b t (h c) -> b h t c'), t=T, h=self.num_head, c =self.chunk_size)
        v = rearrange(v, ('b t (h c) -> b h t c'), t=T, h=self.num_head, c =self.chunk_size)


        retention = q @ k.transpose(-1, -2)

        # b h t c , b h c t -> b h t t

        retention = retention  * self.decay_mask   # b h t t* h t t



        inner_retention = retention @ v


        past_kv = repeat(past_kv, 'n q v -> B n q v', B=B)
        pb, pn, pq, pv = past_kv.shape

        padding = torch.zeros(pb, pn, pq, self.chunk_size)

        past_kv = past_kv+ padding



        dm = repeat(self.decay_mask, 'h c d -> B h c d', B=B)
        pp = q @ past_kv
        cross_retention = pp.transpose(-1, -2) @ dm
        cross_retention = cross_retention.transpose(-1, -2)


        retention = inner_retention + cross_retention



        current_kv = self.gamma.view(self.num_head, 1, 1) * past_kv + (k.transpose(-1, -2) @ v)
        output = self.gn(retention.transpose(-1,-2))
        output = rearrange(output, 'b c h t -> b t (c h)')
        return output, current_kv.mean(dim=0)


class GatedMultiScaleRetention(nn.Module):
    def __init__(self, chunk_size, num_head, block_size):
        super().__init__()
        self.wg = nn.Linear(n_embed,  n_embed, bias = False)
        self.act = nn.SiLU()
        self.y= ChunkwiseRetention(num_head = n_head, chunk_size = n_embed//n_head, block_size=block_size)
        self.wo = nn.Linear(n_embed,  n_embed, bias = False)
        self.past = torch.zeros(num_head, chunk_size, chunk_size)

    def forward(self, x):
        wgx = self.wg(x)
        wgx = self.act(wgx)
        y, past = self.y(wgx, self.past)
        self.past = past.detach()
        y = wgx * y
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4* n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
         nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        self.sa_head= GatedMultiScaleRetention(num_head = n_head, chunk_size = n_embed//n_head, block_size=block_size)
        self.ffw=  FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x+self.ffw(self.ln2(x))
        return x


class RetNet(nn.Module):
    def __init__(self, block_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokes):
        for _ in range(max_new_tokes):
            b, s = idx.shape
            bk = min(s, block_size)
            idx_cond =  torch.cat((torch.zeros(b, block_size-bk, dtype=int), idx), dim=1)[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx