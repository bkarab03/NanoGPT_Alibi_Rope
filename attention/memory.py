import math

import torch
from torch import nn
from torch.nn import functional as F


# B is the batch size
# T is the sequence length (time dimension)
# C is the feature size (or number of channels)
class Memory(nn.Module):

    def __init__(self, config):
        super().__init__()  # Don't forget this line

        self.config = config
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
        self.model_type = config.model_type
        self.pos_enc_type = config.pos_enc_type
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size)) # todo investigate

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attentionXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence

    def forward(self, x, memory=None):
        original_T = x.size(1)
        # if memory is not None:
        #     x = torch.cat([memory, x], dim=1)


        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # if self.flash:

        if False:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Update the bias tensor dynamically to reflect the new T

            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Apply mask
            if self.model_type == "GPT":
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

            # Apply softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Multiply by values
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # y = y[:, -original_T:, :]

        # Create new memory from the latest hidden states
        mem_len = 512  # Or however long you want the memory to be
        new_memory = x[:, -mem_len:, :]

        return self.resid_dropout(self.c_proj(y)), new_memory
