import math

from attention.base import Attention
from embeddings.alibi import get_alibi_biases, get_slopes
from embeddings.rotary_embeddings import RotaryPositionalEmbeddings
from torch.nn import functional as F
import torch


class GroupedSelfAttention(Attention):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_groups = 4  # Define the number of query groups

    def forward(self, x):
        q, k, v, B, T, C = super().forward(x)

        self.num_groups = 4 # or any factor of T
        chunk_size = T // self.num_groups

        assert T % self.num_groups == 0, f"Sequence length T={T} must be divisible by the number of groups={self.num_groups}"

        query_chunks = [q[:, :, t * chunk_size: (t + 1) * chunk_size, :] for t in range(self.num_groups)]

        outputs = []
        for i, Q_chunk in enumerate(query_chunks):
            if self.flash:
                output = torch.nn.functional.scaled_dot_product_attention(Q_chunk, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                if self.pos_enc_type == "rope":
                    self.rope = RotaryPositionalEmbeddings(d=C // self.n_head)
                    k, Q_chunk = self.apply_rotary_embeddings(k, Q_chunk)

                att = (Q_chunk @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

                if self.pos_enc_type == "alibi":
                    att = self.apply_alibi_embeddings(T, att)

                if self.model_type == "GPT":
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size

                    # Slice the bias accordingly
                    bias_chunk = self.bias[:, :, start_idx:end_idx, start_idx:end_idx]
                    bias_chunk_repeated = bias_chunk.repeat(1, 1, 1, self.n_head)

                    # print("att shape:", att.shape)
                    # print("bias_chunk shape:", bias_chunk.shape)

                    # Apply the mask
                    att = att.masked_fill(bias_chunk_repeated == 0, float('-inf'))

                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)

                output = att @ v

            outputs.append(output)

        # Concatenate the grouped attention outputs
        y = torch.cat(outputs, dim=1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        return self.resid_dropout(self.c_proj(y))


def apply_alibi_embeddings(self, T, att):
    self.register_buffer('slopes', get_slopes(self.n_head))
    if self.alibi_biases is None or self.alibi_biases.shape[1] < T:
        self.alibi_biases = get_alibi_biases(att.size(-1), self.bias[:, :, :T, :T][:, :, 0, 0])
    att += self.alibi_biases[:T, :T, None, :]
    return att
