import math

import torch
from torch import nn as nn


def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head

    * `n_heads` is the number of heads in the attention layer $n$

    The slope for first head is

    $$\frac{1}{2^{\frac{8}{n}}} = 2^{-\frac{8}{n}}$$

    The slopes for the rest of the heads are in a geometric series with a ratio same as above.

    For instance when the number of heads is $8$ the slopes are
    $$\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$$
    """

    # Get the closest power of 2 to `n_heads`.
    # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
    # and then add the remaining slopes.
    n = 2 ** math.floor(math.log2(n_heads))
    # $2^{-\frac{8}{n}}$
    m_0 = 2.0 ** (-8.0 / n)
    # $2^{-1\frac{8}{n}}, 2^{-2 \frac{8}{n}}, 2^{-3 \frac{8}{n}}, \dots$
    m = torch.pow(m_0, torch.arange(1, 1 + n))

    # If `n_heads` is not a power of 2, then we add the remaining slopes.
    # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
    # And pick the slopes upto `n_heads`.
    if n < n_heads:
        # $2^{-\frac{8}{2n}}$
        m_hat_0 = 2.0 ** (-4.0 / n)
        # $2^{-1\frac{8}{2n}}, 2^{-3 \frac{8}{2n}}, 2^{-5 \frac{8}{2n}}, \dots$
        # Note that we take steps by $2$ to avoid slopes added previously.
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        # Concatenate the slopes with the remaining slopes.
        m = torch.cat([m, m_hat])

    return m


@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    """
    ## Calculate the attention biases matrix

    * `n_heads` is the number of heads in the attention layer
    * `mask` is the attention mask of shape `[seq_len_q, seq_len_k]`

    This returns a matrix of shape `[seq_len_q, seq_len_k, n_heads, ]` with ALiBi attention biases.
    """

    # Get slopes $m$ for each head
    m = get_slopes(n_heads).to(mask.device)

    # Calculate distances $[0, 1, \dots, N]$
    # Here we calculate the distances using the mask.
    #
    # Since it's causal mask we can just use $[0, 1, \dots, N]$ too.
    # `distance = torch.arange(mask.shape[1], dtype=torch.long, device=mask.device)[None, :]`
    distance = mask.cumsum(dim=-1)

    # Multiply them pair-wise to get the AliBi bias matrix
    return distance[:, :, None] * m[None, None, :]


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[0]
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return torch.cat((x_rope, x_pass), dim=-1)


def sinusoidal_positional_encoding(pos, dim, device):
    # pos: shape (t), dim: embedding dimensionality
    pos = pos[:, None].to(device)  # Ensure pos is on the same device as the model
    div_term = torch.exp(torch.arange(0., dim, 2.) * -(math.log(10000.0) / dim)).to(device) # Ensure div_term is on the same device
    pos_emb = torch.empty(pos.size(0), dim, device=device)
    pos_emb[:, 0::2] = torch.sin(pos * div_term)
    pos_emb[:, 1::2] = torch.cos(pos * div_term)
    return pos_emb
