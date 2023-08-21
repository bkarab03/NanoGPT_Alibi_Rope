import math

import torch

def sinusoidal_positional_encoding(pos, dim, device):
    # pos: shape (t), dim: embedding dimensionality
    pos = pos[:, None].to(device)  # Ensure pos is on the same device as the model
    # print(device)
    div_term = torch.exp(torch.arange(0., dim, 2.) * -(math.log(10000.0) / dim)).to(device) # Ensure div_term is on the same device
    pos_emb = torch.empty(pos.size(0), dim, device=device)
    pos_emb[:, 0::2] = torch.sin(pos * div_term)
    pos_emb[:, 1::2] = torch.cos(pos * div_term)
    return pos_emb
