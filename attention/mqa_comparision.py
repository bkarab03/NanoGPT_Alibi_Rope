# Credits to https://github.com/hkproj/pytorch-llama-notes/blob/main/mqa_comparison.py
import torch
import torch.nn.functional as F
import math

# Algorithms described in the paper "Fast Transformer Decoding: One Write-Head is All You Need", Noam Shazeer, 2019

def MultiheadAttentionBatched():

    # d_model: Model dimensions (also 'd' in einsum)
    # seq_len_kv: Key-Value sequence length
    # seq_len: Query sequence length (also 'n' in einsum)
    # b: Batch size
    # h: Attention heads
    # d_k: Key dimensions per head (also 'k' in einsum)
    # d_v: Value dimensions per head (also 'v' in einsum)

    d_model, seq_len_kv, seq_len, b, h, d_k, d_v = 512, 10, 10, 32, 8, (512 // 8), (512 // 8)

    X = torch.rand(b, seq_len, d_model)  # Query
    M = torch.rand(b, seq_len_kv, d_model)  # Key and Value
    mask = torch.rand(b, h, seq_len, seq_len_kv)
    P_q = torch.rand(h, d_model, d_k)  # W_q
    P_k = torch.rand(h, d_model, d_k)  # W_k
    P_v = torch.rand(h, d_model, d_v)  # W_v
    P_o = torch.rand(h, d_model, d_v)  # W_o

    Q = torch.einsum("bnd,hdk->bhnk ", X, P_q)
    K = torch.einsum("bmd,hdk->bhmk", M, P_k)
    V = torch.einsum("bmd,hdv->bhmv", M, P_v)

    logits = torch.einsum("bhnk,bhmk->bhnm", Q, K)
    weights = torch.softmax(logits + mask, dim=-1)

    O = torch.einsum("bhnm,bhmv->bhnv ", weights, V)
    Y = torch.einsum("bhnv,hdv->bnd ", O, P_o)
    return Y


def MultiheadSelfAttentionIncremental():
    d_model, b, h, d_k, d_v = 512, 32, 8, (512 // 8), (512 // 8)

    m = 5  # Suppose we have already cached "m" tokens
    prev_K = torch.rand(b, h, m, d_k)
    prev_V = torch.rand(b, h, m, d_v)

    X = torch.rand(b, d_model)  # Query
    M = torch.rand(b, d_model)  # Key and Value
    P_q = torch.rand(h, d_model, d_k)  # W_q
    P_k = torch.rand(h, d_model, d_k)  # W_k
    P_v = torch.rand(h, d_model, d_v)  # W_v
    P_o = torch.rand(h, d_model, d_v)  # W_o

    q = torch.einsum("bd,hdk->bhk", X, P_q)
    new_K = torch.cat([prev_K, torch.einsum("bd,hdk->bhk", M, P_k).unsqueeze(2)], dim=2)
    new_V = torch.cat([prev_V, torch.einsum("bd,hdv->bhv", M, P_v).unsqueeze(2)], dim=2)

    logits = torch.einsum("bhk,bhmk->bhm", q, new_K)
    weights = torch.softmax(logits, dim=-1)
    O = torch.einsum("bhm,bhmv->bhv", weights, new_V)
    y = torch.einsum("bhv,hdv->bd", O, P_o)
    return y, new_K, new_V


def MultiquerySelfAttentionIncremental():
    d, b, h, k, v = 512, 32, 8, (512 // 8), (512 // 8)

    m = 5  # Suppose we have already cached "m" tokens
    prev_K = torch.rand(b, m, k)
    prev_V = torch.rand(b, m, v)

    X = torch.rand(b, d)  # Query
    M = torch.rand(b, d)  # Key and Value
    P_q = torch.rand(h, d, k)  # W_q
    P_k = torch.rand(d, k)  # W_k
    P_v = torch.rand(d, v)  # W_v
    P_o = torch.rand(h, d, v)  # W_o

    q = torch.einsum("bd,hdk->bhk", X, P_q)
    K = torch.cat([prev_K, torch.einsum("bd,dk->bk", M, P_k).unsqueeze(1)], dim=1)
    V = torch.cat([prev_V, torch.einsum("bd,dv->bv", M, P_v).unsqueeze(1)], dim=1)

    logits = torch.einsum("bhk,bmk->bhm", q, K)
    weights = torch.softmax(logits, dim=-1)
    O = torch.einsum("bhm,bmv->bhv", weights, V)
    y = torch.einsum("bhv,hdv->bd", O, P_o)
    return y, K, V

if __name__ == "__main__":
    MultiheadAttentionBatched()
    MultiheadSelfAttentionIncremental()
    MultiquerySelfAttentionIncremental()