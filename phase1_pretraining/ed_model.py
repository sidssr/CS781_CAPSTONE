import math, time, os, json, argparse, zipfile
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + s.eps)).to(x.dtype) * s.w

def rope_cache(T, hd, dev, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(torch.arange(T, device=dev).float(), inv)
    return f.cos(), f.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None].to(x.dtype); sin = sin[None, None].to(x.dtype)
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)

class SelfAttn(nn.Module):
    def __init__(s, d, h, causal):
        super().__init__(); s.h, s.causal = h, causal
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
    def forward(s, x, cs):
        B, T, D = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cs[0][:T], cs[1][:T]); k = apply_rope(k, cs[0][:T], cs[1][:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=s.causal)
        return s.o(y.transpose(1, 2).reshape(B, T, D))

class CrossAttn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.q = nn.Linear(d, d, bias=False); s.kv = nn.Linear(d, 2 * d, bias=False)
        s.o = nn.Linear(d, d, bias=False)
    def forward(s, x, mem):
        B, T, D = x.shape; S = mem.shape[1]; hd = D // s.h
        q = s.q(x).view(B, T, s.h, hd).transpose(1, 2)
        k, v = s.kv(mem).view(B, S, 2, s.h, hd).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return s.o(y.transpose(1, 2).reshape(B, T, D))

class SwiGLU(nn.Module):
    def __init__(s, d):
        super().__init__()
        hdn = int(8 * d / 3 / 64) * 64
        s.w1 = nn.Linear(d, hdn, bias=False); s.w3 = nn.Linear(d, hdn, bias=False)
        s.w2 = nn.Linear(hdn, d, bias=False)
    def forward(s, x): return s.w2(F.silu(s.w1(x)) * s.w3(x))

class EncBlock(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.n1, s.n2 = RMSNorm(d), RMSNorm(d)
        s.attn = SelfAttn(d, h, causal=False); s.mlp = SwiGLU(d)
    def forward(s, x, cs):
        x = x + s.attn(s.n1(x), cs)
        return x + s.mlp(s.n2(x))

class DecBlock(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.n1, s.n2, s.n3 = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        s.attn = SelfAttn(d, h, causal=True); s.cross = CrossAttn(d, h); s.mlp = SwiGLU(d)
    def forward(s, x, cs, mem):
        x = x + s.attn(s.n1(x), cs)
        x = x + s.cross(s.n2(x), mem)
        return x + s.mlp(s.n3(x))

class EncDec(nn.Module):
    def __init__(s, vocab, d, Le, Ld, h):
        super().__init__()
        s.d, s.h = d, h
        s.emb = nn.Embedding(vocab, d)
        s.enc = nn.ModuleList([EncBlock(d, h) for _ in range(Le)])
        s.dec = nn.ModuleList([DecBlock(d, h) for _ in range(Ld)])
        s.enorm, s.dnorm = RMSNorm(d), RMSNorm(d)
        s.head = nn.Linear(d, vocab, bias=False); s.head.weight = s.emb.weight
        s.apply(s._init)
        for n, p in s.named_parameters():
            if n.endswith("o.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * (Le + Ld)))
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, 0.02)
    def encode(s, src):
        x = s.emb(src); cs = rope_cache(src.size(1), s.d // s.h, src.device)
        for b in s.enc: x = b(x, cs)
        return s.enorm(x)
    def decode(s, tgt, mem):
        x = s.emb(tgt); cs = rope_cache(tgt.size(1), s.d // s.h, tgt.device)
        for b in s.dec: x = b(x, cs, mem)
        return s.head(s.dnorm(x))
    def forward(s, src, tgt): return s.decode(tgt, s.encode(src))