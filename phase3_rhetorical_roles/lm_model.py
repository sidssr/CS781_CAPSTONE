import math
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

class Attn(nn.Module):
    def __init__(s, d, h, rope):
        super().__init__()
        s.h, s.rope = h, rope
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.o = nn.Linear(d, d, bias=False)
    def forward(s, x, cs):
        B, T, D = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        if s.rope:
            q = apply_rope(q, cs[0][:T], cs[1][:T]); k = apply_rope(k, cs[0][:T], cs[1][:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, D))

class SwiGLU(nn.Module):
    def __init__(s, d):
        super().__init__()
        hdn = int(8 * d / 3 / 64) * 64
        s.w1 = nn.Linear(d, hdn, bias=False); s.w3 = nn.Linear(d, hdn, bias=False)
        s.w2 = nn.Linear(hdn, d, bias=False)
    def forward(s, x): return s.w2(F.silu(s.w1(x)) * s.w3(x))

class GeluMLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.fc = nn.Linear(d, 4 * d, bias=False); s.proj = nn.Linear(4 * d, d, bias=False)
    def forward(s, x): return s.proj(F.gelu(s.fc(x)))

class Block(nn.Module):
    def __init__(s, d, h, arch):
        super().__init__()
        llama = arch == "llama"
        Norm = RMSNorm if llama else nn.LayerNorm
        s.n1, s.n2 = Norm(d), Norm(d)
        s.attn = Attn(d, h, rope=llama)
        s.mlp = SwiGLU(d) if llama else GeluMLP(d)
    def forward(s, x, cs):
        x = x + s.attn(s.n1(x), cs)
        return x + s.mlp(s.n2(x))

class LM(nn.Module):
    def __init__(s, arch, vocab, d, L, h, ctx, drop=0.0):
        super().__init__()
        s.arch, s.ctx, s.h, s.d = arch, ctx, h, d
        s.emb = nn.Embedding(vocab, d)
        s.pos = nn.Embedding(ctx, d) if arch == "gpt2" else None
        s.blocks = nn.ModuleList([Block(d, h, arch) for _ in range(L)])
        s.norm = RMSNorm(d) if arch == "llama" else nn.LayerNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
        s.head.weight = s.emb.weight
        s.apply(s._init)
        for n, p in s.named_parameters():
            if n.endswith("o.weight") or n.endswith("w2.weight") or n.endswith("proj.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * L))
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, 0.02)
    def forward(s, idx):
        B, T = idx.shape
        x = s.emb(idx)
        cs = None
        if s.arch == "gpt2":
            x = x + s.pos(torch.arange(T, device=idx.device))
        else:
            cs = rope_cache(T, s.d // s.h, idx.device)
        for b in s.blocks: x = b(x, cs)
        return s.head(s.norm(x))