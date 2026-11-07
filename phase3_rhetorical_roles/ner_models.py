import sys
sys.path.insert(0, "/kaggle/working")
import torch, torch.nn as nn, torch.nn.functional as F
from lm_model import LM, RMSNorm, SwiGLU, rope_cache, apply_rope

class SelfAttn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
    def forward(s, x, cs):
        B, T, D = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cs[0][:T], cs[1][:T]); k = apply_rope(k, cs[0][:T], cs[1][:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return s.o(y.transpose(1, 2).reshape(B, T, D))

class EncBlock(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.n1, s.n2 = RMSNorm(d), RMSNorm(d)
        s.attn = SelfAttn(d, h); s.mlp = SwiGLU(d)
    def forward(s, x, cs):
        x = x + s.attn(s.n1(x), cs)
        return x + s.mlp(s.n2(x))

class EncOnly(nn.Module):
    def __init__(s, vocab, d, L, h):
        super().__init__(); s.d, s.h = d, h
        s.emb = nn.Embedding(vocab, d)
        s.enc = nn.ModuleList([EncBlock(d, h) for _ in range(L)])
        s.enorm = RMSNorm(d)

_masks = {}
def local_mask(T, W, dev, bidir):
    k = (T, W, str(dev), bidir)
    if k not in _masks:
        i = torch.arange(T, device=dev)[:, None]; j = torch.arange(T, device=dev)[None, :]
        _masks[k] = ((i - j).abs() < W) if bidir else ((j <= i) & (i - j < W))
    return _masks[k]

class EffAttn(nn.Module):
    def __init__(s, d, h, kvh, window):
        super().__init__()
        s.h, s.kvh, s.hd, s.window, s.bidir = h, kvh, d // h, window, False
        s.q = nn.Linear(d, d, bias=False)
        s.kv = nn.Linear(d, 2 * kvh * s.hd, bias=False)
        s.o = nn.Linear(d, d, bias=False)
    def forward(s, x, cs):
        B, T, D = x.shape
        q = s.q(x).view(B, T, s.h, s.hd).transpose(1, 2)
        k, v = s.kv(x).view(B, T, 2, s.kvh, s.hd).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cs[0][:T], cs[1][:T]); k = apply_rope(k, cs[0][:T], cs[1][:T])
        rep = s.h // s.kvh
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        if s.window and s.window < T:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=local_mask(T, s.window, x.device, s.bidir))
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=not s.bidir)
        return s.o(y.transpose(1, 2).reshape(B, T, D))

class EffBlock(nn.Module):
    def __init__(s, d, h, kvh, window):
        super().__init__()
        s.n1, s.n2 = RMSNorm(d), RMSNorm(d)
        s.attn = EffAttn(d, h, kvh, window); s.mlp = SwiGLU(d)
    def forward(s, x, cs):
        x = x + s.attn(s.n1(x), cs)
        return x + s.mlp(s.n2(x))

class EffLM(nn.Module):
    def __init__(s, vocab, d, L, h, kvh, window, global_every):
        super().__init__(); s.d, s.h = d, h
        s.emb = nn.Embedding(vocab, d)
        s.blocks = nn.ModuleList([EffBlock(d, h, kvh, 0 if (i + 1) % global_every == 0 else window)
                                  for i in range(L)])
        s.norm = RMSNorm(d)
        s.head = nn.Linear(d, vocab, bias=False); s.head.weight = s.emb.weight

def get_blocks(core): return core.blocks if hasattr(core, "blocks") else core.enc
def set_blocks(core, bl):
    if hasattr(core, "blocks"): core.blocks = nn.ModuleList(bl)
    else: core.enc = nn.ModuleList(bl)

def load_backbone(kind, path):
    ck = torch.load(path, map_location="cpu", weights_only=False); a = ck["args"]
    if kind in ("llama", "gpt2"):
        m = LM(a["arch"], 32000, a["d"], a["layers"], a["heads"], a["ctx"]); m.load_state_dict(ck["model"])
    elif kind == "eff":
        m = EffLM(32000, a["d"], a["layers"], a["heads"], a["kv_heads"], a["window"], a["global_every"])
        m.load_state_dict(ck["model"])
    elif kind == "enc":
        m = EncOnly(32000, a["d"], a["enc_layers"], a["heads"])
        m.load_state_dict({k: v for k, v in ck["model"].items() if k.startswith(("emb.", "enc.", "enorm."))})
    else:
        raise ValueError(kind)
    return m

class AdapterWrap(nn.Module):
    def __init__(s, inner, d, bn):
        super().__init__(); s.inner = inner
        s.down = nn.Linear(d, bn); s.up = nn.Linear(bn, d)
        nn.init.zeros_(s.up.weight); nn.init.zeros_(s.up.bias)
    def forward(s, x):
        y = s.inner(x); return y + s.up(F.gelu(s.down(y)))

def apply_tech(model, cfg):
    core, tech = model.core, cfg["tech"]
    if tech == "full": return
    for p in core.parameters(): p.requires_grad = False
    blocks = get_blocks(core)
    if tech == "adapter":
        d = core.emb.weight.shape[1]
        for blk in blocks: blk.mlp = AdapterWrap(blk.mlp, d, cfg["bn"])
    elif tech == "norm":
        for m in core.modules():
            if isinstance(m, (RMSNorm, nn.LayerNorm)):
                for p in m.parameters(): p.requires_grad = True
    elif tech == "partial":
        for blk in list(blocks)[-cfg["last_k"]:]:
            for p in blk.parameters(): p.requires_grad = True
        fn = core.norm if hasattr(core, "norm") else core.enorm
        for p in fn.parameters(): p.requires_grad = True
    else:
        raise ValueError(tech)
