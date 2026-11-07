import sys, math
sys.path.insert(0, "/kaggle/working")
import torch, torch.nn as nn, torch.nn.functional as F
import ner_models as nm, lm_model
from lm_model import rope_cache, apply_rope

KEYMASK = [None]
BIDIR = [False]

def _mask():
    m = KEYMASK[0]
    return None if m is None else m[:, None, None, :]

def _attn(s, x, cs):
    B, T, D = x.shape
    q, k, v = s.qkv(x).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
    if s.rope:
        q = apply_rope(q, cs[0][:T], cs[1][:T]); k = apply_rope(k, cs[0][:T], cs[1][:T])
    if BIDIR[0]: y = F.scaled_dot_product_attention(q, k, v, attn_mask=_mask())
    else: y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return s.o(y.transpose(1, 2).reshape(B, T, D))
lm_model.Attn.forward = _attn

def _enc_attn(s, x, cs):
    B, T, D = x.shape
    q, k, v = s.qkv(x).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
    q = apply_rope(q, cs[0][:T], cs[1][:T]); k = apply_rope(k, cs[0][:T], cs[1][:T])
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=_mask())
    return s.o(y.transpose(1, 2).reshape(B, T, D))
nm.SelfAttn.forward = _enc_attn

def _eff_attn(s, x, cs):
    B, T, D = x.shape
    q = s.q(x).view(B, T, s.h, s.hd).transpose(1, 2)
    k, v = s.kv(x).view(B, T, 2, s.kvh, s.hd).permute(2, 0, 3, 1, 4)
    q = apply_rope(q, cs[0][:T], cs[1][:T]); k = apply_rope(k, cs[0][:T], cs[1][:T])
    rep = s.h // s.kvh
    k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
    if BIDIR[0]:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=_mask())
    elif s.window and s.window < T:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=nm.local_mask(T, s.window, x.device, False))
    else:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return s.o(y.transpose(1, 2).reshape(B, T, D))
nm.EffAttn.forward = _eff_attn

class LoRA2(nn.Module):
    def __init__(s, base, r, alpha, mode):
        super().__init__()
        s.base, s.mode = base, mode
        s.scale = alpha / math.sqrt(r) if mode == "rslora" else alpha / r
        out, inn = base.weight.shape
        s.A = nn.Parameter(torch.empty(r, out if mode == "lorma" else inn))
        nn.init.kaiming_uniform_(s.A, a=math.sqrt(5))
        s.B = nn.Parameter(torch.zeros(out, r))
    def forward(s, x):
        if s.mode == "lorma":
            h = s.base(x)
            return h + s.scale * F.linear(F.linear(h, s.A.to(h.dtype)), s.B.to(h.dtype))
        return s.base(x) + s.scale * F.linear(F.linear(x, s.A.to(x.dtype)), s.B.to(x.dtype))

class IA3(nn.Module):
    def __init__(s, base, where):
        super().__init__()
        s.base, s.where = base, where
        n = base.weight.shape[0] if where == "out" else base.weight.shape[1]
        s.l = nn.Parameter(torch.ones(n))
    def forward(s, x):
        if s.where == "out": return s.base(x) * s.l.to(x.dtype)
        return s.base(x * s.l.to(x.dtype))

class _Box: pass

def apply_tech2(core, cfg):
    tech = cfg["tech"]
    if tech == "full": return
    for p in core.parameters(): p.requires_grad = False
    blocks = nm.get_blocks(core)
    tg = set(cfg["targets"].split(","))
    if tech in ("lora", "rslora", "lorma", "ia3"):
        for blk in blocks:
            for parent in list(blk.modules()):
                for name, ch in list(parent.named_children()):
                    if isinstance(ch, nn.Linear) and name in tg:
                        if tech == "ia3": new = IA3(ch, "in" if name in ("w2", "proj") else "out")
                        else: new = LoRA2(ch, cfg["r"], cfg["alpha"], tech)
                        setattr(parent, name, new)
    elif tech == "prompt":
        pass
    else:
        w = _Box(); w.core = core; nm.apply_tech(w, cfg)

class SentEnc(nn.Module):
    def __init__(s, kind, core, nprompt=0, bidir=0):
        super().__init__()
        s.kind, s.core, s.P, s.bidir = kind, core, nprompt, bool(bidir)
        if nprompt:
            idx = torch.randint(0, core.emb.weight.shape[0], (nprompt,))
            s.prompt = nn.Parameter(core.emb.weight[idx].detach().clone())
    def forward(s, idx, mask):
        c, P = s.core, s.P
        N, T = idx.shape
        x = c.emb(idx)
        if P: x = torch.cat([s.prompt[None].expand(N, -1, -1).to(x.dtype), x], 1)
        T2 = x.shape[1]
        KEYMASK[0] = torch.cat([torch.ones(N, P, dtype=torch.bool, device=idx.device), mask], 1) if P else mask
        BIDIR[0] = s.bidir
        if s.kind == "enc":
            cs = rope_cache(T2, c.d // c.h, idx.device)
            for b in c.enc: x = b(x, cs)
            h = c.enorm(x)
        else:
            if s.kind == "gpt2":
                x = x + c.pos(torch.arange(T2, device=idx.device)); cs = None
            else:
                cs = rope_cache(T2, c.d // c.h, idx.device)
            for b in c.blocks: x = b(x, cs)
            h = c.norm(x)
        KEYMASK[0] = None
        h = h[:, P:]
        m = mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        last = h[torch.arange(N, device=h.device), mask.sum(1) - 1]
        return torch.cat([mean, last], -1)

class RRModel(nn.Module):
    def __init__(s, enc, d, nl, ctx=1, ds=384, ndom=2, drop=0.1, layers=1, crf=0):
        super().__init__()
        s.enc, s.ctx, s.crf = enc, ctx, crf
        s.proj = nn.Sequential(nn.Linear(2 * d, ds), nn.GELU(), nn.Dropout(drop))
        s.dom = nn.Embedding(ndom, ds)
        if ctx:
            s.posf = nn.Linear(9, ds)
            s.gru = nn.GRU(ds, ds // 2, num_layers=layers, batch_first=True, bidirectional=True,
                           dropout=drop if layers > 1 else 0.0)
        s.drop = nn.Dropout(drop)
        s.out = nn.Linear(2 * ds if ctx else ds, nl)
        if crf:
            s.trans = nn.Parameter(torch.zeros(nl, nl))
            s.start = nn.Parameter(torch.zeros(nl)); s.end = nn.Parameter(torch.zeros(nl))
    def forward(s, idx, mask, dom, rel):
        h = s.proj(s.enc(idx, mask)) + s.dom(dom)
        if s.ctx:
            k = torch.arange(1, 5, device=rel.device).float()
            f = torch.cat([rel[:, None], torch.sin(math.pi * rel[:, None] * k), torch.cos(math.pi * rel[:, None] * k)], -1)
            h = h + s.posf(f.to(h.dtype))
            g = s.gru(h[None])[0][0]
            h = torch.cat([h, g.to(h.dtype)], -1)
        return s.out(s.drop(h))
    def crf_nll(s, e, y):
        e = e.float(); N = len(y)
        sc = e[torch.arange(N, device=e.device), y].sum() + s.start[y[0]] + s.end[y[-1]]
        if N > 1: sc = sc + s.trans[y[:-1], y[1:]].sum()
        a = s.start + e[0]
        for i in range(1, N): a = torch.logsumexp(a[:, None] + s.trans, 0) + e[i]
        return torch.logsumexp(a + s.end, 0) - sc
    @torch.no_grad()
    def marginals(s, e):
        e = e.float(); N = e.shape[0]
        al = [s.start + e[0]]
        for i in range(1, N): al.append(torch.logsumexp(al[-1][:, None] + s.trans, 0) + e[i])
        be = [None] * N; be[-1] = s.end
        for i in range(N - 2, -1, -1): be[i] = torch.logsumexp(s.trans + (e[i + 1] + be[i + 1])[None, :], 1)
        return F.softmax(torch.stack(al) + torch.stack(be), -1)
