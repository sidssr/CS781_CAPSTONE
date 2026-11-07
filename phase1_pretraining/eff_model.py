import sys
sys.path.insert(0, "/kaggle/working")
from ed_model import *
from ed_lib import _controls

_masks = {}
def local_mask(T, W, dev):
    k = (T, W, str(dev))
    if k not in _masks:
        i = torch.arange(T, device=dev)[:, None]
        j = torch.arange(T, device=dev)[None, :]
        _masks[k] = (j <= i) & (i - j < W)
    return _masks[k]

class EffAttn(nn.Module):
    def __init__(s, d, h, kvh, window):
        super().__init__()
        s.h, s.kvh, s.hd, s.window = h, kvh, d // h, window
        s.q = nn.Linear(d, d, bias=False)
        s.kv = nn.Linear(d, 2 * kvh * s.hd, bias=False)
        s.o = nn.Linear(d, d, bias=False)
    def forward(s, x, cs):
        B, T, D = x.shape
        q = s.q(x).view(B, T, s.h, s.hd).transpose(1, 2)
        k, v = s.kv(x).view(B, T, 2, s.kvh, s.hd).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cs[0][:T], cs[1][:T])
        k = apply_rope(k, cs[0][:T], cs[1][:T])
        rep = s.h // s.kvh
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        if s.window and s.window < T:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=local_mask(T, s.window, x.device))
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, D))

class EffBlock(nn.Module):
    def __init__(s, d, h, kvh, window):
        super().__init__()
        s.n1, s.n2 = RMSNorm(d), RMSNorm(d)
        s.attn = EffAttn(d, h, kvh, window)
        s.mlp = SwiGLU(d)
    def forward(s, x, cs):
        x = x + s.attn(s.n1(x), cs)
        return x + s.mlp(s.n2(x))

class EffLM(nn.Module):
    def __init__(s, vocab, d, L, h, kvh, window, global_every):
        super().__init__()
        s.d, s.h = d, h
        s.emb = nn.Embedding(vocab, d)
        s.blocks = nn.ModuleList([
            EffBlock(d, h, kvh, 0 if (i + 1) % global_every == 0 else window)
            for i in range(L)])
        s.norm = RMSNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
        s.head.weight = s.emb.weight
        s.apply(s._init)
        for n, p in s.named_parameters():
            if n.endswith("o.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * L))
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)
    def forward(s, idx):
        x = s.emb(idx)
        cs = rope_cache(idx.size(1), s.d // s.h, idx.device)
        for b in s.blocks:
            x = b(x, cs)
        return s.head(s.norm(x))

def get_batch(data, B, T, dev):
    ix = np.random.randint(0, len(data) - T - 1, B)
    x = torch.from_numpy(np.stack([data[i:i + T].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + T + 1].astype(np.int64) for i in ix]))
    return x.to(dev), y.to(dev)

@torch.no_grad()
def evaluate(model, val, a, dev, iters=40):
    model.eval()
    tot = 0
    for _ in range(iters):
        x, y = get_batch(val, a.mb, a.ctx, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x)
        tot += F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.view(-1)).item()
    model.train()
    return tot / iters

@torch.no_grad()
def generate(model, tok, prompts, ctx=1024, max_new=220, rep_pen=1.1, ngram=4, bs=16, dev="cuda"):
    EOS = tok.token_to_id("<eos>")
    ids = [tok.encode(p).ids[-(ctx - max_new):] for p in prompts]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = [None] * len(ids)
    for s in range(0, len(order), bs):
        bidx = order[s:s + bs]
        L = min(len(ids[i]) for i in bidx)
        cur = torch.tensor([ids[i][-L:] for i in bidx], device=dev)
        gen = [[] for _ in bidx]
        ban = [{} for _ in bidx]
        for _ in range(max_new):
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(cur[:, -ctx:])[:, -1].float()
            lg = _controls(lg, gen, ban, rep_pen, ngram, EOS)
            nxt = lg.argmax(-1)
            for j in range(len(bidx)):
                gen[j].append(nxt[j].item())
                if ngram and len(gen[j]) >= ngram:
                    ban[j].setdefault(tuple(gen[j][-ngram:-1]), set()).add(gen[j][-1])
            cur = torch.cat([cur, nxt[:, None]], 1)
        for j, i in enumerate(bidx):
            out[i] = " ".join(tok.decode(gen[j]).split()[:128])
    return out

def load(name, ckpt="best.pt", dev="cuda"):
    ck = torch.load(f"/kaggle/working/{name}/{ckpt}", map_location=dev)
    a = ck["args"]
    m = EffLM(32000, a["d"], a["layers"], a["heads"], a["kv_heads"], a["window"], a["global_every"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, a