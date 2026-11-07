import sys, os, json, time, math, random, argparse, zipfile, glob, pickle, gc, re
sys.path.insert(0, "/kaggle/working")
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
import ner_models as nm, lm_model
from ed_model import EncDec, rope_cache

MB = os.environ.get("MB", "/kaggle/working/mb2/")
BASES = {"llama": MB + "llama_best.pt", "gpt2": MB + "gpt2_best.pt", "eff": MB + "eff_best.pt", "encdec": MB + "encdec_best.pt"}
DD = os.environ.get("DD6", "/kaggle/working/hf/Phase_6_Summ")
W = os.environ.get("W6", "/kaggle/working")
p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--kind", required=True)  # llama | gpt2 | eff | encdec
p.add_argument("--tech", default="lorma")  # full | lora | lorma | houlsby | pfeiffer | prompt
p.add_argument("--r", type=int, default=16); p.add_argument("--alpha", type=float, default=32)
p.add_argument("--targets", default=""); p.add_argument("--bn", type=int, default=64); p.add_argument("--nprompt", type=int, default=20)
p.add_argument("--keep", default=""); p.add_argument("--keep_enc", default=""); p.add_argument("--keep_dec", default="")
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--prompt_lr", type=float, default=1e-2)
p.add_argument("--H", type=int, default=256); p.add_argument("--T", type=int, default=384); p.add_argument("--S", type=int, default=192)
p.add_argument("--bs", type=int, default=4); p.add_argument("--accum", type=int, default=4)
p.add_argument("--budget_min", type=float, default=20); p.add_argument("--epochs", type=int, default=30); p.add_argument("--epoch_docs", type=int, default=2000)
p.add_argument("--val_n", type=int, default=150); p.add_argument("--gen_batch", type=int, default=16)
p.add_argument("--max_new", type=int, default=160); p.add_argument("--min_new", type=int, default=30); p.add_argument("--ngram", type=int, default=3)
p.add_argument("--teacher", default=""); p.add_argument("--kd_alpha", type=float, default=0.5); p.add_argument("--kd_T", type=float, default=2.0)
p.add_argument("--max_train", type=int, default=0); p.add_argument("--max_test", type=int, default=0); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
PAD, EOS = 0, 1

if not os.path.exists(DD):
    from huggingface_hub import snapshot_download
    snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset", allow_patterns=["Phase_6_Summ/*"], local_dir=W + "/hf")
tok = Tokenizer.from_file(glob.glob(os.environ.get("TOKGLOB", MB + "tokenizer*.json"), recursive=True)[0])

def rd(path):
    D = json.load(open(path))
    if isinstance(D, dict): D = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "document": v} for k, v in D.items()]
    return D
HC, TC = 4000, 8000
def enc_doc(text):
    if len(text) <= HC + TC: ids = tok.encode(text).ids
    else: ids = tok.encode(text[:HC]).ids[:512] + tok.encode(text[-TC:]).ids[-768:]
    return np.array(ids or [PAD], dtype=np.uint16)
CACHE = W + "/p6_tok.pkl"
if os.path.exists(CACHE):
    train, test = pickle.load(open(CACHE, "rb"))
else:
    train = []
    for x in rd(f"{DD}/Train/train.json"):
        train.append(dict(id=str(x["id"]), ids=enc_doc(x["document"]), summ=np.array(tok.encode(x["summary"]).ids[:400] or [PAD], dtype=np.uint16), ref=x["summary"]))
    test = [dict(id=str(x["id"]), ids=enc_doc(x["document"])) for x in rd(f"{DD}/Test/test.json")]
    pickle.dump((train, test), open(CACHE, "wb")); print("cached tokenized data", len(train), len(test), flush=True)
if os.environ.get("PREP_ONLY"): print("prep done", len(train), len(test)); sys.exit(0)
rng = random.Random(1234); order = list(range(len(train))); rng.shuffle(order)
val = [train[i] for i in order[:a.val_n]]; tr = [train[i] for i in order[a.val_n:]]
if a.max_train: tr = tr[:a.max_train]
if a.max_test: test = test[:a.max_test]
print(f"train {len(tr)} | val {len(val)} | test {len(test)}", flush=True)

# ---------------- model wrapper ----------------
class LoRALinear(nn.Module):
    def __init__(s, base, r, alpha, mode):
        super().__init__(); s.base, s.mode, s.scale = base, mode, alpha / r
        out, inn = base.weight.shape
        s.A = nn.Parameter(torch.empty(r, out if mode == "lorma" else inn)); nn.init.kaiming_uniform_(s.A, a=math.sqrt(5))
        s.B = nn.Parameter(torch.zeros(out, r))
    def forward(s, x):
        if s.mode == "lorma":
            h = s.base(x); return h + s.scale * F.linear(F.linear(h, s.A.to(h.dtype)), s.B.to(h.dtype))
        return s.base(x) + s.scale * F.linear(F.linear(x, s.A.to(x.dtype)), s.B.to(x.dtype))
class AdapterAfter(nn.Module):
    def __init__(s, inner, d, bn):
        super().__init__(); s.inner = inner; s.down = nn.Linear(d, bn); s.up = nn.Linear(bn, d)
        nn.init.zeros_(s.up.weight); nn.init.zeros_(s.up.bias)
    def forward(s, *args, **kw):
        y = s.inner(*args, **kw); return y + s.up(F.gelu(s.down(y)))

def lm_hidden(core, kind, idx, prompt=None):
    x = core.emb(idx); P = 0
    if prompt is not None:
        x = torch.cat([prompt[None].expand(len(x), -1, -1).to(x.dtype), x], 1); P = prompt.shape[0]
    T = x.shape[1]
    if kind == "gpt2": x = x + core.pos(torch.arange(T, device=idx.device)); cs = None
    else: cs = rope_cache(T, core.d // core.h, idx.device)
    for b in core.blocks: x = b(x, cs)
    return core.norm(x), P

class Summ(nn.Module):
    def __init__(s, kind, core, ctx, tech, nprompt):
        super().__init__(); s.kind, s.core, s.ctx = kind, core, ctx
        s.dec_only = kind != "encdec"
        if tech == "prompt":
            idx = torch.randint(0, 32000, (nprompt,)); s.prompt = nn.Parameter(core.emb.weight[idx].detach().clone())
        else: s.prompt = None
    def pad_docs(s, D):
        Ld = a.H + a.T
        if s.dec_only:  # left pad to fixed length
            x = torch.full((len(D), Ld), PAD, dtype=torch.long)
            for i, d in enumerate(D): x[i, Ld - len(d):] = torch.from_numpy(d.astype(np.int64))
        else:
            L = max(len(d) for d in D); x = torch.full((len(D), L), PAD, dtype=torch.long)
            for i, d in enumerate(D): x[i, :len(d)] = torch.from_numpy(d.astype(np.int64))
        return x.to(dev)
    def encode(s, src):
        c = s.core; x = c.emb(src)
        if s.prompt is not None: x = torch.cat([s.prompt[None].expand(len(x), -1, -1).to(x.dtype), x], 1)
        cs = rope_cache(x.shape[1], c.d // c.h, src.device)
        for b in c.enc: x = b(x, cs)
        return c.enorm(x)
    def forward(s, D, yin):
        src = s.pad_docs(D)
        if not s.dec_only: return s.core.decode(yin, s.encode(src))
        h, P = lm_hidden(s.core, s.kind, torch.cat([src, yin], 1), s.prompt)
        return s.core.head(h[:, P + src.shape[1]:])

def build(cfg):
    kind = cfg["kind"]; ck = torch.load(BASES[kind], map_location="cpu", weights_only=False); ar = ck["args"]
    if kind in ("llama", "gpt2"):
        core = lm_model.LM(ar["arch"], 32000, ar["d"], ar["layers"], ar["heads"], ar["ctx"]); core.load_state_dict(ck["model"])
    elif kind == "eff":
        core = nm.EffLM(32000, ar["d"], ar["layers"], ar["heads"], ar["kv_heads"], ar["window"], ar["global_every"]); core.load_state_dict(ck["model"])
    else:
        core = EncDec(32000, ar["d"], ar["enc_layers"], ar["dec_layers"], ar["heads"]); core.load_state_dict(ck["model"])
    ctx = ar.get("ctx", 1024) if kind != "encdec" else 10 ** 9
    if cfg.get("keep"):
        bl = list(core.blocks); core.blocks = nn.ModuleList([bl[i] for i in cfg["keep"]])
    if cfg.get("keep_enc"):
        bl = list(core.enc); core.enc = nn.ModuleList([bl[i] for i in cfg["keep_enc"]])
    if cfg.get("keep_dec"):
        bl = list(core.dec); core.dec = nn.ModuleList([bl[i] for i in cfg["keep_dec"]])
    tech = cfg["tech"]
    blocks = list(core.enc) + list(core.dec) if kind == "encdec" else list(core.blocks)
    d = core.emb.weight.shape[1]
    if tech != "full":
        for q in core.parameters(): q.requires_grad = False
        if tech in ("lora", "lorma"):
            tg = set((cfg.get("targets") or ("qkv,o,q,kv" if kind in ("encdec", "eff") else "qkv,o")).split(","))
            for blk in blocks:
                for parent in list(blk.modules()):
                    for nme, ch in list(parent.named_children()):
                        if isinstance(ch, nn.Linear) and nme in tg: setattr(parent, nme, LoRALinear(ch, cfg["r"], cfg["alpha"], tech))
        elif tech in ("houlsby", "pfeiffer"):
            names = ("attn", "cross", "mlp") if tech == "houlsby" else ("mlp",)
            for blk in blocks:
                for nme in names:
                    if hasattr(blk, nme): setattr(blk, nme, AdapterAfter(getattr(blk, nme), d, cfg["bn"]))
    return Summ(kind, core, ctx, tech, cfg["nprompt"] if tech == "prompt" else 0)

ints = lambda s_: [int(x) for x in s_.split(",")] if s_ else None
cfg = dict(vars(a)); cfg["keep"], cfg["keep_enc"], cfg["keep_dec"] = ints(a.keep), ints(a.keep_enc), ints(a.keep_dec)
model = build(cfg).to(dev)
need = a.H + a.T + a.S + a.nprompt + 1
assert model.dec_only is False or need <= model.ctx, f"sequence {need} exceeds model ctx {model.ctx}; reduce --H/--T/--S"
tr_params = [q for q in model.parameters() if q.requires_grad]
n_tr, n_all = sum(q.numel() for q in tr_params), sum(q.numel() for q in model.parameters())
print(f"{a.kind}/{a.tech}: trainable {n_tr/1e6:.2f}M of {n_all/1e6:.1f}M", flush=True)
teacher = None
if a.teacher:
    ct = json.load(open(f"{a.teacher}/cfg.json")); teacher = build(ct)
    teacher.load_state_dict(torch.load(f"{a.teacher}/best.pt", map_location="cpu"), strict=False); teacher.to(dev).eval()
    for q in teacher.parameters(): q.requires_grad = False
    print("teacher:", ct["kind"], ct["tech"], f"{sum(q.numel() for q in teacher.parameters())/1e6:.1f}M params", flush=True)

AC = dict(device_type=dev, dtype=torch.float16 if dev == "cuda" else torch.bfloat16, enabled=(dev == "cuda"))
def mkdoc(d):
    ids = d["ids"]
    return np.concatenate([ids[:a.H], ids[-a.T:]]) if len(ids) > a.H + a.T else ids
def batch(docs):
    D = [mkdoc(d) for d in docs]
    Y = [np.concatenate([d["summ"][:a.S - 1].astype(np.int64), [EOS]]) for d in docs]; L = max(len(y) for y in Y)
    y = torch.full((len(Y), L), -100, dtype=torch.long); yin = torch.full((len(Y), L), PAD, dtype=torch.long)
    for i, t in enumerate(Y):
        y[i, :len(t)] = torch.from_numpy(t); yin[i, 0] = EOS; yin[i, 1:len(t)] = torch.from_numpy(t[:-1])
    return D, yin.to(dev), y.to(dev)

core_ids = {id(q) for q in model.core.parameters()}
groups = [{"params": [q for q in tr_params if id(q) in core_ids], "lr": a.lr}, {"params": [q for q in tr_params if id(q) not in core_ids], "lr": a.prompt_lr}]
groups = [dict(g, base=g["lr"]) for g in groups if g["params"]]
opt = torch.optim.AdamW(groups, weight_decay=0.01); scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
out = f"{W}/p6_out/{a.name}"; os.makedirs(out, exist_ok=True); json.dump(cfg, open(f"{out}/cfg.json", "w"))
snap = lambda: {n: q.detach().cpu().clone() for n, q in model.named_parameters() if q.requires_grad}
budget = a.budget_min * 60; total = a.epochs * math.ceil(min(a.epoch_docs, len(tr)) / (a.bs * a.accum)); step = 0; t_start = time.time()
def prog(): return max(step / max(1, total), (time.time() - t_start) / budget)
def opt_step():
    m = min(1.0, (step + 1) / 15) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog()))))
    for g in opt.param_groups: g["lr"] = g["base"] * m
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr_params, 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

@torch.no_grad()
def val_loss():
    model.eval(); tot, n = 0.0, 0
    for s in range(0, len(val), 8):
        D, yin, y = batch(val[s:s + 8])
        with torch.autocast(**AC): lg = model(D, yin)
        tot += F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum").item(); n += int((y != -100).sum())
    return tot / n

best_v, best_sd, best_ep, hist = 1e9, None, 0, []
for ep in range(a.epochs):
    model.train(); ls = []; t0 = time.time(); pool = random.sample(tr, min(a.epoch_docs, len(tr))); nb = 0; stop = False
    for s in range(0, len(pool), a.bs):
        D, yin, y = batch(pool[s:s + a.bs])
        with torch.autocast(**AC): lg = model(D, yin)
        lg = lg.float(); mask = (y != -100)
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100)
        if teacher is not None:
            with torch.no_grad(), torch.autocast(**AC): tl = teacher(D, yin).float()
            T_ = a.kd_T
            kd = (F.kl_div(F.log_softmax(lg[mask] / T_, -1), F.softmax(tl[mask] / T_, -1), reduction="batchmean")) * T_ * T_
            loss = a.kd_alpha * loss + (1 - a.kd_alpha) * kd
        scaler.scale(loss / a.accum).backward(); ls.append(loss.item()); nb += 1
        if nb % a.accum == 0:
            opt_step(); step += 1
            if prog() >= 1: stop = True; break
    if nb % a.accum != 0 and not stop: opt_step(); step += 1
    v = val_loss(); hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(ls)), val_loss=v))
    print(f"epoch {ep+1} train {np.mean(ls):.4f} val_loss {v:.4f} ppl {math.exp(v):.1f} ({time.time()-t0:.0f}s, {(time.time()-t_start)/60:.1f}/{a.budget_min} min)", flush=True)
    if v < best_v: best_v, best_sd, best_ep = v, snap(), ep + 1
    if stop or prog() >= 1: break
train_min = (time.time() - t_start) / 60
model.load_state_dict(best_sd, strict=False); torch.save(best_sd, f"{out}/best.pt")

# ---------------- generation ----------------
@torch.no_grad()
def generate(docs, max_new, min_new):
    model.eval(); res = {}; idxs = sorted(range(len(docs)), key=lambda i: len(docs[i]["ids"]))
    for s in range(0, len(idxs), a.gen_batch):
        bi = idxs[s:s + a.gen_batch]; D = [mkdoc(docs[i]) for i in bi]; B = len(D)
        src = model.pad_docs(D); ys = torch.full((B, 1), EOS, dtype=torch.long, device=dev)
        mem = None
        with torch.autocast(**AC):
            if not model.dec_only: mem = model.encode(src)
        gen = [[] for _ in range(B)]; seen = [dict() for _ in range(B)]; done = [False] * B
        for t in range(max_new):
            with torch.autocast(**AC):
                if model.dec_only:
                    h, P = lm_hidden(model.core, model.kind, torch.cat([src, ys], 1), model.prompt); lg = model.core.head(h[:, -1])
                else: lg = model.core.decode(ys, mem)[:, -1]
            lg = lg.float(); lg[:, PAD] = -1e9
            if t < min_new: lg[:, EOS] = -1e9
            for b in range(B):
                g = gen[b]
                if len(g) >= a.ngram - 1:
                    for tkn in seen[b].get(tuple(g[-(a.ngram - 1):]), ()): lg[b, tkn] = -1e9
            nxt = lg.argmax(-1)
            for b in range(B):
                if done[b]: continue
                v = int(nxt[b])
                if v == EOS: done[b] = True; continue
                gen[b].append(v)
                if len(gen[b]) >= a.ngram: seen[b].setdefault(tuple(gen[b][-a.ngram:-1]), set()).add(v)
            ys = torch.cat([ys, nxt[:, None]], 1)
            if all(done): break
        for b, i in enumerate(bi): res[i] = tok.decode(gen[b])
    return [res[i] for i in range(len(docs))]

def words(s_): return re.findall(r"[a-z0-9]+", s_.lower())
def rouge(ref, hyp):
    r, h = words(ref), words(hyp)
    if not r or not h: return 0.0, 0.0
    ov = sum((Counter(r) & Counter(h)).values()); p1, r1 = ov / len(h), ov / len(r); f1 = 0 if ov == 0 else 2 * p1 * r1 / (p1 + r1)
    prev = [0] * (len(h) + 1)
    for x in r:
        cur = [0]
        for j, y in enumerate(h): cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    l = prev[-1]; pl, rl = l / len(h), l / len(r); fl = 0 if l == 0 else 2 * pl * rl / (pl + rl)
    return f1, fl

t1 = time.time(); vh = generate(val, a.max_new, a.min_new); gen_s = (time.time() - t1) / max(1, len(val))
sc = [rouge(v["ref"], h) for v, h in zip(val, vh)]
tri = lambda h: (lambda w: len(set(zip(w, w[1:], w[2:]))) / max(1, len(w) - 2))(words(h))
json.dump([dict(id=v["id"], ref=v["ref"], hyp=h) for v, h in zip(val, vh)], open(f"{out}/val_gen.json", "w"))
print(f"VAL ROUGE-1 {np.mean([s_[0] for s_ in sc]):.4f} ROUGE-L {np.mean([s_[1] for s_ in sc]):.4f} | avg words hyp {np.mean([len(words(h)) for h in vh]):.0f} ref {np.mean([len(words(v['ref'])) for v in val]):.0f} | distinct-3gram {np.mean([tri(h) for h in vh]):.3f}", flush=True)
print("SAMPLE:", vh[0][:400], flush=True)
th = generate(test, a.max_new, a.min_new)
pred = [{"id": d["id"], "summary": h} for d, h in zip(test, th)]; assert len(pred) == len(test)
json.dump(pred, open(f"{out}/prediction.json", "w"))
with zipfile.ZipFile(f"{W}/p6_{a.name}.zip", "w", zipfile.ZIP_DEFLATED) as zf: zf.write(f"{out}/prediction.json", arcname="prediction.json")
res = dict(name=a.name, kind=a.kind, tech=a.tech, trainable_M=n_tr / 1e6, total_M=n_all / 1e6, best_epoch=best_ep, best_val_loss=best_v, val_rouge1=float(np.mean([s_[0] for s_ in sc])),
           val_rougeL=float(np.mean([s_[1] for s_ in sc])), val_avg_words_hyp=float(np.mean([len(words(h)) for h in vh])), val_avg_words_ref=float(np.mean([len(words(v["ref"])) for v in val])),
           val_distinct3=float(np.mean([tri(h) for h in vh])), train_min=train_min, gen_s_per_doc=gen_s, peak_mem_GB=(torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0),
           H=a.H, T=a.T, S=a.S, max_new=a.max_new, history=hist, teacher=a.teacher or None)
json.dump(res, open(f"{out}/results.json", "w")); print("RESULT", json.dumps({k: v for k, v in res.items() if k != "history"}), flush=True)
