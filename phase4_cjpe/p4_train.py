import sys, os, json, time, math, random, argparse, zipfile, glob
sys.path.insert(0, "/kaggle/working")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
import ner_models as nm, p3_models as pm

MB = os.environ.get("MB", "/kaggle/input/datasets/ssrathoria26/mymodel/")
BASES = {"llama": MB + "llama_best.pt", "gpt2": MB + "gpt2_best.pt", "eff": MB + "eff_best.pt", "enc": MB + "encdec_best.pt"}
DD = os.environ.get("DD4", "/kaggle/working/hf/Phase_4_CJPE")
W = os.environ.get("W4", "/kaggle/working")
p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--kind", required=True)
p.add_argument("--tech", default="lorma")  # full | lorma | adapter | bitfit | partial | norm
p.add_argument("--r", type=int, default=16); p.add_argument("--alpha", type=float, default=32)
p.add_argument("--targets", default="qkv,o"); p.add_argument("--bn", type=int, default=64)
p.add_argument("--last_k", type=int, default=4); p.add_argument("--keep", default="")
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--head_lr", type=float, default=2e-3)
p.add_argument("--chunk", type=int, default=256); p.add_argument("--K", type=int, default=6)
p.add_argument("--sel", default="firstlast")  # firstlast | last | first | spread
p.add_argument("--bs", type=int, default=4); p.add_argument("--accum", type=int, default=4)
p.add_argument("--budget_min", type=float, default=25); p.add_argument("--epoch_docs", type=int, default=3000)
p.add_argument("--epochs", type=int, default=20); p.add_argument("--dev_max", type=int, default=1500)
p.add_argument("--cw_pow", type=float, default=1.0); p.add_argument("--ls", type=float, default=0.05)
p.add_argument("--teacher", default=""); p.add_argument("--kd_alpha", type=float, default=0.5); p.add_argument("--kd_T", type=float, default=2.0)
p.add_argument("--head_chars", type=int, default=4000); p.add_argument("--tail_chars", type=int, default=16000)
p.add_argument("--max_train", type=int, default=0); p.add_argument("--max_test", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

if not os.path.exists(DD):
    from huggingface_hub import snapshot_download
    snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset", allow_patterns=["Phase_4_CJPE/*"], local_dir=W + "/hf")
tok = Tokenizer.from_file(glob.glob(os.environ.get("TOKGLOB", "/kaggle/input/**/tokenizer*.json"), recursive=True)[0])

def rd(path):
    D = json.load(open(path))
    if isinstance(D, dict):
        D = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in D.items()]
    return D
def lab_of(x):
    for k in ("label", "labels", "y", "judgment", "judgement", "outcome"):
        if k in x: return int(x[k])
    return None
def mk(x, has_y):
    t = x["text"]
    if isinstance(t, list): t = " ".join(t)
    if len(t) > a.head_chars + a.tail_chars: t = t[:a.head_chars] + " " + t[-a.tail_chars:]
    return dict(id=str(x["id"]), text=t, y=lab_of(x) if has_y else None)
def _unused(docs):
    out = tok.encode_batch([d["text"] for d in docs])
    for d, e in zip(docs, out): d["ids"] = e.ids or [0]; del d["text"]
    return docs
import pickle, gc
CACHE = W + "/p4_tok.pkl"
if os.path.exists(CACHE):
    train, dv, test, te_n = pickle.load(open(CACHE, "rb"))
else:
    def conv(path, has_y):
        D = rd(path); out = []
        for s0 in range(0, len(D), 1000):
            part = [mk(x, has_y) for x in D[s0:s0 + 1000]]
            for d, e in zip(part, tok.encode_batch([d["text"] for d in part])): d["ids"] = np.array(e.ids or [0], dtype=np.uint16); del d["text"]
            out += part
        del D; gc.collect(); return out
    train = conv(f"{DD}/Train/train.json", True); dv = conv(f"{DD}/Train/dev.json", True); test = conv(f"{DD}/Test/test.json", False)
    te_n = len(test); pickle.dump((train, dv, test, te_n), open(CACHE, "wb"))
    print("cached tokenized data", flush=True)
if a.max_train: random.shuffle(train); train = train[:a.max_train]
if a.max_test: test = test[:a.max_test]
te_raw = test
if os.environ.get("PREP_ONLY"): print("prep done", len(train), len(dv), len(test)); sys.exit(0)
ytr = np.array([d["y"] for d in train]); cnt = np.bincount(ytr, minlength=2).astype(float)
print(f"train {len(train)} (class counts {cnt.tolist()}) | dev {len(dv)} | test {len(test)}", flush=True)
cw = np.maximum(cnt, 1) ** -a.cw_pow; cw = cw / (cw * cnt).sum() * cnt.sum()
cw = torch.tensor(cw, dtype=torch.float, device=dev)

def build(cfg):
    core = nm.load_backbone(cfg["kind"], BASES[cfg["kind"]])
    if cfg.get("keep"):
        bl = list(nm.get_blocks(core)); nm.set_blocks(core, [bl[i] for i in cfg["keep"]])
    t = cfg["tech"]
    if t == "bitfit":
        for q in core.parameters(): q.requires_grad = False
        for n, q in core.named_parameters():
            if n.endswith(".bias") or "norm" in n.split(".")[-2] or ".n1." in n or ".n2." in n or ".ln" in n: q.requires_grad = True
    elif t == "norm":
        nm.apply_tech(type("W", (), {"core": core})(), cfg)
    else:
        pm.apply_tech2(core, cfg)
    return DocModel(pm.SentEnc(cfg["kind"], core, 0, 0), core.emb.weight.shape[1], cfg["K"]), core

class DocModel(nn.Module):
    def __init__(s, enc, d, K, ds=384, drop=0.1):
        super().__init__()
        s.enc = enc
        s.proj = nn.Sequential(nn.Linear(2 * d, ds), nn.GELU(), nn.Dropout(drop))
        s.pos = nn.Embedding(64, ds)
        s.att = nn.Sequential(nn.Linear(ds, 128), nn.Tanh(), nn.Linear(128, 1))
        s.drop = nn.Dropout(drop); s.out = nn.Linear(2 * ds, 2)
    def forward(s, idx, mask, counts):
        h = s.proj(s.enc(idx, mask)); N = h.shape[0]
        pos = torch.cat([torch.arange(c) for c in counts]).to(h.device)
        h = h + s.pos(pos)
        outs, o = [], 0
        for c in counts:
            x = h[o:o + c]; o += c
            w = torch.softmax(s.att(x).float(), 0).to(x.dtype)
            outs.append(torch.cat([(w * x).sum(0), x.mean(0)]))
        return s.out(s.drop(torch.stack(outs)))

def pick(ids, mode, K, C):
    ch = [ids[i:i + C] for i in range(0, len(ids), C)]
    n = len(ch)
    if n <= K: return ch
    if mode == "last": return ch[-K:]
    if mode == "first": return ch[:K]
    if mode == "spread": return [ch[int(round(i))] for i in np.linspace(0, n - 1, K)]
    return ch[:1] + ch[-(K - 1):]

def batch(docs, mode, ctxmax):
    cs, counts = [], []
    for d in docs:
        c = pick(d["ids"], mode, a.K, ctxmax); cs.append(c); counts.append(len(c))
    flat = [t for c in cs for t in c]; T = max(len(t) for t in flat)
    x = torch.zeros(len(flat), T, dtype=torch.long); m = torch.zeros(len(flat), T, dtype=torch.bool)
    for i, t in enumerate(flat): x[i, :len(t)] = torch.from_numpy(np.asarray(t, dtype=np.int64)); m[i, :len(t)] = True
    return x.to(dev), m.to(dev), counts

cfg = dict(vars(a)); cfg["keep"] = [int(x) for x in a.keep.split(",")] if a.keep else None
model, core = build(cfg); model = model.to(dev)
CTX = a.chunk
if hasattr(core, "pos") and core.pos is not None: CTX = min(CTX, core.pos.weight.shape[0])
tr = [q for q in model.parameters() if q.requires_grad]
n_tr, n_all = sum(q.numel() for q in tr), sum(q.numel() for q in model.parameters())
print(f"{a.kind}/{a.tech}: trainable {n_tr/1e6:.2f}M of {n_all/1e6:.1f}M  chunk {CTX} K {a.K} sel {a.sel}", flush=True)

teacher = None
if a.teacher:
    ct_ = json.load(open(f"{a.teacher}/cfg.json")); teacher, _ = build(ct_)
    teacher.load_state_dict(torch.load(f"{a.teacher}/best.pt", map_location="cpu"), strict=False)
    teacher.to(dev).eval()
    for q in teacher.parameters(): q.requires_grad = False
    T_CTX = min(ct_["chunk"], CTX) if CTX < ct_["chunk"] else ct_["chunk"]
    n_teacher = sum(q.numel() for q in teacher.parameters())
    print("teacher:", ct_["kind"], ct_["tech"], f"{n_teacher/1e6:.1f}M params", flush=True)

core_ids = {id(q) for q in core.parameters()}
groups = [{"params": [q for q in tr if id(q) in core_ids], "lr": a.lr}, {"params": [q for q in tr if id(q) not in core_ids], "lr": a.head_lr}]
groups = [dict(g, base=g["lr"]) for g in groups if g["params"]]
opt = torch.optim.AdamW(groups, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
AC = dict(device_type=dev, dtype=torch.float16 if dev == "cuda" else torch.bfloat16, enabled=(dev == "cuda"))

@torch.no_grad()
def probs(mdl, docs, mode, ctx):
    mdl.eval(); out = []
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]["ids"])); res = {}
    for s in range(0, len(order), 8):
        idx = order[s:s + 8]; x, m, c = batch([docs[i] for i in idx], mode, ctx)
        with torch.autocast(**AC): lg = mdl(x, m, c)
        pr = F.softmax(lg.float(), -1)[:, 1].cpu().numpy()
        for i, v in zip(idx, pr): res[i] = float(v)
    return np.array([res[i] for i in range(len(docs))])

def mf1(g, q):
    f = []
    for c in (0, 1):
        tp = ((g == c) & (q == c)).sum(); P = tp / max(1, (q == c).sum()); R = tp / max(1, (g == c).sum())
        f.append(0 if P + R == 0 else 2 * P * R / (P + R))
    return float(np.mean(f))
def prf(g, q):
    P = [((g == c) & (q == c)).sum() / max(1, (q == c).sum()) for c in (0, 1)]; R = [((g == c) & (q == c)).sum() / max(1, (g == c).sum()) for c in (0, 1)]
    return dict(P_macro=float(np.mean(P)), R_macro=float(np.mean(R)), P=[float(x) for x in P], R=[float(x) for x in R])
def best_thr(g, pr):
    grid = np.linspace(0.05, 0.95, 91); sc = [mf1(g, (pr >= t).astype(int)) for t in grid]
    j = int(np.argmax(sc)); return float(grid[j]), float(sc[j])

random.seed(a.seed)
dv_eval = dv if len(dv) <= a.dev_max else random.sample(dv, a.dev_max)
gdv = np.array([d["y"] for d in dv_eval])
steps_per_epoch = math.ceil(min(a.epoch_docs, len(train)) / (a.bs * a.accum))
total = a.epochs * steps_per_epoch; budget = a.budget_min * 60
out = f"{W}/p4_out/{a.name}"; os.makedirs(out, exist_ok=True); json.dump(cfg, open(f"{out}/cfg.json", "w"))
snap = lambda: {n: q.detach().cpu().clone() for n, q in model.named_parameters() if q.requires_grad}
best_f, best_sd, best_ep, best_thr_, hist = -1, None, 0, 0.5, []
t_start = time.time(); step = 0; stop = False
def prog(): return max(step / max(1, total), (time.time() - t_start) / budget)
def mult():
    pg = min(1.0, prog()); w = 0.06
    return min(1.0, (step + 1) / 15) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * pg)))
def opt_step():
    for g in opt.param_groups: g["lr"] = g["base"] * mult()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr, 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

for ep in range(a.epochs):
    model.train(); ls = []; t0 = time.time()
    pool = random.sample(train, min(a.epoch_docs, len(train)))
    nb = 0
    for s in range(0, len(pool), a.bs):
        docs = pool[s:s + a.bs]; x, m, c = batch(docs, "firstlast" if a.sel == "firstlast" else a.sel, CTX)
        y = torch.tensor([d["y"] for d in docs], device=dev)
        with torch.autocast(**AC): lg = model(x, m, c)
        lg = lg.float(); loss = F.cross_entropy(lg, y, weight=cw, label_smoothing=a.ls)
        if teacher is not None:
            tx, tm, tc = batch(docs, a.sel, T_CTX)
            with torch.no_grad(), torch.autocast(**AC): tl = teacher(tx, tm, tc).float()
            T = a.kd_T
            kd = F.kl_div(F.log_softmax(lg / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * T * T
            loss = a.kd_alpha * loss + (1 - a.kd_alpha) * kd
        scaler.scale(loss / a.accum).backward(); ls.append(loss.item()); nb += 1
        if nb % a.accum == 0:
            opt_step(); step += 1
            if prog() >= 1: stop = True; break
    if nb % a.accum != 0 and not stop: opt_step(); step += 1
    pr = probs(model, dv_eval, a.sel, CTX); thr, f = best_thr(gdv, pr); f05 = mf1(gdv, (pr >= 0.5).astype(int))
    hist.append(dict(epoch=ep + 1, loss=float(np.mean(ls)), f1_at_0_5=f05, f1_tuned=f, thr=thr))
    print(f"epoch {ep+1} loss {np.mean(ls):.4f} dev macroF1 @0.5 {f05:.4f} tuned {f:.4f} (thr {thr:.2f}) ({time.time()-t0:.0f}s, {(time.time()-t_start)/60:.1f}/{a.budget_min} min)", flush=True)
    if f > best_f: best_f, best_sd, best_ep, best_thr_ = f, snap(), ep + 1, thr
    if stop or prog() >= 1: break

model.load_state_dict(best_sd, strict=False); torch.save(best_sd, f"{out}/best.pt")
pdv = probs(model, dv, a.sel, CTX); thr = best_thr_
gall = np.array([d["y"] for d in dv]); qdv = (pdv >= thr).astype(int)
t1 = time.time(); pte = probs(model, test, a.sel, CTX); infer_s = (time.time() - t1) / max(1, len(test))
pred = [{"id": i, "pred": int(v >= thr)} for i, v in zip([d["id"] for d in test], pte)]
assert len(pred) == len(te_raw) and all(p_["pred"] in (0, 1) for p_ in pred)
json.dump(pred, open(f"{out}/prediction.json", "w"))
np.savez_compressed(f"{W}/probs4_{a.name}.npz", dev=pdv, dev_y=gall, dev_ids=np.array([d["id"] for d in dv]), test=pte, test_ids=np.array([d["id"] for d in test]), thr=thr)
z = f"{W}/p4_{a.name}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf: zf.write(f"{out}/prediction.json", arcname="prediction.json")
# section / long-document ablation (same trained model, different chunk selections on dev)
abl = {}
sub = dv if len(dv) <= 800 else dv[:800]; gsub = np.array([d["y"] for d in sub])
for mode in ("firstlast", "last", "first", "spread"):
    pm_ = probs(model, sub, mode, CTX); abl[mode] = dict(f1_at_thr=mf1(gsub, (pm_ >= thr).astype(int)), f1_tuned=best_thr(gsub, pm_)[1])
res = dict(name=a.name, kind=a.kind, tech=a.tech, sel=a.sel, K=a.K, chunk=CTX, trainable_M=n_tr / 1e6, total_M=n_all / 1e6,
           best_epoch=best_ep, thr=thr, dev_macroF1=mf1(gall, qdv), dev_macroF1_at_0_5=mf1(gall, (pdv >= 0.5).astype(int)), **prf(gall, qdv),
           confusion=[[int(((gall == i) & (qdv == j)).sum()) for j in (0, 1)] for i in (0, 1)], train_class_counts=cnt.tolist(),
           test_pred_pos_rate=float(np.mean([p_["pred"] for p_ in pred])), dev_pos_rate=float(gall.mean()),
           train_min=(time.time() - t_start) / 60, infer_s_per_doc=infer_s, peak_mem_GB=(torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0),
           section_ablation_dev800=abl, history=hist, teacher=a.teacher or None)
json.dump(res, open(f"{out}/results.json", "w"))
print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("history",)}), flush=True)
