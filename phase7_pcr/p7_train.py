import sys, os, json, time, math, random, argparse, zipfile, glob, pickle, gc
sys.path.insert(0, "/kaggle/working")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
import ner_models as nm, p3_models as pm

MB = os.environ.get("MB", "/kaggle/working/mb2/")
BASES = {"llama": MB + "llama_best.pt", "gpt2": MB + "gpt2_best.pt", "eff": MB + "eff_best.pt", "enc": MB + "encdec_best.pt"}
DD = os.environ.get("DD7", "/kaggle/working/hf/Phase_7_PCR/"); W = os.environ.get("W7", "/kaggle/working")
p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--kind", required=True)  # llama | gpt2 | eff | enc
p.add_argument("--tech", default="lorma")  # full | lorma | lora | ia3 | partial | prompt
p.add_argument("--r", type=int, default=16); p.add_argument("--alpha", type=float, default=32); p.add_argument("--targets", default="")
p.add_argument("--last_k", type=int, default=4); p.add_argument("--nprompt", type=int, default=20); p.add_argument("--keep", default="")
p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--head_lr", type=float, default=2e-3); p.add_argument("--prompt_lr", type=float, default=1e-2)
p.add_argument("--CL", type=int, default=256); p.add_argument("--K", type=int, default=4)  # train: chunks per doc
p.add_argument("--eval_chunks", type=int, default=24); p.add_argument("--mine_chunks", type=int, default=4)
p.add_argument("--bq", type=int, default=4); p.add_argument("--nneg", type=int, default=3); p.add_argument("--accum", type=int, default=2)
p.add_argument("--dim", type=int, default=256); p.add_argument("--tau", type=float, default=0.05)
p.add_argument("--budget_min", type=float, default=20); p.add_argument("--eval_every", type=int, default=150); p.add_argument("--mine_at", type=float, default=0.5)
p.add_argument("--teacher", default=""); p.add_argument("--kd_alpha", type=float, default=0.5)
p.add_argument("--max_q", type=int, default=0); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
PAD = 0

if not os.path.exists(DD):
    from huggingface_hub import snapshot_download
    snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset", allow_patterns=["Phase_7_PCR/*"], local_dir=W + "/hf")
tok = Tokenizer.from_file(glob.glob(MB + "tokenizer*.json")[0])

def load_split(sub, name):
    q = json.load(open(f"{DD}{sub}/{name}.json")); c = json.load(open(f"{DD}{sub}/{name}_candidates.json"))
    asl = lambda d: d if isinstance(d, list) else [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in d.items()]
    def enc(docs):
        out = {}
        for s in range(0, len(docs), 200):
            part = docs[s:s + 200]
            txt = [(" ".join(x["text"]) if isinstance(x["text"], list) else x["text"])[:90000] for x in part]
            for x, e in zip(part, tok.encode_batch(txt)): out[str(x["id"])] = np.array(e.ids[:16000] or [PAD], dtype=np.uint16)
        return out
    q, c = asl(q), asl(c)
    rel = {str(x["id"]): [str(r) for r in x.get("relevant_candidates", [])] for x in q}
    return dict(q=enc(q), c=enc(c), rel=rel)
CACHE = W + "/p7_tok.pkl"
if os.path.exists(CACHE): S = pickle.load(open(CACHE, "rb"))
else:
    S = {"train": load_split("Train", "train"), "dev": load_split("Dev", "dev"), "test": load_split("Test", "test")}
    pickle.dump(S, open(CACHE, "wb")); print("cached", {k: (len(v["q"]), len(v["c"])) for k, v in S.items()}, flush=True)
if os.environ.get("PREP_ONLY"): print("prep done", {k: (len(v["q"]), len(v["c"])) for k, v in S.items()}); sys.exit(0)
TR, DV, TE = S["train"], S["dev"], S["test"]
tq = [q for q in TR["q"] if any(r in TR["c"] for r in TR["rel"][q])]
if a.max_q: tq = tq[:a.max_q]
print(f"train queries {len(tq)} pool {len(TR['c'])} | dev q {len(DV['q'])} pool {len(DV['c'])} | test q {len(TE['q'])} pool {len(TE['c'])}", flush=True)

# ---------------- model ----------------
DEF_T = {"llama": "qkv,o", "gpt2": "qkv,o", "eff": "q,kv,o", "enc": "qkv,o"}
IA3_T = {"llama": "qkv,w1,w3,w2", "gpt2": "qkv,fc,proj", "eff": "kv,w1,w3,w2", "enc": "qkv,w1,w3,w2"}
class Retr(nn.Module):
    def __init__(s, enc, d, dim, tau):
        super().__init__(); s.enc, s.d = enc, d
        s.proj = nn.Linear(d, dim); s.logit_scale = nn.Parameter(torch.tensor(math.log(1 / tau)))
    def chunk_vecs(s, x, m):
        h = s.enc(x, m)[:, :s.d]  # mean-pooled part
        return F.normalize(s.proj(h.float()), dim=-1)
def build(cfg):
    core = nm.load_backbone(cfg["kind"], BASES[cfg["kind"]])
    if cfg.get("keep"):
        bl = list(nm.get_blocks(core)); nm.set_blocks(core, [bl[i] for i in cfg["keep"]])
    c2 = dict(cfg); c2["targets"] = cfg.get("targets") or (IA3_T if cfg["tech"] == "ia3" else DEF_T)[cfg["kind"]]
    pm.apply_tech2(core, c2)
    enc = pm.SentEnc(cfg["kind"], core, cfg["nprompt"] if cfg["tech"] == "prompt" else 0, 0)
    return Retr(enc, core.emb.weight.shape[1], cfg["dim"], cfg["tau"]), core
cfg = dict(vars(a)); cfg["keep"] = [int(x) for x in a.keep.split(",")] if a.keep else None
model, core = build(cfg); model = model.to(dev)
trp = [q for q in model.parameters() if q.requires_grad]
n_tr, n_all = sum(q.numel() for q in trp), sum(q.numel() for q in model.parameters())
print(f"{a.kind}/{a.tech}: trainable {n_tr/1e6:.2f}M of {n_all/1e6:.1f}M", flush=True)
teacher = None
if a.teacher:
    ct = json.load(open(f"{a.teacher}/cfg.json")); teacher, _ = build(ct)
    teacher.load_state_dict(torch.load(f"{a.teacher}/best.pt", map_location="cpu"), strict=False); teacher.to(dev).eval()
    for q in teacher.parameters(): q.requires_grad = False
    print("teacher:", ct["kind"], ct["tech"], f"{sum(q.numel() for q in teacher.parameters())/1e6:.1f}M params", flush=True)
AC = dict(device_type=dev, dtype=torch.float16 if dev == "cuda" else torch.bfloat16, enabled=(dev == "cuda"))

def chunks_of(ids, K, mode):
    CL = a.CL; n = max(1, len(ids) // CL)
    if mode == "rand": sel = random.sample(range(n), min(K, n))
    else: sel = sorted(set(int(round(i)) for i in np.linspace(0, n - 1, min(K, n))))
    return [ids[i * CL:(i + 1) * CL] for i in sel]
def pad_batch(chs):
    T = max(len(c) for c in chs); x = torch.zeros(len(chs), T, dtype=torch.long); m = torch.zeros(len(chs), T, dtype=torch.bool)
    for i, c in enumerate(chs): x[i, :len(c)] = torch.from_numpy(c.astype(np.int64)); m[i, :len(c)] = True
    return x.to(dev), m.to(dev)
def embed_docs(mdl, docs, K, mode, bs_chunks=96):
    """differentiable doc embeddings: mean of normalized chunk vectors, renormalized"""
    flat, owner = [], []
    for i, ids in enumerate(docs):
        cs = chunks_of(ids, K, mode); flat += cs; owner += [i] * len(cs)
    owner = torch.tensor(owner, device=dev); out = []
    for s in range(0, len(flat), bs_chunks):
        x, m = pad_batch(flat[s:s + bs_chunks])
        with torch.autocast(**AC): out.append(mdl.chunk_vecs(x, m))
    V = torch.cat(out); D = torch.zeros(len(docs), V.shape[1], device=dev, dtype=V.dtype).index_add(0, owner, V)
    return F.normalize(D, dim=-1)
@torch.no_grad()
def embed_all(mdl, ids_dict, K, mode, order=None):
    mdl.eval(); keys = list(ids_dict.keys()); E = []
    keys_sorted = sorted(keys, key=lambda k: len(ids_dict[k])); res = {}
    for s in range(0, len(keys_sorted), 16):
        ks = keys_sorted[s:s + 16]; e = embed_docs(mdl, [ids_dict[k] for k in ks], K, mode).float().cpu()
        for k, v in zip(ks, e): res[k] = v
    return res
def retrieve(mdl, split, K, ks=7):
    qe = embed_all(mdl, split["q"], K, "spread"); ce = embed_all(mdl, split["c"], K, "spread")
    qk, ck = list(qe), list(ce); sim = torch.stack([qe[k] for k in qk]) @ torch.stack([ce[k] for k in ck]).T
    top = sim.topk(min(ks, len(ck)), dim=1).indices.numpy()
    return qk, ck, sim.numpy(), {q: [ck[j] for j in top[i]] for i, q in enumerate(qk)}
def f1at7(pred, rel):
    tp = sum(len(set(pred[q]) & set(rel[q])) for q in pred); P = tp / (7 * len(pred)); R = tp / max(1, sum(len(rel[q]) for q in pred))
    return (0 if tp == 0 else 2 * P * R / (P + R)), P, R

def mine(mdl):
    mdl.eval(); qe = embed_all(mdl, {q: TR["q"][q] for q in tq}, a.mine_chunks, "spread"); ce = embed_all(mdl, TR["c"], a.mine_chunks, "spread")
    ck = list(ce); C = torch.stack([ce[k] for k in ck]); hn = {}
    for q in tq:
        s = (C @ qe[q]).numpy(); order = np.argsort(-s)[:60]; rel = set(TR["rel"][q])
        hn[q] = [ck[j] for j in order if ck[j] not in rel][:30]
    return hn

groups = [{"params": [q for q in trp if id(q) in {id(z) for z in core.parameters()}], "lr": a.lr},
          {"params": [q for q in trp if id(q) in {id(z) for z in model.enc.parameters()} and id(q) not in {id(z) for z in core.parameters()}], "lr": a.prompt_lr},
          {"params": [q for q in trp if id(q) not in {id(z) for z in model.enc.parameters()}], "lr": a.head_lr}]
groups = [dict(g, base=g["lr"]) for g in groups if g["params"]]
opt = torch.optim.AdamW(groups, weight_decay=0.01); scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
out = f"{W}/p7_out/{a.name}"; os.makedirs(out, exist_ok=True); json.dump(cfg, open(f"{out}/cfg.json", "w"))
snap = lambda: {n: q.detach().cpu().clone() for n, q in model.named_parameters() if q.requires_grad}
budget = a.budget_min * 60; t_start = time.time(); step = 0
def prog(): return (time.time() - t_start) / budget
def opt_step():
    m = min(1.0, (step + 1) / 15) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog()))))
    for g in opt.param_groups: g["lr"] = g["base"] * m
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trp, 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

z0 = retrieve(model, DV, a.eval_chunks); f0 = f1at7(z0[3], DV["rel"]); print(f"zero-shot (before training) dev F1@7 {f0[0]:.4f} P {f0[1]:.3f} R {f0[2]:.3f}", flush=True)
hn = mine(model); mined2 = False; pool_ids = list(TR["c"].keys())
best_f, best_sd, best_step, hist = -1, None, 0, []; ls = []; nb = 0
t_last = time.time()
while prog() < 1:
    model.train()
    qs = random.sample(tq, a.bq); qdocs, cdocs, tgt = [], [], []
    for qi, q in enumerate(qs):
        rel = [r for r in TR["rel"][q] if r in TR["c"]]; pos = random.choice(rel); tgt.append(len(cdocs)); cdocs.append(TR["c"][pos]); qdocs.append(TR["q"][q])
        cand = [c for c in hn.get(q, []) if c not in set(rel)] or pool_ids
        for n in random.sample(cand, min(a.nneg, len(cand))): cdocs.append(TR["c"][n])
    with torch.autocast(**AC): pass
    qv = embed_docs(model, qdocs, a.K, "rand"); cv = embed_docs(model, cdocs, a.K, "rand")
    sim = (qv.float() @ cv.float().T) * model.logit_scale.exp().clamp(max=100)
    loss = F.cross_entropy(sim, torch.tensor(tgt, device=dev))
    if teacher is not None:
        with torch.no_grad():
            tqv = embed_docs(teacher, qdocs, a.K, "rand"); tcv = embed_docs(teacher, cdocs, a.K, "rand")
            ts = (tqv.float() @ tcv.float().T) / a.tau
        ss = (qv.float() @ cv.float().T) / a.tau
        kd = F.kl_div(F.log_softmax(ss, -1), F.softmax(ts, -1), reduction="batchmean")
        loss = a.kd_alpha * loss + (1 - a.kd_alpha) * kd
    scaler.scale(loss / a.accum).backward(); ls.append(loss.item()); nb += 1
    if nb % a.accum == 0:
        opt_step(); step += 1
        if step % 25 == 0: print(f"step {step} loss {np.mean(ls[-50:]):.4f} ({(time.time()-t_start)/60:.1f}/{a.budget_min} min)", flush=True)
        if not mined2 and prog() >= a.mine_at: hn = mine(model); mined2 = True; print("re-mined hard negatives", flush=True)
        if step % a.eval_every == 0:
            z = retrieve(model, DV, a.eval_chunks); f = f1at7(z[3], DV["rel"]); hist.append(dict(step=step, loss=float(np.mean(ls[-50:])), f1=f[0], P=f[1], R=f[2]))
            print(f"== step {step} dev F1@7 {f[0]:.4f} P {f[1]:.3f} R {f[2]:.3f}", flush=True)
            if f[0] > best_f: best_f, best_sd, best_step = f[0], snap(), step
z = retrieve(model, DV, a.eval_chunks); f = f1at7(z[3], DV["rel"]); hist.append(dict(step=step, loss=float(np.mean(ls[-50:])), f1=f[0], P=f[1], R=f[2]))
print(f"== final step {step} dev F1@7 {f[0]:.4f} P {f[1]:.3f} R {f[2]:.3f}", flush=True)
if f[0] > best_f or best_sd is None: best_f, best_sd, best_step = f[0], snap(), step
train_min = (time.time() - t_start) / 60
model.load_state_dict(best_sd, strict=False); torch.save(best_sd, f"{out}/best.pt")
dqk, dck, dsim, dpred = retrieve(model, DV, a.eval_chunks); df1 = f1at7(dpred, DV["rel"])
t1 = time.time(); tqk, tck, tsim, tpred = retrieve(model, TE, a.eval_chunks); infer_s = (time.time() - t1) / (len(TE["q"]) + len(TE["c"]))
assert len(tpred) == len(TE["q"]) and all(len(set(v)) == 7 and all(c in TE["c"] for c in v) for v in tpred.values())
json.dump(tpred, open(f"{out}/prediction.json", "w"))
with zipfile.ZipFile(f"{W}/p7_{a.name}.zip", "w", zipfile.ZIP_DEFLATED) as zf: zf.write(f"{out}/prediction.json", arcname="prediction.json")
np.savez_compressed(f"{W}/sims7_{a.name}.npz", dev=dsim.astype(np.float16), dev_q=np.array(dqk), dev_c=np.array(dck), test=tsim.astype(np.float16), test_q=np.array(tqk), test_c=np.array(tck))
# per-K ablation of how many chunks of the document are used at inference
abl = {}
for K in (4, 8, 12, a.eval_chunks):
    zk = retrieve(model, DV, K); abl[K] = f1at7(zk[3], DV["rel"])[0]
# error analysis: how many of the 7 retrieved are relevant per query + recall of relevant docs within top-7
hits = [len(set(dpred[q]) & set(DV["rel"][q])) for q in dpred]
res = dict(name=a.name, kind=a.kind, tech=a.tech, trainable_M=n_tr / 1e6, total_M=n_all / 1e6, best_step=best_step, dev_F1at7=df1[0], dev_P=df1[1], dev_R=df1[2], zero_shot_F1=f0[0],
           chunks_ablation_dev=abl, queries_with_zero_hits=int(sum(h == 0 for h in hits)), mean_hits=float(np.mean(hits)), train_min=train_min,
           embed_s_per_doc=infer_s, peak_mem_GB=(torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0), history=hist, teacher=a.teacher or None)
json.dump(res, open(f"{out}/results.json", "w")); print("RESULT", json.dumps({k: v for k, v in res.items() if k != "history"}), flush=True)
