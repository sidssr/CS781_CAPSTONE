import sys, os, json, time, math, random, argparse, zipfile, glob
sys.path.insert(0, "/kaggle/working")
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
import ner_models as nm, p3_models as pm

MB = "/kaggle/input/datasets/ssrathoria26/mymodel/"
BASES = {"llama": MB + "llama_best.pt", "gpt2": MB + "gpt2_best.pt", "eff": MB + "eff_best.pt", "enc": MB + "encdec_best.pt"}
DD = "/kaggle/working/hf/Phase_3_RR"
p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--kind", required=True)
p.add_argument("--base", default=""); p.add_argument("--tech", default="lorma")
p.add_argument("--r", type=int, default=16); p.add_argument("--alpha", type=float, default=32)
p.add_argument("--targets", default="qkv,o"); p.add_argument("--bn", type=int, default=64)
p.add_argument("--last_k", type=int, default=4); p.add_argument("--keep", default="")
p.add_argument("--nprompt", type=int, default=20)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--head_lr", type=float, default=2e-3)
p.add_argument("--prompt_lr", type=float, default=1e-2)
p.add_argument("--chunk", type=int, default=64); p.add_argument("--maxlen", type=int, default=80)
p.add_argument("--accum", type=int, default=2); p.add_argument("--ctx", type=int, default=1)
p.add_argument("--cw_pow", type=float, default=0.5); p.add_argument("--ls", type=float, default=0.05)
p.add_argument("--teacher", default=""); p.add_argument("--kd_alpha", type=float, default=0.5)
p.add_argument("--kd_T", type=float, default=2.0)
p.add_argument("--bidir", type=int, default=0); p.add_argument("--crf", type=int, default=0)
p.add_argument("--gru_layers", type=int, default=1); p.add_argument("--aux", type=float, default=0.5)
p.add_argument("--nodev", type=int, default=0); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda"; random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

if not os.path.exists(DD):
    from huggingface_hub import snapshot_download
    snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset", allow_patterns=["Phase_3_RR/*"], local_dir="/kaggle/working/hf")
tok = Tokenizer.from_file(glob.glob("/kaggle/input/**/tokenizer*.json", recursive=True)[0])
LAB = json.load(open(DD + "/Train/label_vocab.json")); NL = len(LAB); INV = {v: k for k, v in LAB.items()}
DOMS = ["CL", "IT"]
PROBS = {}

def mk(i, text, labs, dom):
    ids = [e.ids[:a.maxlen] or [0] for e in tok.encode_batch(text)]
    y = np.array([LAB[l] for l in labs]) if labs else None
    return dict(id=i, ids=ids, y=y, dom=DOMS.index(dom), n=len(text))
def load(dom, split):
    D = json.load(open(f"{DD}/Train/{dom}_{split}.json"))
    return [mk(x["id"], x["text"], x["labels"], dom) for x in D]
def load_test(dom):
    D = json.load(open(f"{DD}/Test/{dom}_test.json"))
    return [mk(k, v, None, dom) for k, v in D.items()]
train = load("CL", "train") + load("IT", "train")
dv = load("CL", "dev") + load("IT", "dev")
test = load_test("CL") + load_test("IT")
assert len({d["id"] for d in train + dv + test}) == len(train + dv + test), "duplicate doc ids"
if a.nodev: train = train + dv
print(f"train docs {len(train)} sents {sum(d['n'] for d in train)} | dev docs {len(dv)} | test docs {len(test)}", flush=True)

cnt = np.bincount(np.concatenate([d["y"] for d in train]), minlength=NL).astype(float)
cw = (np.maximum(cnt, 1) ** -a.cw_pow); cw = cw / (cw * cnt).sum() * cnt.sum()
cw = torch.tensor(cw, dtype=torch.float, device=dev)

def build(cfg):
    base = cfg.get("base") or BASES[cfg["kind"]]
    core = nm.load_backbone(cfg["kind"], base)
    if cfg.get("keep"):
        bl = list(nm.get_blocks(core)); nm.set_blocks(core, [bl[i] for i in cfg["keep"]])
    pm.apply_tech2(core, cfg)
    enc = pm.SentEnc(cfg["kind"], core, cfg["nprompt"] if cfg["tech"] == "prompt" else 0, cfg.get("bidir", 0))
    return pm.RRModel(enc, core.emb.weight.shape[1], NL, ctx=cfg["ctx"], layers=cfg.get("gru_layers", 1), crf=cfg.get("crf", 0))

def batch(d, s, e):
    ids = d["ids"][s:e]; N = len(ids); T = max(len(t) for t in ids)
    x = torch.zeros(N, T, dtype=torch.long); m = torch.zeros(N, T, dtype=torch.bool)
    for i, t in enumerate(ids): x[i, :len(t)] = torch.tensor(t); m[i, :len(t)] = True
    rel = torch.arange(s, e).float() / max(1, d["n"] - 1)
    return x.to(dev), m.to(dev), torch.full((N,), d["dom"], device=dev), rel.to(dev)

@torch.no_grad()
def predict(model, d):
    model.eval(); n, C = d["n"], a.chunk
    st = list(range(0, max(1, n - C + 1), C // 2))
    if st[-1] + C < n: st.append(n - C)
    pr = np.zeros((n, NL)); ct = np.zeros(n)
    for s in st:
        e = min(n, s + C)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(*batch(d, s, e))
        pb = model.marginals(lg) if model.crf else F.softmax(lg.float(), -1)
        pr[s:e] += pb.cpu().numpy(); ct[s:e] += 1
    pr = pr / ct[:, None]; PROBS[d["id"]] = pr
    return pr.argmax(-1)

def macro(g, p_):
    cm = np.zeros((NL, NL)); np.add.at(cm, (g, p_), 1)
    tp = np.diag(cm); rs, cs_ = cm.sum(1), cm.sum(0); use = (rs + cs_) > 0
    P = np.where(cs_ > 0, tp / np.maximum(cs_, 1), 0); R = np.where(rs > 0, tp / np.maximum(rs, 1), 0)
    Fm = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-9), 0)
    return Fm[use].mean(), P[use].mean(), R[use].mean(), {INV[i]: round(float(Fm[i]), 3) for i in range(NL) if use[i]}, cm

def evaluate(model, docs):
    out = {}; G, Pp = [], []
    for dom in (0, 1):
        g, q = [], []
        for d in docs:
            if d["dom"] == dom: g.append(d["y"]); q.append(predict(model, d))
        g, q = np.concatenate(g), np.concatenate(q); G.append(g); Pp.append(q)
        out[DOMS[dom]] = macro(g, q)
    out["ALL"] = macro(np.concatenate(G), np.concatenate(Pp))
    return out

cfg = dict(vars(a)); cfg["keep"] = [int(x) for x in a.keep.split(",")] if a.keep else None
model = build(cfg).to(dev)
tr = [q for q in model.parameters() if q.requires_grad]
n_tr, n_all = sum(q.numel() for q in tr), sum(q.numel() for q in model.parameters())
print(f"{a.kind}/{a.tech} ctx={a.ctx}: trainable {n_tr/1e6:.2f}M of {n_all/1e6:.1f}M", flush=True)

teacher = None
if a.teacher:
    ct_ = json.load(open(f"{a.teacher}/cfg.json")); teacher = build(ct_)
    teacher.load_state_dict(torch.load(f"{a.teacher}/best.pt", map_location="cpu"), strict=False)
    teacher.to(dev).eval()
    for q in teacher.parameters(): q.requires_grad = False
    print("teacher:", ct_["kind"], ct_["tech"], flush=True)

enc_ids = {id(q) for q in model.enc.core.parameters()}
pr_ids = {id(q) for q in model.enc.parameters() if id(q) not in enc_ids}
crf_ids = {id(q) for n, q in model.named_parameters() if n in ("trans", "start", "end")}
groups = [{"params": [q for q in tr if id(q) in crf_ids], "lr": a.head_lr * 5},
          {"params": [q for q in tr if id(q) in enc_ids], "lr": a.lr},
          {"params": [q for q in tr if id(q) in pr_ids], "lr": a.prompt_lr},
          {"params": [q for q in tr if id(q) not in enc_ids and id(q) not in pr_ids and id(q) not in crf_ids], "lr": a.head_lr}]
groups = [dict(g, base=g["lr"]) for g in groups if g["params"]]
opt = torch.optim.AdamW(groups, weight_decay=0.01); scaler = torch.amp.GradScaler("cuda")

C = a.chunk
def items():
    it = []
    for d in train:
        for _ in range(max(1, math.ceil(d["n"] / C))):
            s = random.randint(0, max(0, d["n"] - C)); it.append((d, s, min(d["n"], s + C)))
    random.shuffle(it); return it
n_items = sum(max(1, math.ceil(d["n"] / C)) for d in train)
total = a.epochs * math.ceil(n_items / a.accum); warm = max(1, int(0.1 * total))
def mult(s):
    if s < warm: return (s + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.95 + 0.05
def opt_step(step):
    for g in opt.param_groups: g["lr"] = g["base"] * mult(step)
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr, 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

out = f"/kaggle/working/p3_out/{a.name}"; os.makedirs(out, exist_ok=True)
json.dump(cfg, open(f"{out}/cfg.json", "w"))
snap = lambda: {n: q.detach().cpu().clone() for n, q in model.named_parameters() if q.requires_grad}
best_f, best_sd, best_ep, best_r, step, ep_times = -1, None, 0, None, 0, []
t_start = time.time()
for ep in range(a.epochs):
    model.train(); t0 = time.time(); ls = []; i = -1
    for i, (d, s, e) in enumerate(items()):
        inp = batch(d, s, e); y = torch.from_numpy(d["y"][s:e]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(*inp)
        lg = lg.float()
        ce = F.cross_entropy(lg, y, weight=cw, label_smoothing=a.ls)
        loss = (model.crf_nll(lg, y) / len(y) + a.aux * ce) if model.crf else ce
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                tl = teacher(*inp).float()
            T = a.kd_T
            kd = F.kl_div(F.log_softmax(lg / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * T * T
            loss = a.kd_alpha * loss + (1 - a.kd_alpha) * kd
        scaler.scale(loss / a.accum).backward(); ls.append(loss.item())
        if (i + 1) % a.accum == 0: opt_step(step); step += 1
    if (i + 1) % a.accum != 0: opt_step(step); step += 1
    ep_times.append(time.time() - t0)
    if a.nodev:
        print(f"epoch {ep+1}/{a.epochs} loss {np.mean(ls):.4f} ({ep_times[-1]:.0f}s)", flush=True)
        best_sd, best_ep = snap(), ep + 1
    else:
        r = evaluate(model, dv); f = r["ALL"][0]
        print(f"epoch {ep+1}/{a.epochs} loss {np.mean(ls):.4f} devF1 ALL {f:.4f} CL {r['CL'][0]:.4f} IT {r['IT'][0]:.4f} ({ep_times[-1]:.0f}s)", flush=True)
        if f > best_f: best_f, best_sd, best_ep, best_r = f, snap(), ep + 1, r

model.load_state_dict(best_sd, strict=False); torch.save(best_sd, f"{out}/best.pt")
pred = {}
for d in test:
    lab = [INV[int(k)] for k in predict(model, d)]
    assert len(lab) == d["n"]; pred[d["id"]] = lab
json.dump(pred, open(f"{out}/prediction.json", "w"))
sv = {}
for tag, docs in (("dev", dv), ("test", test)):
    for d in docs:
        predict(model, d); sv[f"{tag}|{d['id']}"] = PROBS[d["id"]].astype(np.float32)
        if d["y"] is not None: sv[f"{tag}|{d['id']}|y"] = d["y"]
np.savez_compressed(f"/kaggle/working/probs_{a.name}.npz", **sv)
z = f"/kaggle/working/p3_{a.name}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf: zf.write(f"{out}/prediction.json", arcname="prediction.json")
res = {"name": a.name, "kind": a.kind, "tech": a.tech, "ctx": a.ctx, "trainable_M": n_tr / 1e6, "total_M": n_all / 1e6,
       "best_epoch": best_ep, "sec_per_epoch": float(np.mean(ep_times)), "total_min": (time.time() - t_start) / 60,
       "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
if best_r:
    for k, v in best_r.items(): res[k] = dict(F1=v[0], P=v[1], R=v[2], per_label=v[3], cm=v[4].tolist())
json.dump(res, open(f"{out}/results.json", "w"))
print("RESULT", json.dumps({k: (v if not isinstance(v, dict) else {x: round(y, 4) for x, y in v.items() if x in ("F1", "P", "R")}) for k, v in res.items()}), flush=True)
if best_r: print("per-label (ALL):", best_r["ALL"][3])
