import sys, os, json, time, math, random, argparse, zipfile, glob
sys.path.insert(0, "/kaggle/working")
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from ner_models import build_model, set_bidir
from ner_data import *

p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--kind", required=True)
p.add_argument("--base", required=True); p.add_argument("--tech", default="lorma")
p.add_argument("--r", type=int, default=16); p.add_argument("--alpha", type=float, default=32)
p.add_argument("--targets", default="qkv,o"); p.add_argument("--bn", type=int, default=64)
p.add_argument("--last_k", type=int, default=4); p.add_argument("--keep", default="")
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--head_lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=4); p.add_argument("--accum", type=int, default=2)
p.add_argument("--W", type=int, default=768); p.add_argument("--bidir", type=int, default=1)
p.add_argument("--o_weight", type=float, default=0.3)
p.add_argument("--teacher", default=""); p.add_argument("--kd_alpha", type=float, default=0.5)
p.add_argument("--kd_T", type=float, default=2.0)
p.add_argument("--nodev", type=int, default=0); p.add_argument("--max_docs", type=int, default=0)
p.add_argument("--seed", type=int, default=0); p.add_argument("--tok", default="")
a = p.parse_args()

dev = "cuda"; random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
TOKEN = os.environ["HF_TOKEN"]
tokp = a.tok or "/kaggle/working/tok32k/tokenizer.json"
if not os.path.exists(tokp): tokp = glob.glob("/kaggle/input/**/tokenizer*.json", recursive=True)[0]
tok = Tokenizer.from_file(tokp)

docs = []
for fold in ["fold1", "fold2"]:
    docs += make_docs(tok, ld(f"Phase_2_LNER/Train/{fold}-text.json", TOKEN), ld(f"Phase_2_LNER/Train/{fold}-gold.json", TOKEN))
test_docs = make_docs(tok, ld("Phase_2_LNER/Test/fold3-text.json", TOKEN))
dev_docs = [d for i, d in enumerate(docs) if i % 5 == 0]
train_docs = docs if a.nodev else [d for i, d in enumerate(docs) if i % 5 != 0]
if a.max_docs: train_docs, dev_docs = train_docs[:a.max_docs], dev_docs[:3]
print(f"train docs {len(train_docs)} | dev docs {len(dev_docs)} | test docs {len(test_docs)}", flush=True)

cfg = dict(vars(a)); cfg["keep"] = [int(x) for x in a.keep.split(",")] if a.keep else None
model = build_model(cfg, NT).to(dev)
tr = [q for q in model.parameters() if q.requires_grad]
n_tr, n_all = sum(q.numel() for q in tr), sum(q.numel() for q in model.parameters())
print(f"{a.kind}/{a.tech}: trainable {n_tr/1e6:.2f}M of {n_all/1e6:.1f}M params", flush=True)

teacher = None
if a.teacher:
    cfg_t = json.load(open(f"{a.teacher}/cfg.json"))
    teacher = build_model(cfg_t, NT)
    teacher.load_state_dict(torch.load(f"{a.teacher}/best.pt", map_location="cpu"), strict=False)
    teacher.to(dev).eval()
    for q in teacher.parameters(): q.requires_grad = False
    print("teacher loaded:", cfg_t["kind"], cfg_t["tech"], flush=True)

head_ids = {id(q) for q in model.head.parameters()}
groups = [{"params": [q for q in tr if id(q) in head_ids], "lr": a.head_lr},
          {"params": [q for q in tr if id(q) not in head_ids], "lr": a.lr}]
groups = [dict(g, base=g["lr"]) for g in groups if g["params"]]
opt = torch.optim.AdamW(groups, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
cw = torch.ones(NT, device=dev); cw[0] = a.o_weight

W = a.W
n_items = sum(max(1, math.ceil(len(d["ids"]) / W)) for d in train_docs)
total = a.epochs * math.ceil(n_items / (a.bs * a.accum)); warm = max(1, int(0.1 * total))
def mult(s):
    if s < warm: return (s + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.95 + 0.05

def batches():
    items = []
    for d in train_docs:
        n = len(d["ids"])
        for _ in range(max(1, math.ceil(n / W))):
            items.append((d, random.randint(0, max(0, n - W))))
    random.shuffle(items)
    for i in range(0, len(items), a.bs):
        ch = items[i:i + a.bs]
        x = torch.zeros(len(ch), W, dtype=torch.long); y = torch.full((len(ch), W), -100, dtype=torch.long)
        for j, (d, s) in enumerate(ch):
            xi, yi = d["ids"][s:s + W], d["tags"][s:s + W]
            x[j, :len(xi)] = torch.from_numpy(xi); y[j, :len(yi)] = torch.from_numpy(yi)
        yield x, y

def opt_step(step):
    for g in opt.param_groups: g["lr"] = g["base"] * mult(step)
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr, 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

out = f"/kaggle/working/ner_out/{a.name}"; os.makedirs(out, exist_ok=True)
json.dump(cfg, open(f"{out}/cfg.json", "w"))
snap = lambda: {n: q.detach().cpu().clone() for n, q in model.named_parameters() if q.requires_grad}
best_f, best_sd, best_ep, step = -1, None, 0, 0
t_start = time.time(); ep_times = []
for ep in range(a.epochs):
    model.train(); t0 = time.time(); ls = []; i = -1
    for i, (x, y) in enumerate(batches()):
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x)
        lg = lg.float()
        loss = F.cross_entropy(lg.view(-1, NT), y.view(-1), weight=cw, ignore_index=-100)
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                tl = teacher(x).float()
            m = (y != -100).float(); T = a.kd_T
            kd = F.kl_div(F.log_softmax(lg / T, -1), F.softmax(tl / T, -1), reduction="none").sum(-1)
            kd = (kd * m).sum() / m.sum().clamp(min=1) * T * T
            loss = a.kd_alpha * loss + (1 - a.kd_alpha) * kd
        scaler.scale(loss / a.accum).backward(); ls.append(loss.item())
        if (i + 1) % a.accum == 0: opt_step(step); step += 1
    if (i + 1) % a.accum != 0: opt_step(step); step += 1
    ep_times.append(time.time() - t0)
    if a.nodev:
        print(f"epoch {ep+1}/{a.epochs} loss {np.mean(ls):.4f} ({ep_times[-1]:.0f}s)", flush=True)
        best_sd, best_ep = snap(), ep + 1
    else:
        f, per = evaluate(model, dev_docs, W, dev)
        print(f"epoch {ep+1}/{a.epochs} loss {np.mean(ls):.4f} dev macroF1 {f:.4f} ({ep_times[-1]:.0f}s)", flush=True)
        if f > best_f: best_f, best_sd, best_ep, best_per = f, snap(), ep + 1, per

model.load_state_dict(best_sd, strict=False)
torch.save(best_sd, f"{out}/best.pt")
pred = {d["id"]: [[s, e, l] for s, e, l in sorted(predict_spans(model, d, W, dev))] for d in test_docs}
json.dump(pred, open(f"{out}/prediction.json", "w"))
z = f"/kaggle/working/ner_{a.name}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf: zf.write(f"{out}/prediction.json", arcname="prediction.json")
res = {"name": a.name, "kind": a.kind, "tech": a.tech, "trainable_M": n_tr / 1e6, "total_M": n_all / 1e6,
       "best_dev_macroF1": best_f, "best_epoch": best_ep, "per_label": None if a.nodev else best_per,
       "sec_per_epoch": float(np.mean(ep_times)), "total_min": (time.time() - t_start) / 60,
       "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9,
       "test_spans": sum(len(v) for v in pred.values())}
json.dump(res, open(f"{out}/results.json", "w"), indent=1)
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "per_label"}), flush=True)
if not a.nodev: print("per-label F1:", {k: round(v, 3) for k, v in best_per.items()})
print("saved", z)