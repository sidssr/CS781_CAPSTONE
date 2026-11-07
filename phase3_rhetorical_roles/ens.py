import sys, json, argparse, zipfile, itertools
import numpy as np
p = argparse.ArgumentParser()
p.add_argument("--runs", required=True); p.add_argument("--out", default="ens")
p.add_argument("--beta", type=float, default=-1); p.add_argument("--select", type=int, default=1)
a = p.parse_args()
W = "/kaggle/working"; DD = W + "/hf/Phase_3_RR"
LAB = json.load(open(DD + "/Train/label_vocab.json")); NL = len(LAB); INV = {v: k for k, v in LAB.items()}
names = a.runs.split(",")
Z = {n: np.load(f"{W}/probs_{n}.npz") for n in names}
z0 = Z[names[0]]
dids = sorted(k[4:] for k in z0.files if k.startswith("dev|") and not k.endswith("|y"))
tids = sorted(k[5:] for k in z0.files if k.startswith("test|"))
Y = np.concatenate([z0[f"dev|{i}|y"] for i in dids])
DP = {n: np.concatenate([Z[n][f"dev|{i}"] for i in dids]) for n in names}
cnt = np.zeros(NL)
for dom in ("CL", "IT"):
    for d in json.load(open(f"{DD}/Train/{dom}_train.json")):
        for l in d["labels"]: cnt[LAB[l]] += 1
def mf1(g, q):
    cm = np.zeros((NL, NL)); np.add.at(cm, (g, q), 1)
    tp = np.diag(cm); rs, cs = cm.sum(1), cm.sum(0); use = (rs + cs) > 0
    P = np.where(cs > 0, tp / np.maximum(cs, 1), 0); R = np.where(rs > 0, tp / np.maximum(rs, 1), 0)
    return (np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-9), 0))[use].mean()
def pred(pr, beta): return (pr * ((cnt + 1) ** -beta)).argmax(-1)
for n in names: print(f"{n:16s} dev F1 {mf1(Y, pred(DP[n], 0)):.4f}")
chosen = names
if a.select:
    cur = [max(names, key=lambda n: mf1(Y, pred(DP[n], 0)))]; best = mf1(Y, pred(DP[cur[0]], 0))
    while True:
        cand = [(mf1(Y, pred(np.mean([DP[m] for m in cur + [n]], 0), 0)), n) for n in names if n not in cur]
        if not cand: break
        f, n = max(cand)
        if f <= best + 1e-4: break
        cur.append(n); best = f
    chosen = cur
print("chosen:", chosen, "dev F1 (beta 0):", round(best if a.select else mf1(Y, pred(np.mean([DP[m] for m in chosen], 0), 0)), 4))
E = np.mean([DP[m] for m in chosen], 0)
scan = {b: mf1(Y, pred(E, b)) for b in (0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)}
print("beta scan:", {k: round(v, 4) for k, v in scan.items()})
beta = a.beta if a.beta >= 0 else max(scan, key=lambda b: (scan[b] - 0.003 * (b > 0), -b))
print("using beta", beta)
out = {}
for i in tids:
    pr = np.mean([Z[m][f"test|{i}"] for m in chosen], 0)
    out[i] = [INV[int(k)] for k in pred(pr, beta)]
for dom in ("CL", "IT"):
    for k, v in json.load(open(f"{DD}/Test/{dom}_test.json")).items(): assert len(out[k]) == len(v), k
json.dump(out, open(f"{W}/prediction_{a.out}.json", "w"))
with zipfile.ZipFile(f"{W}/p3_{a.out}.zip", "w", zipfile.ZIP_DEFLATED) as zf: zf.write(f"{W}/prediction_{a.out}.json", arcname="prediction.json")
print("wrote", f"{W}/p3_{a.out}.zip", "docs", len(out), "sents", sum(len(v) for v in out.values()))
