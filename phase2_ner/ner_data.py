import json, math, collections
import numpy as np, torch
from huggingface_hub import hf_hub_download

R = "Exploration-Lab/CS-781-Capstone"
LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
TAGS = ["O"] + [p + l for l in LABELS for p in ("B-", "I-")]
T2I = {t: i for i, t in enumerate(TAGS)}
NT = len(TAGS)

def ld(f, token):
    return json.load(open(hf_hub_download(R, f, repo_type="dataset", token=token), encoding="utf-8"))

def clean_offsets(t, enc):
    out = []
    for s, e in enc.offsets:
        while s < e and t[s].isspace(): s += 1
        while e > s and t[e - 1].isspace(): e -= 1
        out.append((s, e))
    return out

def to_tags(offs, gold):
    tags = ["O"] * len(offs)
    for s, e, l, *_ in gold:
        first = True
        for i, (ts, te) in enumerate(offs):
            if ts < te and ts < e and te > s:
                tags[i] = ("B-" if first else "I-") + l
                first = False
    return tags

def make_docs(tok, text, gold=None):
    docs = []
    for k, t in text.items():
        enc = tok.encode(t)
        offs = clean_offsets(t, enc)
        d = {"id": k, "text": t, "ids": np.array(enc.ids, dtype=np.int64), "offs": offs}
        if gold is not None:
            d["tags"] = np.array([T2I[x] for x in to_tags(offs, gold[k])], dtype=np.int64)
            d["gold"] = set((s, e, l) for s, e, l, *_ in gold[k])
        docs.append(d)
    return docs

def windows(n, W, stride):
    if n <= W: return [0]
    st = list(range(0, n - W + 1, stride))
    if st[-1] != n - W: st.append(n - W)
    return st

def decode(offs, tags):
    spans, cur = [], None
    for (ts, te), g in zip(offs, tags):
        if ts == te: continue
        if g == "O":
            if cur: spans.append(tuple(cur)); cur = None
        elif g.startswith("B-") or cur is None or cur[2] != g[2:]:
            if cur: spans.append(tuple(cur))
            cur = [ts, te, g[2:]]
        else:
            cur[1] = te
    if cur: spans.append(tuple(cur))
    return spans

def macro_f1(G, P):
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    for g, p in zip(G, P):
        for x in p: (tp if x in g else fp)[x[2]] += 1
        for x in g:
            if x not in p: fn[x[2]] += 1
    f1 = {}
    for l in LABELS:
        pr = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0
        rc = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0
        f1[l] = 2 * pr * rc / (pr + rc) if pr + rc else 0
    return sum(f1.values()) / len(f1), f1

@torch.no_grad()
def predict_probs(model, doc, W, stride, dev, bs=8):
    model.eval()
    n = len(doc["ids"]); L = min(W, n); st = windows(n, W, stride)
    acc = torch.zeros(n, NT); cnt = torch.zeros(n, 1)
    for i in range(0, len(st), bs):
        chunk = st[i:i + bs]
        x = torch.stack([torch.from_numpy(doc["ids"][s:s + L]) for s in chunk]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x)
        p = lg.float().softmax(-1).cpu()
        for j, s in enumerate(chunk):
            acc[s:s + L] += p[j]; cnt[s:s + L] += 1
    return acc / cnt

def predict_spans(model, doc, W, dev):
    pr = predict_probs(model, doc, W, W // 2, dev)
    return decode(doc["offs"], [TAGS[i] for i in pr.argmax(-1).tolist()])

def evaluate(model, docs, W, dev):
    G = [d["gold"] for d in docs]
    P = [set(predict_spans(model, d, W, dev)) for d in docs]
    return macro_f1(G, P)