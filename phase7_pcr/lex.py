import sys, os, json, zipfile, itertools, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
W = os.environ.get("W7", "/kaggle/working"); DD = os.environ.get("DD7", W + "/hf/Phase_7_PCR/"); MAXDF = float(os.environ.get("MAXDF", 0.5))
names = sys.argv[1:]
def txt(x): return " ".join(x["text"]) if isinstance(x["text"], list) else x["text"]
def asl(d): return d if isinstance(d, list) else [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in d.items()]
def load(sub, name): return asl(json.load(open(f"{DD}{sub}/{name}.json"))), asl(json.load(open(f"{DD}{sub}/{name}_candidates.json")))
def f1at7(top, rel):
    tp = sum(len(set(top[q]) & set(rel[q])) for q in top); P = tp/(7*len(top)); R = tp/max(1, sum(len(rel[q]) for q in top))
    return 0 if tp == 0 else 2*P*R/(P+R)
def z(S): return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-8)
def top7(S, cid, qid): idx = np.argsort(-S, 1)[:, :7]; return {q: [cid[j] for j in idx[i]] for i, q in enumerate(qid)}
def bm25(Ctf, Qtf, k1, b, qmode):
    N = Ctf.shape[0]; df = np.asarray((Ctf > 0).sum(0)).ravel(); idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    dl = np.asarray(Ctf.sum(1)).ravel(); avg = dl.mean(); C = Ctf.tocoo()
    val = C.data * (k1 + 1) / (C.data + k1 * (1 - b + b * dl[C.row] / avg)); Cw = sp.csr_matrix((val, (C.row, C.col)), shape=Ctf.shape)
    Q = Qtf.copy().astype(np.float32); Q.data = (np.log1p(Q.data) if qmode == "log" else np.ones_like(Q.data))
    return (Q.multiply(idf[None, :]).tocsr() @ Cw.T).toarray()
D = {}; 
for s, sub in (("dev", "Dev"), ("test", "Test")):
    q, c = load(sub, s); D[s] = dict(qid=[str(x["id"]) for x in q], cid=[str(x["id"]) for x in c], qt=[txt(x) for x in q], ct=[txt(x) for x in c])
rel = {str(x["id"]): [str(r) for r in x.get("relevant_candidates", [])] for x in load("Dev", "dev")[0]}
_cache = {}
def lex(s, ng, k1, b, qmode):
    if (s, ng) not in _cache:
        d = D[s]; cv = CountVectorizer(ngram_range=ng, min_df=2, max_df=MAXDF, max_features=400000, dtype=np.float32)
        Ctf = cv.fit_transform(d["ct"]); _cache[(s, ng)] = (Ctf, cv.transform(d["qt"]))
    Ctf, Qtf = _cache[(s, ng)]; return bm25(Ctf, Qtf, k1, b, qmode)
res = []
for ng, k1, b, qm in itertools.product([(1, 1), (1, 2)], [0.9, 1.5, 2.5], [0.5, 0.75, 0.95], ["log", "bin"]):
    S = lex("dev", ng, k1, b, qm); f = f1at7(top7(S, D["dev"]["cid"], D["dev"]["qid"]), rel); res.append((f, ng, k1, b, qm)); print(f"ng{ng} k1={k1} b={b} q={qm} dev {f:.4f}", flush=True)
res.sort(key=lambda r: -r[0]); print("TOP5", res[:5]); _, ng, k1, b, qm = res[0]
Sd = lex("dev", ng, k1, b, qm); St = lex("test", ng, k1, b, qm)
def write(tag, S):
    o = D["test"]; pred = top7(S, o["cid"], o["qid"]); assert all(len(set(v)) == 7 for v in pred.values())
    os.makedirs(f"{W}/p7_out/{tag}", exist_ok=True); json.dump(pred, open(f"{W}/p7_out/{tag}/prediction.json", "w"))
    with zipfile.ZipFile(f"{W}/p7_{tag}.zip", "w", zipfile.ZIP_DEFLATED) as zf: zf.write(f"{W}/p7_out/{tag}/prediction.json", arcname="prediction.json")
write("bm25", St); print("wrote p7_bm25.zip")
if names:
    data = {n: np.load(f"{W}/sims7_{n}.npz") for n in names}
    def dense(s):
        d = D[s]; out = []
        for n in names:
            z_ = data[n]; qi = {k: i for i, k in enumerate(z_[s+"_q"])}; ci = {k: i for i, k in enumerate(z_[s+"_c"])}
            out.append(z(z_[s].astype(np.float32)[np.ix_([qi[k] for k in d["qid"]], [ci[k] for k in d["cid"]])]))
        return sum(out)/len(out)
    dd, dt = dense("dev"), dense("test"); best = (-1, 0)
    for w in np.linspace(0, 1, 21):
        f = f1at7(top7((1-w)*dd + w*z(Sd), D["dev"]["cid"], D["dev"]["qid"]), rel)
        if f > best[0]: best = (f, w)
    print(f"bm25 alone dev {res[0][0]:.4f} | fusion dev {best[0]:.4f} at w_bm25={best[1]:.2f}"); write("bm25_fusion", (1-best[1])*dt + best[1]*z(St)); print("wrote p7_bm25_fusion.zip")
