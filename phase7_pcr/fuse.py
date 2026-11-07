# Score fusion of dense LM retrievers (sims7_*.npz) + TF-IDF (official data only). Weights tuned on dev.
import sys, os, json, glob, zipfile, itertools, pickle, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
W = os.environ.get("W7", "/kaggle/working"); DD = os.environ.get("DD7", W + "/hf/Phase_7_PCR/")
names = sys.argv[1:] or [os.path.basename(f)[6:-4] for f in sorted(glob.glob(f"{W}/sims7_*.npz")) if "tfidf" not in f]
def txt(x): return " ".join(x["text"]) if isinstance(x["text"], list) else x["text"]
def asl(d): return d if isinstance(d, list) else [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in d.items()]
def load(sub, name):
    q = asl(json.load(open(f"{DD}{sub}/{name}.json"))); c = asl(json.load(open(f"{DD}{sub}/{name}_candidates.json")))
    return q, c
def f1at7(top, rel):
    tp = sum(len(set(top[q]) & set(rel[q])) for q in top); P = tp / (7 * len(top)); R = tp / max(1, sum(len(rel[q]) for q in top))
    return 0 if tp == 0 else 2 * P * R / (P + R)
def z(S): return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-8)  # per-query standardisation
data = {n: np.load(f"{W}/sims7_{n}.npz") for n in names}
out = {}
for sp, sub, key in (("dev", "Dev", "dev"), ("test", "Test", "test")):
    q, c = load(sub, key); qid = [str(x["id"]) for x in q]; cid = [str(x["id"]) for x in c]
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=float(os.environ.get("MAXDF", 0.5)), ngram_range=(1, 1), max_features=200000, dtype=np.float32)
    C = vec.fit_transform([txt(x) for x in c]); Q = vec.transform([txt(x) for x in q]); T = (Q @ C.T).toarray()
    def align(n):
        d = data[n]; S = d[sp].astype(np.float32); qi = {k: i for i, k in enumerate(d[sp + "_q"])}; ci = {k: i for i, k in enumerate(d[sp + "_c"])}
        return S[np.ix_([qi[k] for k in qid], [ci[k] for k in cid])]
    out[sp] = dict(qid=qid, cid=cid, T=z(T), D={n: z(align(n)) for n in names})
rel = {str(x["id"]): [str(r) for r in x.get("relevant_candidates", [])] for x in load("Dev", "dev")[0]}
def top7(S, cid, qid): idx = np.argsort(-S, 1)[:, :7]; return {q: [cid[j] for j in idx[i]] for i, q in enumerate(qid)}
dv = out["dev"]; res = {}
print("single dev F1@7:", {n: round(f1at7(top7(dv["D"][n], dv["cid"], dv["qid"]), rel), 4) for n in names}, " tfidf:", round(f1at7(top7(dv["T"], dv["cid"], dv["qid"]), rel), 4))
dense = sum(dv["D"].values()) / len(names)
print("dense-ensemble:", round(f1at7(top7(dense, dv["cid"], dv["qid"]), rel), 4))
best = (-1, None)
for w in np.linspace(0, 1, 11):
    f = f1at7(top7((1 - w) * dense + w * dv["T"], dv["cid"], dv["qid"]), rel); print(f"  w_tfidf={w:.1f} dev F1@7 {f:.4f}")
    if f > best[0]: best = (f, w)
print("best", best)
def write(tag, S, sp="test"):
    o = out[sp]; pred = top7(S, o["cid"], o["qid"])
    assert all(len(set(v)) == 7 for v in pred.values())
    os.makedirs(f"{W}/p7_out/{tag}", exist_ok=True); json.dump(pred, open(f"{W}/p7_out/{tag}/prediction.json", "w"))
    with zipfile.ZipFile(f"{W}/p7_{tag}.zip", "w", zipfile.ZIP_DEFLATED) as zf: zf.write(f"{W}/p7_out/{tag}/prediction.json", arcname="prediction.json")
te = out["test"]; dense_t = sum(te["D"].values()) / len(names)
write("tfidf_only", te["T"]); write("dense_ens", dense_t); write("fusion", (1 - best[1]) * dense_t + best[1] * te["T"])
json.dump(dict(names=names, tfidf_dev=f1at7(top7(dv["T"], dv["cid"], dv["qid"]), rel), dense_ens_dev=f1at7(top7(dense, dv["cid"], dv["qid"]), rel), best_w=float(best[1]), fusion_dev=best[0]), open(f"{W}/p7_fusion_results.json", "w"))
print("wrote p7_tfidf_only.zip p7_dense_ens.zip p7_fusion.zip")
