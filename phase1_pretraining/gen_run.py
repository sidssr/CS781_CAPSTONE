import sys, os, json, time, zipfile
sys.path.insert(0, "/kaggle/working")
from gen import *
from huggingface_hub import hf_hub_download

name = sys.argv[1]
p = hf_hub_download("Exploration-Lab/CS-781-Capstone", "Phase_1_Pretraining/Test/test.json",
                    repo_type="dataset", token=os.environ["HF_TOKEN"])
test = json.load(open(p, encoding="utf-8"))
keys = [str(k) for k in test.keys()]
prompts = [test[k]["text"] for k in test.keys()]
N = len(prompts)
order = sorted(range(N), key=lambda i: len(tok.encode(prompts[i]).ids))

part = f"/kaggle/working/partial_{name}.json"
done = json.load(open(part)) if os.path.exists(part) else {}
print(f"start: {len(done)}/{N} already done", flush=True)
model = load(name)
CH = 96; t0 = time.time(); n0 = len(done)
for s in range(0, N, CH):
    idxs = [i for i in order[s:s + CH] if str(i) not in done]
    if not idxs: continue
    outs = generate(model, [prompts[i] for i in idxs], max_new=220, bs=24)
    for i, t in zip(idxs, outs): done[str(i)] = t
    json.dump(done, open(part, "w"))
    el = time.time() - t0; fin = len(done) - n0
    eta = el / fin * (N - len(done)) / 60
    print(f"done {len(done)}/{N} | {el/60:.1f} min | eta {eta:.0f} min", flush=True)

d = f"/kaggle/working/sub_{name}"; os.makedirs(d, exist_ok=True)
pred = {keys[i]: {"text": done[str(i)]} for i in range(N)}
with open(f"{d}/prediction.json", "w", encoding="utf-8") as f:
    json.dump(pred, f, ensure_ascii=False)
z = f"/kaggle/working/submission_{name}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(f"{d}/prediction.json", arcname="prediction.json")
print("saved", z, flush=True)