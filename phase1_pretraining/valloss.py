import os, sys, json, math, glob, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.environ.get("CODE","/kaggle/working"))
from tokenizers import Tokenizer
from lm_model import LM
MB = os.environ.get("MB", "/kaggle/working/mb2/")
tok = Tokenizer.from_file(glob.glob(MB + "tokenizer*.json")[0]); EOS = tok.token_to_id("<eos>")
if EOS is None and os.environ.get("TRAINJSON"): EOS = 1   # only for the fake-data test
assert EOS is not None, "tokenizer has no <eos>"; print("EOS id:", EOS)
src = os.environ.get("TRAINJSON")
if not src:
    from huggingface_hub import hf_hub_download
    src = hf_hub_download("Exploration-Lab/CS-781-Capstone", "Phase_1_Pretraining/Train/train_supreme_court.json", repo_type="dataset", token=os.environ["HF_TOKEN"])
records = json.load(open(src, encoding="utf-8")); print("docs:", len(records))
random.seed(42); idx = list(range(len(records))); random.shuffle(idx)   # same shuffle as prep.py
VAL_TARGET = int(os.environ.get("VAL_TARGET", 3_000_000)); chunks, n = [], 0
for p in range(0, len(idx), 50):                                         # every 50th document = validation
    arr = np.array(tok.encode(records[idx[p]]["text"]).ids + [EOS], dtype=np.int64); chunks.append(arr); n += len(arr)
    if n >= VAL_TARGET: break
val = np.concatenate(chunks); print("val tokens:", len(val), "(training log said 3,006,376)")
del records
dev = "cuda" if torch.cuda.is_available() else "cpu"; res = {}
for name in ("llama", "gpt2"):
    ck = torch.load(MB + f"{name}_best.pt", map_location="cpu", weights_only=False); a = ck["args"]
    m = LM(a["arch"], 32000, a["d"], a["layers"], a["heads"], a["ctx"]); m.load_state_dict(ck["model"]); m.to(dev).eval(); T = a["ctx"]
    starts = list(range(0, len(val) - T - 1, T)); tot = cnt = 0
    with torch.no_grad():
        for s in range(0, len(starts), 16):
            ix = starts[s:s + 16]
            x = torch.from_numpy(np.stack([val[i:i + T] for i in ix])).to(dev); y = torch.from_numpy(np.stack([val[i + 1:i + T + 1] for i in ix])).to(dev)
            with torch.autocast(device_type=dev, dtype=torch.float16 if dev == "cuda" else torch.bfloat16):
                lg = m(x)
            tot += F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.view(-1), reduction="sum").item(); cnt += y.numel()
    vl = tot / cnt; res[name] = dict(val_loss=vl, ppl=math.exp(vl), windows=len(starts)); print(name, f"val_loss {vl:.4f} ppl {math.exp(vl):.2f}", flush=True)
json.dump(res, open(os.environ.get("OUT", "/kaggle/working/p1_val_loss.json"), "w"), indent=1)
