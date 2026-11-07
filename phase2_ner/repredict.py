import sys, os, json, zipfile
sys.path.insert(0, "/kaggle/working")
import torch
from tokenizers import Tokenizer
from ner_models import build_model
from ner_data import *

name = sys.argv[1]
out = f"/kaggle/working/ner_out/{name}"
cfg = json.load(open(f"{out}/cfg.json"))
tok = Tokenizer.from_file("/kaggle/working/tok32k/tokenizer.json")
test_docs = make_docs(tok, ld("Phase_2_LNER/Test/fold3-text.json", os.environ["HF_TOKEN"]))
model = build_model(cfg, NT)
model.load_state_dict(torch.load(f"{out}/best.pt", map_location="cpu"), strict=False)
model.to("cuda").eval()
pred = {d["id"]: [[s, e, l] for s, e, l in predict_spans(model, d, cfg["W"], "cuda")] for d in test_docs}
for k, v in pred.items():
    t = [tuple(x) for x in v]
    assert len(t) == len(set(t)), "duplicate"
    assert all(a[0] < a[1] for a in t), "empty span"
    assert all(t[i][1] <= t[i + 1][0] for i in range(len(t) - 1)), "overlap"
json.dump(pred, open(f"{out}/prediction_fixed.json", "w"))
z = f"/kaggle/working/ner_{name}_fixed.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(f"{out}/prediction_fixed.json", arcname="prediction.json")
print(name, "| docs", len(pred), "| spans", sum(len(v) for v in pred.values()), "| saved", z)