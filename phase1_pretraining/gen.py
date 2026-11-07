import sys, os, json, zipfile, torch
sys.path.insert(0, "/kaggle/working")
from lm import LM
from tokenizers import Tokenizer

tok = Tokenizer.from_file("/kaggle/working/tok32k/tokenizer.json")
EOS = tok.token_to_id("<eos>")

def load(name, ckpt="best.pt", dev="cuda"):
    ck = torch.load(f"/kaggle/working/{name}/{ckpt}", map_location=dev)
    a = ck["args"]
    m = LM(a["arch"], 32000, a["d"], a["layers"], a["heads"], a["ctx"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m

@torch.no_grad()
def generate(model, prompts, max_new=256, rep_pen=1.1, ngram=4, temp=0.0,
             top_k=0, bs=8, dev="cuda"):
    ctx = model.ctx
    ids = [tok.encode(p).ids[-(ctx - max_new):] for p in prompts]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = [None] * len(ids)
    for s in range(0, len(order), bs):
        bidx = order[s:s + bs]
        L = min(len(ids[i]) for i in bidx)
        cur = torch.tensor([ids[i][-L:] for i in bidx], device=dev)
        gen = [[] for _ in bidx]
        ban = [{} for _ in bidx]
        for _ in range(max_new):
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(cur[:, -ctx:])[:, -1].float()
            lg[:, EOS] = -1e9
            for j in range(len(bidx)):
                if rep_pen != 1.0 and gen[j]:
                    ix = torch.tensor(list(set(gen[j])), device=dev)
                    v = lg[j, ix]
                    lg[j, ix] = torch.where(v > 0, v / rep_pen, v * rep_pen)
                if ngram and len(gen[j]) >= ngram - 1:
                    pre = tuple(gen[j][-(ngram - 1):]) if ngram > 1 else ()
                    for t in ban[j].get(pre, ()):
                        lg[j, t] = -1e9
            if temp > 0:
                lg = lg / temp
                if top_k:
                    kth = torch.topk(lg, top_k)[0][:, -1:]
                    lg[lg < kth] = -1e9
                nxt = torch.multinomial(torch.softmax(lg, -1), 1).squeeze(1)
            else:
                nxt = lg.argmax(-1)
            for j in range(len(bidx)):
                gen[j].append(nxt[j].item())
                if ngram and len(gen[j]) >= ngram:
                    ban[j].setdefault(tuple(gen[j][-ngram:-1]), set()).add(gen[j][-1])
            cur = torch.cat([cur, nxt[:, None]], 1)
        for j, i in enumerate(bidx):
            out[i] = " ".join(tok.decode(gen[j]).split()[:128])
    return out

def make_submission(name, ckpt="best.pt", **kw):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Exploration-Lab/CS-781-Capstone",
                        "Phase_1_Pretraining/Test/test.json",
                        repo_type="dataset", token=os.environ["HF_TOKEN"])
    test = json.load(open(p, encoding="utf-8"))
    keys = [str(k) for k in test.keys()]
    prompts = [test[k]["text"] for k in test.keys()]
    print("test prompts:", len(prompts))
    model = load(name, ckpt)
    preds = generate(model, prompts, **kw)
    d = f"/kaggle/working/sub_{name}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/prediction.json", "w", encoding="utf-8") as f:
        json.dump({k: {"text": t} for k, t in zip(keys, preds)}, f, ensure_ascii=False)
    z = f"/kaggle/working/submission_{name}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(f"{d}/prediction.json", arcname="prediction.json")
    print("saved", z)