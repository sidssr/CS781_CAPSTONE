from ed_model import *

def get_batch(data, B, E, D, dev):
    ix = np.random.randint(0, len(data) - E - D - 1, B)
    w = torch.from_numpy(np.stack([data[i:i + E + D].astype(np.int64) for i in ix]))
    return w[:, :E].to(dev), w[:, E - 1:E + D - 1].to(dev), w[:, E:E + D].to(dev)

@torch.no_grad()
def evaluate(model, val, a, dev, iters=40):
    model.eval(); tot = 0
    for _ in range(iters):
        src, tin, tout = get_batch(val, a.mb, a.enc_len, a.dec_len, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(src, tin)
        tot += F.cross_entropy(lg.float().view(-1, lg.size(-1)), tout.reshape(-1)).item()
    model.train(); return tot / iters

def load(name, ckpt="best.pt", dev="cuda"):
    ck = torch.load(f"/kaggle/working/{name}/{ckpt}", map_location=dev)
    a = ck["args"]
    m = EncDec(32000, a["d"], a["enc_layers"], a["dec_layers"], a["heads"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval(); return m, a