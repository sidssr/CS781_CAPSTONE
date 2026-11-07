import sys
sys.path.insert(0, "/kaggle/working")
from ed_lib import *

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="encdec")
    p.add_argument("--d", type=int, default=768); p.add_argument("--enc_layers", type=int, default=5)
    p.add_argument("--dec_layers", type=int, default=5); p.add_argument("--heads", type=int, default=12)
    p.add_argument("--enc_len", type=int, default=768); p.add_argument("--dec_len", type=int, default=256)
    p.add_argument("--tokens", type=float, default=120e6)
    p.add_argument("--mb", type=int, default=8); p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4); p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--data", default="/kaggle/working/data")
    a = p.parse_args()

    dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
    train = np.memmap(f"{a.data}/train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(f"{a.data}/val.bin", dtype=np.uint16, mode="r")
    model = EncDec(32000, a.d, a.enc_layers, a.dec_layers, a.heads).to(dev)
    print(f"encdec: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")
    tok_per_step = a.mb * a.accum * (a.enc_len + a.dec_len)
    total = int(a.tokens // tok_per_step)
    out = f"/kaggle/working/{a.name}"; os.makedirs(out, exist_ok=True)
    start, best = 0, 1e9
    if a.resume and os.path.exists(f"{out}/last.pt"):
        ck = torch.load(f"{out}/last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck["best"]; print("resumed at", start)

    def lr_at(s):
        if s < a.warmup: return a.lr * (s + 1) / a.warmup
        r = (s - a.warmup) / max(1, total - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * r)))

    t0 = time.time(); model.train()
    for step in range(start, total):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        for _ in range(a.accum):
            src, tin, tout = get_batch(train, a.mb, a.enc_len, a.dec_len, dev)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(src, tin)
            loss = F.cross_entropy(lg.float().view(-1, lg.size(-1)), tout.reshape(-1)) / a.accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0:
            el = time.time() - t0
            print(f"step {step+1}/{total} loss {loss.item()*a.accum:.3f} "
                  f"{(step+1-start)*tok_per_step/el:,.0f} tok/s", flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == total:
            vl = evaluate(model, val, a, dev)
            print(f"== step {step+1} val_loss {vl:.4f} ppl {math.exp(vl):.2f}", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1,
                        "best": min(best, vl), "args": vars(a)}, f"{out}/last.pt")
            if vl < best:
                best = vl; torch.save({"model": model.state_dict(), "args": vars(a)}, f"{out}/best.pt")
    print("DONE. best val loss", best)