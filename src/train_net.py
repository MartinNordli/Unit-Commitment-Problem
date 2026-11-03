"""ST-Net with LP inputs: one token per (unit, hour), dilated convolution over the week, attention across units,
one Transformer layer over the week. Same training recipe and validation folds as the LightGBM runs."""
import argparse, math, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from common import *
from features import build, group, CATEGORICAL
from lp_surrogate import PRIORITY, UNITS

IDX = {u: i for i, u in enumerate(UNITS)}; PAIRS = [(IDX[a], IDX[b]) for a, b in PRIORITY]
def priority(raw):
    z = list(raw.unbind(1))
    for par, ch in PAIRS: z[ch] = z[par] - F.softplus(z[ch])
    return torch.stack(z, 1)

class Mixer(nn.Module):
    def __init__(s, d, h, p):
        super().__init__(); s.n = nn.LayerNorm(d); s.a = nn.MultiheadAttention(d, h, dropout=p, batch_first=True); s.b = nn.Parameter(torch.zeros(14, 14)); s.dr = nn.Dropout(p)
    def forward(s, x):
        B, U, T, d = x.shape; h = s.n(x).permute(0, 2, 1, 3).reshape(B * T, U, d)
        h, _ = s.a(h, h, h, attn_mask=s.b, need_weights=False)
        return x + s.dr(h.reshape(B, T, U, d).permute(0, 2, 1, 3))

class TimeBlock(nn.Module):
    def __init__(s, d, dil, p):
        super().__init__(); s.n = nn.GroupNorm(1, d); s.c1 = nn.Conv1d(d, d, 3, padding=dil, dilation=dil); s.c2 = nn.Conv1d(d, d, 3, padding=dil, dilation=dil); s.f = nn.Linear(d, 2 * d); s.dr = nn.Dropout(p)
    def forward(s, x, c):
        B, U, T, d = x.shape; h = x.reshape(B * U, T, d).transpose(1, 2); z = s.c1(F.gelu(s.n(h)))
        g, b = s.f(c.reshape(B * U, d)).chunk(2, -1); z = z * (1 + g[..., None]) + b[..., None]; z = s.c2(s.dr(F.gelu(z)))
        return x + z.transpose(1, 2).reshape(B, U, T, d)

class Net(nn.Module):
    def __init__(s, nf, d=64, blocks=4, heads=4, p=.1):
        super().__init__(); s.inp = nn.Linear(nf, d); s.ue = nn.Parameter(torch.randn(14, d) * .02); s.pos = nn.Parameter(torch.randn(168, d) * .02)
        s.c = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))
        s.tb = nn.ModuleList(TimeBlock(d, 2 ** i, p) for i in range(blocks)); s.mx = nn.ModuleList(Mixer(d, heads, p) for _ in range(blocks))
        s.at = nn.TransformerEncoderLayer(d, heads, 4 * d, p, activation="gelu", batch_first=True, norm_first=True)
        s.on = nn.LayerNorm(d); s.h = nn.Linear(d, 1); s.lv = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
    def forward(s, X):
        B, U, T, _ = X.shape; x = s.inp(X) + s.ue[:, None] + s.pos; c = s.c(x.mean(2)) + s.ue
        for t, m in zip(s.tb, s.mx): x = m(t(x, c))
        d = x.shape[-1]; x = s.at(x.reshape(B * U, T, d)).reshape(B, U, T, d); x = s.on(x)
        return priority(s.h(x).squeeze(-1)), s.lv(torch.cat([x.mean(2), c], -1)).squeeze(-1)

def fit(Xtr, Ytr, seed, a, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(Xtr.shape[-1], a.d, a.blocks, 4, a.dropout).to(dev); ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(a.ema))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd); steps = a.epochs * math.ceil(len(Ytr) / a.batch); wu = max(1, int(.05 * steps))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1., (k + 1) / wu) * .5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    gen = torch.Generator().manual_seed(seed); scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda"); net.train()
    for ep in range(a.epochs):
        for idx in torch.randperm(len(Ytr), generator=gen).split(a.batch):
            idx = idx.to(dev); xb, yb = Xtr[idx], Ytr[idx]
            with torch.amp.autocast("cuda", enabled=dev == "cuda", dtype=torch.bfloat16):
                lg, lv = net(xb); loss = F.binary_cross_entropy_with_logits(lg.float(), yb) + a.aux * F.binary_cross_entropy_with_logits(lv.float(), yb.mean(-1))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.); scaler.step(opt); scaler.update(); sch.step(); ema.update_parameters(net)
    ema.eval(); return ema

@torch.no_grad()
def predict(m, X, dev, bs=64):
    out = []
    for i in range(0, len(X), bs):
        with torch.amp.autocast("cuda", enabled=dev == "cuda", dtype=torch.bfloat16): out.append(m(X[i:i + bs].to(dev))[0].float().cpu().numpy())
    return np.concatenate(out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); A = p.add_argument
    A("--name"); A("--lp", default="lp_v2"); A("--groups", default="base,lp,lps"); A("--drop", default=""); A("--seeds", default="0"); A("--folds", default=""); A("--forward", action="store_true"); A("--test", action="store_true")
    A("--epochs", type=int, default=24); A("--batch", type=int, default=32); A("--lr", type=float, default=1e-3); A("--wd", type=float, default=.01); A("--ema", type=float, default=.995); A("--aux", type=float, default=.3); A("--d", type=int, default=64); A("--blocks", type=int, default=4); A("--dropout", type=float, default=.1); A("--shuffle", action="store_true"); A("--test-only", action="store_true")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; print(dev, vars(a), flush=True)
    c = load_cases(); z = np.load(f"{ROOT}/artifacts/lp/{a.lp}.npz"); X, names = build(c, z)
    keep = [i for i, n in enumerate(names) if group(n) in a.groups.split(",") and n not in a.drop.split(",") and n not in CATEGORICAL]; X = X[..., keep]; names = [names[i] for i in keep]; print(len(names), "features", flush=True)
    Y = c["Y"].astype(np.float32); Fo = forward_folds() if a.forward else folds()
    if a.shuffle: Y = Y[np.random.default_rng(0).permutation(len(Y))]
    if a.folds: Fo = {k: v for k, v in Fo.items() if str(k) in a.folds.split(",")}
    seeds = [int(s) for s in a.seeds.split(",")]; oof = np.full((2922, 14, 168), np.nan, np.float32)
    def scaled(tr):
        m = X[tr].mean((0, 1, 2)); s = X[tr].std((0, 1, 2)); s = np.where(s > 1e-6, s, 1.)
        return lambda x: torch.from_numpy(np.clip((x - m) / s, -6, 6).astype(np.float32))
    for k, (tr, va) in ({} if a.test_only else Fo).items():
        t0 = time.time(); sc = scaled(tr); Xtr = sc(X[tr]).to(dev); Ytr = torch.from_numpy(Y[tr]).to(dev); Xva = sc(X[va]); ps = []
        for s in seeds: ps.append(predict(fit(Xtr, Ytr, s, a, dev), Xva, dev))
        oof[va] = np.mean(ps, 0); print(k, "auc", round(auc(Y[va], oof[va]), 5), "seeds", [round(auc(Y[va], q), 5) for q in ps], round(time.time() - t0), "s", flush=True); del Xtr, Ytr
    if not a.test_only and not a.forward and not a.shuffle: np.save(f"{ROOT}/artifacts/oof/{a.name}.npy", oof)
    if a.forward: np.save(f"{ROOT}/artifacts/fwd/{a.name}.npy", oof)
    if not a.test_only: report(a.name, oof, Y, c, extra={"n_features": len(names), "seeds": a.seeds, "forward": a.forward, "epochs": a.epochs, "d": a.d})
    if a.test:
        tr = np.arange(2922); sc = scaled(tr); Xtr = sc(X[tr]).to(dev); Ytr = torch.from_numpy(Y[tr]).to(dev); Xte = sc(X[2922:])
        np.save(f"{ROOT}/artifacts/preds/{a.name}.npy", np.mean([predict(fit(Xtr, Ytr, s, a, dev), Xte, dev) for s in seeds], 0).astype(np.float32))
