"""Stress tests, re-solving the LP on the perturbed inputs. Trained strictly before 2022, tested on 2022. Labels are kept, so
the change is a measure of sensitivity, not a true score. Same perturbations as A004 and A008 in the earlier project."""
import os, sys, time, json, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, xgboost as xgb, torch
from common import *
from lp_surrogate import Topology, WeekLP
from features import build, group
SCALE = open(f"{ROOT}/artifacts/scale_cols.txt").read().strip().split(",")
_g = {}
def init(): _g["lp"] = WeekLP(Topology(f"{ROOT}/data/extended/Tokke_Vinje_topology.yaml"))
def solve(a):
    pr, q, v, w, mf, mv = a
    r = _g["lp"].solve(pr.astype(float), {n: q[:, k].astype(float) for k, n in enumerate(INFLOW)}, {n: float(v[k]) for k, n in enumerate(RES)}, {n: float(w[k]) for k, n in enumerate(RES)}, {n: mf[:, k].astype(float) for k, n in enumerate(MINFLOW)}, {n: mv[:, k].astype(float) for k, n in enumerate(MINVOL)}, convention="telescoping")
    return {k: r[k] for k in ("u", "pfrac", "V", "dual", "conv", "flows")}
if __name__ == "__main__":
    c = load_cases(); Y = c["Y"].astype(np.float32); tr, va = forward_folds()[2022]; rng = np.random.default_rng(0); va = va[::2]; n = len(va)
    Z = np.load(f"{ROOT}/artifacts/lp/lp4_b10.npz"); Xa, names = build(c, Z); k = [i for i, x in enumerate(names) if group(x) in ("base", "lp", "lps", "mf") and x not in SCALE]; Xa = Xa[..., k]
    dev = "cuda"; P = dict(objective="binary:logistic", device=dev, tree_method="hist", eta=.08, grow_policy="lossguide", max_leaves=63, max_depth=0, min_child_weight=50, subsample=.5, colsample_bytree=.8, reg_lambda=1., max_bin=255, verbosity=0, seed=0)
    gb = xgb.train(P, xgb.DMatrix(Xa[tr].reshape(-1, len(k)), label=Y[tr].reshape(-1)), num_boost_round=300)
    import train_net as tn, argparse
    A = argparse.Namespace(d=64, blocks=4, dropout=.1, ema=.995, lr=1e-3, wd=.01, epochs=12, batch=32, aux=.3); kn = [i for i, x in enumerate([names[j] for j in k]) if x not in ("unit_id", "plant_id")]
    mu = Xa[tr][..., kn].mean((0, 1, 2)); sd = Xa[tr][..., kn].std((0, 1, 2)); sd = np.where(sd > 1e-6, sd, 1.); sc = lambda x: torch.from_numpy(np.clip((x[..., kn] - mu) / sd, -6, 6).astype(np.float32))
    net = tn.fit(sc(Xa[tr]).to(dev), torch.from_numpy(Y[tr]).to(dev), 0, A, dev)
    def predict(x): return gb.predict(xgb.DMatrix(x.reshape(-1, x.shape[-1])), output_margin=True).reshape(len(x), 14, 168), tn.predict(net, sc(x), dev)
    pg0, pn0 = predict(Xa[va]); f = lambda a, b: .65 * a + .35 * b; base = {"xgb": auc(Y[va], pg0), "net": auc(Y[va], pn0), "blend": auc(Y[va], f(pg0, pn0))}; print("unperturbed", base, flush=True)
    def neg(p):
        p = p.copy(); o = np.argsort(p, 1)[:, :17]
        for i in range(len(p)): p[i, o[i]] = -20. - 40. * rng.random(17) * 0 - (p[i, o[i]].max() - p[i, o[i]])  # keeps the order, all below -20
        return p, o
    SETS = {"scale_0.5": (lambda: (c["price"][va] * .5, c["wv_end"][va] * .5)), "scale_2": (lambda: (c["price"][va] * 2, c["wv_end"][va] * 2)), "scale_4": (lambda: (c["price"][va] * 4, c["wv_end"][va] * 4)), "price_noise": (lambda: (c["price"][va] * np.exp(rng.normal(0, .1, (n, 168))), c["wv_end"][va])), "negative_cheapest": (lambda: (neg(c["price"][va])[0], c["wv_end"][va]))}
    res = {"base": base}
    with mp.Pool(7, init) as pool:
        for name, mk in SETS.items():
            t = time.time(); pr, w = mk(); out = pool.map(solve, [(pr[i], c["inflow"][va[i]], c["vol0"][va[i]], w[i], c["minflow"][va[i]], c["minvol"][va[i]]) for i in range(n)], chunksize=4)
            z = {q: np.stack([o[q] for o in out]) for q in out[0]}; z["idx"] = np.arange(n)
            ci = {m: c[m][va] for m in ("run_no", "start", "inflow", "minflow", "minvol", "vol0")}; ci["price"] = pr.astype(np.float32); ci["wv_end"] = w.astype(np.float32)
            X, _ = build(ci, z); pg, pn = predict(X[..., k]); bl = f(pg, pn); r = {"xgb": auc(Y[va], pg) - base["xgb"], "net": auc(Y[va], pn) - base["net"], "blend": auc(Y[va], bl) - base["blend"], "lp_u_change": float(np.abs(z["u"] - Z["u"][va]).mean())}
            if name == "negative_cheapest":
                m = np.zeros((n, 168), bool); o = np.argsort(c["price"][va], 1)[:, :17]; [m[i].__setitem__(o[i], True) for i in range(n)]; sig = lambda a: 1 / (1 + np.exp(-a)); M = np.repeat(m[:, None], 14, 1)
                r.update(p_on_before=float(sig(f(pg0, pn0))[M].mean()), p_on_after=float(sig(bl)[M].mean()), lp_on_before=float(Z["u"][va][M].mean()), lp_on_after=float(z["u"][M].mean()))
            res[name] = r; print(name, {a_: round(b_, 5) for a_, b_ in r.items()}, round(time.time() - t), "s", flush=True)
    json.dump(res, open(f"{ROOT}/artifacts/stress.json", "w"))
