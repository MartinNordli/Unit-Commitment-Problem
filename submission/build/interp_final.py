"""Report material: permutation importance for trees, net and blend, plus partial dependence for the trees."""
import os, sys; sys.path.insert(0, "src"); sys.dont_write_bytecode = True
import json, numpy as np, torch, xgboost as xgb
from types import SimpleNamespace as NS
from common import *
DEV = os.environ.get("UC_DEVICE", "cuda")  # "cpu" when no supported GPU is available
LP = os.environ.get("UC_LP", "lp5q")  # lp5q: LP v5 (x1.0, MIP gap 1e-4), with lp5q_w90 and lp5q_w110; lp4: the v4 files lp4_b10, lp4_w90, lp4_w110
SUF = "" if LP == "lp5q" else "_" + LP
from features import build, group, CATEGORICAL
from train_net import fit, predict
SCALE = open(f"{ROOT}/artifacts/scale_cols.txt").read().strip().split(",")
c = load_cases(); Y = c["Y"].astype(np.float32); tr, va = forward_folds()[2022]
X, names = build(c, np.load(f"{ROOT}/artifacts/lp/{LP if LP != 'lp4' else 'lp4_b10'}.npz"), None, [np.load(f"{ROOT}/artifacts/lp/{LP}_w{w}.npz") for w in (90, 110)])
keep = [i for i, n in enumerate(names) if group(n) in ("base", "lp", "ls")]; X = X[..., keep]; names = [names[i] for i in keep]
kt = [i for i, n in enumerate(names) if n not in SCALE]; kn = [i for i, n in enumerate(names) if n not in CATEGORICAL]
print(len(names), len(kt), len(kn), flush=True)
P = dict(objective="binary:logistic", device=DEV, tree_method="hist", eta=.08, grow_policy="lossguide", max_leaves=63, max_depth=0, min_child_weight=50., subsample=.5, colsample_bytree=.8, reg_lambda=1., max_bin=255, verbosity=0, seed=0)
tn = [names[i] for i in kt]
m = xgb.train(P, xgb.DMatrix(X[tr][..., kt].reshape(-1, len(kt)), label=Y[tr].reshape(-1), feature_names=tn), num_boost_round=300)
dev = DEV; A = NS(d=64, blocks=4, dropout=.1, ema=.995, lr=1e-3, wd=.01, epochs=24, batch=32, aux=.3)
Xn = X[..., kn]; mu = Xn[tr].mean((0, 1, 2)); sd = Xn[tr].std((0, 1, 2)); sd = np.where(sd > 1e-6, sd, 1.)
sc = lambda x: torch.from_numpy(np.clip((x - mu) / sd, -6, 6).astype(np.float32))
net = fit(sc(Xn[tr]).to(dev), torch.from_numpy(Y[tr]).to(dev), 0, A, dev); print("net trained", flush=True)
Xv = X[va]; Yv = Y[va]; N = len(va)
def tree(x): return m.predict(xgb.DMatrix(x[..., kt].reshape(-1, len(kt)), feature_names=tn), output_margin=True).reshape(N, 14, 168)
def nnet(x): return predict(net, sc(x[..., kn]), dev)
def scores(x):
    t, n = tree(x), nnet(x); return dict(xgb=auc(Yv, t), net=auc(Yv, n), blend=auc(Yv, .7 * t + .3 * n))
base = scores(Xv); print("base", base, flush=True)
def shuffled(cols, rng):
    x = Xv.copy(); idx = rng.permutation(N * 14)
    for j in cols: x[..., j] = Xv[..., j].reshape(N * 14, 168)[idx].reshape(N, 14, 168)
    return x
GROUPS = dict(market=["price_asinh", "price_z", "price_rank_week", "price_rank_day", "price_diff_prev", "price_diff_next", "price_dev_7h", "price_dev_25h", "week_price_mean_asinh", "week_price_iqr_asinh"],
    calendar=["hour_local", "weekday_local", "doy_sin", "doy_cos", "horizon_hour"], unit=["unit_id", "plant_id", "unit_index", "p_max", "start_per_mw"])
GROUPS["lp_schedule"] = [n for n in names if group(n) == "lp"]; GROUPS["lp_sensitivity"] = [n for n in names if group(n) == "ls"]
used = set(sum(GROUPS.values(), [])); GROUPS["water_state"] = [n for n in names if n not in used]; print({k: len(v) for k, v in GROUPS.items()}, GROUPS["water_state"], flush=True)
out = dict(base=base, groups=GROUPS, group={}, single={})
for g, cols in GROUPS.items():
    r = [scores(shuffled([names.index(n) for n in cols], np.random.default_rng(10 + i))) for i in range(3)]
    out["group"][g] = {k: float(np.mean([q[k] for q in r]) - base[k]) for k in base}; print(g, out["group"][g], flush=True)
for j, n in enumerate(names):
    r = scores(shuffled([j], np.random.default_rng(100 + j))); out["single"][n] = {k: float(r[k] - base[k]) for k in base}
print("single done", flush=True)
rng = np.random.default_rng(0); s = rng.choice(N * 14 * 168, 20000, replace=False); Xs = Xv[..., kt].reshape(-1, len(kt))[s]; pdp = {}
for n in ["lp_margin_ratio", "ls_mean", "lp_margin", "lp_u", "lp_sys_t", "margin_wv_up", "start_hours_wv", "fill_up"]:
    j = tn.index(n); g = np.unique(np.quantile(Xs[:, j], np.linspace(.01, .99, 25))); y = []
    for v in g:
        z = Xs.copy(); z[:, j] = v; y.append(float(m.predict(xgb.DMatrix(z, feature_names=tn), output_margin=True).mean()))
    pdp[n] = dict(x=g.tolist(), y=y)
out["pdp"] = pdp; json.dump(out, open(f"artifacts/interp_final{SUF}.json", "w")); print("saved", flush=True)
