"""XGBoost on GPU. Reports AUC per seed, so seed noise is measured rather than assumed."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, xgboost as xgb
from common import *
from features import build, group
a = argparse.ArgumentParser(); A = a.add_argument
A("--name"); A("--lp", default="lp4_b10"); A("--groups", default="base,lp,lps,mf"); A("--drop", default=""); A("--seeds", default="0"); A("--rounds", type=int, default=300); A("--folds", default=""); A("--test", action="store_true"); A("--forward", action="store_true"); A("--test-only", action="store_true"); A("--pseudo", type=float, default=0.); A("--eta", type=float, default=.08); A("--leaves", type=int, default=63); A("--mcw", type=float, default=50.); A("--shuffle", action="store_true"); A("--sens", action="store_true"); A("--sens-lp", default="lp4")
a = a.parse_args()
c = load_cases(); sens = [np.load(f"{ROOT}/artifacts/lp/{a.sens_lp}_w{w}.npz") for w in (90, 110)] if a.sens else None
X, names = build(c, np.load(f"{ROOT}/artifacts/lp/{a.lp}.npz"), None, sens)
keep = [i for i, n in enumerate(names) if group(n) in a.groups.split(",") and n not in a.drop.split(",")]; X = X[..., keep]; names = [names[i] for i in keep]; print(len(names), "features", flush=True)
Y = c["Y"]; F = forward_folds() if a.forward else folds()
if a.shuffle: Y = Y[np.random.default_rng(0).permutation(len(Y))]
if a.folds: F = {k: v for k, v in F.items() if str(k) in a.folds.split(",")}
P = dict(objective="binary:logistic", device="cuda", tree_method="hist", eta=a.eta, grow_policy="lossguide", max_leaves=a.leaves, max_depth=0, min_child_weight=a.mcw, subsample=.5, colsample_bytree=.8, reg_lambda=1., max_bin=255, verbosity=0)
seeds = [int(s) for s in a.seeds.split(",")]
def fit(idx, s): return xgb.train({**P, "seed": s}, xgb.DMatrix(X[idx].reshape(-1, X.shape[-1]), label=Y[idx].reshape(-1), feature_names=names), num_boost_round=a.rounds)
def pred(m, x): return m.predict(xgb.DMatrix(x.reshape(-1, x.shape[-1]), feature_names=names), output_margin=True).reshape(len(x), 14, 168)
def fit_pseudo(tr, va, s):
    q = 1 / (1 + np.exp(-pred(fit(tr, s), X[va])))
    Am = np.concatenate([X[tr].reshape(-1, X.shape[-1]), X[va].reshape(-1, X.shape[-1])]); L = np.concatenate([Y[tr].reshape(-1).astype(np.float32), q.reshape(-1).astype(np.float32)]); W = np.concatenate([np.ones(Y[tr].size, np.float32), np.full(q.size, a.pseudo, np.float32)])
    return xgb.train({**P, "seed": s}, xgb.DMatrix(Am, label=L, weight=W, feature_names=names), num_boost_round=a.rounds)
per = np.full((len(seeds), 2922, 14, 168), np.nan, np.float32)
for k, (tr, va) in ({} if a.test_only else F).items():
    t = time.time()
    for j, s in enumerate(seeds): per[j, va] = pred(fit_pseudo(tr, va, s) if a.pseudo > 0 else fit(tr, s), X[va])
    print(k, "auc", round(auc(Y[va], per[:, va].mean(0)), 5), round(time.time() - t), "s", flush=True)
if not a.test_only:
    oof = per.mean(0); ok = np.isfinite(oof).all(axis=(1, 2)); by = [auc(Y[ok], p[ok]) for p in per]
    if not a.folds and not a.shuffle: np.save(f"{ROOT}/artifacts/{'fwd' if a.forward else 'oof'}/{a.name}.npy", oof)
    report(a.name, oof, Y, c, extra=dict(model="xgboost-gpu", seeds=a.seeds, forward=a.forward, n_features=len(names), auc_per_seed=[round(x, 5) for x in by], seed_std=float(np.std(by)), shuffle=a.shuffle))
if a.test: np.save(f"{ROOT}/artifacts/preds/{a.name}.npy", np.mean([pred(fit(np.arange(2922), s), X[2922:]) for s in seeds], 0).astype(np.float32))
