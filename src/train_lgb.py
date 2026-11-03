import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
from common import *
from features import build, CATEGORICAL, group
a = argparse.ArgumentParser(); a.add_argument("--name"); a.add_argument("--lp", default="lp_tele"); a.add_argument("--groups", default="base,lp,lps"); a.add_argument("--drop", default=""); a.add_argument("--keep", default="")
a.add_argument("--seeds", default="0"); a.add_argument("--rounds", type=int, default=300); a.add_argument("--threads", type=int, default=24); a.add_argument("--folds", default=""); a.add_argument("--forward", action="store_true"); a.add_argument("--test", action="store_true"); a.add_argument("--params", default="{}"); a.add_argument("--milp", default=""); a.add_argument("--shuffle", action="store_true"); a = a.parse_args()
c = load_cases(); z = np.load(f"{ROOT}/artifacts/lp/{a.lp}.npz") if a.lp != "none" else None
mz = np.load(f"{ROOT}/artifacts/lp/{a.milp}.npz") if a.milp else None
t0 = time.time(); X, names = build(c, z, mz); print("features", X.shape, round(time.time() - t0), "s", flush=True)
keep = [i for i, n in enumerate(names) if group(n) in a.groups.split(",") and n not in a.drop.split(",") and (not a.keep or n in a.keep.split(",") or n in CATEGORICAL)]
X = X[..., keep]; names = [names[i] for i in keep]; print(len(names), names, flush=True)
params = dict(objective="binary", verbose=-1, learning_rate=.08, num_leaves=63, min_data_in_leaf=500, feature_fraction=.8, bagging_fraction=.5, bagging_freq=1, lambda_l2=1., max_bin=255, num_threads=a.threads, force_row_wise=True, deterministic=True); params.update(json.loads(a.params))
Y = c["Y"]; F = forward_folds() if a.forward else folds()
if a.shuffle:  # control: whole-case label tensors permuted across cases, so any AUC far above chance signals leakage in the pipeline
    Y = Y[np.random.default_rng(0).permutation(len(Y))]
if a.folds: F = {k: v for k, v in F.items() if str(k) in a.folds.split(",")}
cat = [n for n in CATEGORICAL if n in names]; seeds = [int(s) for s in a.seeds.split(",")]
def fit(idx, seed):
    d = lgb.Dataset(X[idx].reshape(-1, X.shape[-1]), label=Y[idx].reshape(-1).astype(np.float32), feature_name=names, categorical_feature=cat, free_raw_data=True)
    return lgb.train({**params, "seed": seed}, d, num_boost_round=a.rounds)
oof = np.full((2922, 14, 168), np.nan, np.float32); yrs = years(c)[:2922]
for k, (tr, va) in F.items():
    t1 = time.time(); ps = []
    for s in seeds:
        m = fit(tr, s); ps.append(m.predict(X[va].reshape(-1, X.shape[-1]), raw_score=True).reshape(len(va), 14, 168))
    oof[va] = np.mean(ps, 0); print(k, "auc", round(auc(Y[va], oof[va]), 5), "train", len(tr), round(time.time() - t1), "s", flush=True)
if not a.forward and not a.shuffle: np.save(f"{ROOT}/artifacts/oof/{a.name}.npy", oof)
if a.forward: np.save(f"{ROOT}/artifacts/fwd/{a.name}.npy", oof)
report(a.name, oof, Y, c, extra={"n_features": len(names), "seeds": a.seeds, "forward": a.forward, "args": vars(a).get("groups")})
imp = m.feature_importance("gain"); o = np.argsort(-imp)[:25]; print("top gain", [(names[i], round(imp[i] / imp.sum(), 3)) for i in o])
if a.test:
    ps = [fit(np.arange(2922), s).predict(X[2922:].reshape(-1, X.shape[-1]), raw_score=True).reshape(725, 14, 168) for s in seeds]
    np.save(f"{ROOT}/artifacts/preds/{a.name}.npy", np.mean(ps, 0).astype(np.float32))
