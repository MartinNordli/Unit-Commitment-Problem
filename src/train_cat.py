import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from catboost import CatBoostClassifier, Pool
from common import *
from features import build, group
a = argparse.ArgumentParser(); a.add_argument("--name"); a.add_argument("--lp", default="lp4_b10"); a.add_argument("--groups", default="base,lp,lps,mf"); a.add_argument("--drop", default=""); a.add_argument("--seeds", default="0"); a.add_argument("--rounds", type=int, default=800); a.add_argument("--lr", type=float, default=.08); a.add_argument("--depth", type=int, default=8); a.add_argument("--forward", action="store_true"); a.add_argument("--test", action="store_true"); a = a.parse_args()
c = load_cases(); X, names = build(c, np.load(f"{ROOT}/artifacts/lp/{a.lp}.npz"))
keep = [i for i, n in enumerate(names) if group(n) in a.groups.split(",") and n not in a.drop.split(",")]; X = X[..., keep]; names = [names[i] for i in keep]; print(len(names), "features", flush=True)
Y = c["Y"]; F = forward_folds() if a.forward else folds(); seeds = [int(s) for s in a.seeds.split(",")]
def fit(idx, s):
    m = CatBoostClassifier(iterations=a.rounds, learning_rate=a.lr, depth=a.depth, l2_leaf_reg=3., border_count=128, bootstrap_type="Bernoulli", subsample=.5, task_type="GPU", devices="0", random_seed=s, verbose=0, gpu_ram_part=.4)
    return m.fit(X[idx].reshape(-1, X.shape[-1]), Y[idx].reshape(-1))
def pred(m, x): return m.predict(x.reshape(-1, x.shape[-1]), prediction_type="RawFormulaVal").reshape(len(x), 14, 168)
oof = np.full((2922, 14, 168), np.nan, np.float32)
for k, (tr, va) in F.items():
    t = time.time(); oof[va] = np.mean([pred(fit(tr, s), X[va]) for s in seeds], 0); print(k, "auc", round(auc(Y[va], oof[va]), 5), round(time.time() - t), "s", flush=True)
if not a.forward: np.save(f"{ROOT}/artifacts/oof/{a.name}.npy", oof)
if a.forward: np.save(f"{ROOT}/artifacts/fwd/{a.name}.npy", oof)
report(a.name, oof, Y, c, extra={"model": "catboost-gpu", "seeds": a.seeds, "forward": a.forward})
if a.test: np.save(f"{ROOT}/artifacts/preds/{a.name}.npy", np.mean([pred(fit(np.arange(2922), s), X[2922:]) for s in seeds], 0).astype(np.float32))
