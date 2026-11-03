"""Report material: adversarial validation, train years against test years. Alternating 60-day blocks, so overlapping cases stay together.
Calendar columns are left out: with alternating blocks they let the classifier memorise which days are test, which pushes held-out AUC well below the real drift."""
import sys; sys.path.insert(0, "src"); sys.dont_write_bytecode = True
import json, numpy as np, xgboost as xgb
from common import *
from features import build, group
SCALE = open(f"{ROOT}/artifacts/scale_cols.txt").read().strip().split(",")
c = load_cases(); X, names = build(c, np.load(f"{ROOT}/artifacts/lp/lp4_b10.npz"), None, [np.load(f"{ROOT}/artifacts/lp/lp4_w{w}.npz") for w in (90, 110)])
y = (np.arange(len(X)) >= 2922).astype(np.float32); block = (np.arange(len(X)) // 60) % 2
CAL = {"hour_local", "weekday_local", "doy_sin", "doy_cos", "horizon_hour", "unit_id", "plant_id", "unit_index", "p_max", "start_per_mw"}
names_ok = [n for n in names if group(n) in ("base", "lp", "ls") and n not in CAL]
SETS = dict(with_absolute=names_ok, no_absolute=[n for n in names_ok if n not in SCALE], lp_only=[n for n in names_ok if group(n) in ("lp", "ls") and n not in SCALE], absolute_only=SCALE)
P = dict(objective="binary:logistic", device="cuda", tree_method="hist", eta=.1, max_depth=6, subsample=.5, colsample_bytree=.8, verbosity=0, seed=0, scale_pos_weight=2922 / 725)
rng = np.random.default_rng(0); out = {}
for s, cols in SETS.items():
    k = [names.index(n) for n in cols]; R = []
    for h in (0, 1):
        tr, va = np.where(block != h)[0], np.where(block == h)[0]
        pick = rng.choice(X[tr].shape[0] * 14 * 168, 2000000, replace=False); A = X[tr][..., k].reshape(-1, len(k))[pick]; L = np.broadcast_to(y[tr][:, None, None], (len(tr), 14, 168)).reshape(-1)[pick]
        m = xgb.train(P, xgb.DMatrix(A, label=L, feature_names=cols), num_boost_round=150)
        R.append(auc(np.broadcast_to(y[va][:, None, None], (len(va), 14, 168)), m.predict(xgb.DMatrix(X[va][..., k].reshape(-1, len(k)), feature_names=cols))))
    g = m.get_score(importance_type="gain"); out[s] = dict(n=len(cols), auc=float(np.mean(R)), top=sorted(g, key=g.get, reverse=True)[:6]); print(s, out[s], flush=True)
json.dump(out, open("artifacts/drift.json", "w"))
