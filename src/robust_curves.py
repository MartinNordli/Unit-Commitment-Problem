"""Over- or underfit? Validation AUC per boosting round on forward folds, train/validation gap, and AUC against training-set size."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, xgboost as xgb
from common import *
from features import build, group
c = load_cases(); Y = c["Y"]; X, names = build(c, np.load(f"{ROOT}/artifacts/lp/lp4_b10.npz")); SCALE = open(f"{ROOT}/artifacts/scale_cols.txt").read().strip().split(",")
k = [i for i, n in enumerate(names) if group(n) in ("base", "lp", "lps", "mf") and n not in SCALE]; X = X[..., k]; yr = np.array([int(s[:4]) for s in c["start"]])[:2922]
P = dict(objective="binary:logistic", device="cuda", tree_method="hist", eta=.08, grow_policy="lossguide", max_leaves=63, max_depth=0, min_child_weight=50, subsample=.5, colsample_bytree=.8, reg_lambda=1., max_bin=255, verbosity=0, seed=0, eval_metric="auc")
out = {"rounds": {}, "size": {}}
for y, (tr, va) in forward_folds().items():
    h = {}; D = xgb.DMatrix(X[tr].reshape(-1, len(k)), label=Y[tr].reshape(-1)); V = xgb.DMatrix(X[va].reshape(-1, len(k)), label=Y[va].reshape(-1))
    xgb.train(P, D, num_boost_round=1200, evals=[(D, "train"), (V, "val")], evals_result=h, verbose_eval=False)
    tv, vv = np.array(h["train"]["auc"]), np.array(h["val"]["auc"]); pts = [49, 99, 199, 299, 499, 799, 1199]
    out["rounds"][y] = {r + 1: (round(float(tv[r]), 5), round(float(vv[r]), 5)) for r in pts}; out["rounds"][y]["best_val_round"] = int(vv.argmax()) + 1; print(y, out["rounds"][y], flush=True)
tr0, va = forward_folds()[2022]
for n in (1, 2, 3, 5, 7):
    tr = tr0[yr[tr0] >= 2022 - n]; m = xgb.train({**P, "eval_metric": "logloss"}, xgb.DMatrix(X[tr].reshape(-1, len(k)), label=Y[tr].reshape(-1)), num_boost_round=300)
    out["size"][n] = round(auc(Y[va], m.predict(xgb.DMatrix(X[va].reshape(-1, len(k))))), 5); print("years of training before 2022:", n, out["size"][n], flush=True)
json.dump(out, open(f"{ROOT}/artifacts/robust_curves.json", "w"))
