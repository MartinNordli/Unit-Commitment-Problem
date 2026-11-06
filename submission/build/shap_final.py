"""Report material: SHAP values of the final XGBoost, trained before 2022 and explained on a sample of held-out 2022 rows."""
import os, sys; sys.path.insert(0, "src"); sys.dont_write_bytecode = True
import json, numpy as np, xgboost as xgb
from common import *
DEV = os.environ.get("UC_DEVICE", "cuda")  # "cpu" when no supported GPU is available
LP = os.environ.get("UC_LP", "lp5q")  # lp5q: LP v5 (x1.0, MIP gap 1e-4), with lp5q_w90 and lp5q_w110; lp4: the v4 files lp4_b10, lp4_w90, lp4_w110
SUF = "" if LP == "lp5q" else "_" + LP
from features import build, group
SCALE = open(f"{ROOT}/artifacts/scale_cols.txt").read().strip().split(",")
c = load_cases(); Y = c["Y"].astype(np.float32); tr, va = forward_folds()[2022]
X, names = build(c, np.load(f"{ROOT}/artifacts/lp/{LP if LP != 'lp4' else 'lp4_b10'}.npz"), None, [np.load(f"{ROOT}/artifacts/lp/{LP}_w{w}.npz") for w in (90, 110)])
k = [i for i, n in enumerate(names) if group(n) in ("base", "lp", "ls") and n not in SCALE]; X = X[..., k]; names = [names[i] for i in k]
P = dict(objective="binary:logistic", device=DEV, tree_method="hist", eta=.08, grow_policy="lossguide", max_leaves=63, max_depth=0, min_child_weight=50., subsample=.5, colsample_bytree=.8, reg_lambda=1., max_bin=255, verbosity=0, seed=0)
m = xgb.train(P, xgb.DMatrix(X[tr].reshape(-1, len(k)), label=Y[tr].reshape(-1), feature_names=names), num_boost_round=300)
rng = np.random.default_rng(0); A = X[va].reshape(-1, len(k)); y = Y[va].reshape(-1); s = rng.choice(len(A), 300000, replace=False)
sv = m.predict(xgb.DMatrix(A[s], feature_names=names), pred_contribs=True)[:, :-1]
unit = np.broadcast_to(np.arange(14)[None, :, None], (len(va), 14, 168)).reshape(-1)[s]
imp = np.abs(sv).mean(0); o = np.argsort(-imp)
tot = imp.sum(); print("top 25 mean |SHAP|:", [(names[i], round(float(imp[i]), 4)) for i in o[:25]])
groups = {g: float(imp[[i for i, n in enumerate(names) if group(n) == g]].sum() / tot) for g in ("base", "lp", "ls")}; print("share by group", groups)
gain = m.get_score(importance_type="gain"); print("top 10 gain", sorted(gain, key=gain.get, reverse=True)[:10])
top = o[:40]; np.savez(f"artifacts/shap_final{SUF}.npz", names=np.array(names), imp=imp, top=top, sv=sv[:20000][:, top].astype(np.float32), x=A[s][:20000][:, top].astype(np.float32), y=y[s][:20000], unit=unit[:20000])
json.dump(dict(groups=groups, gain={k_: float(v) for k_, v in gain.items()}), open(f"artifacts/shap_final{SUF}.json", "w"))
