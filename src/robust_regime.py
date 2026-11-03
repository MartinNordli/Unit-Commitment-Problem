"""How well does a model extrapolate? Harder holdouts than a year: train on early years only, or on one price regime and
predict the other. Compares no LP, LP with absolute prices, and LP without. Water-balance physics should hold across regimes."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, xgboost as xgb, pandas as pd
from common import *
from features import build, group
c = load_cases(); Y = c["Y"]; X, names = build(c, np.load(f"{ROOT}/artifacts/lp/lp4_b10.npz")); s = pd.to_datetime(c["start"][:2922]); yr = s.year.values; mp = c["price"][:2922].mean(1)
SCALE = open(f"{ROOT}/artifacts/scale_cols.txt").read().strip().split(",")
SETS = {"no_lp": [i for i, n in enumerate(names) if group(n) in ("base", "mf") and n not in SCALE], "lp_with_prices": [i for i, n in enumerate(names) if group(n) in ("base", "lp", "lps", "mf")], "lp_no_prices": [i for i, n in enumerate(names) if group(n) in ("base", "lp", "lps", "mf") and n not in SCALE]}
def purge(tr, va):
    sv = s[va].values[:, None]; d = np.abs((s[tr].values[None] - sv) / np.timedelta64(1, "D")).min(0)
    return tr[d > 7]
i = np.arange(2922); hi = mp > 50
H = {"train_2015_18_test_2019_22": (i[yr <= 2018], i[yr >= 2019]), "train_2015_19_test_2021_22": (i[yr <= 2019], i[yr >= 2021]), "train_2015_20_test_2022": (i[yr <= 2020], i[yr == 2022]), "train_price_low_test_high": (purge(i[~hi], i[hi]), i[hi]), "train_price_high_test_low": (purge(i[hi], i[~hi]), i[~hi]), "train_no_2020_like_test_2020": (i[(yr != 2020) & (mp > 15)], i[yr == 2020])}
P = dict(objective="binary:logistic", device="cuda", tree_method="hist", eta=.08, grow_policy="lossguide", max_leaves=63, max_depth=0, min_child_weight=50, subsample=.5, colsample_bytree=.8, reg_lambda=1., max_bin=255, verbosity=0, seed=0)
rows = []
for hn, (tr, va) in H.items():
    row = {"holdout": hn, "n_train": len(tr), "n_val": len(va)}
    for sn, k in SETS.items():
        x = X[..., k]; m = xgb.train(P, xgb.DMatrix(x[tr].reshape(-1, len(k)), label=Y[tr].reshape(-1)), num_boost_round=300); q = m.predict(xgb.DMatrix(x[va].reshape(-1, len(k))), output_margin=True); row[sn] = round(auc(Y[va], q), 5)
    row["lp_alone_pfrac"] = round(auc(Y[va], np.load(f"{ROOT}/artifacts/lp/lp4_b10.npz")["pfrac"][va]), 5); rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/artifacts/robust_regime.csv", index=False)
