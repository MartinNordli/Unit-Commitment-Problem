"""R25: LP v5 against v4 for XGBoost, ST-Net (N6, N7) and the 0.7/0.3 blends, LOYO and forward, per year."""
import sys; sys.path.insert(0, "src"); sys.dont_write_bytecode = True
import json, numpy as np
from common import load_cases, years, auc
c = load_cases(); Y = c["Y"]; yr = years(c)[:2922]
L = lambda d, n: np.load(f"artifacts/{d}/{n}.npy").astype(np.float64)
PAIRS = {"xgb": (("XR_ls", "XR_ls_f"), ("XR5_ls", "XR5_ls_f")), "n6": (("N6_min", "N6f_min"), ("N6_v5", "N6f_v5")), "n7": (("N7_noabs", "N7f_noabs"), ("N7_v5", "N7f_v5"))}
def score(p):
    ok = np.isfinite(p).all((1, 2)); ys = sorted(set(yr[ok]))
    return auc(Y[ok], p[ok]), {int(y): auc(Y[ok & (yr == y)], p[ok & (yr == y)]) for y in ys}
out = {}
def row(name, v4, v5):
    (a4, y4), (a5, y5) = score(v4), score(v5); up = sum(y5[y] > y4[y] for y in y4)
    out[name] = dict(v4=a4, v5=a5, delta=a5 - a4, up=up, n=len(y4), per_year={y: round(y5[y] - y4[y], 5) for y in y4})
    print(f"{name:14s} v4 {a4:.5f}  v5 {a5:.5f}  delta {a5 - a4:+.5f}  years up {up}/{len(y4)}  per year {out[name]['per_year']}")
for view, k in (("loyo", 0), ("forward", 1)):
    d = "oof" if k == 0 else "fwd"; P = {m: (L(d, a[k]), L(d, b[k])) for m, (a, b) in PAIRS.items()}
    for m in P: row(f"{m} {view}", *P[m])
    for net in ("n6", "n7"):
        row(f"blend x+{net} {view}", .7 * P["xgb"][0] + .3 * P[net][0], .7 * P["xgb"][1] + .3 * P[net][1])
json.dump(out, open("artifacts/eval_v5.json", "w"), indent=1)
