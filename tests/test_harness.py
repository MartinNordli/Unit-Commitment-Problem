"""Run: python tests/test_harness.py. Guards the harness, the time alignment and case locality of the features."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np, pandas as pd
from common import *
from harness import verify_lock
from features import build, MODEL_INPUTS
c = load_cases(); ROOT_ = ROOT; ok = []
def t(name, cond): ok.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name, flush=True)
verify_lock(); t("locked files unchanged", True)
s = pd.to_datetime(c["start"]); L, Fw = folds(), forward_folds()
t("loyo: no training case within 7 days of the validation year", all((((s[tr].values[:, None] - s[va].values[None]) / np.timedelta64(1, "D")).__abs__().min() > 7) for tr, va in L.values()))
t("loyo: validation years partition the training period", sorted(np.concatenate([v for _, v in L.values()]).tolist()) == list(range(2922)))
t("forward: every training case ends before validation starts", all(s[tr].max() + pd.Timedelta(hours=168) <= s[va].min() for tr, va in Fw.values()))
t("no test case in any fold", all(max(tr.max(), va.max()) < 2922 for tr, va in list(L.values()) + list(Fw.values())))
import importlib.util as iu
sp = iu.spec_from_file_location("kaggle_metric", f"{ROOT_}/kaggle_metric.py"); km = iu.module_from_spec(sp); sp.loader.exec_module(km)
r = np.random.default_rng(0); idx = r.choice(2922, 60, replace=False); p = r.random((60, 14, 168)) * .3 + c["Y"][idx] * .4
mk = lambda a: pd.DataFrame(a.reshape(60, -1), columns=TARGETS).assign(**{"Run No": c["run_no"][idx]})
t("harness auc equals the Kaggle metric", abs(auc(c["Y"][idx], p) - km.score(mk(c["Y"][idx].astype(int)), mk(p))) < 1e-12)
P = pd.read_csv(f"{D}/kernel/Historical_day_ahead_price_2015_2025.csv", parse_dates=["date"]); P["date"] = pd.to_datetime(P["date"], utc=True); P = P.set_index("date").iloc[:, 0]
V = pd.read_csv(f"{D}/kernel/Historical_volume_2015_2024.csv", parse_dates=["date"]); V["date"] = pd.to_datetime(V["date"], utc=True); V = V.set_index("date")
W = pd.read_csv(f"{D}/kernel/Synthetic_water_value_2015_2024.csv", parse_dates=["date"]); W["date"] = pd.to_datetime(W["date"], utc=True); W = W.set_index("date")
good = True
for i in (0, 1000, 2921, 2922, 3646):
    st = pd.Timestamp(c["start"][i], tz="UTC")
    good &= np.allclose(c["price"][i], P.loc[st: st + pd.Timedelta(hours=167)].values) and np.allclose(c["vol0"][i], V.loc[st, RES].values.astype(float)) and np.allclose(c["wv_end"][i], W.loc[st + pd.Timedelta(days=7), RES].values.astype(float))
t("case windows: price from start, volume at start, water value at start + 7 days", good)
t("test cases have exactly the sample_submission ids", (pd.read_csv(f"{ROOT_}/sample_submission.csv", usecols=["Run No"])["Run No"].values == c["run_no"][2922:]).all())
z0 = np.load(f"{ROOT_}/artifacts/lp/lp4_b10.npz"); ids = np.array([5, 700, 2500, 3000]); sub = lambda d: {k: (d[k][ids] if (k != "Y" and hasattr(d[k], "__len__")) else d[k]) for k in MODEL_INPUTS}
zs = {k: z0[k][ids] for k in ("u", "pfrac", "V", "dual", "conv", "flows")}; zs["idx"] = np.arange(4)
X0, names = build(sub(c), zs)
t("features contain no NaN or inf", np.isfinite(X0).all())
c2 = sub(c); c2["price"] = c2["price"].copy(); c2["price"][1] += 7.; c2["vol0"] = c2["vol0"].copy(); c2["vol0"][1] *= 1.1; X1, _ = build(c2, zs)
ch = np.abs(X1 - X0).sum((1, 2, 3)) > 0
t("changing one case changes only that case's features", ch[1] and not ch[0] and not ch[2] and not ch[3])
try:
    build({k: v for k, v in sub(c).items() if k != "wv_end"}, zs); t("build refuses incomplete inputs", False)
except KeyError: t("build refuses incomplete inputs", True)
X2, _ = build({**sub(c), "Y": np.ones(4), "wv_start": np.zeros((4, 17))}, zs); t("labels and wv_start cannot affect features", (X2 == X0).all())
bad = [n for n, g in ok if not g]; print(f"\n{len(ok) - len(bad)}/{len(ok)} passed"); sys.exit(1 if bad else 0)
