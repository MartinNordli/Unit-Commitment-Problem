"""Data loading: every case as arrays. Folds and metric live in the locked harness."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp_surrogate import UNITS
from harness import auc, folds, forward_folds, report, verify_lock
RES = ["Kjelavatn","Staavatn","Langesae","Foersvatn","Venemo","Botnedalsvatn","Bitdalsvatn","Vaamarvatn","Langeidvatn","Totak","Songavatn","Vinjevatn","Bandak","Bordalsvatn","Byrtevatn","Hyljelihyl","Vatjern"]
INFLOW = RES + ["r_Vest_Vassdraget"]
MINFLOW = ["b_Kjelavatn","b_Hyljelihyl","b_Vinjevatn","b_Byrtevatn","b_Bandak","r_Vest_Vassdraget"]
MINVOL = ["Totak","Byrtevatn","Staavatn"]
TARGETS = [f"result_committed_{u}_t{t}" for u in UNITS for t in range(168)]

def _utc(path):
    df = pd.read_csv(path, parse_dates=["date"]); df["date"] = pd.to_datetime(df["date"], utc=True); return df.set_index("date").sort_index()

def load_cases(cache=True):
    """All 3647 cases as arrays, in Run No order. Contains only what SHOP is given for each case."""
    path = os.path.join(ROOT, "artifacts", "cases.npz")
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=False); return {k: z[k] for k in z.files}
    uc = pd.read_csv(f"{D}/kernel/Unit_commitment_decisions.csv", parse_dates=["starttime"])
    assert list(uc.columns[3:]) == TARGETS
    mp = pd.read_csv(f"{ROOT}/prediction_mapping.csv", parse_dates=["starttime"])
    ids = np.r_[uc["Run No"].values, mp["Run No"].values]
    starts = pd.DatetimeIndex(list(uc.starttime) + list(mp.starttime)).tz_localize("UTC")
    hours = pd.date_range(starts[0], starts[-1] + pd.Timedelta(hours=167), freq="h")
    P = _utc(f"{D}/kernel/Historical_day_ahead_price_2015_2025.csv").iloc[:, 0].reindex(hours)
    assert P.notna().all()
    Q = _utc(f"{D}/kernel/Historical_inflow_1958_2025.csv")[INFLOW].reindex(hours); assert Q.notna().all().all()
    MF = _utc(f"{D}/extended/Constraint_min_flow.csv")[MINFLOW].reindex(hours); assert MF.notna().all().all()
    MV = _utc(f"{D}/extended/Constraint_min_volume.csv")[MINVOL].reindex(hours); assert MV.notna().all().all()
    V = _utc(f"{D}/kernel/Historical_volume_2015_2024.csv")[RES]
    W = _utc(f"{D}/kernel/Synthetic_water_value_2015_2024.csv")[RES]
    pos = ((starts - hours[0]) / pd.Timedelta(hours=1)).astype(int).values
    win = lambda a: np.stack([a[p:p+168] for p in pos])
    out = dict(run_no=ids, start=starts.tz_localize(None).values.astype("datetime64[D]").astype(str),
        price=win(P.values).astype(np.float32), inflow=win(Q.values).astype(np.float32),
        minflow=win(MF.values).astype(np.float32), minvol=win(MV.values).astype(np.float32),
        vol0=V.reindex(starts).values.astype(np.float32),
        wv_end=W.reindex(starts + pd.Timedelta(days=7)).values.astype(np.float32),
        wv_start=W.reindex(starts).values.astype(np.float32),
        Y=uc[TARGETS].values.reshape(len(uc), 14, 168).astype(np.int8))
    assert not np.isnan(out["vol0"]).any() and not np.isnan(out["wv_end"]).any()
    if cache: np.savez(path, **out)
    return out

def years(c): return np.array([int(s[:4]) for s in c["start"]])



