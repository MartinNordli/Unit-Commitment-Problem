"""Solve the LP surrogate for a set of cases in parallel; saves one npz."""
import argparse, os, sys, time, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_cases, RES, INFLOW, MINFLOW, MINVOL, ROOT
from lp_surrogate import Topology, WeekLP

_g = {}
def init(conv, relaxed, tl, gap, shared, cm, bp, ws, rc="slope", iu=(), ig=1e-6):
    _g.update(relaxed=relaxed, tl=tl, gap=gap, shared=shared, cm=cm, bp=bp, ws=ws, rc=rc, iu=iu, ig=ig)
    _g["c"] = load_cases(); _g["lp"] = WeekLP(Topology(f"{ROOT}/data/extended/Tokke_Vinje_topology.yaml")); _g["conv"] = conv

def one(i):
    c, lp = _g["c"], _g["lp"]; t0 = time.time()
    r = lp.solve(c["price"][i].astype(float), {n: c["inflow"][i, :, k].astype(float) for k, n in enumerate(INFLOW)},
                 {n: float(c["vol0"][i, k]) for k, n in enumerate(RES)}, {n: float(c["wv_end"][i, k]) * _g["ws"] for k, n in enumerate(RES)},
                 {n: c["minflow"][i, :, k].astype(float) for k, n in enumerate(MINFLOW)}, {n: c["minvol"][i, :, k].astype(float) for k, n in enumerate(MINVOL)},
                 convention=_g["conv"], relaxed=_g["relaxed"], time_limit=_g["tl"], mip_gap=_g["gap"], shared_loss=_g["shared"], conv_mode=_g["cm"], bypass_cost=_g["bp"],
                 river_cost=_g["rc"], int_units=_g["iu"], int_gap=_g["ig"])
    r["i"] = i; r["time"] = time.time() - t0
    return r

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--out"); a.add_argument("--conv", default="global"); a.add_argument("--step", type=int, default=1); a.add_argument("--n", type=int, default=0); a.add_argument("--procs", type=int, default=4); a.add_argument("--milp", action="store_true"); a.add_argument("--tl", type=float, default=120.); a.add_argument("--gap", type=float, default=.005); a.add_argument("--shared", default="marginal"); a.add_argument("--conv-mode", default="real"); a.add_argument("--bypass", type=float, default=-1); a.add_argument("--wv-scale", type=float, default=1.0)
    a.add_argument("--river-cost", default="slope"); a.add_argument("--int-units", default=""); a.add_argument("--int-gap", type=float, default=1e-6); a.add_argument("--start", type=int, default=0); a = a.parse_args()
    c = load_cases(); idx = np.arange(a.start, len(c["run_no"]), a.step)
    if a.n: idx = idx[:a.n]
    t0 = time.time(); res = []
    with mp.Pool(a.procs, init, (a.conv, not a.milp, a.tl, a.gap, a.shared, a.conv_mode, (None if a.bypass < 0 else a.bypass), a.wv_scale, a.river_cost, tuple(u for u in a.int_units.split(",") if u), a.int_gap)) as p:
        for k, r in enumerate(p.imap(one, idx.tolist(), chunksize=1)):
            res.append(r)
            if k % 100 == 0: print(k, len(idx), round(time.time() - t0), "s  last", round(r["time"], 2), "s", r.get("status"), flush=True)
    bad = [r["i"] for r in res if "u" not in r]; print("failed", len(bad), bad[:10])
    good = [r for r in res if "u" in r]
    np.savez(a.out, idx=np.array([r["i"] for r in good]), u=np.stack([r["u"] for r in good]).astype(np.float32), pfrac=np.stack([r["pfrac"] for r in good]).astype(np.float32),
             V=np.stack([r["V"] for r in good]).astype(np.float32), dual=np.stack([r["dual"] for r in good]).astype(np.float32), conv=np.stack([r["conv"] for r in good]).astype(np.float32),
             flows=np.stack([r["flows"] for r in good]).astype(np.float32), values=np.stack([r["values"] for r in good]).astype(np.float32), time=np.array([r["time"] for r in good]), obj=np.array([r["obj"] for r in good]), int_status=np.array([r["int_status"] for r in good]))
    print("integer solves not proven optimal", int(sum(r["int_status"] > 0 for r in good)))
    print("mean solve s", np.mean([r["time"] for r in good]), "total", time.time() - t0)
