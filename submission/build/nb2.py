# %% [markdown]
# # Short notebook 2: LP surrogate + XGBoost + ST-Net without absolute price levels
#
# **Name:** Martin Nordli Almenningen  
# **Student id:** @@STUDENT_ID@@  
# **Kaggle team name:** @@TEAM@@  
# **Group:** 197
#
# Reproduces one selected Kaggle prediction from the raw data, offline, and writes `submission_2.csv`.
#
# This is the second selected prediction, with a different risk from notebook 1. There the ST-Net also reads absolute price levels. Here no model reads them directly, in case the price levels of 2023-2024 make that information misleading.
#
# 1. Load the raw files into one set of arrays per weekly case.
# 2. Solve a linear program of the weekly hydropower schedule, 3 times per case (water values x1.0, x0.9, x1.1).
# 3. Build features from the raw inputs and the LP solutions.
# 4. Train XGBoost and ST-Net on 2015-2022 and predict 2023-2024.
# 5. Blend the two in logit space and write the file.
#
# **To run:** put this notebook next to the raw data, or set `UC_DATA_DIR`. Files are found by name in any subfolder.
# Install from `requirements.txt` (Python 3.12).
#
# **Reference run:** @@RUNTIME@@
#
# How the hyperparameters and the blend weight were chosen is described in the report.

# %%
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"  # one thread per LP worker; XGBoost and PyTorch get their thread count explicitly below
import math, multiprocessing, time
from pathlib import Path

import numpy as np, pandas as pd, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as F, xgboost as xgb, yaml
from scipy.optimize import linprog
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

DATA_DIR = Path(os.environ.get("UC_DATA_DIR", "."))
WORK_DIR = Path(os.environ.get("UC_WORK_DIR", "work")); WORK_DIR.mkdir(exist_ok=True)  # LP results, so a rerun can resume
OUTPUT = "submission_2.csv"
N_CORES = min(4, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())

DROP = {"price_asinh", "week_price_mean_asinh", "week_price_iqr_asinh", "wv_up_asinh", "wv_down_asinh", "lp_be_asinh"}  # absolute price levels
TREE_GROUPS, TREE_DROP, TREE_SEEDS = @@TREE_GROUPS@@, DROP, (0, 1, 2)
NET_GROUPS, NET_DROP, NET_SEEDS, NET_EPOCHS = @@NET_GROUPS@@, DROP, (0, 1, 2), @@NET_EPOCHS@@
TREE_WEIGHT = @@TREE_WEIGHT@@
N_TRAIN = 2922  # 2015-2022; the last 725 cases are the test set

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(N_CORES)
T_START, TIMES = time.time(), {}
print("device", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", "| cores", N_CORES)

def find(name):
    hits = sorted(Path(d) / name for d, _, files in os.walk(DATA_DIR, followlinks=True) if name in files)
    assert hits, f"{name} not found under {DATA_DIR.resolve()}; set UC_DATA_DIR"
    return hits[0]

# %% [markdown]
# ## 1. Raw data
# One entry per weekly case, in `Run No` order. Only what SHOP receives for a case is used. The water value at the end of the week is the terminal value.

# %%
UNITS = ["Hogga_G1", "Tokke_G1", "Tokke_G2", "Tokke_G3", "Tokke_G4", "Lio_G1", "Byrte_G1", "Vinje_G1",
         "Vinje_G2", "Vinje_G3", "Haukeli_G1", "Songa_G1", "Kjela_G1", "Vesle Kjela_G1"]
RES = ["Kjelavatn", "Staavatn", "Langesae", "Foersvatn", "Venemo", "Botnedalsvatn", "Bitdalsvatn", "Vaamarvatn", "Langeidvatn", "Totak",
       "Songavatn", "Vinjevatn", "Bandak", "Bordalsvatn", "Byrtevatn", "Hyljelihyl", "Vatjern"]
INFLOW = RES + ["r_Vest_Vassdraget"]
MINFLOW = ["b_Kjelavatn", "b_Hyljelihyl", "b_Vinjevatn", "b_Byrtevatn", "b_Bandak", "r_Vest_Vassdraget"]
MINVOL = ["Totak", "Byrtevatn", "Staavatn"]
TARGETS = [f"result_committed_{u}_t{t}" for u in UNITS for t in range(168)]

def read_utc(name):
    df = pd.read_csv(find(name), parse_dates=["date"]); df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date").sort_index()

def load_cases():
    uc = pd.read_csv(find("Unit_commitment_decisions.csv"), parse_dates=["starttime"])
    assert list(uc.columns[3:]) == TARGETS
    mp_ = pd.read_csv(find("prediction_mapping.csv"), parse_dates=["starttime"])
    ids = np.r_[uc["Run No"].values, mp_["Run No"].values]
    starts = pd.DatetimeIndex(list(uc.starttime) + list(mp_.starttime)).tz_localize("UTC")
    hours = pd.date_range(starts[0], starts[-1] + pd.Timedelta(hours=167), freq="h")
    P = read_utc("Historical_day_ahead_price_2015_2025.csv").iloc[:, 0].reindex(hours); assert P.notna().all()
    Q = read_utc("Historical_inflow_1958_2025.csv")[INFLOW].reindex(hours); assert Q.notna().all().all()
    MF = read_utc("Constraint_min_flow.csv")[MINFLOW].reindex(hours); assert MF.notna().all().all()
    MV = read_utc("Constraint_min_volume.csv")[MINVOL].reindex(hours); assert MV.notna().all().all()
    V = read_utc("Historical_volume_2015_2024.csv")[RES]
    W = read_utc("Synthetic_water_value_2015_2024.csv")[RES]
    pos = ((starts - hours[0]) / pd.Timedelta(hours=1)).astype(int).values
    win = lambda a: np.stack([a[p:p + 168] for p in pos])
    out = dict(run_no=ids, start=starts.tz_localize(None).values.astype("datetime64[D]").astype(str),
               price=win(P.values).astype(np.float32), inflow=win(Q.values).astype(np.float32),
               minflow=win(MF.values).astype(np.float32), minvol=win(MV.values).astype(np.float32),
               vol0=V.reindex(starts).values.astype(np.float32),
               wv_end=W.reindex(starts + pd.Timedelta(days=7)).values.astype(np.float32),
               Y=uc[TARGETS].values.reshape(len(uc), 14, 168).astype(np.int8))
    assert not np.isnan(out["vol0"]).any() and not np.isnan(out["wv_end"]).any()
    return out

CASES = load_cases()
print("cases", len(CASES["run_no"]), "train", len(CASES["Y"]))

# %% [markdown]
# ## 2. LP surrogate of the weekly schedule
# Maximise market income plus the value of stored water at the end of the week, minus flow and switching costs, subject to the water balance of the whole cascade, min flow, min volume, and relaxed commitment `u` in [0, 1].
# The value of a reservoir telescopes: its water value times the conversion of the first plant below, plus the value of the reservoir below that plant.

# %%
M3S_TO_MM3H = 0.0036
K_SEG = 3
# Fixed order among identical units, verified on the training labels.
PRIORITY = [("Vinje_G1", "Vinje_G2"), ("Vinje_G2", "Vinje_G3"), ("Tokke_G1", "Tokke_G3"), ("Tokke_G3", "Tokke_G2"), ("Tokke_G3", "Tokke_G4")]
LP_PRIORITY = PRIORITY + [("Tokke_G4", "Tokke_G2")]
BIDIRECTIONAL = {"Totak_Vaamarvatn": 150.0, "Bordalsvatn_Foersvatn": 15.0}  # tunnels between lakes at the same level
PLANT_ORDER = {"Tokke": ["Tokke_G1", "Tokke_G3", "Tokke_G4", "Tokke_G2"], "Vinje": ["Vinje_G1", "Vinje_G2", "Vinje_G3"]}
NODE_RIVER = "r_Vest_Vassdraget"  # has its own inflow and min flow, passes everything on to Bandak


class Topology:
    def __init__(self, path):
        spec = yaml.safe_load(open(path))
        m = spec["model"]
        self.res, self.riv, self.plant, self.gen, self.tun = m["reservoir"], m["river"], m["plant"], m["generator"], m["tunnel"]
        self.res_names = list(self.res)
        self.edges = {}
        for c in spec["connections"]:
            self.edges.setdefault((c["from_type"], c["from"]), []).append((c["to_type"], c["to"]))
        self.gen_plant = {g: n for (t, g), o in self.edges.items() if t == "generator" for (tt, n) in o}
        self.plant_gens = {p: [g for g in UNITS if self.gen_plant[g] == p] for p in self.plant}
        self.plant_up = {p: self._plant_up(p) for p in self.plant}
        self.plant_down = {p: self._follow(("plant", p)) for p in self.plant}
        self.first_below = {r: self._first_below(r) for r in self.res}
        self.res_vol = {r: np.asarray(v["vol_head"]["x"], float) for r, v in self.res.items()}
        self.res_lvl = {r: np.asarray(v["vol_head"]["y"], float) for r, v in self.res.items()}

    def _plant_up(self, p):
        ups = []
        for (t, s), o in self.edges.items():
            if t == "reservoir" and ("plant", p) in o:
                ups.append(s)
            if t == "tunnel" and ("plant", p) in o:
                ups += [x for (tt, x), oo in self.edges.items() if tt == "reservoir" and ("tunnel", s) in oo]
        return sorted(set(ups), key=lambda r: -self.res[r]["max_vol"])

    def _follow(self, node):
        """Reservoir that eventually receives water leaving this node (None if it leaves the system)."""
        frontier = list(self.edges.get(node, []))
        while frontier:
            t, n = frontier.pop(0)
            if t == "reservoir":
                return n
            frontier += self.edges.get((t, n), [])
        return None

    def _first_below(self, r):
        node, seen = ("reservoir", r), set()
        while node not in seen:
            seen.add(node)
            out = self.edges.get(node, [])
            pl = [n for t, n in out if t == "plant"]
            if pl:
                return pl[0]
            nxt = [o for o in out if o[0] in ("tunnel", "reservoir")]
            if not nxt:
                rank = {"r": 0, "b": 1, "s": 2}
                rv = sorted((o for o in out if o[0] == "river"), key=lambda o: rank.get(o[1].split("_")[0], 3))
                nxt = rv[:1]
            if not nxt:
                return None
            node = nxt[0]
        return None

    def level(self, r, vol):
        return float(np.interp(vol, self.res_vol[r], self.res_lvl[r]))

    def gross_head(self, p, vols):
        return max(self.level(self.plant_up[p][0], vols[self.plant_up[p][0]]) - float(self.plant[p]["outlet_line"]), 1.0)

    def unit_loss(self, g):
        """Marginal head loss of a unit: adding the k-th unit lowers the head of all k running units."""
        gi = self.gen[g]
        pl = self.gen_plant[g]
        p = self.plant[pl]
        main = float(p["main_loss"][0])
        pen = float(p["penstock_loss"][int(gi["penstock"]) - 1])
        if pl == "Songa":  # feed tunnels act in parallel
            feed = [float(self.tun[t]["loss_factor"]) for t in ("Songavatn_Songa", "Bitdalsvatn_Songa")]
            main += 1.0 / sum(1.0 / np.sqrt(f) for f in feed) ** 2
        order = PLANT_ORDER.get(pl, self.plant_gens[pl])
        k = order.index(g) + 1
        n = len([x for x in order[:k] if self.gen[x]["penstock"] == gi["penstock"]])
        return main * (k**3 - (k - 1) ** 3) + pen * (n**3 - (n - 1) ** 3)

    def unit_curve(self, g, gross_head):
        """Concave piecewise-linear P(Q) for one unit: Qmin, Pmin, widths[K], slopes[K] (MW per m3/s)."""
        gi = self.gen[g]
        loss = self.unit_loss(g)
        curves = gi["turb_eff_curves"]
        refs = np.array([c["ref"] for c in curves], float)
        qs = np.linspace(min(c["x"][0] for c in curves), max(c["x"][-1] for c in curves), 80)
        hn = np.maximum(gross_head - loss * qs**2, 1.0)
        eta = np.zeros_like(qs)
        for i, (q, h) in enumerate(zip(qs, hn)):  # turbine efficiency between the two curves nearest in net head
            vals = np.array([np.interp(q, c["x"], c["y"]) for c in curves])
            eta[i] = np.interp(h, refs, vals) if len(refs) > 1 else vals[0]
        pw = 9.81e-3 * hn * qs * eta / 100
        ge = np.interp(pw, gi["gen_eff_curve"]["x"], gi["gen_eff_curve"]["y"]) / 100
        pw = pw * ge
        pmin, pmax = float(gi["p_min"]), float(gi["p_max"])
        pw_mono = np.maximum.accumulate(pw)
        qmin = float(np.interp(pmin, pw_mono, qs))
        reachable = pw_mono[-1] >= pmax
        qmax = float(np.interp(pmax, pw_mono, qs)) if reachable else float(qs[np.argmax(pw)])
        pmax_eff = pmax if reachable else float(pw_mono[-1])
        if qmax <= qmin + 1e-6:
            qmax = qmin + 1e-3
        mask = (qs > qmin) & (qs < qmax)
        qb = np.r_[qmin, qs[mask], qmax]
        pb = np.r_[pmin, pw[mask], pmax_eff]
        hull = [0]  # upper concave hull
        for i in range(1, len(qb)):
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                if (pb[b] - pb[a]) * (qb[i] - qb[a]) <= (pb[i] - pb[a]) * (qb[b] - qb[a]):
                    hull.pop()
                else:
                    break
            hull.append(i)
        qh, ph = qb[hull], pb[hull]
        if len(qh) - 1 > K_SEG:  # keep the start, the best-efficiency point and the end
            best = int(np.argmax(ph / qh))
            mids = sorted({0, best, (best + len(qh) - 1) // 2, len(qh) - 1})
            while len(mids) < K_SEG + 1:
                mids = sorted(set(mids) | {(mids[0] + mids[1]) // 2 + 1})
            qh, ph = qh[mids], ph[mids]
        w, s = np.zeros(K_SEG), np.zeros(K_SEG)
        n = len(qh) - 1
        w[:n] = np.diff(qh)
        s[:n] = np.diff(ph) / np.maximum(np.diff(qh), 1e-9)
        return qmin, pmin, w, s


class WeekLP:
    def __init__(self, topo, T=168):
        self.topo, self.T = topo, T
        self.rivers = [r for r in topo.riv if r != NODE_RIVER]
        self.tunnels = list(topo.tun)
        self.flows = self.rivers + self.tunnels
        self.R = topo.res_names
        self.G = UNITS
        nG, nF, nR = len(self.G), len(self.flows), len(self.R)
        off = 0
        self.iu = off; off += nG * T
        self.isw = off; off += nG * T
        self.id = off; off += nG * K_SEG * T
        self.if_ = off; off += nF * T
        self.iv = off; off += nR * T
        self.isl_v = off; off += len(MINVOL) * T
        self.isl_f = off; off += len(MINFLOW) * T
        self.n = off
        self.fsrc, self.fdst = {}, {}  # where every flow comes from and goes to
        for f in self.rivers:
            self.fsrc[f] = next((s for (t, s), o in topo.edges.items() if t == "reservoir" and ("river", f) in o), None)
            self.fdst[f] = topo._follow(("river", f))
        for f in self.tunnels:
            self.fsrc[f] = next((s for (t, s), o in topo.edges.items() if t == "reservoir" and ("tunnel", f) in o), None)
            self.fdst[f] = next((n for t, n in topo.edges.get(("tunnel", f), []) if t == "reservoir"), None)
        self.songa_tunnels = [f for f in self.tunnels if ("plant", "Songa") in topo.edges.get(("tunnel", f), [])]
        self.vest_in_flows = [f for f in self.rivers if ("river", NODE_RIVER) in topo.edges.get(("river", f), [])]
        self.vest_in_plants = [p for p in topo.plant if ("river", NODE_RIVER) in topo.edges.get(("plant", p), [])]

    def U(self, g, t): return self.iu + g * self.T + t
    def SW(self, g, t): return self.isw + g * self.T + t
    def D(self, g, k, t): return self.id + (g * K_SEG + k) * self.T + t
    def F(self, f, t): return self.if_ + f * self.T + t
    def V(self, r, t): return self.iv + r * self.T + t

    def solve(self, price, inflow, vol0, wv_end, minflow, minvol):
        topo, T = self.topo, self.T
        nG = len(self.G)
        c = np.zeros(self.n)
        lb, ub = np.zeros(self.n), np.full(self.n, np.inf)
        ub[self.iu:self.iu + nG * T] = 1.0
        rows_eq, cols_eq, vals_eq, rhs_eq = [], [], [], []
        rows_in, cols_in, vals_in, rhs_in = [], [], [], []
        ne, ni = 0, 0
        curves = [topo.unit_curve(g, topo.gross_head(topo.gen_plant[g], vol0)) for g in self.G]
        Tidx = np.arange(T)

        for gi, g in enumerate(self.G):
            qmin, pmin, w, s = curves[gi]
            c[self.U(gi, 0):self.U(gi, 0) + T] -= price * pmin
            for k in range(K_SEG):
                i0 = self.D(gi, k, 0)
                c[i0:i0 + T] -= price * s[k]
                # d_k <= w_k * u
                rows_in.append(np.arange(ni, ni + T)); cols_in.append(i0 + Tidx); vals_in.append(np.ones(T))
                rows_in.append(np.arange(ni, ni + T)); cols_in.append(self.U(gi, 0) + Tidx); vals_in.append(np.full(T, -w[k]))
                rhs_in.append(np.zeros(T)); ni += T
            cost = float(topo.gen[g].get("startcost_const", 0))
            # |u_t - u_{t-1}| * cost; start and stop cost are equal, so count each transition once
            c[self.SW(gi, 1):self.SW(gi, 0) + T] += cost
            ub[self.SW(gi, 0)] = 0.0
            for sign in (1.0, -1.0):
                m = T - 1
                rr = np.arange(ni, ni + m)
                rows_in += [rr, rr, rr]
                cols_in += [self.U(gi, 0) + Tidx[1:], self.U(gi, 0) + Tidx[:-1], self.SW(gi, 0) + Tidx[1:]]
                vals_in += [np.full(m, sign), np.full(m, -sign), np.full(m, -1.0)]
                rhs_in.append(np.zeros(m)); ni += m

        for a, b in LP_PRIORITY:  # u_b <= u_a
            ga, gb = self.G.index(a), self.G.index(b)
            rr = np.arange(ni, ni + T)
            rows_in += [rr, rr]
            cols_in += [self.U(gb, 0) + Tidx, self.U(ga, 0) + Tidx]
            vals_in += [np.ones(T), -np.ones(T)]
            rhs_in.append(np.zeros(T)); ni += T

        # Flow bounds and costs
        for fi, f in enumerate(self.flows):
            i0 = self.F(fi, 0)
            if f in topo.tun:
                if f in BIDIRECTIONAL:
                    lb[i0:i0 + T], ub[i0:i0 + T] = -BIDIRECTIONAL[f], BIDIRECTIONAL[f]
                elif f not in self.songa_tunnels:  # Venemo -> Totak, gravity limited
                    src, dst = self.fsrc[f], self.fdst[f]
                    dh = max(topo.level(src, vol0[src]) - topo.level(dst, vol0[dst]), 0.5)
                    ub[i0:i0 + T] = np.sqrt(dh / float(topo.tun[f]["loss_factor"]))
                continue
            info = topo.riv[f]
            if "flow_cost_const" in info:
                c[i0:i0 + T] += float(info["flow_cost_const"])
            elif "flow_cost_curve" in info:
                cx, cy = info["flow_cost_curve"]["x"], info["flow_cost_curve"]["y"]
                c[i0:i0 + T] += float(cy[1] - cy[0]) / float(cx[1] - cx[0])  # slope over the range these rivers carry
            if "max_flow_const" in info:
                ub[i0:i0 + T] = float(info["max_flow_const"])

        # Reservoir balances
        res_in = {r: [] for r in self.R}
        res_out = {r: [] for r in self.R}
        for fi, f in enumerate(self.flows):
            if self.fsrc.get(f):
                res_out[self.fsrc[f]].append(fi)
            if self.fdst.get(f):
                res_in[self.fdst[f]].append(fi)
        for ri, r in enumerate(self.R):
            lb[self.V(ri, 0):self.V(ri, 0) + T] = 0.0
            ub[self.V(ri, 0):self.V(ri, 0) + T] = max(float(topo.res[r]["max_vol"]), 0.0)
            rr = np.arange(ne, ne + T)
            rows_eq.append(rr); cols_eq.append(self.V(ri, 0) + Tidx); vals_eq.append(np.ones(T))
            rows_eq.append(rr[1:]); cols_eq.append(self.V(ri, 0) + Tidx[:-1]); vals_eq.append(-np.ones(T - 1))
            for fi in res_in[r]:
                rows_eq.append(rr); cols_eq.append(self.F(fi, 0) + Tidx); vals_eq.append(np.full(T, -M3S_TO_MM3H))
            for fi in res_out[r]:
                rows_eq.append(rr); cols_eq.append(self.F(fi, 0) + Tidx); vals_eq.append(np.full(T, M3S_TO_MM3H))
            plants_out = [p for p in topo.plant if p != "Songa" and topo.plant_up[p][0] == r]
            plants_in = [p for p in topo.plant if topo.plant_down[p] == r]
            for plist, sign in ((plants_out, 1.0), (plants_in, -1.0)):
                for p in plist:
                    for g in topo.plant_gens[p]:
                        gi = self.G.index(g)
                        qmin = curves[gi][0]
                        rows_eq.append(rr); cols_eq.append(self.U(gi, 0) + Tidx); vals_eq.append(np.full(T, sign * M3S_TO_MM3H * qmin))
                        for k in range(K_SEG):
                            rows_eq.append(rr); cols_eq.append(self.D(gi, k, 0) + Tidx); vals_eq.append(np.full(T, sign * M3S_TO_MM3H))
            b = M3S_TO_MM3H * inflow.get(r, np.zeros(T)).copy()
            if r == "Bandak":  # r_Vest_Vassdraget's own inflow goes straight to Bandak
                b = b + M3S_TO_MM3H * inflow[NODE_RIVER]
            b[0] += vol0[r]
            rhs_eq.append(b); ne += T

        # Songa: two tunnels feed the plant
        rr = np.arange(ne, ne + T)
        for f in self.songa_tunnels:
            rows_eq.append(rr); cols_eq.append(self.F(self.flows.index(f), 0) + Tidx); vals_eq.append(np.ones(T))
        gi = self.G.index("Songa_G1")
        rows_eq.append(rr); cols_eq.append(self.U(gi, 0) + Tidx); vals_eq.append(np.full(T, -curves[gi][0]))
        for k in range(K_SEG):
            rows_eq.append(rr); cols_eq.append(self.D(gi, k, 0) + Tidx); vals_eq.append(-np.ones(T))
        rhs_eq.append(np.zeros(T)); ne += T

        # Min volume with slack
        for si, r in enumerate(MINVOL):
            ri = self.R.index(r)
            rr = np.arange(ni, ni + T)
            i0 = self.isl_v + si * T
            rows_in += [rr, rr]; cols_in += [self.V(ri, 0) + Tidx, i0 + Tidx]; vals_in += [-np.ones(T), -np.ones(T)]
            rhs_in.append(-minvol[r]); ni += T
            c[i0:i0 + T] += 5e5  # per Mm3: far above any water value
        # Min flow with slack
        for si, f in enumerate(MINFLOW):
            rr = np.arange(ni, ni + T)
            i0 = self.isl_f + si * T
            rows_in.append(rr); cols_in.append(i0 + Tidx); vals_in.append(-np.ones(T))
            c[i0:i0 + T] += 2000.0
            if f == NODE_RIVER:
                for x in self.vest_in_flows:
                    rows_in.append(rr); cols_in.append(self.F(self.flows.index(x), 0) + Tidx); vals_in.append(-np.ones(T))
                for p in self.vest_in_plants:
                    for g in topo.plant_gens[p]:
                        gi = self.G.index(g)
                        rows_in.append(rr); cols_in.append(self.U(gi, 0) + Tidx); vals_in.append(np.full(T, -curves[gi][0]))
                        for k in range(K_SEG):
                            rows_in.append(rr); cols_in.append(self.D(gi, k, 0) + Tidx); vals_in.append(-np.ones(T))
                rhs_in.append(inflow[NODE_RIVER] - minflow[f])
            else:
                rows_in.append(rr); cols_in.append(self.F(self.flows.index(f), 0) + Tidx); vals_in.append(-np.ones(T))
                rhs_in.append(-minflow[f])
            ni += T

        # Terminal value of stored water
        def conv_of(pl):
            """MWh per Mm3 at the best point of the plant's first unit, with its head losses."""
            lead = PLANT_ORDER.get(pl, topo.plant_gens[pl])[0]
            q0, p0, w_, s_ = curves[self.G.index(lead)]
            qs_, ps_ = q0 + np.r_[0, np.cumsum(w_)], p0 + np.r_[0, np.cumsum(w_ * s_)]
            return float(np.max(ps_ / qs_)) / M3S_TO_MM3H

        def telescoped(r):
            fp = topo.first_below[r]
            if fp is None:
                return 0.0
            down = topo.plant_down[fp]
            return wv_end[r] * conv_of(fp) + (telescoped(down) if down else 0.0)

        for r in self.R:
            if topo.first_below[r] is not None:
                c[self.V(self.R.index(r), T - 1)] -= telescoped(r)

        Aeq = sp.csr_matrix((np.concatenate(vals_eq), (np.concatenate(rows_eq), np.concatenate(cols_eq))), shape=(ne, self.n))
        Aub = sp.csr_matrix((np.concatenate(vals_in), (np.concatenate(rows_in), np.concatenate(cols_in))), shape=(ni, self.n))
        res = linprog(c, A_ub=Aub, b_ub=np.concatenate(rhs_in), A_eq=Aeq, b_eq=np.concatenate(rhs_eq), bounds=np.c_[lb, ub],
                      method="highs", options={"time_limit": 120.0, "presolve": True})
        assert res.x is not None, res.message
        x = res.x
        u = np.stack([x[self.U(g, 0):self.U(g, 0) + T] for g in range(nG)])
        p = np.zeros((nG, T))
        for gi in range(nG):
            qmin, pmin, w, s = curves[gi]
            p[gi] = pmin * u[gi]
            for k in range(K_SEG):
                p[gi] += s[k] * x[self.D(gi, k, 0):self.D(gi, k, 0) + T]
        return dict(u=u, pfrac=p / np.array([float(topo.gen[g]["p_max"]) for g in self.G])[:, None],
                    V=np.stack([x[self.V(r, 0):self.V(r, 0) + T] for r in range(len(self.R))]),
                    dual=-res.eqlin.marginals[:len(self.R) * T].reshape(len(self.R), T),  # EUR per Mm3
                    conv=np.array([conv_of(topo.gen_plant[g]) for g in self.G]),
                    flows=np.stack([x[self.F(f, 0):self.F(f, 0) + T] for f in range(len(self.flows))]))

TOPO = Topology(find("Tokke_Vinje_topology.yaml"))
LP = WeekLP(TOPO)

# %% [markdown]
# Solve all 3647 cases three times. The two runs with scaled water values show how fragile each decision is.

# %%
_LP_ARGS = {}

def solve_case(i):
    c, ws = CASES, _LP_ARGS["ws"]
    r = LP.solve(c["price"][i].astype(float), {n: c["inflow"][i, :, k].astype(float) for k, n in enumerate(INFLOW)},
                 {n: float(c["vol0"][i, k]) for k, n in enumerate(RES)}, {n: float(c["wv_end"][i, k]) * ws for k, n in enumerate(RES)},
                 {n: c["minflow"][i, :, k].astype(float) for k, n in enumerate(MINFLOW)}, {n: c["minvol"][i, :, k].astype(float) for k, n in enumerate(MINVOL)})
    return {k: r[k].astype(np.float32) for k in _LP_ARGS["keep"]}

def solve_all(name, ws, keep):
    path = WORK_DIR / f"{name}.npz"
    if path.exists():
        z = np.load(path); return {k: z[k] for k in z.files}
    t0 = time.time(); _LP_ARGS.update(ws=ws, keep=keep); idx = range(len(CASES["run_no"]))
    if "fork" in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context("fork").Pool(N_CORES) as pool:
            res = list(pool.imap(solve_case, idx, chunksize=1))
    else:  # e.g. Windows: one core, within the time limit
        res = [solve_case(i) for i in idx]
    out = {k: np.stack([r[k] for r in res]) for k in keep}
    np.savez(path, **out); print(name, "solved in", round(time.time() - t0), "s", flush=True)
    return out

t0 = time.time()
Z = solve_all("lp_base", 1.0, ("u", "pfrac", "V", "dual", "conv", "flows"))
SENS = [solve_all("lp_w90", 0.9, ("u",)), solve_all("lp_w110", 1.1, ("u",))]
TIMES["lp"] = time.time() - t0

# %% [markdown]
# ## 3. Features
# One row per (case, unit, hour). Market and calendar features, unit and reservoir state, environmental constraints, and the LP solution with its neighbours in time, plant and system.

# %%
PLANTS = list(TOPO.plant)
UP = [TOPO.plant_up[TOPO.gen_plant[u]][0] for u in UNITS]
DOWN = [TOPO.plant_down[TOPO.gen_plant[u]] for u in UNITS]
ABOVE = {p: [q for q in PLANTS if TOPO.plant_down[q] and TOPO.first_below[TOPO.plant_down[q]] == p] for p in PLANTS}
BELOW = {p: (TOPO.first_below[TOPO.plant_down[p]] if TOPO.plant_down[p] else None) for p in PLANTS}
PMAX = np.array([float(TOPO.gen[u]["p_max"]) for u in UNITS]); START = np.array([float(TOPO.gen[u]["startcost_const"]) for u in UNITS])
CATEGORICAL = ["unit_id", "plant_id"]

def _rank(a, axis=-1):
    return np.argsort(np.argsort(a, axis=axis, kind="stable"), axis=axis, kind="stable") / max(a.shape[axis] - 1, 1)

def _roll(a, w):
    """Centred moving mean over the last axis, shrinking at the edges."""
    pad = np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a, axis=-1)], axis=-1); T = a.shape[-1]
    lo = np.clip(np.arange(T) - w // 2, 0, T); hi = np.clip(np.arange(T) + w // 2 + 1, 0, T)
    return (pad[..., hi] - pad[..., lo]) / (hi - lo)

def _shift(a, k):
    out = np.roll(a, k, axis=-1)
    if k > 0: out[..., :k] = a[..., :1]
    elif k < 0: out[..., k:] = a[..., -1:]
    return out

def build(c, z, sens):
    """Returns X (N, 14, 168, F) float32 and the column names."""
    N = len(c["run_no"]); T = 168; G = 14
    p = c["price"].astype(np.float64)
    q25, q50, q75 = np.percentile(p, [25, 50, 75], axis=1); scale = (q75 - q25 + 1.0)[:, None]
    hrs = pd.to_datetime(c["start"]).tz_localize("UTC")
    local_all = pd.DatetimeIndex(hrs).tz_convert("Europe/Oslo")
    off = np.array([(l.utcoffset().total_seconds() / 3600) for l in local_all]).astype(int)
    hour = (np.arange(T)[None] + off[:, None]) % 24
    wday = ((local_all.dayofweek.values[:, None] * 24 + off[:, None] + np.arange(T)[None]) // 24) % 7
    doy = hrs.dayofyear.values
    day_rank = np.concatenate([_rank(p[:, d * 24:(d + 1) * 24], 1) for d in range(7)], axis=1)
    F_, names = [], []
    def add(name, a):
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1: a = np.broadcast_to(a[:, None, None], (N, G, T))
        elif a.shape == (N, T): a = np.broadcast_to(a[:, None, :], (N, G, T))
        elif a.shape == (N, G): a = np.broadcast_to(a[:, :, None], (N, G, T))
        F_.append(a); names.append(name)
    # market and calendar
    add("price_asinh", np.arcsinh(p / 10)); add("price_z", (p - q50[:, None]) / scale); add("price_rank_week", _rank(p, 1)); add("price_rank_day", day_rank)
    add("price_diff_prev", np.diff(p, axis=1, prepend=p[:, :1]) / scale); add("price_diff_next", np.diff(p, axis=1, append=p[:, -1:]) / scale)
    add("price_dev_7h", (p - _roll(p, 7)) / scale); add("price_dev_25h", (p - _roll(p, 25)) / scale)
    add("week_price_mean_asinh", np.arcsinh(p.mean(1) / 10)); add("week_price_iqr_asinh", np.arcsinh((q75 - q25) / 10))
    add("hour_local", hour); add("weekday_local", wday); add("doy_sin", np.sin(2 * np.pi * doy / 365.25)); add("doy_cos", np.cos(2 * np.pi * doy / 365.25)); add("horizon_hour", np.broadcast_to(np.arange(T), (N, T)))
    # unit
    ri = {r: i for i, r in enumerate(RES)}; qi = {r: i for i, r in enumerate(INFLOW)}
    wv = c["wv_end"].astype(np.float64); vol = c["vol0"].astype(np.float64)
    maxv = {r: float(TOPO.res[r]["max_vol"]) for r in RES}
    wu = np.stack([wv[:, ri[r]] for r in UP], 1); wd = np.stack([wv[:, ri[d]] if d else np.zeros(N) for d in DOWN], 1)
    fu = np.stack([vol[:, ri[r]] / maxv[r] for r in UP], 1); fd = np.stack([vol[:, ri[d]] / maxv[d] if d else np.zeros(N) for d in DOWN], 1)
    iu = np.stack([c["inflow"][:, :, qi[r]].mean(1) for r in UP], 1)
    uid = np.broadcast_to(np.arange(G), (N, G)); pid = np.broadcast_to(np.array([PLANTS.index(TOPO.gen_plant[u]) for u in UNITS]), (N, G))
    add("unit_id", uid); add("plant_id", pid); add("unit_index", np.broadcast_to(np.array([int(u[-1]) for u in UNITS]), (N, G)))
    add("p_max", np.broadcast_to(PMAX, (N, G))); add("start_per_mw", np.broadcast_to(START / PMAX, (N, G)))
    add("wv_up_asinh", np.arcsinh(wu / 10)); add("wv_down_asinh", np.arcsinh(wd / 10)); add("fill_up", fu); add("fill_down", fd); add("inflow_up_asinh", np.arcsinh(iu))
    add("margin_wv_up", (p[:, None] - wu[:, :, None]) / scale[:, None]); add("price_over_wv_up", p[:, None] / (np.abs(wu)[:, :, None] + 1))
    add("econ_frac_wv", (p[:, None] > wu[:, :, None]).mean(2)); add("start_hours_wv", START[None] / (PMAX[None] * (np.abs(wu) + 1)))
    # environmental constraints
    need = np.maximum(c["minflow"][:, :, MINFLOW.index("r_Vest_Vassdraget")] - c["inflow"][:, :, INFLOW.index("r_Vest_Vassdraget")], 0).astype(np.float64)
    add("mf_need_vest_t", need); add("mf_need_vest_week", need.mean(1)); add("mf_need_vest_max", need.max(1))
    gap = []
    for r in UP:
        if r in MINVOL: gap.append((vol[:, ri[r]] - c["minvol"][:, :, MINVOL.index(r)].mean(1)) / maxv[r])
        else: gap.append(vol[:, ri[r]] / maxv[r])
    add("mf_gap_up", np.stack(gap, 1))
    # LP solution
    u = z["u"].astype(np.float64); pf = z["pfrac"].astype(np.float64); V = z["V"].astype(np.float64); dual = z["dual"].astype(np.float64); conv = z["conv"].astype(np.float64)
    ro = TOPO.res_names; rx = {r: i for i, r in enumerate(ro)}
    lam_up = np.stack([dual[:, rx[r]] for r in UP], 1); lam_dn = np.stack([dual[:, rx[d]] if d else np.zeros((N, T)) for d in DOWN], 1)
    be = (lam_up - lam_dn) / conv[:, :, None]  # the LP's marginal cost of running the unit, EUR/MWh
    margin = p[:, None] - be
    add("lp_u", u); add("lp_p", pf); add("lp_margin", margin / scale[:, None]); add("lp_margin_ratio", p[:, None] / (np.abs(be) + 1)); add("lp_be_asinh", np.arcsinh(be / 10))
    add("lp_be_gap_wv", (be - wu[:, :, None]) / scale[:, None])
    for k in (1, 2): add(f"lp_u_prev{k}", _shift(u, k)); add(f"lp_u_next{k}", _shift(u, -k))
    for w in (5, 13, 25): add(f"lp_u_roll{w}", _roll(u, w))
    add("lp_margin_roll5", _roll(margin, 5) / scale[:, None])
    add("lp_week_u", u.mean(2)); add("lp_week_p", pf.mean(2)); add("lp_day_u", _roll(u, 25)); add("lp_switches", np.abs(np.diff(u, axis=2)).sum(2))
    # hours to the nearest change in the LP schedule, signed by whether the LP has the unit on
    on = u > 0.5; ch = np.diff(on.astype(np.int8), axis=2, prepend=on[:, :, :1].astype(np.int8)) != 0
    last = np.where(ch, np.arange(T), -1000); last = np.maximum.accumulate(last, axis=2)
    nxt = np.where(ch, np.arange(T), 1000); nxt = np.minimum.accumulate(nxt[:, :, ::-1], axis=2)[:, :, ::-1]
    dist = np.minimum(np.arange(T) - last, nxt - np.arange(T)).clip(0, 48)
    add("lp_dist_switch", dist * (2 * on - 1))
    # same plant, neighbouring plants, whole system
    plant_of = [TOPO.gen_plant[x] for x in UNITS]
    cap = {pl: sum(PMAX[i] for i, x in enumerate(plant_of) if x == pl) for pl in PLANTS}
    pl_mw = {pl: (pf * PMAX[None, :, None])[:, [i for i, x in enumerate(plant_of) if x == pl]].sum(1) / cap[pl] for pl in PLANTS}
    add("lp_plant_t", np.stack([pl_mw[x] for x in plant_of], 1)); add("lp_plant_week", np.stack([pl_mw[x].mean(1) for x in plant_of], 1))
    zero = np.zeros((N, T))
    above = np.stack([sum((pl_mw[q] for q in ABOVE[x]), zero) for x in plant_of], 1); below = np.stack([pl_mw[BELOW[x]] if BELOW[x] else zero for x in plant_of], 1)
    add("lp_above_t", above); add("lp_below_t", below); add("lp_above_week", above.mean(2)); add("lp_below_week", below.mean(2))
    sys_t = (pf * PMAX[None, :, None]).sum(1) / PMAX.sum(); add("lp_sys_t", sys_t); add("lp_sys_week", sys_t.mean(1))
    for pl in PLANTS: add(f"lps_{pl.replace(' ', '')}", pl_mw[pl].mean(1))
    # reservoir trajectory
    vmax = np.array([maxv[r] for r in ro])[None, :, None]; Vf = V / vmax
    up_f = np.stack([Vf[:, rx[r]] for r in UP], 1); dn_f = np.stack([Vf[:, rx[d]] if d else zero for d in DOWN], 1)
    add("lp_fill_up_t", up_f); add("lp_fill_down_t", dn_f); add("lp_fill_up_end", up_f[:, :, -1]); add("lp_fill_down_end", dn_f[:, :, -1])
    add("lp_up_full", (up_f > .995).mean(2)); add("lp_up_empty", (up_f < .005).mean(2)); add("lp_dn_full", (dn_f > .995).mean(2))
    spill = [i for i, f in enumerate(LP.flows) if f in TOPO.riv and f.split("_")[0] in ("s", "b")]
    src = np.array([rx[LP.fsrc[LP.flows[i]]] for i in spill]); fl = z["flows"][:, spill].astype(np.float64)
    res_sp = np.stack([fl[:, src == k].sum(1) for k in range(len(ro))], 1)
    sp_up = np.stack([res_sp[:, rx[r]] for r in UP], 1); add("lp_bypass_up_week", np.arcsinh(sp_up.mean(2))); add("lp_bypass_up_t", np.arcsinh(sp_up))
    # the same LP with water values scaled by 0.9 and 1.1: how fragile is each decision?
    lo, hi = sens[0]["u"].astype(np.float64), sens[1]["u"].astype(np.float64)
    add("ls_u_cheap", lo); add("ls_u_dear", hi); add("ls_band", lo - hi); add("ls_week_cheap", lo.mean(2)); add("ls_week_dear", hi.mean(2)); add("ls_week_band", (lo - hi).mean(2)); add("ls_mean", (lo + u + hi) / 3)
    return np.stack(F_, -1), names

def group(name):
    for g in ("ls", "mf", "lps", "lp"):
        if name.startswith(g + "_"): return g
    return "base"

t0 = time.time()
X_ALL, NAMES = build(CASES, Z, SENS)
Y = CASES["Y"]
del Z, SENS
tree_keep = [i for i, n in enumerate(NAMES) if group(n) in TREE_GROUPS and n not in TREE_DROP]
net_keep = [i for i, n in enumerate(NAMES) if group(n) in NET_GROUPS and n not in NET_DROP and n not in CATEGORICAL]
TIMES["features"] = time.time() - t0
print("features: trees", len(tree_keep), "net", len(net_keep))

# %% [markdown]
# ## 4a. XGBoost
# Hyperparameters and their origin are in the report. Output is in logit space.

# %%
t0 = time.time()
X = X_ALL[..., tree_keep]; tree_names = [NAMES[i] for i in tree_keep]
XGB_PARAMS = dict(objective="binary:logistic", device=DEVICE, tree_method="hist", eta=.08, grow_policy="lossguide", max_leaves=63, max_depth=0,
                  min_child_weight=50., subsample=.5, colsample_bytree=.8, reg_lambda=1., max_bin=255, verbosity=0, nthread=N_CORES)
dtrain = xgb.DMatrix(X[:N_TRAIN].reshape(-1, X.shape[-1]), label=Y.reshape(-1), feature_names=tree_names)
dtest = xgb.DMatrix(X[N_TRAIN:].reshape(-1, X.shape[-1]), feature_names=tree_names)
tree_preds = []
for s in TREE_SEEDS:
    m = xgb.train({**XGB_PARAMS, "seed": s}, dtrain, num_boost_round=300)
    tree_preds.append(m.predict(dtest, output_margin=True).reshape(-1, 14, 168)); print("xgb seed", s, round(time.time() - t0), "s", flush=True)
tree_logit = np.mean(tree_preds, 0)
del dtrain, dtest, X, m
TIMES["xgboost"] = time.time() - t0

# %% [markdown]
# ## 4b. ST-Net
# One token per (unit, hour): dilated convolutions over the week, attention across units, one Transformer layer over the week. Output respects the fixed priority among identical units.

# %%
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
IDX = {u: i for i, u in enumerate(UNITS)}; PAIRS = [(IDX[a], IDX[b]) for a, b in PRIORITY]

def priority(raw):
    z = list(raw.unbind(1))
    for par, ch in PAIRS: z[ch] = z[par] - F.softplus(z[ch])
    return torch.stack(z, 1)

class Mixer(nn.Module):
    def __init__(s, d, h, p):
        super().__init__(); s.n = nn.LayerNorm(d); s.a = nn.MultiheadAttention(d, h, dropout=p, batch_first=True); s.b = nn.Parameter(torch.zeros(14, 14)); s.dr = nn.Dropout(p)
    def forward(s, x):
        B, U, T, d = x.shape; h = s.n(x).permute(0, 2, 1, 3).reshape(B * T, U, d)
        h, _ = s.a(h, h, h, attn_mask=s.b, need_weights=False)
        return x + s.dr(h.reshape(B, T, U, d).permute(0, 2, 1, 3))

class TimeBlock(nn.Module):
    def __init__(s, d, dil, p):
        super().__init__(); s.n = nn.GroupNorm(1, d); s.c1 = nn.Conv1d(d, d, 3, padding=dil, dilation=dil); s.c2 = nn.Conv1d(d, d, 3, padding=dil, dilation=dil); s.f = nn.Linear(d, 2 * d); s.dr = nn.Dropout(p)
    def forward(s, x, c):
        B, U, T, d = x.shape; h = x.reshape(B * U, T, d).transpose(1, 2); z = s.c1(F.gelu(s.n(h)))
        g, b = s.f(c.reshape(B * U, d)).chunk(2, -1); z = z * (1 + g[..., None]) + b[..., None]; z = s.c2(s.dr(F.gelu(z)))
        return x + z.transpose(1, 2).reshape(B, U, T, d)

class Net(nn.Module):
    def __init__(s, nf, d=64, blocks=4, heads=4, p=.1):
        super().__init__(); s.inp = nn.Linear(nf, d); s.ue = nn.Parameter(torch.randn(14, d) * .02); s.pos = nn.Parameter(torch.randn(168, d) * .02)
        s.c = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))
        s.tb = nn.ModuleList(TimeBlock(d, 2 ** i, p) for i in range(blocks)); s.mx = nn.ModuleList(Mixer(d, heads, p) for _ in range(blocks))
        s.at = nn.TransformerEncoderLayer(d, heads, 4 * d, p, activation="gelu", batch_first=True, norm_first=True)
        s.on = nn.LayerNorm(d); s.h = nn.Linear(d, 1); s.lv = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
    def forward(s, X):
        B, U, T, _ = X.shape; x = s.inp(X) + s.ue[:, None] + s.pos; c = s.c(x.mean(2)) + s.ue
        for t, m in zip(s.tb, s.mx): x = m(t(x, c))
        d = x.shape[-1]; x = s.at(x.reshape(B * U, T, d)).reshape(B, U, T, d); x = s.on(x)
        return priority(s.h(x).squeeze(-1)), s.lv(torch.cat([x.mean(2), c], -1)).squeeze(-1)

def fit_net(Xtr, Ytr, seed, epochs, batch=32, lr=1e-3, wd=.01, ema_decay=.995, aux=.3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(Xtr.shape[-1]).to(DEVICE); ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd); steps = epochs * math.ceil(len(Ytr) / batch); wu = max(1, int(.05 * steps))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1., (k + 1) / wu) * .5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    gen = torch.Generator().manual_seed(seed); scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda"); net.train()
    for ep in range(epochs):
        for idx in torch.randperm(len(Ytr), generator=gen).split(batch):
            idx = idx.to(DEVICE); xb, yb = Xtr[idx], Ytr[idx]
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda", dtype=AMP_DTYPE):
                lg, lv = net(xb); loss = F.binary_cross_entropy_with_logits(lg.float(), yb) + aux * F.binary_cross_entropy_with_logits(lv.float(), yb.mean(-1))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.); scaler.step(opt); scaler.update(); sch.step(); ema.update_parameters(net)
    ema.eval(); return ema

@torch.no_grad()
def predict_net(m, X, bs=64):
    out = []
    for i in range(0, len(X), bs):
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda", dtype=AMP_DTYPE): out.append(m(X[i:i + bs].to(DEVICE))[0].float().cpu().numpy())
    return np.concatenate(out)

t0 = time.time()
X = X_ALL[..., net_keep]; del X_ALL
mean_, std_ = X[:N_TRAIN].mean((0, 1, 2)), X[:N_TRAIN].std((0, 1, 2)); std_ = np.where(std_ > 1e-6, std_, 1.)
scaled = lambda x: torch.from_numpy(np.clip((x - mean_) / std_, -6, 6).astype(np.float32))
Xtr = scaled(X[:N_TRAIN]).to(DEVICE); Ytr = torch.from_numpy(Y.astype(np.float32)).to(DEVICE); Xte = scaled(X[N_TRAIN:])
net_preds = []
for s in NET_SEEDS:
    net_preds.append(predict_net(fit_net(Xtr, Ytr, s, NET_EPOCHS), Xte)); print("net seed", s, round(time.time() - t0), "s", flush=True)
net_logit = np.mean(net_preds, 0)
gpu_peak = torch.cuda.max_memory_reserved() / 2**30 if DEVICE == "cuda" else 0.
TIMES["st_net"] = time.time() - t0

# %% [markdown]
# ## 5. Blend and submission

# %%
sub = pd.read_csv(find("sample_submission.csv"))
blend = TREE_WEIGHT * tree_logit.astype(np.float64) + (1 - TREE_WEIGHT) * net_logit.astype(np.float64)
prob = 1 / (1 + np.exp(-blend))
assert list(sub.columns[1:]) == TARGETS and (sub["Run No"].values == CASES["run_no"][N_TRAIN:]).all()
assert prob.shape == (725, 14, 168) and np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
out = pd.DataFrame(prob.reshape(725, -1), columns=TARGETS); out.insert(0, "Run No", sub["Run No"].values)
out.to_csv(OUTPUT, index=False, float_format="%.8f")
check = pd.read_csv(OUTPUT); assert check.shape == sub.shape and list(check.columns) == list(sub.columns)
np.savez(WORK_DIR / "logits.npz", tree=tree_logit, net=net_logit)

print("wrote", OUTPUT, check.shape, "mean p", round(float(prob.mean()), 4))
print("minutes", {k: round(v / 60, 1) for k, v in TIMES.items()}, "total", round((time.time() - T_START) / 60, 1))
print("peak ST-Net GPU GB", round(gpu_peak, 1))
