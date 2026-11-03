"""Case-level features: market and hydraulic basics plus the LP surrogate's solution. Fully vectorised.
Every column is computed only from what SHOP gets for that case, plus the LP solved on the same inputs."""
import numpy as np, pandas as pd
from common import RES, INFLOW, ROOT
from lp_surrogate import Topology, WeekLP, UNITS

TOPO = Topology(f"{ROOT}/data/extended/Tokke_Vinje_topology.yaml")
PLANTS = list(TOPO.plant)
UP = [TOPO.plant_up[TOPO.gen_plant[u]][0] for u in UNITS]
DOWN = [TOPO.plant_down[TOPO.gen_plant[u]] for u in UNITS]
ABOVE = {p: [q for q in PLANTS if TOPO.plant_down[q] and TOPO.first_below[TOPO.plant_down[q]] == p] for p in PLANTS}
BELOW = {p: (TOPO.first_below[TOPO.plant_down[p]] if TOPO.plant_down[p] else None) for p in PLANTS}
PMAX = np.array([float(TOPO.gen[u]["p_max"]) for u in UNITS]); PMIN = np.array([float(TOPO.gen[u]["p_min"]) for u in UNITS])
START = np.array([float(TOPO.gen[u]["startcost_const"]) for u in UNITS])
CATEGORICAL = ["unit_id", "plant_id"]
MODEL_INPUTS = ("run_no", "start", "price", "inflow", "minflow", "minvol", "vol0", "wv_end")  # exactly what SHOP gets for a case

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

def build(c, z, m=None, sens=None):
    """Returns X (N, 14, 168, F) float32 and the column names."""
    c = {k: c[k] for k in MODEL_INPUTS}  # labels, wv_start and anything else are structurally out of reach
    N = len(c["run_no"]); T = 168; G = 14
    p = c["price"].astype(np.float64)
    q25, q50, q75 = np.percentile(p, [25, 50, 75], axis=1); scale = (q75 - q25 + 1.0)[:, None]
    hrs = pd.to_datetime(c["start"]).tz_localize("UTC")
    start_h = np.array([d for d in (hrs.hour)])
    local_all = pd.DatetimeIndex(hrs).tz_convert("Europe/Oslo")
    off = np.array([(l.utcoffset().total_seconds() / 3600) for l in local_all]).astype(int)
    hour = (np.arange(T)[None] + off[:, None]) % 24
    wday = ((local_all.dayofweek.values[:, None] * 24 + off[:, None] + np.arange(T)[None]) // 24) % 7
    doy = hrs.dayofyear.values
    day_rank = np.concatenate([_rank(p[:, d * 24:(d + 1) * 24], 1) for d in range(7)], axis=1)
    F, names = [], []
    def add(name, a):
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1: a = np.broadcast_to(a[:, None, None], (N, G, T))
        elif a.shape == (N, T): a = np.broadcast_to(a[:, None, :], (N, G, T))
        elif a.shape == (N, G): a = np.broadcast_to(a[:, :, None], (N, G, T))
        F.append(a); names.append(name)
    # --- market and calendar
    add("price_asinh", np.arcsinh(p / 10)); add("price_z", (p - q50[:, None]) / scale); add("price_rank_week", _rank(p, 1)); add("price_rank_day", day_rank)
    add("price_diff_prev", np.diff(p, axis=1, prepend=p[:, :1]) / scale); add("price_diff_next", np.diff(p, axis=1, append=p[:, -1:]) / scale)
    add("price_dev_7h", (p - _roll(p, 7)) / scale); add("price_dev_25h", (p - _roll(p, 25)) / scale)
    add("week_price_mean_asinh", np.arcsinh(p.mean(1) / 10)); add("week_price_iqr_asinh", np.arcsinh((q75 - q25) / 10))
    add("hour_local", hour); add("weekday_local", wday); add("doy_sin", np.sin(2 * np.pi * doy / 365.25)); add("doy_cos", np.cos(2 * np.pi * doy / 365.25)); add("horizon_hour", np.broadcast_to(np.arange(T), (N, T)))
    # --- unit
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
    # --- environmental constraints: the flow the river below Lio must get from upstream, and room above min volume
    from common import MINFLOW, MINVOL
    need = np.maximum(c["minflow"][:, :, MINFLOW.index("r_Vest_Vassdraget")] - c["inflow"][:, :, INFLOW.index("r_Vest_Vassdraget")], 0).astype(np.float64)
    add("mf_need_vest_t", need); add("mf_need_vest_week", need.mean(1)); add("mf_need_vest_max", need.max(1))
    gap = []
    for r in UP:
        if r in MINVOL: gap.append((vol[:, ri[r]] - c["minvol"][:, :, MINVOL.index(r)].mean(1)) / maxv[r])
        else: gap.append(vol[:, ri[r]] / maxv[r])
    add("mf_gap_up", np.stack(gap, 1))
    if z is None: return np.stack(F, -1), names
    # --- LP solution
    assert (z["idx"] == np.arange(N)).all()
    u = z["u"].astype(np.float64); pf = z["pfrac"].astype(np.float64); V = z["V"].astype(np.float64); dual = z["dual"].astype(np.float64); conv = z["conv"].astype(np.float64)
    ro = TOPO.res_names; rx = {r: i for i, r in enumerate(ro)}
    lam_up = np.stack([dual[:, rx[r]] for r in UP], 1); lam_dn = np.stack([dual[:, rx[d]] if d else np.zeros((N, T)) for d in DOWN], 1)
    be = (lam_up - lam_dn) / conv[:, :, None]                     # LP's marginal cost of running the unit, EUR/MWh
    margin = p[:, None] - be
    add("lp_u", u); add("lp_p", pf); add("lp_margin", margin / scale[:, None]); add("lp_margin_ratio", p[:, None] / (np.abs(be) + 1)); add("lp_be_asinh", np.arcsinh(be / 10))
    add("lp_be_gap_wv", (be - wu[:, :, None]) / scale[:, None])
    for k in (1, 2): add(f"lp_u_prev{k}", _shift(u, k)); add(f"lp_u_next{k}", _shift(u, -k))
    for w in (5, 13, 25): add(f"lp_u_roll{w}", _roll(u, w))
    add("lp_margin_roll5", _roll(margin, 5) / scale[:, None])
    add("lp_week_u", u.mean(2)); add("lp_week_p", pf.mean(2)); add("lp_day_u", _roll(u, 25)); add("lp_switches", np.abs(np.diff(u, axis=2)).sum(2))
    # hours to the nearest change in the LP schedule (signed by whether the LP has the unit on)
    on = u > 0.5; ch = np.diff(on.astype(np.int8), axis=2, prepend=on[:, :, :1].astype(np.int8)) != 0
    last = np.where(ch, np.arange(T), -1000); last = np.maximum.accumulate(last, axis=2)
    nxt = np.where(ch, np.arange(T), 1000); nxt = np.minimum.accumulate(nxt[:, :, ::-1], axis=2)[:, :, ::-1]
    dist = np.minimum(np.arange(T) - last, nxt - np.arange(T)).clip(0, 48)
    add("lp_dist_switch", dist * (2 * on - 1))
    # same plant, neighbouring plants, and whole system
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
    lp = WeekLP(TOPO); spill = [i for i, f in enumerate(lp.flows) if f in TOPO.riv and f.split("_")[0] in ("s", "b")]
    src = np.array([rx[lp.fsrc[lp.flows[i]]] for i in spill]); fl = z["flows"][:, spill].astype(np.float64)
    res_sp = np.stack([fl[:, src == k].sum(1) for k in range(len(ro))], 1)
    sp_up = np.stack([res_sp[:, rx[r]] for r in UP], 1); add("lp_bypass_up_week", np.arcsinh(sp_up.mean(2))); add("lp_bypass_up_t", np.arcsinh(sp_up))
    if m is not None:  # integer schedule from the MILP
        mu = m["u"].astype(np.float64); mp = m["pfrac"].astype(np.float64)
        add("mp_u", mu); add("mp_p", mp); add("mp_week_u", mu.mean(2)); add("mp_week_p", mp.mean(2)); add("mp_switches", np.abs(np.diff(mu, axis=2)).sum(2))
        for k in (1, 2): add(f"mp_u_prev{k}", _shift(mu, k)); add(f"mp_u_next{k}", _shift(mu, -k))
        for w in (5, 13): add(f"mp_u_roll{w}", _roll(mu, w))
        add("mp_gap_lp", mu - u)
    if sens is not None:  # the same LP with water values scaled by 0.9 and 1.1: how fragile is each decision?
        lo, hi = sens[0]["u"].astype(np.float64), sens[1]["u"].astype(np.float64)
        add("ls_u_cheap", lo); add("ls_u_dear", hi); add("ls_band", lo - hi); add("ls_week_cheap", lo.mean(2)); add("ls_week_dear", hi.mean(2)); add("ls_week_band", (lo - hi).mean(2)); add("ls_mean", (lo + u + hi) / 3)
    return np.stack(F, -1), names

def group(name):
    if name.startswith("ls_"): return "ls"
    if name.startswith("mf_"): return "mf"
    if name.startswith("mp_"): return "mp"
    if name.startswith("lps_"): return "lps"
    if name.startswith("lp_"): return "lp"
    return "base"
