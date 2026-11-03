"""LP surrogate of the SHOP weekly scheduling problem for Tokke-Vinje.

Uses only what SHOP is given for a case: 168 h of price, inflow and constraints, the start volumes,
the end-of-horizon water values, and the topology file. No labels. The result is a relaxed schedule
that respects the water balance of the entire cascade, which is the part tree and neural models can
only approximate.

Maximise   sum_t price_t * P_t  +  sum_r value_r * V_r,end  -  flow costs  -  switching costs
subject to reservoir balances, volume bounds, min flow, min volume, relaxed commitment u in [0, 1].

P(Q) is a concave piecewise-linear curve per unit, built from the turbine and generator efficiency
curves at the head implied by the start volume (head is held fixed within the week).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import yaml
from scipy.optimize import linprog

M3S_TO_MM3H = 0.0036
K_SEG = 3
MWH_PER_MM3_PER_M = 1e6 * 9.81 * 1000.0 / 3.6e9

UNITS = ["Hogga_G1", "Tokke_G1", "Tokke_G2", "Tokke_G3", "Tokke_G4", "Lio_G1", "Byrte_G1", "Vinje_G1",
         "Vinje_G2", "Vinje_G3", "Haukeli_G1", "Songa_G1", "Kjela_G1", "Vesle Kjela_G1"]
# Fixed priority among identical units, verified on the training labels (see EDA).
PRIORITY = [("Vinje_G1", "Vinje_G2"), ("Vinje_G2", "Vinje_G3"), ("Tokke_G1", "Tokke_G3"),
            ("Tokke_G3", "Tokke_G2"), ("Tokke_G3", "Tokke_G4")]
LP_PRIORITY = PRIORITY + [("Tokke_G4", "Tokke_G2")]  # holds 98 % of hours: fine for the LP, too strict for a net
# Tunnels joining lakes at the same regulated level can flow either way.
BIDIRECTIONAL = {"Totak_Vaamarvatn": 150.0, "Bordalsvatn_Foersvatn": 15.0}
PLANT_ORDER = {"Tokke": ["Tokke_G1", "Tokke_G3", "Tokke_G4", "Tokke_G2"], "Vinje": ["Vinje_G1", "Vinje_G2", "Vinje_G3"]}
NODE_RIVER = "r_Vest_Vassdraget"  # has its own inflow and min flow, passes everything on to Bandak


class Topology:
    def __init__(self, path: str):
        spec = yaml.safe_load(open(path))
        m = spec["model"]
        self.res = m["reservoir"]
        self.riv = m["river"]
        self.plant = m["plant"]
        self.gen = m["generator"]
        self.tun = m["tunnel"]
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

    def _eff(self, g):
        return max(max(c["y"]) for c in self.gen[g]["turb_eff_curves"]) / 100 * max(self.gen[g]["gen_eff_curve"]["y"]) / 100

    def conv(self, p, vols):
        eff = np.mean([self._eff(g) for g in self.plant_gens[p]])
        return MWH_PER_MM3_PER_M * self.gross_head(p, vols) * eff

    def chain(self, p):
        path = [p]
        d = self.plant_down[p]
        while d is not None:
            n = self.first_below.get(d)
            if n is None or n in path:
                break
            path.append(n)
            d = self.plant_down[n]
        return path

    def unit_loss(self, g, shared=True):
        """Head loss coefficient (m per (m3/s)^2 of this unit's own flow).

        With shared=True a unit sees the flow of the units committed before it, because losses in a shared
        tunnel grow with the square of the total flow: the k-th unit sees k^2 * main_loss. Feed tunnels of
        Songa act in parallel, and their loss is added to the main loss.
        """
        gi = self.gen[g]
        pl = self.gen_plant[g]
        p = self.plant[pl]
        main = float(p["main_loss"][0])
        pen = float(p["penstock_loss"][int(gi["penstock"]) - 1])
        if not shared:
            return main + pen
        if pl == "Songa":
            feed = [float(self.tun[t]["loss_factor"]) for t in ("Songavatn_Songa", "Bitdalsvatn_Songa")]
            main += 1.0 / sum(1.0 / np.sqrt(f) for f in feed) ** 2
        order = PLANT_ORDER.get(pl, self.plant_gens[pl])
        k = order.index(g) + 1
        same_pen = [x for x in order[:k] if self.gen[x]["penstock"] == gi["penstock"]]
        n = len(same_pen)
        if shared == "avg":
            return main * k**2 + pen * n**2
        # Adding the k-th unit lowers the head of all k running units, so its marginal loss scales as k^3 - (k-1)^3.
        return main * (k**3 - (k - 1) ** 3) + pen * (n**3 - (n - 1) ** 3)

    def unit_curve(self, g, gross_head, shared_loss=True):
        """Concave piecewise-linear P(Q) for one unit: Qmin, Pmin, widths[K], slopes[K] (MW per m3/s)."""
        gi = self.gen[g]
        loss = self.unit_loss(g, shared_loss)
        curves = gi["turb_eff_curves"]
        refs = np.array([c["ref"] for c in curves], float)
        qs = np.linspace(min(c["x"][0] for c in curves), max(c["x"][-1] for c in curves), 80)
        hn = np.maximum(gross_head - loss * qs**2, 1.0)
        # Interpolate turbine efficiency between the two curves nearest in net head.
        eta = np.zeros_like(qs)
        for i, (q, h) in enumerate(zip(qs, hn)):
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
    def __init__(self, topo: Topology, T: int = 168):
        self.topo, self.T = topo, T
        self.rivers = [r for r in topo.riv if r != NODE_RIVER]
        self.tunnels = list(topo.tun)
        self.flows = self.rivers + self.tunnels
        self.R = topo.res_names
        self.G = UNITS
        nG, nF, nR = len(self.G), len(self.flows), len(self.R)
        self.minvol_res = ["Totak", "Byrtevatn", "Staavatn"]
        self.minflow = ["b_Kjelavatn", "b_Hyljelihyl", "b_Vinjevatn", "b_Byrtevatn", "b_Bandak", "r_Vest_Vassdraget"]
        off = 0
        self.iu = off; off += nG * T
        self.isw = off; off += nG * T
        self.id = off; off += nG * K_SEG * T
        self.if_ = off; off += nF * T
        self.iv = off; off += nR * T
        self.isl_v = off; off += len(self.minvol_res) * T
        self.isl_f = off; off += len(self.minflow) * T
        self.n = off
        # Where every flow comes from and goes to
        self.fsrc, self.fdst = {}, {}
        for f in self.rivers:
            self.fsrc[f] = next((s for (t, s), o in topo.edges.items() if t == "reservoir" and ("river", f) in o), None)
            self.fdst[f] = self._flow_dst(("river", f))
        for f in self.tunnels:
            self.fsrc[f] = next((s for (t, s), o in topo.edges.items() if t == "reservoir" and ("tunnel", f) in o), None)
            self.fdst[f] = next((n for t, n in topo.edges.get(("tunnel", f), []) if t == "reservoir"), None)
        self.songa_tunnels = [f for f in self.tunnels if ("plant", "Songa") in topo.edges.get(("tunnel", f), [])]
        self.res_to_plant = {topo.plant_up[p][0]: p for p in topo.plant if p != "Songa"}
        self.vest_in_flows = [f for f in self.rivers if ("river", NODE_RIVER) in topo.edges.get(("river", f), [])]
        self.vest_in_plants = [p for p in topo.plant if ("river", NODE_RIVER) in topo.edges.get(("plant", p), [])]

    def _flow_dst(self, node):
        return self.topo._follow(node)

    def U(self, g, t): return self.iu + g * self.T + t
    def SW(self, g, t): return self.isw + g * self.T + t
    def D(self, g, k, t): return self.id + (g * K_SEG + k) * self.T + t
    def F(self, f, t): return self.if_ + f * self.T + t
    def V(self, r, t): return self.iv + r * self.T + t

    def solve(self, price, inflow, vol0, wv_end, minflow, minvol, convention="global", value_scale=None,
              startcost_scale=1.0, relaxed=True, time_limit=120.0, mip_gap=0.005, shared_loss="marginal", conv_mode="real", bypass_cost=None):
        """price (T,), inflow dict name->(T,), vol0 dict, wv_end dict, minflow dict->(T,), minvol dict->(T,)."""
        topo, T = self.topo, self.T
        nG = len(self.G)
        c = np.zeros(self.n)
        lb, ub = np.zeros(self.n), np.full(self.n, np.inf)
        ub[self.iu:self.iu + nG * T] = 1.0
        rows_eq, cols_eq, vals_eq, rhs_eq = [], [], [], []
        rows_in, cols_in, vals_in, rhs_in = [], [], [], []
        ne, ni = 0, 0

        curves = []
        for g in self.G:
            curves.append(topo.unit_curve(g, topo.gross_head(topo.gen_plant[g], vol0), shared_loss))
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
            cost = float(topo.gen[g].get("startcost_const", 0)) * startcost_scale
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
            if f in self.topo.tun:
                if f in BIDIRECTIONAL:
                    lb[i0:i0 + T], ub[i0:i0 + T] = -BIDIRECTIONAL[f], BIDIRECTIONAL[f]
                elif f not in self.songa_tunnels:  # Venemo -> Totak, gravity limited
                    src, dst = self.fsrc[f], self.fdst[f]
                    dh = max(topo.level(src, vol0[src]) - topo.level(dst, vol0[dst]), 0.5)
                    ub[i0:i0 + T] = np.sqrt(dh / float(topo.tun[f]["loss_factor"]))
                continue
            info = topo.riv[f]
            if "flow_cost_const" in info:
                cost = float(info["flow_cost_const"])
                c[i0:i0 + T] += bypass_cost if f.startswith("b_") and bypass_cost is not None else cost
            elif "flow_cost_curve" in info:
                cx, cy = info["flow_cost_curve"]["x"], info["flow_cost_curve"]["y"]
                # Slope over the range these rivers actually carry (0-40 m3/s).
                c[i0:i0 + T] += float(cy[1] - cy[0]) / float(cx[1] - cx[0])
            if "max_flow_const" in info:
                ub[i0:i0 + T] = float(info["max_flow_const"])

        # Reservoir balances (equality)
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
            # V_t - V_{t-1}
            rows_eq.append(rr); cols_eq.append(self.V(ri, 0) + Tidx); vals_eq.append(np.ones(T))
            rows_eq.append(rr[1:]); cols_eq.append(self.V(ri, 0) + Tidx[:-1]); vals_eq.append(-np.ones(T - 1))
            for fi in res_in[r]:
                rows_eq.append(rr); cols_eq.append(self.F(fi, 0) + Tidx); vals_eq.append(np.full(T, -M3S_TO_MM3H))
            for fi in res_out[r]:
                rows_eq.append(rr); cols_eq.append(self.F(fi, 0) + Tidx); vals_eq.append(np.full(T, M3S_TO_MM3H))
            # plant discharge in/out
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
            # r_Vest_Vassdraget's own inflow goes straight to Bandak
            if r == "Bandak":
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

        # Min volume with slack: -V - sl <= -minvol
        for si, r in enumerate(self.minvol_res):
            ri = self.R.index(r)
            rr = np.arange(ni, ni + T)
            i0 = self.isl_v + si * T
            rows_in += [rr, rr]; cols_in += [self.V(ri, 0) + Tidx, i0 + Tidx]; vals_in += [-np.ones(T), -np.ones(T)]
            rhs_in.append(-minvol[r]); ni += T
            c[i0:i0 + T] += 5e5  # per Mm3: far above any water value
        # Min flow with slack
        for si, f in enumerate(self.minflow):
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
        values = {}

        def real_conv(pl):
            """MWh per Mm3 at the best point of the plant's first unit, with its head losses."""
            lead = PLANT_ORDER.get(pl, topo.plant_gens[pl])[0]
            q0, p0, w_, s_ = curves[self.G.index(lead)]
            qs_, ps_ = q0 + np.r_[0, np.cumsum(w_)], p0 + np.r_[0, np.cumsum(w_ * s_)]
            return float(np.max(ps_ / qs_)) / M3S_TO_MM3H

        conv_of = (real_conv if conv_mode == "real" else (lambda pl: topo.conv(pl, vol0)))

        def telescoped(r):
            # A plant's margin nets out to wv_up: value(r) = wv_r * conv(first plant) + value(reservoir below it).
            fp = topo.first_below[r]
            if fp is None:
                return 0.0
            down = topo.plant_down[fp]
            return wv_end[r] * conv_of(fp) + (telescoped(down) if down else 0.0)

        for r in self.R:
            fp = topo.first_below[r]
            if fp is None:
                values[r] = 0.0
                continue
            if convention == "local":
                values[r] = wv_end[r] * topo.conv(fp, vol0)
            elif convention == "global":
                values[r] = wv_end[r] * sum(topo.conv(p, vol0) for p in topo.chain(fp))
            elif convention == "telescoping":
                values[r] = telescoped(r)
            else:
                raise ValueError(convention)
            if value_scale:
                values[r] *= value_scale[r]
            c[self.V(self.R.index(r), T - 1)] -= values[r]

        Aeq = sp.csr_matrix((np.concatenate(vals_eq), (np.concatenate(rows_eq), np.concatenate(cols_eq))), shape=(ne, self.n))
        Aub = sp.csr_matrix((np.concatenate(vals_in), (np.concatenate(rows_in), np.concatenate(cols_in))), shape=(ni, self.n))
        kw = {}
        opts = {"time_limit": time_limit, "presolve": True}
        if not relaxed:
            integ = np.zeros(self.n, dtype=np.uint8)
            integ[self.iu:self.iu + nG * T] = 1
            kw["integrality"] = integ
            opts["mip_rel_gap"] = mip_gap
        res = linprog(c, A_ub=Aub, b_ub=np.concatenate(rhs_in), A_eq=Aeq, b_eq=np.concatenate(rhs_eq),
                      bounds=np.c_[lb, ub], method="highs", options=opts, **kw)
        if res.x is None:
            return dict(status=res.status, message=res.message)
        x = res.x
        u = np.stack([x[self.U(g, 0):self.U(g, 0) + T] for g in range(nG)])
        p = np.zeros((nG, T))
        q = np.zeros((nG, T))
        for gi in range(nG):
            qmin, pmin, w, s = curves[gi]
            p[gi] = pmin * u[gi]
            q[gi] = qmin * u[gi]
            for k in range(K_SEG):
                d = x[self.D(gi, k, 0):self.D(gi, k, 0) + T]
                p[gi] += s[k] * d
                q[gi] += d
        pfrac = p / np.array([float(topo.gen[g]["p_max"]) for g in self.G])[:, None]
        V = np.stack([x[self.V(r, 0):self.V(r, 0) + T] for r in range(len(self.R))])
        dual = -res.eqlin.marginals[:len(self.R) * T].reshape(len(self.R), T) if relaxed else np.zeros((len(self.R), T))  # EUR per Mm3
        conv = np.array([conv_of(topo.gen_plant[g]) for g in self.G])
        flows = np.stack([x[self.F(f, 0):self.F(f, 0) + T] for f in range(len(self.flows))])
        return dict(status=res.status, obj=-res.fun, u=u, pfrac=pfrac, q=q, V=V, dual=dual, conv=conv, flows=flows,
                    values=np.array([values[r] for r in self.R]), nit=res.nit)
