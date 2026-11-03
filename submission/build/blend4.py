"""Round 4: gate inputs and tree + net blends, forward and LOYO. Reads saved predictions only."""
import sys; sys.path.insert(0, 'src'); sys.dont_write_bytecode = True
import json, numpy as np
from common import *
c = load_cases(); Y = c['Y']; yr = np.array([int(s[:4]) for s in c['start']])[:2922]
FW = [2019, 2020, 2021, 2022]; ok = np.isin(yr, FW)
L = lambda d, n: np.load(f'artifacts/{d}/{n}.npy').astype(np.float64)
def S(p, m): return round(auc(Y[m], p[m]), 5)
def yrs(p, ys): return {y: round(auc(Y[yr == y], p[yr == y]), 5) for y in ys}
nets = {'base e24 s012': ('N4f_e24_scale', 'N4_v4_scale'), 'base e24 s345': ('N4f2_s345', 'N4b_s345'), 'base e12 s012': ('N9f_e12_scale', 'N9_e12_scale'), 'base e12 s345': ('N9f2_e12_s345', 'N9b_e12_s345'), 'ls e24 s012': ('N5f_ls', 'N5_ls'), 'min e24 s012': ('N6f_min', 'N6_min')}
trees = {'XR_ref': ('XR_ref_f', 'XR_ref'), 'XR_ls': ('XR_ls_f', 'XR_ls'), 'XR_sens2': ('XR_sens_f2', 'XR_sens2')}
all8 = np.ones(2922, bool)
for tn, (tf, tl) in trees.items():
    T = L('fwd', tf), L('oof', tl); print(tn, 'alone fwd', S(T[0], ok), 'loyo', S(T[1], all8))
    for nn, (nf, nl) in nets.items():
        N = L('fwd', nf), L('oof', nl)
        print(f'  {tn} + {nn}: alone fwd {S(N[0], ok)} loyo {S(N[1], all8)} |', ' '.join(f'w{w}: f{S(w*T[0]+(1-w)*N[0], ok)} l{S(w*T[1]+(1-w)*N[1], all8)}' for w in (.6, .7, .8)), flush=True)
