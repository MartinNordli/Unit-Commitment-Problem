import sys; sys.path.insert(0,'src')
import numpy as np
from common import *
c=load_cases(); Y=c['Y']; yr=years(c)[:2922]
for n in sys.argv[1:]:
    z=np.load(f'artifacts/lp/{n}.npz'); u=z['u'][:2922]; pf=z['pfrac'][:2922]
    print(f'{n:14s} t={z["time"].mean():.2f}s max={z["time"].max():.0f} u={auc(Y,u):.4f} pfrac={auc(Y,pf):.4f} weekMAE={np.abs(u.mean(2)-Y.mean(2)).mean():.4f} acc={(u.round()==Y).mean():.4f} worstyr={min(auc(Y[yr==y],pf[yr==y]) for y in range(2015,2023)):.4f}')
    print('   week mae',{UNITS[g][:7]:round(float(np.abs(u[:,g].mean(1)-Y[:,g].mean(1)).mean()),3) for g in range(14)},flush=True)
