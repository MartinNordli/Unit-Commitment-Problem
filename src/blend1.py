"""Nested logit blends on the OOF files: weights for each year are chosen on the other seven only."""
import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd, itertools
from common import *
c=load_cases(); Y=c['Y']; yr=years(c)[:2922]; YRS=list(range(2015,2023))
P='/cluster/home/martnal/ucCommitmentProblem/Unit-Commitment-Problem/artifacts/oof/'
lg=lambda p:np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
def prior(n): return lg(pd.read_parquet(P+n+'__all.parquet').drop(columns='Run No').values.reshape(2922,14,168).astype(float))
def zs(a): return (a-a.mean())/a.std()
M={n:np.load(f'artifacts/oof/{n}.npy').astype(np.float64) for n in ['L7_v4_mf','L8_v4','N3_v4_mf','N2_stnet_v3','L5_v3']}
M['E026_old']=prior('E026_system_state'); M['E019_old']=prior('E019_stnet')
print('logit corr L7/N3',np.corrcoef(M['L7_v4_mf'].ravel(),M['N3_v4_mf'].ravel())[0,1].round(4), 'std',M['L7_v4_mf'].std().round(2),M['N3_v4_mf'].std().round(2))
base=auc(Y,M['L7_v4_mf']); print('L7',round(base,5),'N3',round(auc(Y,M['N3_v4_mf']),5))
def nested(names,step=.05,pow_=1.):
    grid=[w for w in itertools.product(np.arange(0,1+1e-9,step),repeat=len(names)-1) if sum(w)<=1+1e-9]
    A=[M[n] for n in names]; out=np.zeros_like(A[0]); W=[]
    sc={}
    for w in grid:
        ww=list(w)+[1-sum(w)]; b=sum(a*x for a,x in zip(ww,A)); sc[w]={y:auc(Y[yr==y],b[yr==y]) for y in YRS}
    for y in YRS:
        best=max(grid,key=lambda w:np.mean([sc[w][k] for k in YRS if k!=y])); ww=list(best)+[1-sum(best)]; W.append(np.round(ww,2).tolist()); out[yr==y]=sum(a*x for a,x in zip(ww,A))[yr==y]
    return out,W
for names in (['L7_v4_mf','N3_v4_mf'],['L5_v3','N2_stnet_v3'],['L7_v4_mf','N2_stnet_v3'],['L7_v4_mf','N3_v4_mf','N2_stnet_v3'],['L7_v4_mf','N3_v4_mf','E019_old'],['L7_v4_mf','N3_v4_mf','E026_old']):
    b,W=nested(names,step=.1 if len(names)>2 else .05); per={y:round(auc(Y[yr==y],b[yr==y]),5) for y in YRS}
    d={y:per[y]-auc(Y[yr==y],M['L7_v4_mf'][yr==y]) for y in YRS}
    print(names,'nested',round(auc(Y,b),5),'worst',min(per.values()),'years>L7',sum(v>0 for v in d.values()),'/8','weights',W[0],W[-1]); print('   ',per,flush=True)
    if names==['L7_v4_mf','N3_v4_mf']: np.save('artifacts/oof/B1_L7_N3.npy',b.astype(np.float32)); B=b
print('per unit, blend',{UNITS[g][:9]:round(auc(Y[:,g],B[:,g]),4) for g in range(14)})
