"""R16: blend across model families, scored strictly forward. Weights for year Y are chosen on earlier forward years only, 2019 has none and uses equal weights."""
import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from common import *
c=load_cases(); Y=c['Y']; yr=np.array([int(s[:4]) for s in c['start']])[:2922]; YRS=[2019,2020,2021,2022]; ok=np.isin(yr,YRS)
F=lambda n:np.load(f'artifacts/fwd/{n}.npy').astype(np.float64)
M={k:F(v) for k,v in dict(xgb='XR_ref_f',xgb_p='XR_scale_f',lgb='L10f',cat='C2f',net='N4f_e24_scale',net_np='N3f3_e24',net12='N9f_e12_scale').items()}
print('single',{k:round(auc(Y[ok],v[ok]),5) for k,v in M.items()}); print('logit corr with xgb',{k:round(float(np.corrcoef(M['xgb'][ok].ravel(),v[ok].ravel())[0,1]),4) for k,v in M.items()},flush=True)
def sc(names,w): 
    b=sum(a*M[n] for a,n in zip(w,names)); return b
def forward_nested(names,step):
    grid=[w+(1-sum(w),) for w in itertools.product(np.arange(0,1+1e-9,step),repeat=len(names)-1) if sum(w)<=1+1e-9]
    S={w:{y:auc(Y[yr==y],sc(names,w)[yr==y]) for y in YRS} for w in grid}; out=np.zeros_like(Y,dtype=float); W={}
    for y in YRS:
        prev=[k for k in YRS if k<y]; w=(tuple([1/len(names)]*len(names)) if not prev else max(grid,key=lambda v:np.mean([S[v][k] for k in prev]))); W[y]=[round(float(x),2) for x in w]; out[yr==y]=sc(names,w)[yr==y]
    per={y:round(auc(Y[yr==y],out[yr==y]),5) for y in YRS}; return round(auc(Y[ok],out[ok]),5),per,W
for names in (['xgb','net'],['lgb','net'],['xgb','lgb','net'],['xgb','xgb_p','net'],['xgb','cat','net'],['xgb','lgb','cat','net'],['xgb','xgb_p','lgb','net'],['xgb','lgb','net','net_np']):
    a,p,w=forward_nested(names,.1); print(names,a,'worst',min(p.values()),p,'weights 2022',w[2022],flush=True)
eq={'trees_mean':np.mean([M['xgb'],M['xgb_p'],M['lgb'],M['cat']],0)}; print('equal mean of 4 tree models',round(auc(Y[ok],eq['trees_mean'][ok]),5))
for w in (.5,.6,.65,.7,.8): print(' fixed trees',w,'+ net',round(1-w,2),round(auc(Y[ok],(w*eq['trees_mean']+(1-w)*M['net'])[ok]),5),{y:round(auc(Y[yr==y],(w*eq['trees_mean']+(1-w)*M['net'])[yr==y]),5) for y in YRS})
