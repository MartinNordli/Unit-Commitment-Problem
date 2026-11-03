import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from common import *
from lp_surrogate import Topology, UNITS
from sklearn.linear_model import HuberRegressor
c=load_cases(); Y=c['Y']; yr=years(c)[:2922]; price=c['price'][:2922]; topo=Topology('data/extended/Tokke_Vinje_topology.yaml')
R={r:i for i,r in enumerate(RES)}
print('--- A. alignment: pooled within-week AUC of on-status vs price shifted by k hours (units that switch)')
for g in ['Tokke_G1','Vinje_G1','Songa_G1']:
    gi=UNITS.index(g); out={}
    m=(Y[:,gi].sum(1)>24)&(Y[:,gi].sum(1)<144)
    for k in (-3,-2,-1,0,1,2,3):
        a=np.roll(price[m],k,axis=1)[:,4:-4]; a=(a-a.mean(1,keepdims=True))/(a.std(1,keepdims=True)+1e-6)
        out[k]=round(auc(Y[m,gi][:,4:-4],a),4)
    print(g,int(m.sum()),out)
print('--- B. implied break-even per plant: best price threshold in weeks where the unit both runs and stops')
lead={'Hogga':'Hogga_G1','Tokke':'Tokke_G1','Lio':'Lio_G1','Byrte':'Byrte_G1','Vinje':'Vinje_G1','Haukeli':'Haukeli_G1','Songa':'Songa_G1','Kjela':'Kjela_G1','Vesle Kjela':'Vesle Kjela_G1'}
for plant,g in lead.items():
    gi=UNITS.index(g); up=topo.plant_up[plant][0]; dn=topo.plant_down[plant]
    rows=[]
    for i in range(2922):
        y=Y[i,gi]
        if y.sum()<12 or y.sum()>156: continue
        o=np.argsort(price[i]); p=price[i][o]; yy=y[o]
        # threshold maximising accuracy: on above, off below
        acc=np.r_[0,np.cumsum(1-yy)]+np.r_[(yy[::-1].cumsum())[::-1],0]
        k=int(np.argmax(acc)); th=p[k-1] if k>=len(p) else (p[k] if k==0 else .5*(p[k-1]+p[k]))
        rows.append((i,th,acc[k]/168))
    if len(rows)<30: print(plant,'too few',len(rows)); continue
    a=np.array(rows); idx=a[:,0].astype(int); th=a[:,1]
    wu=c['wv_end'][idx,R[up]]; wd=c['wv_end'][idx,R[dn]] if dn else np.zeros(len(idx)); wus=c['wv_start'][idx,R[up]]
    vols=[{n:float(c['vol0'][i,k]) for k,n in enumerate(RES)} for i in idx[:50]]
    cv=np.mean([topo.conv(plant,v) for v in vols]); fb=topo.first_below[dn] if dn else None
    cvd=np.mean([topo.conv(fb,v) for v in vols]) if fb else 0
    Eu=np.mean([sum(topo.conv(p,v) for p in topo.chain(plant)) for v in vols]); Ed=np.mean([sum(topo.conv(p,v) for p in topo.chain(fb)) for v in vols]) if fb else 0
    X=np.c_[wu,wd]; ok=np.abs(th)<np.percentile(np.abs(th),99)
    h=HuberRegressor(fit_intercept=True,epsilon=1.5,max_iter=2000).fit(X[ok],th[ok])
    pred=lambda a,b:a*wu+b*wd
    r2=lambda p:round(1-((th-p)**2)[ok].sum()/((th-th[ok].mean())**2)[ok].sum(),3)
    print(f"{plant:12s} n={len(idx):4d} acc={a[:,2].mean():.3f} fit: {h.coef_[0]:.2f}*up {h.coef_[1]:+.2f}*down {h.intercept_:+.1f} R2={r2(h.predict(X)):.3f} | local a={1:.2f} b={cvd/cv:.2f} R2={r2(pred(1,cvd/cv))} | global a={Eu/cv:.2f} b={Ed/cv:.2f} R2={r2(pred(Eu/cv,Ed/cv))} | wv_up only R2={r2(wu)} | wv_start R2={r2(wus)} | corr(th,wu)={np.corrcoef(th[ok],wu[ok])[0,1]:.3f}")
print('--- C. overlap smoothing on the E021a OOF, same clock hour across neighbouring runs (predictions only)')
P='/cluster/home/martnal/ucCommitmentProblem/Unit-Commitment-Problem/artifacts/oof/'
def oof(n): return pd.read_parquet(P+n+'__all.parquet').drop(columns='Run No').values.reshape(2922,14,168).astype(float)
lg=lambda p:np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
base=.4*lg(oof('E013_neighbour_plants'))+.6*lg(oof('E019_stnet'))
def smooth(L,lags,alpha):
    """Mean logit for the same clock hour from runs starting up to `lags` days away; weight alpha^|lag|."""
    n,U,T=L.shape; num=L.copy(); den=np.ones_like(L)
    for k in range(1,lags+1):
        w=alpha**k; h=24*k
        num[k:,:,:T-h]+=w*L[:-k,:,h:]; den[k:,:,:T-h]+=w      # earlier run sees this clock hour later in its horizon
        num[:-k,:,h:]+=w*L[k:,:,:T-h]; den[:-k,:,h:]+=w       # later run
    return num/den
b0=auc(Y,base); print('base',round(b0,5),{y:round(auc(Y[yr==y],base[yr==y]),4) for y in range(2015,2023)})
for lags,alpha in [(1,.5),(1,1.),(2,.5),(2,1.),(3,.5),(3,1.),(6,.5),(6,1.)]:
    s=smooth(base,lags,alpha); d={y:auc(Y[yr==y],s[yr==y])-auc(Y[yr==y],base[yr==y]) for y in range(2015,2023)}
    print(f'lags={lags} alpha={alpha}: {auc(Y,s):.5f} delta={auc(Y,s)-b0:+.5f} years up={sum(v>0 for v in d.values())}/8 min={min(d.values()):+.5f}')
print('--- D. where the error is: E021a AUC per unit, and mean logloss per unit')
print({UNITS[g][:9]:round(auc(Y[:,g],base[:,g]),4) for g in range(14)})
