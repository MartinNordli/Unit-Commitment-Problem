import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from common import *
c=load_cases(); Y=c['Y']; yr=years(c)[:2922]
P='/cluster/home/martnal/ucCommitmentProblem/Unit-Commitment-Problem/artifacts/oof/'
def oof(n): return pd.read_parquet(P+n+'__all.parquet').drop(columns='Run No').values.reshape(2922,14,168).astype(float)
lg=lambda p:np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
A={n:lg(oof(n)) for n in ['E013_neighbour_plants','E019_stnet','E022_stnet_seeds345','E026_system_state']}
st=(A['E019_stnet']+A['E022_stnet_seeds345'])/2
base=.4*A['E013_neighbour_plants']+.6*A['E019_stnet']
print('E013',auc(Y,A['E013_neighbour_plants']),'E026',auc(Y,A['E026_system_state']),'stnet3',auc(Y,A['E019_stnet']),'stnet6',auc(Y,st),'E021a',auc(Y,base))
print('E021a per yr',{y:round(auc(Y[yr==y],base[yr==y]),4) for y in range(2015,2023)})
for n in sys.argv[1:]:
    z=np.load(f'artifacts/lp/{n}.npz'); u=z['u'][:2922]; pf=z['pfrac'][:2922]; assert (z['idx'][:2922]==np.arange(2922)).all()
    print('==',n,'time',z['time'].mean().round(2),'max',z['time'].max().round(1),'obj nan',np.isnan(z['obj']).sum(), 'frac fractional',((u>.01)&(u<.99)).mean().round(3))
    print(' auc u',round(auc(Y,u),4),'pfrac',round(auc(Y,pf),4),'level(u mean over week)',round(auc(Y,np.broadcast_to(u.mean(2,keepdims=True),u.shape)),4),'true-level',round(auc(Y,np.broadcast_to(Y.mean(2,keepdims=True),u.shape).astype(float)),4))
    print(' per yr',{y:round(auc(Y[yr==y],u[yr==y]),4) for y in range(2015,2023)})
    print(' per unit',{UNITS[g][:9]:round(auc(Y[:,g],u[:,g]),3) for g in range(14)})
    print(' week share corr',{UNITS[g][:9]:round(np.corrcoef(u[:,g].mean(1),Y[:,g].mean(1))[0,1],2) for g in range(14)})
    print(' week share MAE',round(np.abs(u.mean(2)-Y.mean(2)).mean(),4),'base model MAE',round(np.abs(1/(1+np.exp(-base)).mean(2)-Y.mean(2)).mean() if 0 else np.abs((1/(1+np.exp(-base))).mean(2)-Y.mean(2)).mean(),4))
    s=lg(np.clip(u,.02,.98))
    for w in (.05,.1,.2,.3,.5):
        print('  blend E021a + %.2f*LP logit'%w, round(auc(Y,base+w*s),5))
