import os,sys,time,multiprocessing as mp; os.environ['OMP_NUM_THREADS']='1'; sys.path.insert(0,'src')
import numpy as np
from common import *
from lp_surrogate import Topology, WeekLP
c=load_cases(); Y=c['Y']; yr=years(c)[:2922]; lp=WeekLP(Topology('data/extended/Tokke_Vinje_topology.yaml'))
def one(a):
    i,tl=a; t=time.time()
    r=lp.solve(c['price'][i].astype(float),{n:c['inflow'][i,:,k].astype(float) for k,n in enumerate(INFLOW)},{n:float(c['vol0'][i,k]) for k,n in enumerate(RES)},{n:float(c['wv_end'][i,k]) for k,n in enumerate(RES)},{n:c['minflow'][i,:,k].astype(float) for k,n in enumerate(MINFLOW)},{n:c['minvol'][i,:,k].astype(float) for k,n in enumerate(MINVOL)},convention='telescoping',relaxed=False,time_limit=tl)
    return i,r.get('u'),r.get('pfrac'),time.time()-t,r.get('status')
idx=np.arange(0,2922,12)
for tl in (10,60):
    with mp.Pool(48) as p: out=p.map(one,[(int(i),tl) for i in idx],chunksize=1)
    g=[o for o in out if o[1] is not None]; I=np.array([o[0] for o in g]); U=np.stack([o[1] for o in g]); PF=np.stack([o[2] for o in g])
    z=np.load('artifacts/lp/lp_tele.npz'); print(f'time_limit {tl}: solved {len(g)}/{len(idx)}, mean {np.mean([o[3] for o in out]):.1f}s, statuses',np.unique([o[4] for o in out],return_counts=True))
    print(' MILP u',round(auc(Y[I],U),4),'pfrac',round(auc(Y[I],PF),4),'week MAE',round(np.abs(U.mean(2)-Y[I].mean(2)).mean(),4),'| LP u',round(auc(Y[I],z['u'][I]),4),'pfrac',round(auc(Y[I],z['pfrac'][I]),4),'week MAE',round(np.abs(z['u'][I].mean(2)-Y[I].mean(2)).mean(),4),'| exact-hour accuracy MILP',round((U.round()==Y[I]).mean(),4),'LP',round((z['u'][I].round()==Y[I]).mean(),4))
    print(' per unit MILP',{UNITS[g_][:9]:round(auc(Y[I,g_],U[:,g_]),3) for g_ in range(14)})
