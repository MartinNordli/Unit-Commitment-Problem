import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from common import *
from lp_surrogate import Topology
pd.set_option('display.width',230); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
c=load_cases(); Y=c['Y']; yr=years(c)[:2922]; z=np.load('artifacts/lp/lp_v3.npz'); u=z['u'][:2922]; pf=z['pfrac'][:2922]; dual=z['dual'][:2922]; V=z['V'][:2922]; conv=z['conv'][:2922]; topo=Topology('data/extended/Tokke_Vinje_topology.yaml'); price=c['price'][:2922]; ro=topo.res_names
R={r:i for i,r in enumerate(RES)}; X={r:i for i,r in enumerate(ro)}
print('Tokke ordering: P(G4>=G2)',(Y[:,4]>=Y[:,2]).mean().round(4),'P(G2>=G4)',(Y[:,2]>=Y[:,4]).mean().round(4),'P(G3>=G4)',(Y[:,3]>=Y[:,4]).mean().round(4),'P(G3>=G2)',(Y[:,3]>=Y[:,2]).mean().round(4),'P(G1>=G3)',(Y[:,1]>=Y[:,3]).mean().round(4),'P(G1>=G2)',(Y[:,1]>=Y[:,2]).mean().round(4))
print('Vinje ordering: ',(Y[:,7]>=Y[:,8]).mean().round(4),(Y[:,8]>=Y[:,9]).mean().round(4))
for n in ['Lio_G1','Kjela_G1','Haukeli_G1','Vinje_G3','Hogga_G1']:
    g=UNITS.index(n); pl=topo.gen_plant[n]; up=topo.plant_up[pl][0]; dn=topo.plant_down[pl]; e=u[:,g].mean(1)-Y[:,g].mean(1); mv=float(topo.res[up]['max_vol'])
    o=np.argsort(-np.abs(e)); picked=[]
    for i in o:
        if all(abs(i-j)>21 for j in picked): picked.append(i)
        if len(picked)>=9: break
    rows=[]
    for i in picked:
        be=(dual[i,X[up]]-(dual[i,X[dn]] if dn else 0))/conv[i,g]
        rows.append(dict(start=c['start'][i],lp=u[i,g].mean(),true=Y[i,g].mean(),price=price[i].mean(),p10=np.percentile(price[i],10),p90=np.percentile(price[i],90),wv_up=c['wv_end'][i,R[up]],wv_dn=c['wv_end'][i,R[dn]] if dn else 0,be=be.mean(),fill0=c['vol0'][i,R[up]]/mv,fillE=V[i,X[up]][-1]/mv,fill_dn=(c['vol0'][i,R[dn]]/float(topo.res[dn]['max_vol']) if dn else 0),fillE_dn=(V[i,X[dn]][-1]/float(topo.res[dn]['max_vol']) if dn else 0),inflow=c['inflow'][i,:,INFLOW.index(up)].mean(),minvol=(c['minvol'][i,:,MINVOL.index(up)].mean() if up in MINVOL else 0)))
    print('---',n,'mae',round(np.abs(e).mean(),3),'up',up,'dn',dn,'corr(fill0,e)',round(np.corrcoef(c['vol0'][:2922,R[up]],e)[0,1],2)); print(pd.DataFrame(rows).round(2).to_string())
    # error vs true margin: how often true-on happens when price below wv_up
    wu=c['wv_end'][:2922,R[up]][:,None]; m=price-wu
    b=[-1e9,-30,-10,-3,0,3,10,30,1e9]; d=pd.DataFrame({'m':np.digitize(m.ravel(),b),'y':Y[:,g].ravel(),'l':u[:,g].ravel()}).groupby('m').mean().round(3); print(' P(on) by price-wv_up bin:',{f'{b[k-1]}..{b[k]}':(float(d.y[k]),float(d.l[k])) for k in d.index})
