import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from common import *
from lp_surrogate import Topology
from sklearn.linear_model import HuberRegressor
pd.set_option('display.width',230); pd.set_option('display.max_columns',40)
c=load_cases(); Y=c['Y']; yr=years(c)[:2922]; z=np.load('artifacts/lp/lp_v2.npz'); u=z['u'][:2922]; oof=np.load('artifacts/oof/L2_v2.npy'); topo=Topology('data/extended/Tokke_Vinje_topology.yaml'); price=c['price'][:2922]
R={r:i for i,r in enumerate(RES)}
print('--- A. structural check on early years only (2015-2018): implied threshold fit')
lead={'Hogga':'Hogga_G1','Tokke':'Tokke_G1','Lio':'Lio_G1','Byrte':'Byrte_G1','Vinje':'Vinje_G1','Haukeli':'Haukeli_G1','Songa':'Songa_G1','Kjela':'Kjela_G1','Vesle Kjela':'Vesle Kjela_G1'}
for plant,g in lead.items():
    gi=UNITS.index(g); up=topo.plant_up[plant][0]; dn=topo.plant_down[plant]; rows=[]
    for i in np.where(yr<=2018)[0]:
        y=Y[i,gi]
        if y.sum()<12 or y.sum()>156: continue
        o=np.argsort(price[i]); p=price[i][o]; yy=y[o]; acc=np.r_[0,np.cumsum(1-yy)]+np.r_[(yy[::-1].cumsum())[::-1],0]; k=int(np.argmax(acc)); rows.append((i,p[k-1] if k>=len(p) else (p[k] if k==0 else .5*(p[k-1]+p[k]))))
    a=np.array(rows); idx=a[:,0].astype(int); th=a[:,1]; wu=c['wv_end'][idx,R[up]]; wd=c['wv_end'][idx,R[dn]] if dn else np.zeros(len(idx)); ok=np.abs(th)<np.percentile(np.abs(th),99)
    h=HuberRegressor(epsilon=1.5,max_iter=2000).fit(np.c_[wu,wd][ok],th[ok]); print(f' {plant:12s} n={len(idx):4d} {h.coef_[0]:.2f}*up {h.coef_[1]:+.2f}*down {h.intercept_:+.1f}')
print('--- B. per unit: LP vs SHOP vs model')
rows=[]
for g,n in enumerate(UNITS):
    y=Y[:,g]; l=u[:,g]; rows.append(dict(unit=n,true_share=y.mean(),lp_share=l.mean(),lp_on_true_off=((l>.5)&(y==0)).mean(),lp_off_true_on=((l<.5)&(y==1)).mean(),lp_week_mae=np.abs(l.mean(1)-y.mean(1)).mean(),lp_auc=auc(y,l),model_auc=auc(y,oof[:,g]),model_week_mae=np.abs((1/(1+np.exp(-oof[:,g]))).mean(1)-y.mean(1)).mean()))
print(pd.DataFrame(rows).round(3).to_string())
print('--- C. LP week-share bias (lp - true) per unit per year')
print(pd.DataFrame({y_:[(u[yr==y_,g].mean(1)-Y[yr==y_,g].mean(1)).mean() for g in range(14)] for y_ in range(2015,2023)},index=UNITS).round(2).to_string())
print('--- D. LP week-share bias by month'); mon=np.array([int(s[5:7]) for s in c['start'][:2922]])
print(pd.DataFrame({m:[(u[mon==m,g].mean(1)-Y[mon==m,g].mean(1)).mean() for g in range(14)] for m in range(1,13)},index=UNITS).round(2).to_string())
print('--- E. what predicts the LP week-share error, per weak unit: corr of (lp-true) with case state')
for n in ['Hogga_G1','Lio_G1','Haukeli_G1','Kjela_G1','Vinje_G3','Songa_G1','Vesle Kjela_G1']:
    g=UNITS.index(n); pl=topo.gen_plant[n]; up=topo.plant_up[pl][0]; dn=topo.plant_down[pl]; e=u[:,g].mean(1)-Y[:,g].mean(1); mv=float(topo.res[up]['max_vol'])
    d={'fill_up':c['vol0'][:2922,R[up]]/mv,'wv_up':c['wv_end'][:2922,R[up]],'price':price.mean(1),'p_minus_wv':price.mean(1)-c['wv_end'][:2922,R[up]],'inflow_up':c['inflow'][:2922,:,INFLOW.index(up)].mean(1),'true_share':Y[:,g].mean(1),'lp_share':u[:,g].mean(1)}
    if dn: d['fill_dn']=c['vol0'][:2922,R[dn]]/float(topo.res[dn]['max_vol'])
    print(f' {n:15s} mae={np.abs(e).mean():.3f}',{k:round(float(np.corrcoef(e,v)[0,1]),2) for k,v in d.items()})
print('--- F. model error decomposition: remaining per-year per-unit AUC of L2')
print(pd.DataFrame({y_:[auc(Y[yr==y_,g],oof[yr==y_,g]) for g in range(14)] for y_ in range(2015,2023)},index=UNITS).round(3).to_string())
print('--- G. worst 12 cases by mean abs error, L2'); p=1/(1+np.exp(-oof)); e=np.abs(p-Y).mean((1,2)); o=np.argsort(-e)[:12]
for i in o: print(' ',c['start'][i],'err',round(e[i],3),'price mean',round(price[i].mean(),1),'worst units',[UNITS[g][:8] for g in np.argsort(-np.abs(p[i]-Y[i]).mean(1))[:3]], 'lp err',round(np.abs(u[i]-Y[i]).mean(),3))
