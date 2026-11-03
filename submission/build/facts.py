"""Report material: numbers stated in the report text, printed once for checking."""
import sys; sys.path.insert(0,'src'); sys.dont_write_bytecode=True
import numpy as np, pandas as pd, json
from common import *
from features import TOPO, UP, DOWN, PMAX, PLANTS
pd.set_option('display.width',220); pd.set_option('display.max_columns',40)
c=load_cases(); Y=c['Y']; yr=years(c); P=c['price']; ytr=yr[:2922]
print('cases',len(yr),'train',Y.shape,'pos',Y.mean().round(4))
uc=pd.read_csv('data/kernel/Unit_commitment_decisions.csv',usecols=['starttime','total_calculation_time'],parse_dates=['starttime'])
print(uc.total_calculation_time.describe().round(1).to_dict()); print(uc.groupby(uc.starttime.dt.year).total_calculation_time.median().round(1).to_dict())
print('on per unit',dict(zip(UNITS,Y.mean((0,2)).round(3)))); print('on per year',{int(y):round(float(Y[ytr==y].mean()),3) for y in range(2015,2023)})
h=pd.read_csv('data/kernel/Historical_day_ahead_price_2015_2025.csv',parse_dates=['date']).set_index('date').iloc[:,0]
h.index=pd.to_datetime(h.index,utc=True); h=h['2015':'2024-12-31']
g=h.groupby(h.index.year); print(pd.DataFrame({'min':g.min(),'mean':g.mean(),'p50':g.median(),'max':g.max(),'neg':g.apply(lambda s:(s<0).mean()*100)}).round(2))
print('train neg in cases',(P[:2922]<0).mean(),'test',(P[2922:]<0).mean(), 'min',P[2922:].min())
print('agree lag',[(l,float((Y[l:,:,:168-24*l]==Y[:-l,:,24*l:]).mean().round(4))) for l in (1,2,3,6)])
sw=np.abs(np.diff(Y,axis=2)).sum(2); print('switches/week',dict(zip(UNITS,sw.mean(0).round(2)))); print('constant',dict(zip(UNITS,((Y.sum(2)==0)|(Y.sum(2)==168)).mean(0).round(3)))); print('constant all',((Y.sum(2)==0)|(Y.sum(2)==168)).mean().round(3))
i=UNITS.index
for a,b in [('Vinje_G1','Vinje_G2'),('Vinje_G2','Vinje_G3'),('Tokke_G1','Tokke_G3'),('Tokke_G3','Tokke_G2'),('Tokke_G3','Tokke_G4'),('Tokke_G4','Tokke_G2')]: print(a,'>=',b,round(float((Y[:,i(a)]>=Y[:,i(b)]).mean()),4))
v=pd.read_csv('data/kernel/Historical_volume_2015_2024.csv',index_col=0); print('const vol',[r for r in v if v[r].nunique()==1])
ri={r:k for k,r in enumerate(RES)}
wv=c['wv_end'];print('wv corr min',np.corrcoef(wv.T).min().round(3),'median',np.median(np.corrcoef(wv.T)).round(3),'range',wv.min().round(1),wv.max().round(1),'neg wv',(wv<0).mean())
print('corr week price vs mean wv',np.corrcoef(P.mean(1),wv.mean(1))[0,1].round(3))
print('min flow nonzero',(c['minflow']>0).mean((0,1)).round(3).tolist(),MINFLOW); print('minvol nonzero',(c['minvol']>0).mean((0,1)).round(3).tolist(),MINVOL)
rows=[]
for pl in PLANTS:
    u0=[k for k,u in enumerate(UNITS) if TOPO.gen_plant[u]==pl][0]; up=UP[u0]; dn=DOWN[u0]
    on=Y[:,u0].astype(bool); mix=(on.sum(1)>=12)&(on.sum(1)<=156)
    th=np.array([(np.median(P[k][on[k]])+np.median(P[k][~on[k]]))/2 for k in np.where(mix)[0]])
    Wu=wv[:2922,ri[up]][mix]; Wd=(wv[:2922,ri[dn]][mix] if dn else np.zeros(mix.sum()))
    A=np.c_[Wu,Wd,np.ones(len(Wu))]; b=np.linalg.lstsq(A,th,rcond=None)[0]; r2=1-((th-A@b)**2).sum()/((th-th.mean())**2).sum()
    rows.append((pl,int(mix.sum()),*b.round(2),round(r2,3),round(1-((th-Wu)**2).sum()/((th-th.mean())**2).sum(),3)))
print(pd.DataFrame(rows,columns=['plant','n','a_up','a_down','c','r2','r2_a1']))
for n in ['lp_local','lp_global','lp_tele','lp_v2','lp_v3','lp4_b10','lp4_b100','lp4_b1000','lp4_w90','lp4_w110','milp_v2','mp4_b10']:
    z=np.load(f'artifacts/lp/{n}.npz'); idx=z['idx']; ok=idx<2922
    print(n,'n',int(ok.sum()),'u',round(auc(Y[idx[ok]],z['u'][ok]),4),'pfrac',round(auc(Y[idx[ok]],z['pfrac'][ok]),4),'frac',round(float(((z['u'][ok]>.01)&(z['u'][ok]<.99)).mean()),3),'t',round(float(z['time'].mean()),2))
X=np.load('artifacts/oof/XR_ls.npy');N=np.load('artifacts/oof/N6_min.npy'); B=.7*X+.3*N; print('blend',auc(Y,B)); z=np.load('artifacts/lp/lp4_b10.npz'); U=z['u'][:2922]; PF=z['pfrac'][:2922]
print('corr logits',np.corrcoef(X.ravel(),N.ravel())[0,1].round(3))
print(pd.DataFrame({'lp':[auc(Y[:,g],PF[:,g]) for g in range(14)],'xgb':[auc(Y[:,g],X[:,g]) for g in range(14)],'net':[auc(Y[:,g],N[:,g]) for g in range(14)],'blend':[auc(Y[:,g],B[:,g]) for g in range(14)]},index=UNITS).round(4))
print('peryear',{int(y):round(auc(Y[ytr==y],B[ytr==y]),5) for y in range(2015,2023)})
lvl=lambda A:np.broadcast_to(A.mean(2,keepdims=True),A.shape)
print('true-level auc',auc(Y,lvl(Y)),'level blend',auc(Y,lvl(B)))
s=pd.read_csv('artifacts/submission_1.csv'); print(s.shape,list(s.columns)[:3]); T=s.iloc[:,1:].values.reshape(725,14,168); P2=P[2922:]; neg=np.broadcast_to(P2[:,None]<0,T.shape); print('p on neg',T[neg].mean().round(4),'others',T[~neg].mean().round(4),'neg hours',neg.mean().round(4)); ch=(P2<np.percentile(P2,10,axis=1,keepdims=True))&(P2>=0); ch=np.broadcast_to(ch[:,None],T.shape); print('p on cheap nonneg',T[ch].mean().round(4))
zt=z['pfrac'][2922:]; print('lp pfrac on neg',zt[neg].mean().round(4), 'u',z['u'][2922:][neg].mean().round(4))
