"""Sanity of the LP on the unlabeled test years: does it behave as physics says, and how far do its features drift?"""
import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from common import *
from features import build, group
pd.set_option('display.width',220); pd.set_option('display.max_columns',40)
c=load_cases(); z=np.load('artifacts/lp/lp4_b10.npz'); yr=years(c); u=z['u']; pf=z['pfrac']; p=c['price']
print('solved all cases:',len(z['idx'])==3647,'max time',z['time'].max().round(1),'any nan u',np.isnan(u).any())
print('LP on-share by year per unit'); print(pd.DataFrame({y:u[yr==y].mean((0,2)) for y in sorted(set(yr))},index=[x[:10] for x in UNITS]).round(2).to_string())
te=yr>=2023; print('price<0 hours: train',(p[~te]<0).mean().round(5),'test',(p[te]<0).mean().round(5),'min',p[te].min())
for lo,hi in [(-100,-20),(-20,-5),(-5,0),(0,5),(5,20),(20,60),(60,1000)]:
    m=(p>=lo)&(p<hi)
    print(f' price {lo:5d}..{hi:5d}: train n={int((m&~te[:,None]).sum()):7d} LP on={u.mean(1)[m&~te[:,None]].mean() if (m&~te[:,None]).any() else float("nan"):.3f} | test n={int((m&te[:,None]).sum()):7d} LP on={u.mean(1)[m&te[:,None]].mean() if (m&te[:,None]).any() else float("nan"):.3f}')
print('total LP MW share on negative-price test hours by unit',{UNITS[g][:8]:round(float(pf[:,g][(p<0)&te[:,None]].mean()),2) for g in range(14)})
X,names=build(c,z)
from scipy.stats import ks_2samp
r=np.random.default_rng(0); tr_i=r.choice(np.where(~te)[0],300,replace=False); te_i=r.choice(np.where(te)[0],300,replace=False); ks=[]
for k,n in enumerate(names):
    ks.append((ks_2samp(X[tr_i,:,::6,k].ravel(),X[te_i,:,::6,k].ravel()).statistic,n))
ks.sort(reverse=True); print('largest train/test KS:',[(n,round(s,3)) for s,n in ks[:18]]); print('LP feature KS median',np.median([s for s,n in ks if n.startswith('lp')]).round(3),'max',max(s for s,n in ks if n.startswith('lp')).round(3))
# Adversarial AUC at the weekly level: can trees tell test from train?
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
def adv(cols,label):
    idx=[names.index(n) for n in cols]; W=X[:,:,:,idx].mean(2); A=W.reshape(-1,len(idx)); y=np.repeat(te.astype(int),14); g=np.repeat(np.arange(3647)//30,14); o=np.zeros(len(y))
    for a,b in GroupKFold(5).split(A,y,g): o[b]=lgb.LGBMClassifier(n_estimators=150,learning_rate=.05,num_leaves=31,verbose=-1,n_jobs=4,subsample=.7,subsample_freq=1,colsample_bytree=.7).fit(A[a],y[a]).predict_proba(A[b])[:,1]
    print(' adversarial AUC',label,round(roc_auc_score(y,o),4),flush=True)
S="price_asinh,week_price_mean_asinh,week_price_iqr_asinh,wv_up_asinh,wv_down_asinh,lp_be_asinh".split(',')
adv(names,'all features'); adv([n for n in names if n not in S],'without absolute price features'); adv([n for n in names if n.startswith('lp') and n not in S],'LP features only')
