"""Nested blends of all LP v4 models, then the first submission file, validated against sample_submission."""
import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd, itertools
from common import *
c=load_cases(); Y=c['Y']; yr=years(c)[:2922]; YRS=list(range(2015,2023))
O=lambda n:np.load(f'artifacts/oof/{n}.npy').astype(np.float64); T=lambda n:np.load(f'artifacts/preds/{n}.npy').astype(np.float64)
gb=['L9_v4_mf_s3','X1_v4_mf','C1_v4_mf']; nn=['N3_v4_mf','N4_v4_scale']
G=np.mean([O(n) for n in gb],0); N=np.mean([O(n) for n in nn],0)
print({n:round(auc(Y,O(n)),5) for n in gb+nn},'corr G/N',np.corrcoef(G.ravel(),N.ravel())[0,1].round(4),'GBDT mean',round(auc(Y,G),5),'NN mean',round(auc(Y,N),5),flush=True)
print('logit std G',G.std().round(2),'N',N.std().round(2),'test G',np.mean([T(n) for n in ['L9_v4_mf_s3','X2_v4_mf','C1_v4_mf']],0).std().round(2),'test N',np.mean([T(n) for n in nn],0).std().round(2))
ws=np.arange(0,1.001,.05); sc={w:{y:auc(Y[yr==y],(w*G+(1-w)*N)[yr==y]) for y in YRS} for w in ws}
b=np.zeros_like(G); W=[]
for y in YRS:
    w=max(ws,key=lambda v:np.mean([sc[v][k] for k in YRS if k!=y])); W.append(round(float(w),2)); b[yr==y]=(w*G+(1-w)*N)[yr==y]
per={y:round(auc(Y[yr==y],b[yr==y]),5) for y in YRS}; print('nested GBDT+NN',round(auc(Y,b),5),'worst',min(per.values()),'GBDT weight by fold',W); print(per)
report('B2_gbdt3_nn2_nested',b,Y,c,extra={'weights':W}); np.save('artifacts/oof/B2.npy',b.astype(np.float32))
w=float(np.median(W)); P=w*np.mean([T(n) for n in ['L9_v4_mf_s3','X2_v4_mf','C1_v4_mf']],0)+(1-w)*np.mean([T(n) for n in nn],0); p=1/(1+np.exp(-P))
s=pd.read_csv('sample_submission.csv',nrows=None); assert list(s.columns[1:])==TARGETS and (s['Run No'].values==c['run_no'][2922:]).all() and p.shape==(725,14,168) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
out=pd.DataFrame(p.reshape(725,-1).astype(np.float32),columns=TARGETS); out.insert(0,'Run No',s['Run No'].values); out.to_csv('artifacts/submission_B2.csv',index=False,float_format='%.6f')
r=pd.read_csv('artifacts/submission_B2.csv'); assert r.shape==s.shape and list(r.columns)==list(s.columns)
print('submission written, w=',w,'shape',r.shape,'mean p',round(float(p.mean()),4))
pe=np.load('/cluster/home/martnal/ucCommitmentProblem/Unit-Commitment-Problem/artifacts/test_preds/E019_stnet__757679e.parquet',allow_pickle=True) if False else None
old=pd.read_parquet('/cluster/home/martnal/ucCommitmentProblem/Unit-Commitment-Problem/artifacts/test_preds/E013_neighbour_plants__757679e.parquet').drop(columns='Run No').values.reshape(725,14,168); old2=pd.read_parquet('/cluster/home/martnal/ucCommitmentProblem/Unit-Commitment-Problem/artifacts/test_preds/E019_stnet__757679e.parquet').drop(columns='Run No').values.reshape(725,14,168)
lg=lambda x:np.log(np.clip(x,1e-5,1-1e-5)/(1-np.clip(x,1e-5,1-1e-5))); e21=.4*lg(old)+.6*lg(old2)
print('test logit corr: B2 vs old E021a (LB 0.98906)',np.corrcoef(P.ravel(),e21.ravel())[0,1].round(4),'| OOF same pair',np.corrcoef(b.ravel(),(.4*lg(pd.read_parquet("/cluster/home/martnal/ucCommitmentProblem/Unit-Commitment-Problem/artifacts/oof/E013_neighbour_plants__all.parquet").drop(columns="Run No").values.reshape(2922,14,168))+.6*lg(pd.read_parquet("/cluster/home/martnal/ucCommitmentProblem/Unit-Commitment-Problem/artifacts/oof/E019_stnet__all.parquet").drop(columns="Run No").values.reshape(2922,14,168))).ravel())[0,1].round(4))
