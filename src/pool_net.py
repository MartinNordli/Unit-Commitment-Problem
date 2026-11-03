"""R14/R16 follow-up: ST-Net seed sets compared and pooled, and tree + net blends, scored forward."""
import sys; sys.path.insert(0,'src')
import numpy as np
from common import *
from harness import auc
c=load_cases(); Y=c['Y']; yr=np.array([int(s[:4]) for s in c['start']])[:2922]; YRS=[2019,2020,2021,2022]; ok=np.isin(yr,YRS)
F=lambda n:np.load(f'artifacts/fwd/{n}.npy').astype(np.float64)
a,b=F('N4f_e24_scale'),F('N4f2_s345'); e12=F('N9f_e12_scale'); np3=F('N3f3_e24'); x=F('XR_ref_f'); s=F('XR_sens_f'); n6=(a+b)/2
S=lambda p:(round(auc(Y[ok],p[ok]),5),{y:round(auc(Y[yr==y],p[yr==y]),5) for y in YRS})
print('net seeds012',S(a)); print('net seeds345',S(b)); print('net 6 seeds',S(n6)); print('net e12 s012',S(e12)); print('net noprice e24 s012',S(np3))
for nm,t in (('xgb_ref',x),('xgb_sens',s)):
    print(nm,S(t))
    for w in (.6,.7,.8):
        for nn,net in (('net012',a),('net345',b),('net6',n6)): print(f' {nm} {w} + {nn}',S(w*t+(1-w)*net)[0])
print('sens .7+net6 years',S(.7*s+.3*n6))
