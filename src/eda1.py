import pandas as pd, numpy as np, re
pd.set_option('display.width',220); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',200)
D='data/kernel/'
uc=pd.read_csv(D+'Unit_commitment_decisions.csv',parse_dates=['starttime'])
cols=[c for c in uc.columns if c.startswith('result_committed')]
print(uc.shape, uc.starttime.min(), uc.starttime.max(), uc['Run No'].is_monotonic_increasing, uc.starttime.diff().value_counts().head().to_dict())
print('nan',uc[cols].isna().sum().sum(),'vals',np.unique(uc[cols].values)[:10])
gens=[];
for c in cols:
    g=re.match(r'result_committed_(.+)_t\d+$',c).group(1)
    if g not in gens: gens.append(g)
print(gens, len(cols)/len(gens))
assert list(cols)==[f'result_committed_{g}_t{t}' for g in gens for t in range(168)]
Y=uc[cols].values.reshape(len(uc),14,168).astype(np.int8)
print('calc time',uc.total_calculation_time.describe().to_dict())
yr=uc.starttime.dt.year.values
print('overall pos',Y.mean())
print(pd.DataFrame({g:[Y[yr==y,i].mean() for y in sorted(set(yr))] for i,g in enumerate(gens)},index=sorted(set(yr))).round(3).T)
print('by hour-of-run mean',np.round(Y.mean((0,1))[:30],3))
# switching
sw=np.abs(np.diff(Y,axis=2)).sum(2)
print('switches per week per gen',dict(zip(gens,sw.mean(0).round(2))))
print('constant-in-week frac',dict(zip(gens,((Y.sum(2)==0)|(Y.sum(2)==168)).mean(0).round(3))))
# run overlap consistency
same=[]
for lag in [1,2,3,6]:
    same.append((lag,(Y[lag:,:,:168-24*lag]==Y[:-lag,:,24*lag:]).mean().round(4)))
print('agreement of absolute hour across runs (lag days)',same)
# first 24h vs. later
for h0,h1 in [(0,24),(24,72),(72,120),(120,168)]:
    print('agree lag1 hour range',h0,h1,(Y[1:,:,h0:h1]==Y[:-1,:,h0+24:h1+24] if h1+24<=168 else Y[1:,:,h0:144]==Y[:-1,:,h0+24:168]).mean().round(4))
# gen correlations
print(pd.DataFrame(np.corrcoef(Y.transpose(1,0,2).reshape(14,-1)),index=gens,columns=[g[:7] for g in gens]).round(2))
# identical unit symmetry
print('Tokke counts dist',np.bincount(Y[:,1:5].sum(1).ravel(),minlength=5)/Y[:,1:5].sum(1).size)
print('P(G_i on)',Y[:,1:5].mean((0,2)).round(3), 'Vinje',Y[:,7:10].mean((0,2)).round(3))
# order
print('Tokke nested? G1>=G2',(Y[:,1]>=Y[:,2]).mean(),(Y[:,2]>=Y[:,3]).mean(),(Y[:,3]>=Y[:,4]).mean(),'Vinje',(Y[:,7]>=Y[:,8]).mean(),(Y[:,8]>=Y[:,9]).mean())
# timeseries
p=pd.read_csv(D+'Historical_day_ahead_price_2015_2025.csv',parse_dates=['date']).set_index('date').iloc[:,0]
print('price',p.index.min(),p.index.max(),'nan',p.isna().sum(),'dups',p.index.duplicated().sum(),'gaps',(p.index.to_series().diff().dropna()!=pd.Timedelta('1h')).sum())
print(p.groupby(p.index.year).describe().round(1))
q=pd.read_csv(D+'Historical_inflow_1958_2025.csv',parse_dates=['date']).set_index('date')
print('inflow',q.index.min(),q.index.max(),'nan',q.isna().sum().sum(),'gaps',(q.index.to_series().diff().dropna()!=pd.Timedelta('1h')).sum())
print(q.groupby(q.index.year).mean().loc[2013:].round(1).T)
v=pd.read_csv(D+'Historical_volume_2015_2024.csv',parse_dates=['date']).set_index('date')
wv=pd.read_csv(D+'Synthetic_water_value_2015_2024.csv',parse_dates=['date']).set_index('date')
print('vol',v.index.min(),v.index.max(),v.isna().sum().sum(),(v.index.to_series().diff().dropna()!=pd.Timedelta('1D')).sum())
print('wv',wv.index.min(),wv.index.max(),wv.isna().sum().sum(),(wv.index.to_series().diff().dropna()!=pd.Timedelta('1D')).sum())
print(v.groupby(v.index.year).mean().round(1).T)
print(wv.groupby(wv.index.year).mean().round(1).T)
print('WV corr',wv.corr().round(2).iloc[:5,:8])
print('WV unique per day (how many distinct)',wv.nunique(axis=1).describe().round(1).to_dict())
print('WV sample',wv.iloc[[0,400,1200,2600,3200,3600]].T.round(1))
print('vol std of daily diff / zeros',(v.diff().abs()<1e-9).mean().round(3).to_dict())
# min constraints
mf=pd.read_csv('data/extended/Constraint_min_flow.csv',parse_dates=['date']).set_index('date')
mv=pd.read_csv('data/extended/Constraint_min_volume.csv',parse_dates=['date']).set_index('date')
print('minflow',mf.index.min(),mf.index.max(),mf.describe().round(2).loc[['min','mean','max']].T)
print('minvol',mv.index.min(),mv.index.max(),mv.describe().round(2).loc[['min','mean','max']].T)
