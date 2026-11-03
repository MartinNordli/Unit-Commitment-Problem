import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from common import *
pd.set_option('display.width',200)
c=load_cases(); Y=c['Y']; yr=years(c)[:2922]
for n in sys.argv[1:]:
    u=np.load(f'artifacts/lp/{n}.npz')['u'][:2922]
    print(n,'bias by unit',{UNITS[g][:9]:round(float((u[:,g]-Y[:,g]).mean()),3) for g in range(14)})
    print(n,'week mae by unit',{UNITS[g][:9]:round(float(np.abs(u[:,g].mean(1)-Y[:,g].mean(1)).mean()),3) for g in range(14)})
