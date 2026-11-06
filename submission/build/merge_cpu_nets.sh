#!/bin/bash
# Merge the per-fold CPU ST-Net runs of R25 and print the gate for v5 against v4.
cd /cluster/home/martnal/sintef; PY=${PY:-/tmp/sintef_env_martnal/venv/bin/python}
for t in v4 v5; do for n in N6 N7; do
  $PY -B submission/build/merge_folds.py ${n}c_$t oof "{\"device\": \"cpu\", \"seeds\": \"0,1,2\", \"epochs\": 24}" | cut -c1-200
  $PY -B submission/build/merge_folds.py ${n}fc_$t fwd "{\"device\": \"cpu\", \"seeds\": \"0,1,2\", \"epochs\": 24}" | cut -c1-200
done; done
for n in N6 N7; do echo "== $n"; $PY -B src/compare.py ${n}c_v5 ${n}c_v4 --noise 0.0001 --cand-fwd ${n}fc_v5 --ref-fwd ${n}fc_v4; done
