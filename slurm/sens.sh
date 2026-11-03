#!/bin/bash
cd /cluster/home/martnal/sintef; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1; PY=/tmp/sintef_env_martnal/venv/bin/python
for w in 90:0.9 110:1.1; do n=${w%%:*}; v=${w##*:}; $PY src/run_lp.py --out artifacts/lp/lp4_w$n.npz --conv telescoping --procs 3 --wv-scale $v | tail -1; done
