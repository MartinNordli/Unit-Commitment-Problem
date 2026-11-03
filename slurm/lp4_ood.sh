#!/bin/bash
cd /cluster/home/martnal/sintef; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1; PY=/tmp/sintef_env_martnal/venv/bin/python
for b in 10 100 1000; do $PY src/run_lp.py --out artifacts/lp/lp4_b$b.npz --conv telescoping --procs 7 --bypass $b | tail -1; done
OMP_NUM_THREADS=7 $PY src/lp_score.py lp_v3 lp4_b10 lp4_b100 lp4_b1000
for b in 10 100 1000; do $PY src/run_lp.py --out artifacts/lp/mp4_b$b.npz --conv telescoping --procs 7 --bypass $b --milp --tl 30 --gap 0.002 | tail -1; done
OMP_NUM_THREADS=7 $PY src/lp_score.py mp4_b10 mp4_b100 mp4_b1000
