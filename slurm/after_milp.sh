#!/bin/bash
# When the MILP at cost 10 is written, stop the unneeded MILP runs at 100 and 1000, score it, then solve the shifted-water-value LPs.
cd /cluster/home/martnal/sintef; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1; PY=/tmp/sintef_env_martnal/venv/bin/python
until [ -f artifacts/lp/mp4_b10.npz ] && ! pgrep -f "mp4_b10.npz" >/dev/null; do sleep 10; done
pkill -f slurm/lp4_ood.sh; pkill -f "run_lp.py --out artifacts/lp/mp4_b"; sleep 2; rm -f artifacts/lp/mp4_b100.npz artifacts/lp/mp4_b1000.npz
OMP_NUM_THREADS=4 $PY src/lp_score.py mp4_b10
