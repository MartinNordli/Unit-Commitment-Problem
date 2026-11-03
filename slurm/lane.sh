#!/bin/bash
# usage: lane.sh <gpu> <file with one command per line>; runs them in order on one A100
export CUDA_VISIBLE_DEVICES=$1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
cd /cluster/home/martnal/sintef; PY=/tmp/sintef_env_martnal/venv/bin/python
while IFS= read -r line; do [ -z "$line" ] && continue; n=$(echo "$line" | awk '{print $1}'); shift_cmd=$(echo "$line" | cut -d' ' -f2-); echo "$(date +%H:%M) start $n"; eval "$PY $shift_cmd" > logs/$n.out 2>&1; echo "$(date +%H:%M) done $n rc=$?"; done < "$2"
