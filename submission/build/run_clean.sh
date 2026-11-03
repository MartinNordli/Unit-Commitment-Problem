#!/bin/bash
# Run a short notebook from an empty directory with only the raw data, on the cores of this job step, and log memory.
# usage: run_clean.sh <notebook file> <test dir>
NB="$1"; T="$2"; R=/cluster/home/martnal/sintef; V=/tmp/sintef_env_martnal/venv
rm -rf "$T"; mkdir -p "$T"; cp -r $R/data "$T/data"; cp $R/prediction_mapping.csv $R/sample_submission.csv "$T/"; cp "$R/submission/$NB" "$T/"; cd "$T"
echo "cores: $(nproc)  start $(date)" > run.log; export PATH=$V/bin:$PATH PYTHONDONTWRITEBYTECODE=1
( while sleep 15; do K=$(pgrep -u $USER -f "ipykernel_launcher.*" -n); [ -z "$K" ] && continue; echo "$(date +%H:%M:%S) kernel_rss_mb $(( $(ps -o rss= -p $K) / 1024 )) tree_rss_mb $(( $(ps -o rss= --ppid $K -p $K | awk '{s+=$1} END {print s+0}') / 1024 )) gpu_mb $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, -v k=$K '$1==k {print $2+0}')"; done ) > mem.log &
M=$!
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$NB" >> run.log 2>&1
echo "rc=$? end $(date)" >> run.log; kill $M
mkdir -p $R/logs/clean; cp run.log mem.log $R/logs/clean/; cp "$NB" *.csv $R/logs/clean/ 2>/dev/null; cp work/logits.npz $R/logs/clean/ 2>/dev/null
