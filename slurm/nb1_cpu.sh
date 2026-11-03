#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --job-name=nb1_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nb1_cpu_%j.out
# Worst-case test of Short notebook 1: no GPU, 4 cores, 32 GB enforced by Slurm, 12 h limit, empty directory.
R=/cluster/home/martnal/sintef; T=/tmp/nb1cpu_$SLURM_JOB_ID; E=/tmp/nb1cpu_env_$SLURM_JOB_ID
mkdir -p $E; trap "rm -rf $T $E" EXIT; tar -xf $R/artifacts/env.tar -C $E
mkdir -p $T; cp -r $R/data $T/data; cp $R/prediction_mapping.csv $R/sample_submission.csv "$R/submission/Short notebook 1.ipynb" $T/; cd $T
export PATH=$E/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
echo "host $(hostname) cores $(nproc) cpu $(lscpu | grep 'Model name' | cut -d: -f2 | xargs) start $(date)"
( while sleep 5; do K=$(pgrep -u $USER -f "nb1cpu_env_$SLURM_JOB_ID.*ipykernel_launcher" | head -1); [ -n "$K" ] && echo "$(date +%H:%M:%S) group_rss_mb $(ps -u $USER -o rss=,args= | grep "nb1cpu_env_$SLURM_JOB_ID.*ipykernel_launcher" | grep -v grep | awk '{s+=$1} END {print int(s/1024)}') cgroup_peak_mb $(( $(cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.peak 2>/dev/null || echo 0) / 1048576 ))"; done ) > $R/logs/nb1_cpu_mem_$SLURM_JOB_ID.log &
M=$!
python -m jupyter nbconvert --to notebook --execute --output executed.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "Short notebook 1.ipynb"
RC=$?; echo "rc=$RC end $(date)"; kill $M; [ $RC -eq 0 ] || exit $RC
mkdir -p $R/logs/nb1_cpu; cp executed.ipynb submission_1.csv $R/logs/nb1_cpu/; cp work/logits.npz $R/logs/nb1_cpu/
sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed,State 2>/dev/null | tail -n 3
