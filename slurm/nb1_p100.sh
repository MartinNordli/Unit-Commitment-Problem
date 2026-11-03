#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --job-name=nb1_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/nb1_p100_%j.out
# Short_notebook_1 on an old GPU (P100 16 GB): 4 cores, 32 GB enforced by Slurm, 12 h limit, empty directory.
R=/cluster/home/martnal/sintef; T=/tmp/nb1p100_$SLURM_JOB_ID; E=/tmp/nb1p100_env_$SLURM_JOB_ID
mkdir -p $E; trap "rm -rf $T $E" EXIT; tar -xf $R/artifacts/env.tar -C $E
mkdir -p $T; cp -r $R/data $T/data; cp $R/prediction_mapping.csv $R/sample_submission.csv $R/submission/Short_notebook_1.ipynb $T/; cd $T
export PATH=$E/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1
echo "host $(hostname) cores $(nproc) gpu $(nvidia-smi -L) start $(date)"
( while sleep 5; do echo "$(date +%H:%M:%S) gpu_mb $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) rss_mb $(ps -u $USER -o rss=,args= | grep "nb1p100_env_$SLURM_JOB_ID.*ipykernel_launcher" | grep -v grep | awk '{s+=$1} END {print int(s/1024)}')"; done ) > $R/logs/nb1_p100_mem_$SLURM_JOB_ID.log &
M=$!
python -m nbconvert --to notebook --execute --output executed.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 Short_notebook_1.ipynb
RC=$?; echo "rc=$RC end $(date)"; kill $M; [ $RC -eq 0 ] || exit $RC
mkdir -p $R/logs/nb1_p100; cp executed.ipynb submission_1.csv $R/logs/nb1_p100/; cp work/logits.npz $R/logs/nb1_p100/
sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed,State 2>/dev/null | tail -n 3
