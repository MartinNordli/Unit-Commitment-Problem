#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Run a notebook from an empty directory with only the raw data: 4 cores, 32 GB enforced by Slurm, 12 h limit.
# usage: sbatch --job-name=<name> --gres=gpu:<type>:1 slurm/nb_test.sh <notebook path>. Results in logs/<name>/.
R=/cluster/home/martnal/sintef; NB=$(basename "$1"); T=/tmp/nbt_$SLURM_JOB_ID; E=/tmp/nbt_env_$SLURM_JOB_ID
mkdir -p $E; trap "rm -rf $T $E" EXIT; tar -xf $R/artifacts/env.tar -C $E
mkdir -p $T; cp -r $R/data $T/data; cp $R/prediction_mapping.csv $R/sample_submission.csv "$R/$1" $T/; cd $T
export PATH=$E/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1
echo "host $(hostname) cores $(nproc) gpu $(nvidia-smi -L) start $(date)"
( while sleep 5; do echo "$(date +%H:%M:%S) gpu_mb $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)"; done ) > $R/logs/${SLURM_JOB_NAME}_mem_$SLURM_JOB_ID.log &
M=$!
python -m nbconvert --to notebook --execute --output executed.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$NB"
RC=$?; echo "rc=$RC end $(date)"; kill $M; [ $RC -eq 0 ] || exit $RC
mkdir -p $R/logs/$SLURM_JOB_NAME; cp executed.ipynb submission_*.csv work/logits.npz $R/logs/$SLURM_JOB_NAME/
