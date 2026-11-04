#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Run a notebook with no GPU from an empty directory: 4 cores, 32 GB enforced by Slurm, 12 h limit.
# usage: sbatch --job-name=<name> slurm/nb_cpu.sh <notebook path>. Results in logs/<name>/.
R=/cluster/home/martnal/sintef; NB=$(basename "$1"); T=/tmp/nbc_$SLURM_JOB_ID; E=/tmp/nbc_env_$SLURM_JOB_ID
mkdir -p $E; trap "rm -rf $T $E" EXIT; tar -xf $R/artifacts/env.tar -C $E
mkdir -p $T; cp -r $R/data $T/data; cp $R/prediction_mapping.csv $R/sample_submission.csv "$R/$1" $T/; cd $T
export PATH=$E/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
echo "host $(hostname) cores $(nproc) cpu $(lscpu | grep 'Model name' | cut -d: -f2 | xargs) start $(date)"
python -m nbconvert --to notebook --execute --output executed.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$NB"
RC=$?; echo "rc=$RC end $(date)"; [ $RC -eq 0 ] || exit $RC
mkdir -p $R/logs/$SLURM_JOB_NAME; cp executed.ipynb submission_*.csv work/logits.npz $R/logs/$SLURM_JOB_NAME/
