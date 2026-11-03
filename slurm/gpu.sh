#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x_%j.out
export PYTHONDONTWRITEBYTECODE=1
ENVD=/tmp/sintef_env_$SLURM_JOB_ID; mkdir -p $ENVD; trap "rm -rf $ENVD" EXIT
tar -xf /cluster/home/martnal/sintef/artifacts/env.tar -C $ENVD
export PY=$ENVD/venv/bin/python
cd /cluster/home/martnal/sintef
nvidia-smi -L
eval "$@"
