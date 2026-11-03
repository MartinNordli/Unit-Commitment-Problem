#!/bin/bash
# usage: g.sh <gpu 0|1> <log name> <python args...>  -- runs on the given A100 of the interactive allocation
G=$1; L=$2; shift 2
cd /cluster/home/martnal/sintef
nohup srun --jobid=25896375 --overlap --ntasks=1 --cpus-per-task=8 --gres=gpu:a100:2 --export=ALL,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=2 bash -c "export CUDA_VISIBLE_DEVICES=$G; exec /tmp/sintef_env_martnal/venv/bin/python $*" > logs/$L.out 2>&1 &
