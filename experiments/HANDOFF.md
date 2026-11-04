# Handoff (written 2026-11-03 19:37, when allocation 25896375 ended)

Read first: this file, `experiments/ledger.md` (R1-R17), the three task PDFs in the repo root. Answer the user in Norwegian.

## Rules still in force
- Group 197, member Martin Nordli Almenningen. Kaggle team name: `[197] sur salamander`. Student id 562274, confirmed by the user 2026-11-03. Do not copy "g.93".
- `src/harness.py`, `harness/folds.json`, `kaggle_metric.py`, `src/lp_surrogate.py` are locked (`harness.lock`). Changes need the user's approval.
- Gate: `src/compare.py`, noise 0.0001, >=6/8 years up, forward delta > -noise, clean commit. Write the hypothesis in the ledger BEFORE the run. Commit src before launching, or results come back dirty and the gate refuses them.
- Selection on forward + LOYO. Public LB is a sanity check only. The user uploads to Kaggle. Never push.
- The user wants GPUs in use as much as possible; only LP solves and light analysis on CPU. Do not let them sit idle after a round ends (happened 19:06-19:33): set a waiter.

## Getting started on a new allocation
- The job id 25896375 is hard-coded in launch commands, and `slurm/g.sh`: replace it. Env was on node-local disk: if `/tmp/sintef_env_martnal/venv` is missing, `tar -xf artifacts/env.tar -C /tmp/sintef_env_martnal`.
- Launch: `srun --jobid=<J> --overlap --ntasks=1 --cpus-per-task=8 --cpu-bind=none --gres=gpu:a100:2 bash slurm/lane.sh <gpu> <file>`. `--cpus-per-task=1` pins everything to one core, and CUDA_VISIBLE_DEVICES must be set inside the step (lane.sh does). Verify with nvidia-smi. No allocation: `sbatch slurm/gpu.sh`.

## Where things stand (updated 2026-11-03 21:50)
- Round 4 is done, see the ledger. Final model: XGBoost `base,lp,ls` + ST-Net `base,lp,ls` e24, logit blend 0.7/0.3. Forward 0.99379, LOYO 0.994.
- Notebook 1 is done and verified from an empty directory: `submission/Short_notebook_1.ipynb`, `submission/requirements.txt`, output `artifacts/submission_1.csv`. Built from `submission/build/nb1.py` with `mknb.py`; `run_clean.sh` does the clean run. Trees reproduce the research run exactly, the net to Spearman 0.99999. Reruns on the H100 are byte-identical.
- `artifacts/submission_1.csv` is on Kaggle: public LB 0.99638, the best so far (B2 0.99626). Notebook 1 repeats the best Kaggle score.

## Next steps, in order
All notebooks, the report and the test runs are done (2026-11-04). See the ledger, "Final notebook runs" and "Final code without GPU".
Files to deliver on Canvas: `submission/Short_notebook_1.ipynb`, `submission/Short_notebook_2.ipynb`, `submission/requirements.txt`, `submission/Report.ipynb`.
Left for the user, deadline 2026-11-08 22:00 Oslo (strict): select `artifacts/submission_1.csv` and `artifacts/submission_2.csv` on Kaggle, deliver on Canvas, check exam registration and that the Kaggle team name is `[197] sur salamander`, read through the report.
The notebook headers were patched after the run, without rerunning. `slurm/nb_test.sh` and `slurm/nb_cpu.sh` run a notebook from an empty directory.

## Known limitations to state honestly
- LP structure was chosen seeing all years' labels; no untouched holdout. Negative price behaviour (2.2% of test hours, 0.014% of train) cannot be verified. Lio, Vinje G3, Hogga are the weakest units.
- Cleanup candidates (my own, about 250 MB each): `lp_global, lp_local, lp_tele, lp_v2, lp_v3, lp4_b100, lp4_b1000, milp_v2, mp4_b10` in `artifacts/lp`. Check that nothing reproducible needs them. `/cluster` inode quota is tight (about 1.97M of 2M).
