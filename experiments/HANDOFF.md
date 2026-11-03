# Handoff (written 2026-11-03 19:37, when allocation 25896375 ended)

Read first: this file, `experiments/ledger.md` (R1-R17), the three task PDFs in the repo root. Answer the user in Norwegian.

## Rules still in force
- Group 197, member Martin Nordli Almenningen. Kaggle team name must contain [197]. Student id 562274 comes from the prior project: confirm with the user. Do not copy "g.93".
- `src/harness.py`, `harness/folds.json`, `kaggle_metric.py`, `src/lp_surrogate.py` are locked (`harness.lock`). Changes need the user's approval.
- Gate: `src/compare.py`, noise 0.0001, >=6/8 years up, forward delta > -noise, clean commit. Write the hypothesis in the ledger BEFORE the run. Commit src before launching, or results come back dirty and the gate refuses them.
- Selection on forward + LOYO. Public LB is a sanity check only. The user uploads to Kaggle. Never push.
- The user wants GPUs in use as much as possible; only LP solves and light analysis on CPU. Do not let them sit idle after a round ends (happened 19:06-19:33): set a waiter.

## Getting started on a new allocation
- The job id 25896375 is hard-coded in launch commands, and `slurm/g.sh`: replace it. Env was on node-local disk: if `/tmp/sintef_env_martnal/venv` is missing, `tar -xf artifacts/env.tar -C /tmp/sintef_env_martnal`.
- Launch: `srun --jobid=<J> --overlap --ntasks=1 --cpus-per-task=8 --cpu-bind=none --gres=gpu:a100:2 bash slurm/lane.sh <gpu> <file>`. `--cpus-per-task=1` pins everything to one core, and CUDA_VISIBLE_DEVICES must be set inside the step (lane.sh does). Verify with nvidia-smi. No allocation: `sbatch slurm/gpu.sh`.

## Where things stand
- Submitted so far: B2 (`artifacts/submission_B2.csv`), public LB 0.99626, nested LOYO 0.99377. Not from a notebook, so it cannot be final as is.
- Best candidate: XGBoost `base,lp,ls` (`ls_*` is LP at water value x0.9, x1.1: `lp4_w90.npz`, `lp4_w110.npz`) + ST-Net with prices. Blend with the 79-feature tree model: forward 0.9936 vs 0.99338 without `ls_*`; blend with the 66-feature model not yet scored. Not yet through the gate.
- Features to drop if confirmed: `mf_*`, `lps_*`. Scale columns: keep for ST-Net, neutral for trees.
- Rejected: eta 0.04/800 rounds, d=96, MILP features, pseudo-labels, overlap smoothing, higher bypass cost, 6 net seeds in blend.

## Next steps, in order
1. Forward is done and clean: `XR_ls_f` 0.99341 (66 features), `XR_sens_f2` 0.99338 (79). Run LOYO from clean code: `XR_ls` (`--groups base,lp,ls --sens`, drop scale columns, seeds 0,1,2) and, as reference, `XR_sens2` (lane lines in `logs/lanes/a2.txt`), about 6 min each on a free GPU. Then run the gate against XR_ref. If `XR_ls` passes, it is the tree feature set.
2. `train_net.py` calls `build(c, z)` without `sens`: add `--sens`, test `ls_*` on ST-Net (register first). Run e12 seeds 3-5 to settle e12 vs e24 (R10 correction).
3. Fix final models, train on all 8 years, write test logits, blend with fixed weight about .7/.3 (flat .6-.8). Weight must be explained in the report.
4. Short notebook 1 (best) and 2 (differently risked hedge, for example trees only): self-contained, from raw data incl. LP solves, offline, requirements.txt, header with names, id, team. Verify runtime from a clean directory: limit 12 h on about 4 cores, 32 GB, one GPU, not high-end. Check ST-Net memory on a smaller GPU. All LP sets: 3 x 3647 solves at about 1.3 s.
5. Report: EDA (>=4 of the listed kinds), >=2 predictor types, feature engineering, interpretation (SHAP/importance), all models tried including failures, robustness. Clearly titled sections.
6. Deadline 2026-11-08 22:00 Oslo (strict): select 2 Kaggle predictions, deliver notebooks and report.

## Known limitations to state honestly
- LP structure was chosen seeing all years' labels; no untouched holdout. Negative price behaviour (2.2% of test hours, 0.014% of train) cannot be verified. Lio, Vinje G3, Hogga are the weakest units.
- Cleanup candidates (my own, about 250 MB each): `lp_global, lp_local, lp_tele, lp_v2, lp_v3, lp4_b100, lp4_b1000, milp_v2, mp4_b10` in `artifacts/lp`. Check that nothing reproducible needs them. `/cluster` inode quota is tight (about 1.97M of 2M).
