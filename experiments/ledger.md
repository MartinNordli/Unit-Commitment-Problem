# Experiment ledger
Hypotheses are written here before they run. Numbers come from `experiments.jsonl`, written by the harness.
Gate (`src/compare.py`): beat the reference by more than measured seed noise, improve at least 6 of 8 years, lose no more than the noise on forward validation, from committed code.

## Locked on 2026-11-03, before this round
LP structure v4 (`src/lp_surrogate.py`, in `harness.lock`): telescoping values, head losses, priority order, YAML costs. No further LP changes without the human's approval.
Known limit: these choices were made looking at labels from all years, so OOF and forward scores are somewhat optimistic. No part of 2015-2022 is untouched. Public LB 0.99626 for B2 is one independent check.

## Registered 2026-11-03
- R1 Seed noise: XGBoost, 5 seeds, LOYO. Sets the noise level for the gate.
- R2 Label control on LP v4: shuffled whole-case labels. Expected about 0.59, the level base rates alone give.
- R3 Drop `mf_*`: gain was +0.00003. Expected to fail the gate as an addition, so simpler model preferred.
- R4 Drop `lps_*`: unit shares of all plants. Expected small loss.
- R5 Add `ls_*`: LP at water value x0.9 and x1.1. Hypothesis: flags near-ties, which are the main remaining LP error. Expected +0.0002 to +0.001.
- R6 Keep absolute price features: expected no gain, more drift.
- R7 Regime holdouts (`robust_regime.py`): LP features should lose much less than no-LP when trained on one regime and tested on another.
- R8 Learning curves (`robust_curves.py`): is 300 rounds over- or underfit?
- R9 Stress (`stress.py`): compare with A008 (blend scale_4 -0.00245, negative_cheapest -0.02291, price_noise -0.00653).
- R10 ST-Net epochs, chosen nested: budget for each year taken from the forward result of the year before, never the year itself.

## Results R1-R4, R6-R10 (2026-11-03)
- R1 Seed std XGBoost: 0.00003 LOYO, 0.00006-0.00009 forward. Gate noise set to 0.0001.
- R2 Shuffled labels on LP v4: 0.589. No leakage visible.
- R3 Without `mf_*`: LOYO -0.00001, forward -0.00007. Inside noise: `mf_*` would fail the gate as an addition.
- R4 Without `lps_*`: LOYO -0.00005, forward -0.00005. Inside noise.
- R6 With absolute prices, XGBoost: LOYO +0.00002, forward +0.00005, inside noise. ST-Net forward: +0.0007 (0.99195 vs 0.99125, 3 seeds each). Hypothesis (no gain) rejected for ST-Net; earlier lean towards removing them was wrong.
- R7 Regime holdouts: no-LP 0.910-0.932, LP 0.990-0.993. Absolute prices were better in 5 of 6 holdouts.
- R8 Curves: validation still rises slowly to about 1000 rounds (300 to 1000: 0.0000 to +0.0002); never falls, so no overfit in rounds. 7 years of training beat 1-3 years by about 0.002.
- R9 Stress, blend: scale_4 -0.00259, negative_cheapest -0.00716, price_noise -0.00775. A008: -0.00245, -0.02291, -0.00653. Different test set (2022 only), so indicative.
- R10 ST-Net epochs at 3 seeds, forward: without prices e12 0.99114, e16 0.99132, e24 0.99125; with prices e12 0.99157, e24 0.99195. e24 wins 4 of 4 years with prices. The earlier claim that 24 epochs overfits came from comparing 1 seed against 3 and does not hold.

## Registered 2026-11-03, second round
- R11 Add `ls_*` (= R5). R12 Drop both `mf_*` and `lps_*`. R13 XGBoost eta 0.04, 800 rounds. R14 ST-Net seeds 3-5, does a larger seed ensemble help? R15 ST-Net d=96. R16 Forward blend of all families.

## Round 3 (registered 2026-11-03, before running)
- R17 `ls_*` without `mf_*` and `lps_*`: R12 says those two groups add nothing, R11 says `ls_*` adds. Expected: same as R11 within noise, with fewer features.
- Note: R11-R15 report dirty=true because src/blend3.py was untracked when they ran. No trainer imports it. The gate does not accept them; whatever goes forward is rerun from a clean commit.

## Results R11-R16 (2026-11-03). All dirty=true (see note above): indicative, not yet accepted by the gate.
- R11 `ls_*` added, XGBoost 3 seeds vs XR_ref (5 seeds): LOYO +0.00023, 6/8 years, forward +0.00035, 3/4 years. 2020 +0.00095 and 2022 -0.00006. Hypothesis holds if the clean rerun agrees.
- R12 Drop `mf_*` and `lps_*` together: LOYO 0.00000, forward 0.00000, 59 vs 72 features. Simpler model preferred.
- R13 eta 0.04, 800 rounds: LOYO +0.00008, forward +0.00005. Inside noise, rejected: more time for nothing measurable.
- R14 ST-Net seeds 3-5: LOYO 0.99261 vs 0.99270, forward 0.99160 vs 0.99195. So 3-seed ST-Net ensembles differ by up to 0.00035. Six seeds pooled, forward: 0.99203.
- Correction to R10: e24 s345 0.99160 is level with e12 s012 0.99157, so "e24 beats e12" is not established. Same-seed e12 s345 not run. The with-prices gain shrinks to +0.00035 to +0.0007, and most of it comes from 2020 (0.98986 vs 0.98666).
- R15 ST-Net d=96, forward: 0.99132 vs 0.99195, 1/4 years. Rejected.
- R16 Forward-nested blends: xgb+net 0.99329, xgb+cat+net 0.99334, adding LightGBM or the no-price net does not help. Weights flat over trees 0.6-0.8.
- Follow-up (`src/pool_net.py`, weights not nested, picked from flat range): xgb_ref .7 + net 0.99338; xgb_sens .7 + net seeds012/345/6 seeds 0.99361/0.99355/0.9936. Net seed noise nearly vanishes in the blend, six seeds add nothing there.

## Results R17 and clean R11 forward (2026-11-03, commit 11585b1, dirty=false)
- R17 `XR_ls_f` (`base,lp,ls`, 66 features): forward 0.99341, +0.00037 vs XR_ref_f, 3/4 years (2021 -0.00003). Hypothesis holds: same as R11 within noise with fewer features.
- R11 clean rerun `XR_sens_f2`: 0.99338, identical per year to the dirty run, so the dirty flag changed nothing and GPU XGBoost reproduces exactly with the same seeds.
- LOYO from clean code still to run for both before the gate can pass them.

## Optional proposal, not required to explore (2026-11-03)
Synthetic data was discussed and judged unlikely to raise the score in general: noise changes the correct label, rolling cases already act as time shifts, and LP-labelled cases hold none of the SHOP-minus-LP residual the model learns.
- P1 Negative prices: push the cheapest hours of training weeks below zero and label them with the MILP/LP or a physical rule. Negative prices are 2.2 % of test hours and 0.014 % of train. Forward and LOYO cannot confirm it, because the validation years hardly contain such hours. Only suited to a hedge in notebook 2, and the report must say it could not be validated on labels.
- P2 Scale price and water values together as augmentation on ST-Net. Testable through the gate. Expected zero or slightly negative, since absolute prices carry information for the net (R6, R10).

## Round 4 (registered 2026-11-03, before running)
- R18 LOYO from clean code: `XR_ls` (66 features) and `XR_sens2` (79) against XR_ref. Expected about +0.0002. The 6/8 years criterion is close (dirty R11 had exactly 6/8), so a fail is possible. If `XR_ls` passes it is the tree feature set; if not, trees stay on `XR_min`-type features.
- R19 `ls_*` on ST-Net: `N5_ls`, `N5f_ls` (`base,lp,lps,mf,ls`, e24, seeds 0,1,2) vs N4_v4_scale, N4f_e24_scale. Expected +0.0001 to +0.0005 forward, smaller in the blend.
- R20 Settle e12 vs e24 on the same seeds: `N9b_e12_s345`, `N9f2_e12_s345`, against N4b_s345, N4f2_s345. Expected: no clear winner.
- R21 ST-Net without `mf_*` and `lps_*`: `N6_min`, `N6f_min` (`base,lp,ls`, e24, seeds 0,1,2). Expected level with R19 within net seed noise (up to 0.00035), and then preferred for simplicity.
- Note: net seed noise is larger than the gate's 0.0001, so net choices are also judged by their effect in the blend.

## Results round 4 (2026-11-03, commit 1d71092, dirty=false)
- R18 `XR_ls` LOYO 0.99349 vs XR_ref 0.99326: +0.00023, 7/8 years, forward +0.00037. PASS. `XR_sens2` also passes (6/8) but adds nothing over `XR_ls` (LOYO 0.00000, forward -0.00003, 79 vs 66 features). `XR_ls` is the tree feature set.
- R19 `ls_*` on ST-Net keeping `mf_*`, `lps_*`: `N5_ls` LOYO +0.00006, 4/8 years, forward +0.00017. FAIL.
- R20 e24 beats e12 on both seed sets: LOYO 0.99270 and 0.99261 vs 0.99243 and 0.99235; forward 0.99195 and 0.99160 vs 0.99157 and 0.99149; blended LOYO about 0.9938 vs 0.9936. Settled: e24. The R10 correction stands for the forward view alone, but LOYO and same-seed comparison now agree.
- R21 ST-Net `base,lp,ls`: `N6_min` LOYO 0.99334, +0.00064, 7/8 years, forward 0.99278, +0.00083, 4/4 years. PASS, and well beyond net seed noise (up to 0.00035). Larger than expected: removing `mf_*` and `lps_*` helps the net, unlike the trees.
- Blend of `XR_ls` + `N6_min`, trees 0.6/0.7/0.8: forward 0.99379/0.99379/0.99373, LOYO 0.99404/0.994/0.99391. Flat, so the weight stays at the previously fixed 0.7. For comparison B2 had nested LOYO 0.99377. Weight not nested; picked from a flat range set before this round.

## Final model for notebook 1 (fixed 2026-11-03)
XGBoost `base,lp,ls` without absolute price columns, seeds 0,1,2, 300 rounds, eta 0.08 + ST-Net `base,lp,ls` with them, e24, d=64, seeds 0,1,2. Logit blend 0.7/0.3. Trained on 2015-2022.

## R22 (registered 2026-11-03, before running)
Stress test of the final blend (`submission/build/stress2.py`), same perturbations as R9, ST-Net now with absolute prices, e24, weight 0.7/0.3, 1 seed each. Hypothesis: `negative_cheapest` and `scale_*` get worse for the net than in R9 because it now sees absolute prices, and the blend inherits part of that. Sensitivity only, not a true score. Informs the hedge in notebook 2.

## Results R22 (2026-11-03)
Final blend, change in AUC on perturbed 2022 inputs: scale_0.5 -0.00007, scale_2 +0.00052, scale_4 -0.00231, price_noise -0.00754, negative_cheapest -0.00722. R9 had -0.00259, -0.00775 and -0.00716 for the last three (scale_4, price_noise, negative_cheapest). Hypothesis mostly rejected: the blend did not get worse. The net alone is worse on scale_4 (-0.00400) but better than trees on negative_cheapest (-0.00618 vs -0.00792). On negative_cheapest p(on) falls from 0.137 to 0.008.

## Notebook 1 clean run (2026-11-03)
From an empty directory, 4 cores, H100: 50.6 minutes, LP 47.6. All LP schedules identical to the research artifacts. XGBoost logits identical. ST-Net logits max difference 0.125, Spearman 0.99999, because GPU training is not bit-exact. Peak about 11 GB RAM, about 4 GB GPU (sampled every 15 s). float16 and CPU paths tested on random data at real shapes; CPU estimate 4-5 hours in total. Not run on a smaller GPU.

## Public LB (2026-11-03)
`submission_1.csv` scored 0.99638, against 0.99626 for B2. `submission_2.csv` scored 0.99635. Independent sanity check only, not used for selection. Notebook 1 now repeats the best Kaggle score. The hedge costs 0.00003 on public, in line with forward (equal) and LOYO (-0.00001).

## R23 (registered 2026-11-03, before running)
Hedge for notebook 2. Notebook 1 bets that absolute price levels help the net (R6, R21), and test years have a different price distribution. Candidate: same blend, but ST-Net without absolute price columns as well, so nothing in the model reads the price level directly: `N7_noabs`, `N7f_noabs` (`base,lp,ls`, drop the 6 absolute columns, e24, seeds 0,1,2). Expected net forward 0.0003 to 0.0007 below `N6f_min`, blend less. Accepted as hedge if blend forward is within 0.0005 of notebook 1's 0.99379 and `scale_4` in the stress test is no worse. It is a hedge, so it does not need to beat notebook 1 and does not go through the gate.
Optional proposal P1 stays unexplored.

## Results R23 so far (2026-11-03, commit 5f0c925)
`N7f_noabs` net forward 0.99256 vs `N6f_min` 0.99278, smaller loss than expected. Blend with `XR_ls`, trees 0.6/0.7/0.8: 0.99377/0.99379/0.99374, level with notebook 1 at 0.7 (0.99379). Stress: scale_4 -0.00177 vs -0.00231, negative_cheapest -0.00668 vs -0.00722, price_noise -0.00759 vs -0.00754, unperturbed 0.99234 vs 0.99254 on half of 2022. Both criteria met: accepted as the hedge for notebook 2. LOYO still running, for the record.

## Worst-case notebook test
First attempt, job 25904579, did not run the notebook: `jupyter` has a fixed interpreter path in its first line, while Slurm reported COMPLETED. Fixed and resubmitted.

## Results R23 final and notebook 2 clean run (2026-11-03)
`N7_noabs` LOYO 0.99320 vs `N6_min` 0.99334. Blend with `XR_ls` at 0.7: LOYO 0.99399 vs 0.994. The net alone fails the gate against `N6_min`, as expected for a hedge that is not meant to pass it. Test predictions differ from submission 1 at Spearman 0.99889.
Notebook 2 from an empty directory, 4 cores, H100: 42.3 minutes, LP 39.2. XGBoost logits identical to the research run, ST-Net Spearman 0.99999, max logit difference 0.145. Output `artifacts/submission_2.csv`.

## R24 (registered 2026-11-03, before running)
Interpretation for the report (`submission/build/interp_final.py`). No model selection. Trained before 2022, measured on 2022, seed 0. Permutation importance for XGBoost, ST-Net and blend: each (case, unit) gets its week-long feature track from another random (case, unit), for single features and for six groups (market, calendar, unit identity, water state, LP schedule, LP sensitivity). Partial dependence for XGBoost on top features. Hypothesis: both models lean mainly on the LP schedule; the net loses more than the trees when market price features are shuffled, because it has the absolute columns. Sensitivity only, since shuffled inputs are not physically consistent.

## Results R24 (2026-11-04)
Unshuffled on 2022: XGBoost 0.9924, net 0.99096, blend 0.99268. Group loss, xgb/net/blend: LP schedule -0.238/-0.239/-0.216, LP sensitivity -0.018/-0.024/-0.016, unit -0.004/-0.082/-0.003, water state -0.002/-0.003/-0.001, market -0.0005/-0.0087/-0.0014, calendar -0.001/-0.001/-0.001. Hypothesis holds. Not predicted: the net loses 0.065 on `unit_index` alone, cause unknown.
Adversarial AUC, train vs test, without calendar and unit columns: with absolute prices 0.953, without 0.921, LP columns only 0.869, absolute only 0.922. The first version with calendar columns was wrong (0.92, 0.68, 0.87, a superset below its subset), since alternating blocks let the classifier memorise dates.

## Filenames and old GPU test (2026-11-04)
The task PDF's text layer shows spaces, but the rendered names have underscores, so the notebooks are now `Short_notebook_1.ipynb` and `Short_notebook_2.ipynb`. Worst-case test without GPU: job 25904586, started under the old name, same content. Test on P100 16 GB: `slurm/nb1_p100.sh`, job 25906288, results in `logs/nb1_p100/`.

## Worst-case notebook test, result (2026-11-04)
Job 25904586: notebook 1 from an empty directory, no GPU, 4 cores (Xeon Gold 6348, 2.6 GHz), 32 GB enforced by Slurm, succeeded. 170.9 minutes: LP 36.7, features 0.2, XGBoost 6.3, ST-Net 127.7. Slurm MaxRSS 18.5 GB. Earlier estimate of 4-5 hours was too high. Not bit-equal to the GPU run: against `artifacts/submission_1.csv` Spearman 0.99908, largest logit difference 1.26, 99th percentile 0.48. Trees alone Spearman 0.99858, net alone 0.99944, since XGBoost on CPU and float32 training differ from GPU. No score yet for this file: `logs/nb1_cpu/submission_1.csv`. A real 4-core laptop can be slower than this server CPU; that was not tested.

## P100 test (2026-11-04)
Job 25906288 on Tesla P100 16 GB failed after the LP: `XGBoostError: This program was not compiled for SM 60`. So the notebooks crash on old GPUs, and the earlier statement "a GPU with at least 6 GB is enough" was wrong. Fix in both notebooks: XGBoost and ST-Net each fall back to the CPU if the GPU fails, for an unsupported GPU or out of memory. No change to what runs on a working GPU. Rerun from an empty directory: both notebooks on H100 to confirm unchanged results, notebook 1 on P100 to test the fallback.
