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
