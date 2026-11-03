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
