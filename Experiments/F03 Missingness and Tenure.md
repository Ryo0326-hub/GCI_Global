---
type: experiment
status: screened
created: 2026-10-09
tags: [gci, experiment]
---
# F03 Missingness and Tenure

[[03 Experiments]] · [[06 Decisions]]

Hypothesis: Missingness and cleaned employment/life-stage features add conditional signal.

Configuration: `('missingness', 'tenure')`

Comparison: same input hashes, five folds, seed 42. Audit disabled during exploration.

- [x] Run the controlled LightGBM experiment.
- [x] Link its generated report below.
- [x] Record the decision and next step.

Report: [[Experiments/Runs/20261009T050051427044Z_F03_missingness_tenure]]. LightGBM OOF AUC **0.749427**, **+0.000728** versus B01 LightGBM.
Decision: Small standalone gain, but adding this group to affordability reduced AUC from 0.757137 to 0.756631. Defer the combination; CatBoost was not screened. [[Experiments/F04 Combined Features]] · [[research/FEATURE_SCREEN]].
