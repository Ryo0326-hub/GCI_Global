---
type: experiment
status: completed
created: 2026-10-09
tags: [gci, experiment]
---
# B01 All Features

[[03 Experiments]] · [[06 Decisions]]

Hypothesis: Compare LightGBM and CatBoost using every supplied predictor with minimal cleaning.

Configuration: `()`

Comparison: same input hashes, five folds, seed 42. Audit disabled during exploration.

- [x] Run the controlled experiment.
- [x] Link its generated report below.
- [x] Record the decision and next step.

Report: [[Experiments/Runs/20261009T041309638772Z_B01_all_features]]
Decision: [[06 Decisions]]. Keep CatBoost as the provisional baseline. Confirm the Colab workflow, then test the scores feature group against B01 on the same folds.
