---
type: experiment
status: screened
created: 2026-10-09
tags: [gci, experiment]
---
# F01 Credit Scores

[[03 Experiments]] · [[06 Decisions]]

Hypothesis: Available-score summaries and pairwise interactions improve joint score representation.

Configuration: `('scores',)`

Comparison: same input hashes, five folds, seed 42. Audit disabled during exploration.

- [x] Run the controlled LightGBM experiment.
- [x] Link its generated report below.
- [x] Record the decision and next step.

Report: [[Experiments/Runs/20261009T045958594430Z_F01_scores]]. LightGBM OOF AUC **0.748759**, only **+0.000060** versus B01 LightGBM.
Decision: Defer this group. The gain is too small to prioritize; CatBoost was not screened. Affordability is the next candidate. [[research/FEATURE_SCREEN]].
