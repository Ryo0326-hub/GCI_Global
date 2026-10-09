---
type: experiment
status: awaiting_confirmation
created: 2026-10-09
tags: [gci, experiment]
---
# F02 Affordability

[[03 Experiments]] · [[06 Decisions]]

Hypothesis: Credit/annuity/income/goods and household ratios improve risk ranking.

Configuration: `('affordability',)`

Comparison: same input hashes, five folds, seed 42. Audit disabled during exploration.

- [x] Run the controlled LightGBM experiment.
- [x] Link its generated report below.
- [x] Record the decision and next step.
- [ ] Confirm the group with CatBoost.
- [ ] Confirm the strongest candidate with another fold seed.

Report: [[Experiments/Runs/20261009T050024803961Z_F02_affordability]]. LightGBM OOF AUC **0.757137**, **+0.008438** versus B01 LightGBM, improving all five folds. Audit unevaluated; public score pending.
Decision: Keep affordability alone for confirmation and subsequent tuning. The combined missingness/tenure group scored lower. [[research/FEATURE_SCREEN]] · [[02 Runbook|Colab configuration]].

The frozen five-fold training reproduced its CSV exactly. [CSV in Drive](https://drive.google.com/file/d/1x-DY1CpWPt4lBadoksqwhIXexL6TODh3/view) · [Matching code ZIP](https://drive.google.com/file/d/1LEvfOpsbYZt32mGrFW_y-gBpB6sbYH8x/view). Public scoring is pending.
