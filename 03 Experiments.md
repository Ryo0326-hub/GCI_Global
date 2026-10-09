---
type: experiment_board
tags: [gci, experiments]
---
# Experiments

[[00 Dashboard]] · [[06 Decisions]]

## Ready

- [[Experiments/F01 Credit Scores]]
- [[Experiments/F02 Affordability]]
- [[Experiments/F03 Missingness and Tenure]]
- [[Experiments/T01 Optuna]]
- [[Experiments/E01 Blend]]

## Completed

- [[Experiments/B01 All Features]]: CatBoost 0.752544 vs LightGBM 0.748699 development OOF AUC. CSV/ZIP verified; audit untouched. Colab execution still needs confirmation.

## Run evidence

![[Experiments/Runs/Latest Run]]

Generated run notes live in Experiments/Runs. The machine-readable ledger is `output/experiments.csv`. Use the Experiment template for a new idea and record a decision after it finishes. Later candidates: peer features, another fold seed, XGBoost, and a bounded neural experiment.

## Comparison rules

Compare the same dataset bytes and fold assignments. Change one feature group or model setting at a time. Check pooled OOF AUC and fold changes. Confirm the strongest candidates using another seed. A small improvement selected from many trials may be noise.
