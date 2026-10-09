---
type: experiment_board
tags: [gci, experiments]
---
# Experiments

[[00 Dashboard]] · [[06 Decisions]]

## Ready

- F02 confirmation: CatBoost with affordability, followed by a second fold seed.
- [[Experiments/T01 Optuna]]
- [[Experiments/E01 Blend]]

## Completed

- [[Experiments/B01 All Features]]: CatBoost 0.752544 vs LightGBM 0.748699 development OOF AUC. CSV/ZIP verified; audit untouched. Colab execution still needs confirmation.
- [[Experiments/F01 Credit Scores]]: LightGBM 0.748759; no material gain.
- [[Experiments/F02 Affordability]]: LightGBM 0.757137; strongest local candidate, confirmation pending.
- [[Experiments/F03 Missingness and Tenure]]: LightGBM 0.749427; small standalone gain.
- [[Experiments/F04 Combined Features]]: LightGBM 0.756631; below affordability alone, defer combination.

These feature screens compared LightGBM only; they do not establish CatBoost outcomes. All audit scores remain unevaluated. [[research/FEATURE_SCREEN|Full comparison and next step]].

## Run evidence

![[Experiments/Runs/Latest Run]]

Generated run notes live in Experiments/Runs. The machine-readable ledger is `output/experiments.csv`. Use the Experiment template for a new idea and record a decision after it finishes. Later candidates: peer features, another fold seed, XGBoost, and a bounded neural experiment.

## Comparison rules

Compare the same dataset bytes and fold assignments. Change one feature group or model setting at a time. Check pooled OOF AUC and fold changes. Confirm the strongest candidates using another seed. A small improvement selected from many trials may be noise.
