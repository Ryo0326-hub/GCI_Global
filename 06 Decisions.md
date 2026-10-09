---
type: decision_log
tags: [gci, decisions]
---
# Decisions

[[00 Dashboard]] · [[03 Experiments]]

## Current champion

Provisional baseline: **B01 CatBoost**, development OOF AUC **0.752544**. [[Experiments/Runs/20261009T041309638772Z_B01_all_features|Run evidence]]. This has no recorded leaderboard score and is not a frozen final model.

## 2026-10-09: baseline model comparison

Five fixed development folds, seed 42, 136,961 development rows and 34,241 reserved audit rows. Both models used all 32 supplied predictors with employment-placeholder cleaning and an indicator. CatBoost scored 0.752544 pooled OOF AUC vs LightGBM 0.748699 and led on every fold. Select CatBoost's five-model average for the initial candidate, with weights CatBoost 1.0 and LightGBM 0.0. Audit labels were not evaluated.

The 61,500-row CSV has SHA-256 `0b1e655e888da5817c926605be88a003e60d344767bb4890f7eda50085c41108`. Saved-fold inference reproduced those exact bytes. CSV and matching comp.zip are in Drive. No Omnicampus upload occurred.

Next: verify one Colab execution, then test F01 score summaries/interactions independently. Keep the same partition and folds; compare both models with their B01 results before choosing further feature groups or tuning. See [[research/SETUP_VALIDATION]].

## 2026-10-09: project setup

Use a new comp.ipynb, preserve the tutorial, and keep project notes beside the code as an Obsidian vault. Begin with LightGBM and CatBoost. Reserve audit data, save OOF evidence, and retain run artifacts. Feature groups and Optuna are supported but optional. Neural models require a later bounded experiment.

## Decision format

Date, candidate run, comparison run, controlled change, OOF/fold evidence, audit status, decision, reason, next experiment. Use the Decision template for an additional note and link it here.
