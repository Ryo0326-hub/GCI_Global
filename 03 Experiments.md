---
type: experiment_board
tags: [gci, experiments]
---
# Experiments

[[00 Dashboard]] · [[06 Decisions]] · [[research/TOP_SOLUTIONS_2026-11-17]]

## Current stage

The bounded model and feature comparisons are complete. Three designs were frozen before one comparison on the 34,241 reserved audit rows. That check selected **50% shallow CPU CatBoost + 50% tuned pricing XGBoost**. The completed final five-fold run used all 171,202 labelled rows as its training population, with final-training OOF AUC 0.764534. Its CSV and matching ZIP passed saved-model and Drive readback checks. Audit feedback will not be used for additional tuning.

## Development evidence

All first-seed results below use seed 42 and the same 136,961 development rows. Second-seed results use 20261117 with the same reserved audit partition.

| Candidate | Development OOF AUC | Interpretation |
| --- | ---: | --- |
| Historical F02 affordability LightGBM | 0.757137 | Public 0.76253 |
| GPU CatBoost raw | 0.750751 | Native GPU baseline |
| GPU CatBoost affordability | 0.757276 | Same-backend feature improvement |
| Tuned GPU CatBoost | 0.758229 | Second seed 0.758608 |
| GPU CatBoost category crosses | 0.757862 | No improvement; tested blends also weaker |
| Tuned LightGBM + peer profiles | 0.758555 | Second seed 0.758829 |
| Pricing hypotheses + LightGBM profiles | 0.759832 | Second seed 0.760046 |
| Shallow CPU CatBoost affordability | 0.760200 | S17 v1; public 0.76365 |
| XGBoost profiles | 0.758902 | Additional model family |
| XGBoost profiles + pricing | 0.761092 | Second seed 0.761549 |
| Tuned XGBoost profiles + pricing | 0.761422 | Second seed 0.761574 |
| 50% CatBoost / 50% tuned XGBoost | 0.763037 | Frozen finalist |
| 25% LightGBM / 50% CatBoost / 25% tuned XGBoost | 0.763285 | Highest development blend; frozen finalist |

The searches recorded 40 LightGBM trials (21 completed, 19 pruned), 20 GPU CatBoost trials (14 completed, 6 pruned), and 8 XGBoost trials (7 completed, 1 pruned). Pruning stops unpromising trials early. Five-fold confirmation determines the comparison above; three-fold search scores are kept separate.

Additional loan/household features, GOSS, category crosses and enquiry comparisons did not improve their tested LightGBM baselines. An optional CPU CatBoost second-seed run was cancelled before completion when XGBoost became the stronger standalone candidate; its partial files were never promoted to a submission.

## One reserved audit comparison

| Pre-frozen candidate | Audit ROC AUC |
| --- | ---: |
| Tuned pricing XGBoost | 0.762829 |
| **50% CatBoost / 50% XGBoost** | **0.764923** |
| 25% LightGBM / 50% CatBoost / 25% XGBoost | 0.764597 |

The difference between the two blends is small; this is a selection among three fixed designs, not proof that one will win the leaderboard. The selected features, parameters, backends and weights are in `configs/sprint/final.json`. No model above demonstrates 0.80 AUC.

## Run evidence

![[Experiments/Runs/Latest Run]]

Completed run notes are in `Experiments/Runs`. Immutable artifacts and search records are under `output/sprint/`; the frozen audit record is `research/SPRINT_AUDIT_2026-11-17.json`. Historical October runs and S17 v1 remain preserved. Final-training OOF scores reuse the full labelled population and must not be described as independent audit results.
