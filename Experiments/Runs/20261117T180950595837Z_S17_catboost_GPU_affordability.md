---
type: experiment_run
status: completed
run_id: 20261117T180950595837Z_S17_catboost_GPU_affordability
stage: development
oof_auc: 0.75727621
public_auc:
created: 2026-11-17
tags: [gci, experiment, generated]
---
# 20261117T180950595837Z_S17_catboost_GPU_affordability

[[00 Dashboard]] · [[03 Experiments]] · [[04 Submission Tracker]]

catboost generated 61,500 model probabilities. No competition upload occurred.

| Model | Development OOF AUC |
| --- | --- |
| catboost | 0.757276 |

Selected OOF AUC: **0.757276**. This is a local model-selection result, not a leaderboard score. Audit labels were not evaluated.

- Feature groups: affordability
- Folds / seed: 5 / 42
- Weights: `{"catboost": 1.0}`
- Runtime: 231.7 seconds
- CSV SHA-256: `4a222028f9fa39a8b0fde71404a61cb7d8d0a86487b2210e131ed4323d94a559`
- Run artifacts: `/content/drive/MyDrive/GCI_Global/Competition/output/sprint_gpu/runs/20261117T180950595837Z_S17_catboost_GPU_affordability`

## Decision

- [ ] Compare with the current champion on the same folds.
- [ ] Record keep/reject and the reason in [[06 Decisions]].
- [ ] If submitted, record the public score and upload timestamp in [[04 Submission Tracker]].

## Notes

Write observations here. Generated metrics above are immutable evidence for this run.
