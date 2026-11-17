---
type: experiment_run
status: completed
run_id: 20261117T182101139406Z_S17_catboost_CPU_affordability
stage: development
oof_auc: 0.75900937
public_auc:
created: 2026-11-17
tags: [gci, experiment, generated]
---
# 20261117T182101139406Z_S17_catboost_CPU_affordability

[[00 Dashboard]] · [[03 Experiments]] · [[04 Submission Tracker]]

catboost generated 61,500 model probabilities. No competition upload occurred.

| Model | Development OOF AUC |
| --- | --- |
| catboost | 0.759009 |

Selected OOF AUC: **0.759009**. This is a local model-selection result, not a leaderboard score. Audit labels were not evaluated.

- Feature groups: affordability
- Folds / seed: 5 / 42
- Weights: `{"catboost": 1.0}`
- Runtime: 579.7 seconds
- CSV SHA-256: `7a8316f7742efccc16aaee9194d0bc48f7924db92763f1193bda03ccea25ffed`
- Run artifacts: `/Users/ryokitano/Documents/Competitions/GCI_World/output/sprint/local/runs/20261117T182101139406Z_S17_catboost_CPU_affordability`

## Decision

- [ ] Compare with the current champion on the same folds.
- [ ] Record keep/reject and the reason in [[06 Decisions]].
- [ ] If submitted, record the public score and upload timestamp in [[04 Submission Tracker]].

## Notes

Write observations here. Generated metrics above are immutable evidence for this run.
