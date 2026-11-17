---
type: experiment_run
status: completed
run_id: 20261117T192003394287Z_S17_xgboost_CPU_profiles_tuned
stage: development
oof_auc: 0.75935382
public_auc:
created: 2026-11-17
tags: [gci, experiment, generated]
---
# 20261117T192003394287Z_S17_xgboost_CPU_profiles_tuned

[[00 Dashboard]] · [[03 Experiments]] · [[04 Submission Tracker]]

xgboost generated 61,500 model probabilities. No competition upload occurred.

| Model | Development OOF AUC |
| --- | --- |
| xgboost | 0.759354 |

Selected OOF AUC: **0.759354**. This is a local model-selection result, not a leaderboard score. Audit labels were not evaluated.

- Feature groups: affordability, peer_profile
- Folds / seed: 5 / 42
- Weights: `{"xgboost": 1.0}`
- Runtime: 249.2 seconds
- CSV SHA-256: `1a97251ea636c9e1553fa8917cf4cf307b7838ad8c830535afdd8889bb422356`
- Run artifacts: `/Users/ryokitano/Documents/Competitions/GCI_World/output/sprint/xgboost_tuned/runs/20261117T192003394287Z_S17_xgboost_CPU_profiles_tuned`

## Decision

- [ ] Compare with the current champion on the same folds.
- [ ] Record keep/reject and the reason in [[06 Decisions]].
- [ ] If submitted, record the public score and upload timestamp in [[04 Submission Tracker]].

## Notes

Write observations here. Generated metrics above are immutable evidence for this run.
