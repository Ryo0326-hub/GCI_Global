---
type: experiment_run
status: completed
run_id: 20261117T191448247969Z_S17_xgboost_CPU_pricing_seed2
stage: development
oof_auc: 0.76154924
public_auc:
created: 2026-11-17
tags: [gci, experiment, generated]
---
# 20261117T191448247969Z_S17_xgboost_CPU_pricing_seed2

[[00 Dashboard]] · [[03 Experiments]] · [[04 Submission Tracker]]

xgboost generated 61,500 model probabilities. No competition upload occurred.

| Model | Development OOF AUC |
| --- | --- |
| xgboost | 0.761549 |

Selected OOF AUC: **0.761549**. This is a local model-selection result, not a leaderboard score. Audit labels were not evaluated.

- Feature groups: affordability, peer_profile, pricing_proxies
- Folds / seed: 5 / 20261117
- Weights: `{"xgboost": 1.0}`
- Runtime: 193.5 seconds
- CSV SHA-256: `08625e97c41df2aeffed71e329923135db40f0fcd3642b05c6b2a0da183f9581`
- Run artifacts: `/Users/ryokitano/Documents/Competitions/GCI_World/output/sprint/xgboost_pricing_seed2/runs/20261117T191448247969Z_S17_xgboost_CPU_pricing_seed2`

## Decision

- [ ] Compare with the current champion on the same folds.
- [ ] Record keep/reject and the reason in [[06 Decisions]].
- [ ] If submitted, record the public score and upload timestamp in [[04 Submission Tracker]].

## Notes

Write observations here. Generated metrics above are immutable evidence for this run.
