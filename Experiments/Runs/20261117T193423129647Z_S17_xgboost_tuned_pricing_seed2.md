---
type: experiment_run
status: completed
run_id: 20261117T193423129647Z_S17_xgboost_tuned_pricing_seed2
stage: development
oof_auc: 0.76157357
public_auc:
created: 2026-11-17
tags: [gci, experiment, generated]
---
# 20261117T193423129647Z_S17_xgboost_tuned_pricing_seed2

[[00 Dashboard]] · [[03 Experiments]] · [[04 Submission Tracker]]

xgboost generated 61,500 model probabilities. No competition upload occurred.

| Model | Development OOF AUC |
| --- | --- |
| xgboost | 0.761574 |

Selected OOF AUC: **0.761574**. This is a local model-selection result, not a leaderboard score. Audit labels were not evaluated.

- Feature groups: affordability, peer_profile, pricing_proxies
- Per-model feature/training overrides: `{}`
- Folds / seed: 5 / 20261117
- Weights: `{"xgboost": 1.0}`
- Runtime: 164.5 seconds
- CSV SHA-256: `39b88ee8254d2956c656cd39673f16b0b5a538f464efca5aab3634ac3823d830`
- Run artifacts: `/Users/ryokitano/Documents/Competitions/GCI_World/output/sprint/xgboost_tuned_pricing_seed2/runs/20261117T193423129647Z_S17_xgboost_tuned_pricing_seed2`

## Decision

- [ ] Compare with the current champion on the same folds.
- [ ] Record keep/reject and the reason in [[06 Decisions]].
- [ ] If submitted, record the public score and upload timestamp in [[04 Submission Tracker]].

## Notes

Write observations here. Generated metrics above are immutable evidence for this run.
