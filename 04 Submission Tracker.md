---
type: submission_tracker
tags: [gci, submissions]
---
# Submission Tracker

[[00 Dashboard]] · [[06 Decisions]]

**Latest submitted result: S17 v1 — public ROC AUC 0.76365, rank 2,918.** Ryo supplied completed-scoring screenshots on November 17. The target remains >=0.80 AUC and top 10%; threshold accuracy is not the competition metric. The earlier observed rank-500 score was 0.77032, a changing public benchmark rather than a guaranteed final honours cutoff.

| Result reported / verified | Candidate | Development OOF AUC | Public AUC | Rank | CSV hash prefix |
| --- | --- | ---: | ---: | ---: | --- |
| October 9 | B01 CatBoost | 0.752544 | 0.756 | 668 | `0b1e655e` |
| November 17 verification | F02 affordability LightGBM | 0.757137 | 0.76253 | 3,237 | `41fb2033` |
| November 17 user result | S17 v1 shallow CPU CatBoost | 0.760200 | **0.76365** | **2,918** | `4e44b9f3` |

## S17 v1

[Submission CSV](https://drive.google.com/file/d/1E9FVLIg9pU71J-kSKYORG8CaavSIZICD/view) · [Matching code ZIP](https://drive.google.com/file/d/1mqUESU9r72glIu-IysKQu3wynVsZqJHC/view)

- Run: [[Experiments/Runs/20261117T183627714036Z_S17_catboost_CPU_shallow]].
- CSV SHA-256: `4e44b9f3e3e171999cda3e5d4314739c0068b7f4877f19765cab0cac99132821`.
- ZIP SHA-256: `30dc7f3739312462ad80e2d3c016e1702680f9313cf4e733c1575fc9e7568a75`.
- Five folds over 136,961 development rows; audit reserved. This interim candidate precedes the all-label final refit.
- Saved models reproduce all 61,500 exported probabilities exactly. Both uploaded Drive files were downloaded again and matched their original hashes; the ZIP contains the exact CSV and executed code/configuration.
- Ryo confirmed CSV submission. Matching code-ZIP upload has not been separately confirmed.
- Scoring display: `2026-11-18 04:06:47`, display timezone not verified. The output identifies November 17, 2026 UTC. Exact upload time is unconfirmed.

S17 improved public AUC by **0.00112** and the reported rank by 319 positions. Public feedback is recorded, not used to optimize row-specific predictions or blend weights. A frozen 50% CatBoost / 50% XGBoost ensemble was selected on the reserved audit (AUC 0.764923) and is now in full-data final training. Its public result is not yet known.

## Preserved earlier submissions

**F02 affordability LightGBM:** [[Experiments/Runs/20261009T050024803961Z_F02_affordability]]. [CSV](https://drive.google.com/file/d/1x-DY1CpWPt4lBadoksqwhIXexL6TODh3/view) · [Matching ZIP](https://drive.google.com/file/d/1LEvfOpsbYZt32mGrFW_y-gBpB6sbYH8x/view). The scored CSV downloaded from Omnicampus on November 17 exactly matched SHA-256 `41fb2033764d3b8e36a63c0e1af9ca2d23ff23ef18243b060d4e304890edc47e`. Completed-scoring display: `2026-10-10 03:04:59`; display timezone and actual upload time unconfirmed. Code-ZIP upload unconfirmed.

**B01 CatBoost:** [[Experiments/Runs/20261009T041309638772Z_B01_all_features]]. [CSV](https://drive.google.com/file/d/1nVURqIMihFHbX3NDGP_bBTtNI-I1-u0H/view) · [Matching ZIP](https://drive.google.com/file/d/19uZAJys7QEEjfI4pmbD6Yk-J6EICQR4j/view). CSV SHA-256 `0b1e655e888da5817c926605be88a003e60d344767bb4890f7eda50085c41108`. CSV identity was user-confirmed; screenshot showed public score 0.756 and rank 668. Upload time and code-ZIP upload unconfirmed.

## Before the deadline

- [x] S17 v1 CSV submitted and completed scoring recorded.
- [ ] Confirm matching code ZIP upload for the intended final candidate.
- [ ] Record the final candidate's score, run ID, hash and upload time.
- [ ] Verify the intended CSV is the LAST upload before the deadline.

**Deadline: November 19, 2026, 8 p.m. Toronto time.** A new completed notebook run may change the current export; always pair the CSV and ZIP from the same run. No competition upload is automated.
