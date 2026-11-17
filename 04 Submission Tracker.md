---
type: submission_tracker
tags: [gci, submissions]
---
# Submission Tracker

S17 v2's delivered CSV and ZIP remain unchanged after final review. The [exact executed notebook](https://colab.research.google.com/drive/1zRbaJKregEsrMBcxkpNW4ni8Oe24X6-J) is archived separately; the main notebook has later export/GPU safeguards for future runs. See [[research/SPRINT_REVIEW_2026-11-17]]. No rerun is needed for the delivered candidate.

[[00 Dashboard]] · [[06 Decisions]]

**Latest submitted result: S17 v1 — public ROC AUC 0.76365, rank 2,918.** Ryo supplied completed-scoring screenshots on November 17. The target remains >=0.80 AUC and top 10%; threshold accuracy is not the competition metric. The earlier observed rank-500 score was 0.77032, a changing public benchmark rather than a guaranteed final honours cutoff.

| Result reported / verified | Candidate | Development OOF AUC | Public AUC | Rank | CSV hash prefix |
| --- | --- | ---: | ---: | ---: | --- |
| October 9 | B01 CatBoost | 0.752544 | 0.756 | 668 | `0b1e655e` |
| November 17 verification | F02 affordability LightGBM | 0.757137 | 0.76253 | 3,237 | `41fb2033` |
| November 17 user result | S17 v1 shallow CPU CatBoost | 0.760200 | **0.76365** | **2,918** | `4e44b9f3` |

## S17 v2 — ready, not yet submitted

[CSV](https://drive.google.com/file/d/1NNzbcs0UK3ZUbuHh6GaBXzwLUlYFHHMe/view) · [matching code ZIP](https://drive.google.com/file/d/1ir2pw5r823k55DYfxxTIXifON-keQ_jv/view)

- Final run: [[Experiments/Runs/20261117T193823071888Z_S17_v2_final]]. Equal CPU CatBoost/XGBoost; five fitted folds per family.
- Development selection AUC: **0.763037**. Reserved audit AUC: **0.764923** (one pre-frozen comparison).
- Full-population final-training OOF AUC: **0.764534**. This uses all 171,202 labelled rows across folds and is **not independent audit evidence**.
- Public AUC / rank / competition upload time: **unknown; not submitted by the agent**.
- CSV SHA-256: `9fb3842240f920de8527e6e48eedc635238d49b5cdd6041465bf6b2052465032`.
- Matching ZIP SHA-256: `011a0bf867a55462dde2520e86d8ba4892a5c2b42d0e4c677d24c987419fa399`.
- Exactly 61,500 rows in sample ID order, valid probabilities, no index column. Saved-model predictions are byte-identical, with maximum absolute error 0.
- Both versioned and canonical Drive CSV/ZIP copies were downloaded again and matched their hashes. The notebook embeds the same executed source and final configuration.
- Current Drive exports: `GCI_Global/Competition/output/submission.csv` and `GCI_Global/Competition/comp.zip`. Versioned files above remain stable when later runs replace the current exports.
- [Saved models and run evidence](https://drive.google.com/file/d/14d4sl28TSj9U0sAYF4b6F9OHtClsYuMn/view) · [Obsidian run reports](https://drive.google.com/file/d/18Io3M2AZ3E193W00ZNPpIoQhaKjbG-gs/view).
- Earlier B01 current exports were renamed to historical filenames, preserving file IDs and bytes. F02 and S17 v1 remain unchanged.

Upload the CSV and this matching ZIP, wait for completed scoring, then record its score/rank and upload time. **No notebook rerun is needed to use these files.** This candidate does not yet establish the >=0.80 target or honours placement.

## S17 v1

[Submission CSV](https://drive.google.com/file/d/1E9FVLIg9pU71J-kSKYORG8CaavSIZICD/view) · [Matching code ZIP](https://drive.google.com/file/d/1mqUESU9r72glIu-IysKQu3wynVsZqJHC/view)

- Run: [[Experiments/Runs/20261117T183627714036Z_S17_catboost_CPU_shallow]].
- CSV SHA-256: `4e44b9f3e3e171999cda3e5d4314739c0068b7f4877f19765cab0cac99132821`.
- ZIP SHA-256: `30dc7f3739312462ad80e2d3c016e1702680f9313cf4e733c1575fc9e7568a75`.
- Five folds over 136,961 development rows; audit reserved. This interim candidate precedes the all-label final refit.
- Saved models reproduce all 61,500 exported probabilities exactly. Both uploaded Drive files were downloaded again and matched their original hashes; the ZIP contains the exact CSV and executed code/configuration.
- Ryo confirmed CSV submission. Matching code-ZIP upload has not been separately confirmed.
- Scoring display: `2026-11-18 04:06:47`, display timezone not verified. The output identifies November 17, 2026 UTC. Exact upload time is unconfirmed.

S17 improved public AUC by **0.00112** and the reported rank by 319 positions. Public feedback is recorded, not used to optimize row-specific predictions or blend weights. A frozen 50% CatBoost / 50% XGBoost ensemble was selected on the reserved audit (AUC 0.764923) and has completed full-data final training. S17 v2 is ready; its public result is not yet known.

## Preserved earlier submissions

**F02 affordability LightGBM:** [[Experiments/Runs/20261009T050024803961Z_F02_affordability]]. [CSV](https://drive.google.com/file/d/1x-DY1CpWPt4lBadoksqwhIXexL6TODh3/view) · [Matching ZIP](https://drive.google.com/file/d/1LEvfOpsbYZt32mGrFW_y-gBpB6sbYH8x/view). The scored CSV downloaded from Omnicampus on November 17 exactly matched SHA-256 `41fb2033764d3b8e36a63c0e1af9ca2d23ff23ef18243b060d4e304890edc47e`. Completed-scoring display: `2026-10-10 03:04:59`; display timezone and actual upload time unconfirmed. Code-ZIP upload unconfirmed.

**B01 CatBoost:** [[Experiments/Runs/20261009T041309638772Z_B01_all_features]]. [CSV](https://drive.google.com/file/d/1nVURqIMihFHbX3NDGP_bBTtNI-I1-u0H/view) · [Matching ZIP](https://drive.google.com/file/d/19uZAJys7QEEjfI4pmbD6Yk-J6EICQR4j/view). CSV SHA-256 `0b1e655e888da5817c926605be88a003e60d344767bb4890f7eda50085c41108`. CSV identity was user-confirmed; screenshot showed public score 0.756 and rank 668. Upload time and code-ZIP upload unconfirmed.

## Before the deadline

- [x] S17 v1 CSV submitted and completed scoring recorded.
- [ ] Confirm matching code ZIP upload for the intended final candidate.
- [ ] Record the final candidate's score, run ID, hash and upload time.
- [ ] Verify the intended CSV is the LAST upload before the deadline.

**Deadline: November 19, 2026, 8 p.m. Toronto time.** A new completed notebook run may change the current export; always pair the CSV and ZIP from the same run. No competition upload is automated.
