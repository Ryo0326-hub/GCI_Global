---
type: project_dashboard
status: active
deadline: 2026-11-19T20:00:00-05:00
tags: [gci, home-credit]
---

# GCI World

Build evidence that the next model generalizes better, then submit the strongest reproducible candidate.

[Open comp.ipynb in Colab](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR) · [Competition folder in Drive](https://drive.google.com/drive/folders/106aE1f2Nz4pRZ1EGtryNFhgvo74451sC) · [GitHub project](https://github.com/Ryo0326-hub/GCI_Global)

| Item | Current state |
| --- | --- |
| Stage | S17 v2 final CSV/ZIP verified and ready for manual upload |
| Public result | **S17 v1: 0.76365, rank 2,918**; reported by Ryo with completed-scoring screenshots |
| Target | Aim >=0.80 public ROC AUC and top 10% final rank; earlier observed rank 500 = 0.77032 |
| Deadline | Nov 19, 2026, 8 p.m. Toronto time |
| Final ranking | Last submitted file, evaluated on the full test set |
| Earlier linked submission | B01 CatBoost, local OOF 0.752544 and October 9 public 0.756 |
| Best local evidence | Selected CatBoost/XGBoost blend: development 0.763037; reserved audit 0.764923 |
| Next action | [[docs/superpowers/plans/2026-11-17-honours-sprint\|Astra sprint plan]]: upload the S17 v2 CSV and matching ZIP; record completed scoring |

Research: [[research/TOP_SOLUTIONS_2026-11-17|Lessons from second-, fifth- and seventh-place solutions]]. S17 v1 has been submitted; S17 v2 completed validation and final training and is ready for manual upload.

## Latest run

![[Experiments/Runs/Latest Run]]

## Work

- [[Daily/2026-11-17|November 17 status]] and [[docs/superpowers/plans/2026-11-17-honours-sprint|Astra implementation handoff]].
- [[01 Roadmap]]: milestones and checkboxes.
- [[02 Runbook]]: run, export, reproduce and import reports.
- [[03 Experiments]]: queued experiments and their decisions.
- [[04 Submission Tracker]]: public scores and the last upload.
- [[05 Data and Rules]]: permitted inputs and validation boundaries.
- [[06 Decisions]]: why a feature/model became the champion.
- [[07 How the Submission Works]]: why a completed CSV can exist while other experiments run, and what the new features contribute.
- [[research/SPRINT_REVIEW_2026-11-17|Final review and handoff]]: verification, safeguards and exact executed notebook.
- [[research/CODE_STUDY|Public code study]] and [[COMPETITION_PLAN|Full competition plan]].
- [[research/COLAB_ENVIRONMENT_FIX|Colab setup fix]] and [[research/FEATURE_SCREEN|Feature comparison evidence]].

## Daily routine

Open today's Daily Note from the calendar icon. Choose one experiment from [[03 Experiments]]. Write the hypothesis before running it. Import the run report, then record the decision and next action. Keep public scores separate from local AUC.

## Deeper feature extension

Requested November 17: category combinations, credit-enquiry comparisons, explicit hypothetical loan-pricing scenarios, and XGBoost. These use only supplied predictors. The one frozen audit comparison is complete. See [[research/TOP_SOLUTIONS_2026-11-17]].

**First candidate ready:** [CSV](https://drive.google.com/file/d/1E9FVLIg9pU71J-kSKYORG8CaavSIZICD/view) · [matching code ZIP](https://drive.google.com/file/d/1mqUESU9r72glIu-IysKQu3wynVsZqJHC/view). Local AUC **0.760200**; public AUC **0.76365**, rank **2,918**. Details in [[04 Submission Tracker]].

**Latest submitted result: S17 v1 — public AUC 0.76365, rank 2,918**, reported by Ryo on November 17. Development AUC 0.760200; S17 v2 ready for upload. See [[04 Submission Tracker]].

**Recommended next submission — S17 v2:** [CSV](https://drive.google.com/file/d/1NNzbcs0UK3ZUbuHh6GaBXzwLUlYFHHMe/view) · [matching code ZIP](https://drive.google.com/file/d/1ir2pw5r823k55DYfxxTIXifON-keQ_jv/view). Both files are complete and verified. Public score pending; no rerun needed. The independent audit selected this blend at 0.764923 AUC.
