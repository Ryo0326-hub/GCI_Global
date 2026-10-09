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
| Stage | Public baseline recorded; Colab forced restart removed; feature screening completed |
| Public result | B01 CatBoost: 0.756, rank 668, reported October 9 |
| Target | First 0.80 ROC AUC; 0.85 is a stretch goal. Earlier leader snapshot: 0.79169 |
| Deadline | Nov 19, 2026, 8 p.m. Toronto time |
| Final ranking | Last submitted file, evaluated on the full test set |
| Public champion | B01 CatBoost, local OOF 0.752544 and public 0.756 |
| Best local candidate | F02 affordability LightGBM: OOF 0.757137; public score pending |
| Next action | Reload the updated Colab notebook and Run all. Confirm affordability with CatBoost and another fold seed |

## Latest run

![[Experiments/Runs/Latest Run]]

## Work

- [[01 Roadmap]]: milestones and checkboxes.
- [[02 Runbook]]: run, export, reproduce and import reports.
- [[03 Experiments]]: queued experiments and their decisions.
- [[04 Submission Tracker]]: public scores and the last upload.
- [[05 Data and Rules]]: permitted inputs and validation boundaries.
- [[06 Decisions]]: why a feature/model became the champion.
- [[research/CODE_STUDY|Public code study]] and [[COMPETITION_PLAN|Full competition plan]].
- [[research/COLAB_ENVIRONMENT_FIX|Colab setup fix]] and [[research/FEATURE_SCREEN|Feature comparison evidence]].

## Daily routine

Open today's Daily Note from the calendar icon. Choose one experiment from [[03 Experiments]]. Write the hypothesis before running it. Import the run report, then record the decision and next action. Keep public scores separate from local AUC.
