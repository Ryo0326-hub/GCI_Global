---
type: roadmap
tags: [gci, planning]
---
# Roadmap

[[00 Dashboard]] · [[docs/superpowers/plans/2026-11-17-honours-sprint]]

Target: **>=0.80 public ROC AUC** and final top 10%. The current evidence is below 0.80; neither ranking nor final score is guaranteed. Deadline: **November 19, 2026, 8 p.m. Toronto**.

## Completed

- [x] Verify competition objective, supplied files and submission format.
- [x] Set up the independent Colab notebook, local repository and Obsidian vault.
- [x] Resolve the numerical-package setup and verify Python 3.12/3.13.
- [x] Identify the exact F02 CSV behind public AUC 0.76253.
- [x] Read the observed rank-500 cutoff: 0.77032 on November 17.
- [x] Verify native T4 GPU training and preserve GPU settings through Optuna.
- [x] Confirm affordability models on five development folds.
- [x] Research public top solutions and test applicable feature ideas.
- [x] Deliver a verified first CSV and matching code ZIP in Drive.

## In progress

- [x] Complete the bounded searches and deeper feature confirmations.
- [x] Compare CatBoost, LightGBM and XGBoost using aligned OOF predictions.
- [x] Freeze at most three finalists, including coarse blends.
- [x] Inspect the reserved audit once, then freeze the final selection.
- [x] Refit on all labelled data and verify saved-model inference.
- [x] Deliver the final CSV, matching ZIP and updated notebook to Drive.

## Ryo's upload steps

- [ ] Upload the intended CSV and matching code ZIP to Omnicampus.
- [ ] Wait for completed scoring; record score, rank, hash and upload time.
- [ ] Verify the intended file is the LAST submission before the deadline.

S17 v1 was submitted at public AUC 0.76365. S17 v2 is ready for manual upload; both candidates remain available in [[04 Submission Tracker]]. It does not require another notebook run. Neural or distributed experiments will not delay the final deliverable.
