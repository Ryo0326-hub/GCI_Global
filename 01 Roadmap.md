---
type: roadmap
tags: [gci, planning]
---
# Roadmap

[[00 Dashboard]] · [[COMPETITION_PLAN]]

## Oct 9–11: reliable baselines

- [x] Inspect the rules, tutorial, data and public code.
- [x] Identify the Drive notebook and input/output folders.
- [x] Build an independent notebook and reproducible pipeline.
- [x] Verify the Obsidian dashboard and report import.
- [x] Complete B01 on five development folds with reserved audit data.
- [x] Verify the matching CSV/code ZIP and compare model OOF results.
- [ ] Execute the notebook in Colab and confirm its Drive exports.

## Oct 12–18: feature evidence

- [ ] F01: credit-score combinations.
- [ ] F02: affordability and family ratios.
- [ ] F03: missingness and employment/life-stage features.
- [ ] Combine only feature groups that improve fixed-fold results.

## Oct 19–25: tuning

- [ ] Persistent Optuna search on development rows.
- [ ] Confirm best configurations on five folds and another seed.
- [ ] Test supported peer statistics fitted within folds.

## Oct 26–Nov 8: diversity

- [ ] Compare LightGBM/CatBoost blend with standalone champions.
- [ ] Evaluate XGBoost if additional diversity is needed.
- [ ] Run a bounded MLP/TabM experiment if compute is available.

## Nov 9–15: freeze choices

- [ ] Preselect at most three finalists using development evidence.
- [ ] Inspect the reserved audit once after freezing designs.
- [ ] Record the final features, parameters, seeds and weights.

## Nov 16–19: final reproduction and upload

- [ ] Full-data training after choice freeze.
- [ ] Clean-session reproduction and CSV/ZIP hash checks.
- [ ] Submit and wait for completed scoring.
- [ ] Record the public score and uploaded file hash.
- [ ] Verify the intended file is the LAST submission before 8 p.m. Nov 19.
