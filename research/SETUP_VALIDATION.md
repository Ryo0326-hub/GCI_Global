# Setup verification — October 9, 2026

[[00 Dashboard]] · [[Experiments/Runs/20261009T041309638772Z_B01_all_features|Baseline run]]

The complete baseline ran locally with pinned packages. LightGBM and CatBoost each completed five folds on 136,961 development rows. The 34,241-row audit was reserved and never scored.

| Model | Development OOF AUC |
| --- | --- |
| LightGBM | 0.7486989812 |
| CatBoost | 0.7525440922 |

CatBoost was selected. `output/submission.csv` contains all 61,500 test IDs in sample-submission order, unique IDs, finite probabilities in [0,1], and exactly the required two columns. Loading the saved CatBoost folds and predicting again reproduced the CSV byte for byte.

- CSV SHA-256: `0b1e655e888da5817c926605be88a003e60d344767bb4890f7eda50085c41108`
- Code ZIP SHA-256: `4d68a947e2ffb2362521b32e83a3cd715c01c1633030fe327d0d21766da6a221`
- ZIP checks: valid archive, matching CSV and executed pipeline hashes, frozen configuration, pinned dependencies, notebook and citations. Datasets and fitted models are excluded.
- Four unit tests passed for ID alignment, invalid probabilities, guarded ratios, fold-only peer/category processing and final/audit configuration boundaries.
- A separate 3,000-row development-only integration sample passed LightGBM/CatBoost training, every optional feature group, one persistent Optuna trial, blend diagnostics and CSV/ZIP export. Its small-sample scores are not competition evidence. No original audit rows entered that sample.
- The 21-cell notebook passed schema, Python syntax and embedded-source checks. The full notebook is saved in the existing Drive Colab file. The tutorial remains unchanged.
- The project dashboard is open in the GCI_World Obsidian vault. Built-in Templates, Daily Notes and bookmarks are configured; run reports and manual ZIP import are available.
- The initial project commit `13d80c0` was pushed to [GitHub](https://github.com/Ryo0326-hub/GCI_Global). [GitHub's pipeline/notebook checks passed](https://github.com/Ryo0326-hub/GCI_Global/actions/runs/37884313623). Raw inputs, predictions, model files, local runtime and downloaded reference snapshots are excluded from Git.

Local outputs and matching CSV/ZIP/report exports are available in Drive. Downloaded-byte hashes matched the original local notebook, CSV, code ZIP and report ZIP exactly. The original notebook hash predates the later environment repair.

Later October 9 updates: you confirmed the B01 Drive CSV earned **0.756**, rank **668**. This public result is user-reported and linked to the exported CSV, rather than inferred from local CV. A complete Colab training/export run still needs confirmation. The earlier setup passed fresh Python 3.12 and 3.13 checks and its native model fits also passed in your actual Colab Python **3.13.16** session. The reported crash came from the setup's forced restart. That restart has been removed; numerical work now uses fresh processes. The updated design passed eight tests, notebook checks and a bounded local notebook-flow test covering tuning, both models, blend diagnostics and CSV/ZIP/report export. See [[research/COLAB_ENVIRONMENT_FIX]] and [[research/FEATURE_SCREEN]].
