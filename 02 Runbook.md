---
type: runbook
tags: [gci, workflow]
---
# Runbook

The delivered S17 v2 CSV/ZIP are already complete. Later export and GPU preflight fixes apply to future notebook runs. The exact executed source and notebook for S17 v2 remain inside `comp_S17_v2.zip`; do not substitute the newer root source or rewrite the historical manifest. No retraining is needed to submit the delivered files. The sprint audit is closed.

[[00 Dashboard]] · [[03 Experiments]]

## Run the current notebook

**The November 17 searches and reserved audit are complete.** The final configuration is an equal CPU CatBoost/XGBoost blend; no GPU is required. The verified S17 v2 CSV and matching ZIP in [[04 Submission Tracker]] can be uploaded without rerunning the notebook. The experiment notebook and its outputs are preserved in [the dated Colab copy](https://colab.research.google.com/drive/1JO1rZCLe6dobLMWWvgCYg67TFPKnXVse).

1. Reload the updated [comp.ipynb](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR). If you have unsaved edits, save a separate copy before reloading.
2. Select a T4 under Runtime → Change runtime type if the configuration requests GPU, then Runtime → Run all. The setup cell installs the pinned packages and tests them in a fresh process. Model steps also use fresh processes, so execution continues without a forced restart. Authorize your own Drive mount when prompted.
3. Use the saved sprint configuration. The tiny backend fit must report the requested device; a GPU error does not trigger a hidden CPU fallback.
4. Wait for each fold and the output checks. Save the notebook with Cmd+S / Ctrl+S before the ZIP cell.
5. Download `submission.csv` and `comp.zip`, or find them in the Competition Drive folder.

Keep NumPy **2.2.6**. NumPy 1.26.4 is unsupported on the Python 3.13 runtime seen in the traceback. The pins passed fresh Python 3.12 and 3.13 checks and the setup's native model checks passed in your actual Colab Python 3.13.16 session. The reported crash was caused by the earlier forced restart, which has been removed. Reload the updated notebook and Run all using the current session. Delete the runtime only if the setup explicitly reports a failed fresh-process check. See [[research/COLAB_ENVIRONMENT_FIX]].

## Output locations

| File | Drive location |
| --- | --- |
| Current candidate | GCI_Global/Competition/output/submission.csv |
| Matching code package | GCI_Global/Competition/comp.zip |
| Colab-generated historical evidence | GCI_Global/Competition/output/runs/run_id/ |
| S17 v2 local-run backup | GCI_Global/Competition/S17_v2_run_artifacts.zip |
| Progress-note archive | GCI_Global/Competition/obsidian_reports.zip |

Each run saves OOF predictions, parameters, package versions, input hashes, fold assignments, fitted models and its executed source snapshot. Packaging first checks saved-model predictions against the frozen CSV. Fresh training across platforms can differ numerically; GPU training is also nondeterministic. The saved-model check verifies the recorded CSV under its recorded environment. New runs update the current candidate and preserve earlier run folders. The default blend was frozen and checked on the reserved audit before final training.

## Import Colab reports into Obsidian

Colab writes Markdown to Drive. That does **not** automatically synchronize this Mac vault. Download `obsidian_reports.zip`, then run this from the project folder:

```bash
.venv/bin/python scripts/import_run_reports.py ~/Downloads/obsidian_reports.zip
```

Alternatively, unzip the archive and copy its `Experiments/Runs` notes into this vault's matching folder. The helper preserves edited experiment notes and updates the generated Latest Run summary. The dashboard then shows the new run.

## Controlled experiments

Use a new label and `mode='development'`. Keep seed 42, audit seed 20261009, five confirmation folds and `evaluate_audit=False`. Raw/affordability CatBoost comparisons and the initial LightGBM tuning are complete; consult [[06 Decisions]] before repeating them. [[research/TOP_SOLUTIONS_2026-11-17]] records the public-code ideas and measured results.

Set `RUN_OPTUNA=True` only for development search. It uses three folds; the returned fixed+sampled parameters retain GPU/device settings and are confirmed with five folds. Trial counts are additional on resume. Keep one writer per study, with SQLite on local disk. The notebook backs it up consistently to Drive when the step ends. A lost runtime can lose trials since its last backup.

The notebook builder uses `configs/sprint/final.json` once it exists; otherwise it uses the GPU affordability development configuration. To request CPU deliberately, set CatBoost `task_type='CPU'` and remove `devices`. Do not edit the executed pipeline while a run is active: a changed source fingerprint prevents promotion of that run.

## Audit and final refit

**This sprint's audit has already been used once.** Do not run it again for further tuning. It selected 50% CatBoost / 50% XGBoost at AUC 0.764923; see [[03 Experiments]]. `configs/sprint/final.json` is the frozen final configuration. The guidance below describes the protocol for a future separately designed experiment.

Keep the audit disabled during exploration and early stopping. After recording at most three frozen finalists and their component hashes, use `scripts/audit_frozen.py` for one comparison of the saved models, or use a frozen audit configuration. The saved-model route uses each run's exact source snapshot, checks its training partition, and does no new fitting. Do not evaluate the same candidates again through both routes. Do not retune after inspecting it. Once choices are final, choose final mode, set frozen_choices=True and supply the selected blend weights. Final training includes all labeled rows and no longer provides independent audit evidence.

## Local run

The local Python 3.12 environment is in `.venv`. On this Mac the wrapper reuses the already-installed OpenMP library, without changing global package settings:

```bash
.venv/bin/python scripts/run_local.py --input /Users/ryokitano/Downloads/Competition/input --output output --config configs/sprint/final.json --notes Experiments/Runs
```

To edit the pipeline locally, change `src/gci_pipeline.py` and regenerate the notebook with `.venv/bin/python scripts/build_notebook.py`. The code ZIP refuses to package a saved notebook with a different pipeline version from the executed run.

## Submission

Upload the CSV and matching code ZIP manually to Omnicampus. Record score, timestamp, run ID and CSV hash in [[04 Submission Tracker]]. Verify that scoring finished. The last upload counts. Local CV never proves a public/private leaderboard result.
