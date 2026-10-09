---
type: runbook
tags: [gci, workflow]
---
# Runbook

[[00 Dashboard]] · [[03 Experiments]]

## First Colab run

1. Reload the updated [comp.ipynb](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR). If you have unsaved edits, save a separate copy before reloading.
2. Select Runtime → Run all. The setup cell installs the pinned packages and tests them in a fresh process. Model steps also use fresh processes, so execution continues without a forced restart. Authorize your own Drive mount when prompted.
3. Keep B01's five-fold development configuration and the audit disabled.
4. Wait for each fold and the output checks. Save the notebook with Cmd+S / Ctrl+S before the ZIP cell.
5. Download `submission.csv` and `comp.zip`, or find them in the Competition Drive folder.

Keep NumPy **2.2.6**. NumPy 1.26.4 is unsupported on the Python 3.13 runtime seen in the traceback. The pins passed fresh Python 3.12 and 3.13 checks and the setup's native model checks passed in your actual Colab Python 3.13.16 session. The reported crash was caused by the earlier forced restart, which has been removed. Reload the updated notebook and Run all using the current session. Delete the runtime only if the setup explicitly reports a failed fresh-process check. See [[research/COLAB_ENVIRONMENT_FIX]].

## Output locations

| File | Drive location |
| --- | --- |
| Current candidate | GCI_Global/Competition/output/submission.csv |
| Matching code package | GCI_Global/Competition/comp.zip |
| Historical evidence | GCI_Global/Competition/output/runs/run_id/ |
| Progress-note archive | GCI_Global/Competition/obsidian_reports.zip |

Each run saves OOF predictions, parameters, package versions, input hashes, fold assignments and fitted models. New runs update the current candidate and preserve earlier run folders. A model blend is optional and must be separately validated.

## Import Colab reports into Obsidian

Colab writes Markdown to Drive. That does **not** automatically synchronize this Mac vault. Download `obsidian_reports.zip`, then run this from the project folder:

```bash
.venv/bin/python scripts/import_run_reports.py ~/Downloads/obsidian_reports.zip
```

Alternatively, unzip the archive and copy its `Experiments/Runs` notes into this vault's matching folder. The helper preserves edited experiment notes and updates the generated Latest Run summary. The dashboard then shows the new run.

## Next experiment

Affordability is the strongest completed feature screen. After confirming the setup works, reproduce that candidate by changing these three values in the configuration cell:

```python
label='F02_affordability',
models=('lightgbm',),
feature_groups=('affordability',),
```

Keep every other setting, including seed 42, five folds and `evaluate_audit=False`. Its local OOF AUC was **0.757137**, versus raw-feature LightGBM **0.748699**. The default notebook still reproduces B01 so environment repair does not silently change the experiment. For the next controlled comparison, use a new label and `models=('catboost',)` with the same affordability group, then confirm the strongest model with another seed. Enable Optuna after that confirmation. See [[research/FEATURE_SCREEN]] and [[06 Decisions]].

## Audit and final refit

Keep the audit disabled during exploration and early stopping. After freezing finalists, an audit run can evaluate it. Once choices are final, choose final mode, set frozen_choices=True and supply the selected blend weights. Final training includes all labeled rows and no longer provides independent audit evidence.

## Local run

The local Python 3.12 environment is in `.venv`. On this Mac the wrapper reuses the already-installed OpenMP library, without changing global package settings:

```bash
.venv/bin/python scripts/run_local.py --input /Users/ryokitano/Downloads/Competition/input --output output --notes Experiments/Runs
```

To edit the pipeline locally, change `src/gci_pipeline.py` and regenerate the notebook with `.venv/bin/python scripts/build_notebook.py`. The code ZIP refuses to package a saved notebook with a different pipeline version from the executed run.

## Submission

Upload the CSV and matching code ZIP manually to Omnicampus. Record score, timestamp, run ID and CSV hash in [[04 Submission Tracker]]. Verify that scoring finished. The last upload counts. Local CV never proves a public/private leaderboard result.
