# GCI Global — Home Credit

An independent competition notebook and an Obsidian project vault. The workflow compares LightGBM, CatBoost and XGBoost, supports controlled feature experiments and persistent Optuna searches, and packages each submission with its executed code and configuration.

[Open the new notebook in Colab](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR) · [Competition folder in Drive](https://drive.google.com/drive/folders/106aE1f2Nz4pRZ1EGtryNFhgvo74451sC)

## Start in Colab

1. Reload the updated `comp.ipynb`, then **Runtime → Run all**. The frozen final ensemble uses CPU; select a T4 only for a separate GPU experiment. Setup verifies the pins and continues. Authorize your own Drive mount when prompted.
2. Use the saved sprint configuration in the notebook. Development runs reserve 20% of labelled rows; a frozen final configuration uses all 171,202 labels. The backend preflight must pass before training.
3. Save the notebook before packaging. Check that the CSV contains 61,500 rows and the ZIP verification succeeds.
4. Review the experiment report before uploading the CSV and matching code ZIP to Omnicampus. Record the returned public score in the Submission Tracker.

The notebook uses `MyDrive/GCI_Global/Competition/input/{train.csv,test.csv,sample_submission.csv}` and verifies the original file hashes. The original tutorial is preserved.

The pinned stack uses **NumPy 2.2.6** and passed fresh Python **3.12 and 3.13** checks, including all three model libraries. Do not downgrade to NumPy 1.26.4 on Python 3.13. Setup and notebook controls use standard-library code; numerical steps run in fresh processes with streamed progress. This avoids stale notebook imports and requires no forced kernel restart. See `research/COLAB_ENVIRONMENT_FIX.md` for diagnosis and recovery.

| Output | Drive path under GCI_Global/Competition |
| --- | --- |
| Current candidate | `output/submission.csv` |
| Matching source/configuration package | `comp.zip` |
| Historical models, predictions and run manifest | `output/runs/<run_id>/` |
| Notes for import into the Mac vault | `obsidian_reports.zip` |

## Track progress in Obsidian

Open this repository folder as a vault, then open `00 Dashboard.md`. Roadmap, Runbook, Experiments, Submission Tracker and Decisions are linked from there. Templates, Daily Notes and dashboard bookmarks use built-in Obsidian features.

Colab exports Markdown reports to Drive. Download `obsidian_reports.zip` and import it from this folder:

```bash
.venv/bin/python scripts/import_run_reports.py ~/Downloads/obsidian_reports.zip
```

Edited run notes are preserved. The generated latest-run summary updates the dashboard. This is a manual report import; it does not provide automatic Drive-to-vault synchronization.

## Develop locally

Use Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pytest==8.4.1 nbformat==5.10.4
.venv/bin/python scripts/run_local.py --input /path/to/Competition/input --output output --config configs/sprint/final.json --notes Experiments/Runs
```

LightGBM needs an OpenMP runtime on macOS. The local wrapper can reuse an existing Anaconda OpenMP library when present; Colab does not need this workaround.

The implementation is in `src/gci_pipeline.py`. After changing it, regenerate the self-contained notebook and replace the Drive copy before a Colab run:

```bash
.venv/bin/python scripts/build_notebook.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_notebook.py
.venv/bin/python scripts/check_environment.py
```

Each run stores a frozen `reproduce_config.json`. The ZIP includes a reproduction command, dependency versions, input/code/CSV hashes, the notebook, source, citations and matching CSV. Fitted models and datasets stay outside the code ZIP and Git.

## Experiment order

The November 17 sprint selected a **50% CatBoost / 50% XGBoost** blend after one reserved audit comparison (AUC **0.764923**). Its final configuration preserves affordability features for CatBoost and pricing/peer features for XGBoost. Final training uses all 171,202 labels across five folds. The independent audit is now closed to further model selection.

Baseline → separate feature ablations → Optuna confirmation → model-diversity/blend checks → frozen audit → final full-data training. Keep folds and seeds fixed when comparing a single change. Audit evaluation remains disabled during exploration. Neural models are a later bounded experiment if they provide useful diversity.

The latest user-confirmed public submission is **S17 v1 shallow CPU CatBoost: 0.76365, rank 2,918**. Its development OOF AUC is 0.760200. The prior F02 CSV was linked by exact downloaded hash to public AUC 0.76253. The observed rank 500 score on November 17 was 0.77032. Our target is **>=0.80 ROC AUC** and final top 10%; neither is achieved by a local validation result. Current sprint results and decisions are in `00 Dashboard.md` and `06 Decisions.md`.

GPU backend/device settings survive Optuna and best-parameter application. Studies run with one writer and source/input/configuration identities. Colab uses a local SQLite database and consistent Drive backups at the end of each tuning step. Audit requires frozen choices and explicit weights. Saved-model inference is verified against the exported CSV; new GPU retraining can differ numerically.

Only the last competition upload counts. Deadline: **November 19, 2026, 8 p.m. Toronto** (November 20 01:00 UTC). Keep time for completed scoring and a corrective upload.

See `COMPETITION_PLAN.md`, `research/CODE_STUDY.md`, `research/TOP_SOLUTIONS_2026-11-17.md` and `CITATIONS.md` for the reasoning and primary references. Downloaded reference-code snapshots remain local; their pinned source links are recorded in Git. Only competition-provided data enters training.
