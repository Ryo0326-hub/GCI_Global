# GCI Global — Home Credit

An independent competition notebook and an Obsidian project vault. The workflow compares LightGBM and CatBoost, supports controlled feature experiments and persistent Optuna searches, and packages each submission with its executed code and configuration.

[Open the new notebook in Colab](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR) · [Competition folder in Drive](https://drive.google.com/drive/folders/106aE1f2Nz4pRZ1EGtryNFhgvo74451sC)

## Start in Colab

1. Reload the updated `comp.ipynb` and choose **Runtime → Run all**. Setup verifies the pins and continues. Authorize your own Drive mount when prompted.
2. Keep the first run's B01 configuration. It fits five development folds and reserves 20% of labeled rows for a later audit.
3. Save the notebook before packaging. Check that the CSV contains 61,500 rows and the ZIP verification succeeds.
4. Review the experiment report before uploading the CSV and matching code ZIP to Omnicampus. Record the returned public score in the Submission Tracker.

The notebook uses `MyDrive/GCI_Global/Competition/input/{train.csv,test.csv,sample_submission.csv}` and verifies the original file hashes. The original tutorial is preserved.

The pinned stack uses **NumPy 2.2.6** and passed fresh Python **3.12 and 3.13** checks, including both model libraries. Do not downgrade to NumPy 1.26.4 on Python 3.13. Setup and notebook controls use standard-library code; numerical steps run in fresh processes with streamed progress. This avoids stale notebook imports and requires no forced kernel restart. See `research/COLAB_ENVIRONMENT_FIX.md` for diagnosis and recovery.

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
.venv/bin/python scripts/run_local.py --input /path/to/Competition/input --output output --notes Experiments/Runs
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

Baseline → separate feature ablations → Optuna confirmation → model-diversity/blend checks → frozen audit → final full-data training. Keep folds and seeds fixed when comparing a single change. Audit evaluation remains disabled during exploration. Neural models are a later bounded experiment if they provide useful diversity.

B01 CatBoost earned a user-confirmed public AUC of **0.756**, rank **668**. Its local OOF AUC was **0.752544**. Affordability features raised LightGBM's local OOF AUC from **0.748699** to **0.757137**; this candidate has no public score yet. See `research/FEATURE_SCREEN.md` and `04 Submission Tracker.md`. The next target is **0.80 ROC AUC**, with **0.85** as a stretch goal. The earlier leader screenshot showed 0.79169. Local OOF AUC does not establish a leaderboard result. The last competition upload counts; the supplied deadline is November 20, 2026 at 01:00 UTC.

See `COMPETITION_PLAN.md`, `research/CODE_STUDY.md` and `CITATIONS.md` for the reasoning and primary references. Downloaded reference-code snapshots remain local; their pinned source links are recorded in Git. Only competition-provided data enters training.
