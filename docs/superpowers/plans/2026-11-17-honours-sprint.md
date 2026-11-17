# GCI Honours Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Ryo selected Astra for coding. Steps use checkbox syntax for tracking.

**Goal:** Aim for >=0.80 public ROC AUC; deliver a stronger validated CSV and its matching code ZIP before November 19, 2026, 8 p.m. Toronto time.

**Architecture:** Preserve the existing pipeline and fixed validation. Use Colab GPU for CatBoost and CPU for LightGBM, with CSC resources only when actually accessible/available. Preserve backend settings through Optuna, freeze finalists for audit, then refit on all labels.

**Tech Stack:** Existing pinned Python/NumPy/pandas/SciPy/sklearn/LightGBM/CatBoost/Optuna stack, portable CLI, self-contained Colab notebook and Obsidian reports.

**Spec:** [[docs/superpowers/specs/2026-11-17-honours-sprint-design]]

## Global Constraints

- Deadline: November 19, 2026, 20:00 America/Toronto. Target final artifacts by noon that day.
- Only supplied data; preserve hashes and every test ID. No external labels, pretrained external weights or manual overrides.
- Audit seed 20261009; audit fraction 0.20; first comparison seed 42; five confirmation folds. Audit stays outside tuning/early stopping until finalists freeze.
- November 17 verification: public 0.76253/rank 3,237 belongs to F02 affordability LightGBM; rank 500 score is 0.77032. A >=0.80 score is a target, not an achieved result.
- Tiramisu GPU access is denied by current ACLs; mannitol GPU is occupied in the supplied snapshot. No system/permission changes or interference with other jobs.
- Start shared CPU work with four threads. Never silently fall back from a requested GPU to CPU.
- Preserve historical models, predictions, CSVs, ZIPs and manifests. No automated competition upload; the last uploaded file counts.

## Review Focus

1. Fixed GPU parameters disappear during tuning or best-parameter application.
2. GPU requested but inaccessible; a long job must fail clearly before wasting time.
3. Interrupted jobs must not promote incomplete predictions as a submission.
4. Changed input/config/source must not be disguised as the same historical run.
5. CSV/code ZIP must match: correct IDs, valid probabilities and frozen executed source/configuration.

---

### Task 1: Establish provenance and a working backend

**Files:** Read `04 Submission Tracker.md`, `research/FEATURE_SCREEN.md`, `src/gci_pipeline.py`, historical manifests and the saved notebook; record findings in the tracker and dated daily note.

**Interfaces:** Consumes existing CSV/run hashes and leaderboard evidence; produces current submission identity (or explicitly unresolved status), the observed cutoff, and a measured one-fold backend pilot.

- [x] Identify the CSV that earned 0.76253 and read the score near rank 500 at https://edu.omnicamp.us/en/courses/170/assignments/1038/leader-board/ . Do not guess a model/run identity.
- [x] Allocate Colab GPU and inspect its model/free VRAM, driver and Python. Run a tiny native CatBoost GPU fit with explicit `task_type='GPU'` and the assigned logical device.
- [x] Benchmark one fixed development fold, logging effective parameters, timing and validation AUC. Do not use audit or require CPU/GPU byte equality.
- [x] Cap GPU setup at 30 minutes. If unavailable, continue CPU LightGBM and reduce CatBoost search. CSC GPU allocation is a user/admin action, not a prerequisite.
- [x] Verify finite predictions, correct validation IDs, explicit backend and no audit rows. A pilot is not five-fold score evidence.

### Task 2: Fix GPU parameter propagation through Optuna

**Files:** Modify `src/gci_pipeline.py:tune_model` and `scripts/build_notebook.py` optional-tuning cell as needed; regenerate `comp.ipynb`. Test in `tests/test_pipeline.py` or new `tests/test_tuning.py`.

**Interfaces:** Preserve `tune_model(input_dir, study_dir, cfg, model_name='lightgbm', n_trials=20, timeout=3600) -> dict`. Return fixed selected-model parameters merged with the selected search parameters, so the caller retains its backend.

- [x] Write a failing test: set `cfg.model_params['catboost']={'task_type':'GPU','devices':'0'}`; intercept `fit_cv` during an Optuna trial; assert that both GPU keys remain in its configuration alongside sampled parameters. Stub fitting so the test needs no GPU.
- [x] Assert that returned best parameters and notebook-applied confirmation parameters retain both GPU keys. Also assert that a CPU configuration acquires no GPU keys.
- [x] Run the test and confirm it fails with the current replacement of model parameters.
- [x] Merge fixed model parameters with sampled parameters for each trial; sampled search keys win. Merge fixed parameters into the best configuration returned/applied. Retain LightGBM `subsample_freq=1`.
- [x] Run `python scripts/build_notebook.py`, `python -m pytest -q`, `python scripts/check_notebook.py`, and `python scripts/check_environment.py`. Require passing output before long experiments.
- [x] Run one real GPU trial and a confirmation fit; record that both use the requested backend. Explicit GPU errors must not masquerade as CPU successes.
- [x] Commit this bounded fix while preserving unrelated Obsidian/user changes. GPU nondeterminism is recorded, not treated as a failed byte-equality test.

### Task 3: Compare affordability candidates and tune

**Files:** Reuse `src/gci_pipeline.py`, `scripts/run_local.py` and `scripts/screen_features.py`; create frozen configs in `configs/sprint/`; keep artifacts in new run directories and generated notes in `Experiments/Runs/`.

**Interfaces:** Existing `Config`, `run_experiment(...)`, `tune_model(...)`, `oof.csv`, `fold_assignments.csv`, `manifest.json` and `reproduce_config.json`.

- [x] Preserve B01/F02 evidence. If source hashes change, do not rewrite old manifests. New runs get new labels; common input/fold identities establish comparisons.
- [x] Compare raw and affordability CatBoost on the same backend and folds. Keep affordability LightGBM (OOF 0.757137) as the measured CPU feature candidate.
- [x] Screen only a few extra groups if time remains: fold-trained peer statistics and category/frequency features. Do not automatically add previously unhelpful score summaries or the weaker combined group.
- [x] Search three development folds: initially 20 LightGBM CPU trials and 20 CatBoost GPU trials, two-hour timeout per model. Use pilot timings to decide whether to extend to 40. CPU CatBoost fallback starts at most 10 trials.
- [x] Persist complete trials and identities; one writer per study. Do not share a live SQLite study across hosts. Use SQLite backup for a consistent copy if backing up a live local study to Drive.
- [x] Confirm winners on five development folds. Repeat the strongest with seed 20261117 while retaining audit seed 20261009.
- [x] Verify complete OOF coverage keyed by ID, metrics, runtime and unchanged audit isolation. Small selected gains are not statistical proof.
- [x] Optional XGBoost has a one-hour total cap only after the main candidates work. Defer neural training unless the main artifacts and deadline buffer are secure. Do not add TRL or service infrastructure.

### Task 4: Freeze finalists, audit once, refit all labels

**Files:** Reuse `Config`, `run_experiment` and `select_blend`; record frozen choices in `06 Decisions.md` and `configs/sprint/`.

**Interfaces:** Frozen model/feature/backend configurations and coarse weights produce one audit comparison and final all-data prediction artifacts.

- [x] By November 18 evening, select at most three finalists using development evidence. Compare standalone models and coarse probability blends; the current OOF meta-holdout is diagnostic, not independent audit evidence.
- [x] Freeze features, parameters, backend, seeds and candidate weights before examining audit. Evaluate those finalists once; do not repeatedly tune on audit feedback.
- [ ] Refit selected models on all 171,202 labelled rows with `mode='final'`, `frozen_choices=True`, and explicit selected `blend_weights`. Do not label final-training CV as an independent audit score.
- [ ] Retain fitted models and exact configuration/versions; verify saved-model inference reproduces the frozen CSV under its recorded environment. New GPU retraining may differ numerically.
- [ ] Verify existing final/audit boundary tests still pass and final manifests show full labelled-data use without an independent audit claim.

### Task 5: Verify and deliver the candidate

**Files:** Reuse `validate_submission`, `build_bundle`, notebook packaging and `scripts/import_run_reports.py`; update tracker/dashboard/daily notes.

**Interfaces:** Final manifest and predictions produce `submission.csv`, matching `comp.zip`, historical run artifacts and an Obsidian report archive.

- [ ] Reuse existing ID/probability/bundle tests; add a focused regression only when changed behavior introduces a new contract.
- [ ] Check exactly 61,500 rows, schema `SK_ID_CURR,TARGET`, unique sample-order IDs, finite probabilities in [0,1], and no index column.
- [ ] Build the ZIP from executed source and frozen configuration. Verify archive integrity and CSV/code hashes. Include citations; keep datasets outside Git.
- [ ] Save CSV/ZIP in `GCI_Global/Competition` Drive destinations and verify uploaded bytes. If training remotely, package there or resolve artifact locations explicitly; server absolute paths do not become Colab paths automatically.
- [ ] Target final artifacts by November 19 noon. Ryo uploads CSV and matching ZIP manually, waits for completed scoring, and records score/hash/time.
- [ ] Verify the intended candidate is the last upload before 8 p.m. Toronto time. No public/local result guarantees the final top-10% placement.

## Handoff

Astra is the user-selected coding executor. Start with Tasks 1–2, then affordability CatBoost. Reuse completed baseline evidence. Do not wait for CSC GPU access to start the sprint.

## Implementation extension requested November 17

Ryo requested deeper features and model diversity while searches were running. Added individually tested category combinations, enquiry-window comparisons and explicit pricing hypotheses, plus XGBoost. Pricing hypotheses improved the LightGBM profile candidate; category/enquiry groups did not. Model-specific feature and training options preserve the exact validated component settings in a final ensemble while keeping folds and audit seeds shared. A separate frozen-model audit comparison evaluates saved snapshots without refitting or tuning on the audit.

A first interim CSV and matching ZIP were delivered and read back from Drive while the stronger comparisons continue. Full-data refit and final artifact delivery remain pending.
