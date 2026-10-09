"""Create the project vault using Obsidian's built-in features only."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
notes = {
"00 Dashboard.md": '''---
type: project_dashboard
status: active
deadline: 2026-11-19T20:00:00-05:00
tags: [gci, home-credit]
---
# GCI World

Build evidence that the next model generalizes better, then submit the strongest reproducible candidate.

[Open comp.ipynb in Colab](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR) · [Competition folder in Drive](https://drive.google.com/drive/folders/106aE1f2Nz4pRZ1EGtryNFhgvo74451sC)

| Item | Current state |
| --- | --- |
| Stage | Baseline setup and verification |
| Public target | 0.79169, from the supplied screenshot |
| Deadline | Nov 19, 2026, 8 p.m. Toronto time |
| Final ranking | Last submitted file, evaluated on the full test set |
| Champion | Record a selected run in [[06 Decisions]] |
| Next action | Run B01, compare its models, then test F01 |

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

## Daily routine

Open today's Daily Note from the calendar icon. Choose one experiment from [[03 Experiments]]. Write the hypothesis before running it. Import the run report, then record the decision and next action. Keep public scores separate from local AUC.
''',
"01 Roadmap.md": '''---
type: roadmap
tags: [gci, planning]
---
# Roadmap

[[00 Dashboard]] · [[COMPETITION_PLAN]]

## Oct 9–11: reliable baselines

- [x] Inspect the rules, tutorial, data and public code.
- [x] Identify the Drive notebook and input/output folders.
- [x] Build an independent notebook and reproducible pipeline.
- [ ] Verify the Obsidian dashboard and report import.
- [ ] Complete B01 on all training rows with reserved audit data.
- [ ] Verify the matching CSV/code ZIP and compare model OOF results.

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
''',
"02 Runbook.md": '''---
type: runbook
tags: [gci, workflow]
---
# Runbook

[[00 Dashboard]] · [[03 Experiments]]

## First Colab run

1. Open [comp.ipynb](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR).
2. Select Runtime → Run all. Authorize your own Drive mount when prompted.
3. Keep B01's five-fold development configuration and the audit disabled.
4. Wait for each fold and the output checks. Save the notebook with Cmd+S / Ctrl+S before the ZIP cell.
5. Download `submission.csv` and `comp.zip`, or find them in the Competition Drive folder.

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

Change the label and add one feature group. Keep the same folds/seed for a controlled comparison. Groups: scores, affordability, tenure, missingness, peer, categories, logs. Enable Optuna only after a baseline, and compare tuned parameters with five-fold results. Record both rejected and accepted ideas in [[06 Decisions]].

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
''',
"03 Experiments.md": '''---
type: experiment_board
tags: [gci, experiments]
---
# Experiments

[[00 Dashboard]] · [[06 Decisions]]

## Ready

- [[Experiments/B01 All Features]]
- [[Experiments/F01 Credit Scores]]
- [[Experiments/F02 Affordability]]
- [[Experiments/F03 Missingness and Tenure]]
- [[Experiments/T01 Optuna]]
- [[Experiments/E01 Blend]]

## Run evidence

![[Experiments/Runs/Latest Run]]

Generated run notes live in Experiments/Runs. The machine-readable ledger is `output/experiments.csv`. Use the Experiment template for a new idea and record a decision after it finishes. Later candidates: peer features, another fold seed, XGBoost, and a bounded neural experiment.

## Comparison rules

Compare the same dataset bytes and fold assignments. Change one feature group or model setting at a time. Check pooled OOF AUC and fold changes. Confirm the strongest candidates using another seed. A small improvement selected from many trials may be noise.
''',
"04 Submission Tracker.md": '''---
type: submission_tracker
tags: [gci, submissions]
---
# Submission Tracker

[[00 Dashboard]] · [[06 Decisions]]

**Last uploaded file:** not recorded. **Public score:** not recorded.

| Toronto upload time | Run ID | CSV SHA-256 | Public AUC | Scoring complete? | Notes |
| --- | --- | --- | --- | --- | --- |

Record actual Omnicampus results here. Do not enter local AUC as a public score. The supplied leaderboard leader was 0.79169, and the existing tutorial submission's score is unknown.

- [ ] Selected CSV is the intended last upload.
- [ ] Matching comp.zip was uploaded for code disclosure.
- [ ] Scoring timestamp updated and result recorded.

Final deadline: November 19, 2026, 8 p.m. Toronto time. Keep time for a corrective upload.
''',
"05 Data and Rules.md": '''---
type: data_rules
tags: [gci, validation]
---
# Data and Rules

[[00 Dashboard]] · [[COMPETITION_PLAN]] · [[research/data_audit.json|Data audit]]

171,202 labeled rows. 61,500 test rows. 32 raw predictors. Default rate 8.0729%. ID is excluded from features. The input filenames and Drive file sizes match the local audit. Every training run verifies the exact input hashes before proceeding.

Use only provided train/test data. The supplied EXT_SOURCE fields are permitted. Do not download original Kaggle labels, relational tables or pretrained external weights. No manual conditional prediction overrides. Cite referenced code. Set seeds and preserve dependency versions.

Validation has a fixed 20% audit partition and five development folds. All learned categories, frequency/peer statistics and other fitted transformations stay within training folds. The audit is not used for tuning or early stopping. Final mode uses every labeled row after choices are frozen.

The default baseline retains all customers, preserves categories, and replaces the employment placeholder with missing plus an indicator. Ratio features guard zero denominators. Historical source filters are not copied blindly: test has an income of 117 million that an old filter would delete.

The metric is ROC-AUC. Public scoring uses a subset; the README says final scoring uses the entire test set and the last uploaded CSV. CSV schema is SK_ID_CURR,TARGET with no index column. Code disclosure must reproduce that same candidate.
''',
"06 Decisions.md": '''---
type: decision_log
tags: [gci, decisions]
---
# Decisions

[[00 Dashboard]] · [[03 Experiments]]

## Current champion

Not selected. Finish a complete baseline before choosing a champion.

## 2026-10-09: project setup

Use a new comp.ipynb, preserve the tutorial, and keep project notes beside the code as an Obsidian vault. Begin with LightGBM and CatBoost. Reserve audit data, save OOF evidence, and retain run artifacts. Feature groups and Optuna are supported but optional. Neural models require a later bounded experiment.

## Decision format

Date, candidate run, comparison run, controlled change, OOF/fold evidence, audit status, decision, reason, next experiment. Use the Decision template for an additional note and link it here.
''',
"Templates/Daily.md": '''---
type: daily_note
date: "{{date:YYYY-MM-DD}}"
tags: [gci, daily]
---
# {{date:YYYY-MM-DD}}

[[00 Dashboard]] · [[03 Experiments]]

## Today's experiment

Hypothesis:
Comparison run:

- [ ] Choose one controlled change.
- [ ] Run and import its report.
- [ ] Record keep/reject and why.

## Evidence and next action

Run:
Observation:
Next action:
''',
"Templates/Experiment.md": '''---
type: experiment
status: planned
created: "{{date:YYYY-MM-DD}}"
run_id:
oof_auc:
public_auc:
tags: [gci, experiment]
---
# {{title}}

[[00 Dashboard]] · [[03 Experiments]]

## Hypothesis

## Controlled change

Comparison run:
Feature group/model:
Fold seed:

## Acceptance evidence

Fixed-fold OOF and fold changes, then repeat-seed confirmation if promising. Audit remains disabled during exploration.

## Results and decision

Run report:
Decision:
Reason:
Next experiment:
''',
"Templates/Decision.md": '''---
type: decision
date: "{{date:YYYY-MM-DD}}"
tags: [gci, decision]
---
# {{title}}

[[06 Decisions]]

Candidate:
Comparison:
Evidence:
Audit status:
Decision:
Reason:
Next action:
''',
"Daily/2026-10-09.md": '''---
type: daily_note
date: 2026-10-09
tags: [gci, daily]
---
# October 9

[[00 Dashboard]]

- [x] Create the new notebook/pipeline and project tracking structure.
- [ ] Verify the complete baseline workflow.
- [ ] Review B01's model comparison, then choose the first feature ablation.

Next hypothesis: credit-score combinations may give trees easier access to joint score signal. Test them independently of affordability and missingness.
''',
"Experiments/Runs/Latest Run.md": '''<!-- Generated run summary placeholder. -->
# Latest completed run

No completed full-data baseline is recorded yet. The pipeline will update this note after a successful local run or report import.
''',
}
experiments = [
    ("B01 All Features", "Compare LightGBM and CatBoost using every supplied predictor with minimal cleaning.", "()"),
    ("F01 Credit Scores", "Available-score summaries and pairwise interactions improve joint score representation.", "('scores',)"),
    ("F02 Affordability", "Credit/annuity/income/goods and household ratios improve risk ranking.", "('affordability',)"),
    ("F03 Missingness and Tenure", "Missingness and cleaned employment/life-stage features add conditional signal.", "('missingness', 'tenure')"),
    ("T01 Optuna", "Tuning regularization and tree complexity improves a validated feature set.", "Keep the accepted groups fixed; enable RUN_OPTUNA."),
    ("E01 Blend", "A coarse blend improves a standalone champion through complementary errors.", "Use CHECK_BLEND and audit confirmation after freezing the proposed weights."),
]
for name, hypothesis, setting in experiments:
    notes[f"Experiments/{name}.md"] = f'''---
type: experiment
status: planned
created: 2026-10-09
tags: [gci, experiment]
---
# {name}

[[03 Experiments]] · [[06 Decisions]]

Hypothesis: {hypothesis}

Configuration: `{setting}`

Comparison: same input hashes, five folds, seed 42. Audit disabled during exploration.

- [ ] Run the controlled experiment.
- [ ] Link its generated report below.
- [ ] Record the decision and next step.

Report:
Decision:
'''
for name, text in notes.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text)
for folder in ("Inbox", "Attachments"):
    (root / folder).mkdir(exist_ok=True)
config_dir = root / ".obsidian"
config_dir.mkdir(exist_ok=True)
plugin_path = config_dir / 'core-plugins.json'
plugins = json.loads(plugin_path.read_text()) if plugin_path.exists() else {
    "file-explorer": True, "global-search": True, "switcher": True,
    "graph": True, "backlink": True, "canvas": True, "outgoing-link": True,
    "tag-pane": True, "properties": True, "page-preview": True,
    "command-palette": True, "editor-status": True, "outline": True,
    "word-count": True, "file-recovery": True, "sync": False, "publish": False,
}
plugins.update({"templates": True, "daily-notes": True, "bookmarks": True})
settings = {
    "core-plugins.json": plugins,
    "app.json": {"alwaysUpdateLinks": True, "newFileLocation": "folder", "newFileFolderPath": "Inbox", "attachmentFolderPath": "Attachments"},
    "templates.json": {"folder": "Templates", "dateFormat": "YYYY-MM-DD", "timeFormat": "HH:mm"},
    "daily-notes.json": {"folder": "Daily", "template": "Templates/Daily", "format": "YYYY-MM-DD", "autorun": False},
    "bookmarks.json": {"items": [{"type": "file", "ctime": 1791518400000, "path": "00 Dashboard.md"},
                                    {"type": "file", "ctime": 1791518400001, "path": "03 Experiments.md"},
                                    {"type": "file", "ctime": 1791518400002, "path": "04 Submission Tracker.md"}]},
}
for name, value in settings.items():
    path = config_dir / name
    if path.exists():
        current = json.loads(path.read_text())
        if name == "core-plugins.json":
            current.update({"templates": True, "daily-notes": True, "bookmarks": True})
        elif name == "app.json":
            for key, default in value.items():
                current.setdefault(key, default)
        else:
            continue
        value = current
    path.write_text(json.dumps(value, indent=2) + '\n')
print(f"Created {len(notes)} project notes, native templates/daily notes/bookmarks.")
