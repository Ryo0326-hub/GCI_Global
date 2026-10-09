"""Build the self-contained Colab interface from the tested pipeline."""
from pathlib import Path
import hashlib
import json
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1]
pipeline = (ROOT / "src/gci_pipeline.py").read_text()
requirements = (ROOT / "requirements.txt").read_text()
citations = (ROOT / "CITATIONS.md").read_text()
environment = (ROOT / "src/gci_environment.py").read_text()
pins = dict(line.split("==") for line in requirements.splitlines() if line)
cells = []


def cell(kind, source):
    cell_id = uuid.uuid4().hex[:12]
    item = {"cell_type": kind, "id": cell_id, "metadata": {"id": cell_id}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """# GCI World: Home Credit submission workflow

This is the new competition notebook. The original tutorial is unchanged.

**First run:** choose Runtime → Run all. Setup restarts the session once to clear cached packages. After Colab reconnects, choose Run all again and authorize your own Drive mount. The default compares LightGBM and CatBoost on all 32 supplied predictors and keeps a separate audit set out of model selection.

Outputs in `MyDrive/GCI_Global/Competition`:
- `output/submission.csv`: the current format-checked candidate.
- `comp.zip`: saved notebook, executed pipeline, frozen configuration, dependencies, citations, and matching CSV.
- `output/runs/<run_id>/`: immutable earlier results, fold predictions and fitted models.
- `Obsidian Reports/Experiments/Runs/`: experiment notes and latest-run summary.

Local AUC is not the public leaderboard score. Upload to Omnicampus manually and record its score. The **last** submission counts. Only supplied input files are allowed.
""")
cell("markdown", "## 1. Install and verify the pinned environment\n\nPython 3.13 needs NumPy 2.x. The tested NumPy 2.2.6 pin remains in place. Setup tests imports and both models in a fresh process, then restarts once to clear Colab's cached packages. After reconnecting, choose Run all again. A matching fresh session continues without another restart. CPU is suitable for these tree models.\n")
cell("code", """# Keep this setup cell before importing NumPy, pandas, SciPy or sklearn.
from pathlib import Path
import importlib
import sys

environment_file = Path('/content/gci_environment.py')
environment_file.write_text(""" + repr(environment) + """ )
sys.path.insert(0, '/content')
import gci_environment
gci_environment = importlib.reload(gci_environment)
environment_result = gci_environment.prepare_colab(""" + repr(pins) + ")\n")
cell("markdown", "## 2. Mount Drive and use the tutorial's folder convention\n\nThe folder below was verified in your Drive. Change it only if you move the project.\n")
cell("code", """from google.colab import drive
drive.mount('/content/drive')

%cd "/content/drive/MyDrive/GCI_Global/Competition"

import os
from pathlib import Path
import shutil

current_dir = Path(os.getcwd())
train_file = current_dir / "input" / "train.csv"
test_file = current_dir / "input" / "test.csv"
sample_sub_file = current_dir / "input" / "sample_submission.csv"
for path in (train_file, test_file, sample_sub_file):
    if not path.is_file():
        raise FileNotFoundError(path)

# Work on Colab's local disk; preserve all outputs in Drive.
runtime_dir = Path('/content/gci_work')
runtime_dir.mkdir(parents=True, exist_ok=True)
input_dir = runtime_dir / 'input'
input_dir.mkdir(exist_ok=True)
for path in (train_file, test_file, sample_sub_file):
    shutil.copy2(path, input_dir / path.name)
output_dir = current_dir / 'output'
output_dir.mkdir(exist_ok=True)
notes_dir = current_dir / 'Obsidian Reports' / 'Experiments' / 'Runs'
os.chdir(runtime_dir)
print('Competition folder:', current_dir)
print('CSV output:', output_dir / 'submission.csv')
print('Code ZIP:', current_dir / 'comp.zip')
""")
cell("markdown", "## 3. The reproducible pipeline\n\nThis cell writes the complete training/export implementation. It is included in the ZIP so reproduction does not depend on an unsaved notebook. References appear in the final bibliography.\n")
cell("code", "%%writefile gci_pipeline.py\n" + pipeline)
cell("code", """import importlib
import sys
sys.path.insert(0, str(runtime_dir))
import gci_pipeline as gci
gci = importlib.reload(gci)

train, test, sample_sub, input_hashes = gci.load_inputs(input_dir)
print('Train:', train.shape, 'Test:', test.shape)
print('Default rate:', f'{train.TARGET.mean():.4%}')
print('Input bytes match the audited dataset.')
""")
cell("markdown", """## 4. Configure one experiment

Start with B01. For a controlled feature experiment, change `label` and add one feature group: `scores`, `affordability`, `tenure`, `missingness`, `peer`, `categories`, or `logs`.

Keep `evaluate_audit=False` while experimenting. Use a second `seed` when confirming a candidate. Later, after freezing choices, switch to `mode='final'`, `frozen_choices=True`, and supply the selected `blend_weights`. Final mode includes all labeled rows and has no independent audit score.
""")
cell("code", """config = gci.Config(
    label='B01_all_features',
    mode='development',
    models=('lightgbm', 'catboost'),
    feature_groups=(),
    seed=42,
    n_splits=5,
    max_iterations=2000,
    early_stopping=150,
    threads=4,
    evaluate_audit=False,
    frozen_choices=False,
    blend_weights=None,  # Default picks the stronger standalone model by development OOF AUC.
    model_params={},
)

RUN_OPTUNA = False
OPTUNA_MODEL = 'lightgbm'
OPTUNA_TRIALS = 20
OPTUNA_TIMEOUT_SECONDS = 3600
""")
cell("markdown", "## 5. Optional Optuna tuning\n\nDisabled for the first baseline. Searches only development rows, persists its study, and uses three folds. Confirm the selected parameters with the five-fold run below. The trial count is additional trials when resuming.\n")
cell("code", """if RUN_OPTUNA:
    best_params = gci.tune_model(input_dir, output_dir / 'studies', config,
                                 model_name=OPTUNA_MODEL, n_trials=OPTUNA_TRIALS,
                                 timeout=OPTUNA_TIMEOUT_SECONDS)
    config.model_params[OPTUNA_MODEL] = best_params
    print('Parameters selected for confirmation:', best_params)
else:
    print('Optuna disabled: running the baseline configuration.')
""")
cell("markdown", "## 6. Train, validate and produce submission.csv\n\nEach fold reports progress. Earlier runs are retained even though `output/submission.csv` points to the most recent completed candidate.\n")
cell("code", """manifest = gci.run_experiment(input_dir, output_dir, config, notes_dir=notes_dir)
submission_path = output_dir / 'submission.csv'
submission = gci.pd.read_csv(submission_path)
gci.validate_submission(submission, sample_sub)
display(submission.head())
print('Selected model:', manifest['selected_name'])
print('Local OOF AUC:', manifest['selected_oof_auc'])
print('Submission rows:', len(submission))
print('SHA-256:', manifest['submission_sha256'])
""")
cell("markdown", "## 7. Optional blend diagnostic\n\nThe run initially selects a standalone model. This cell proposes coarse blend weights without changing its submission. Review the meta-validation results and confirm any blend on a frozen audit run.\n")
cell("code", """CHECK_BLEND = False
if CHECK_BLEND:
    proposed_weights, blend_table = gci.select_blend(manifest['run_dir'], config.seed)
    display(blend_table)
    print('Proposed weights:', proposed_weights)
    print('To test them, set config.blend_weights and use a new experiment label.')
""")
cell("markdown", """## 8. Save the notebook and package the matching code ZIP

Press **Ctrl+S / Cmd+S** in Colab before this cell. The ZIP checks that the saved notebook contains the executed pipeline. It includes the frozen run configuration and matching CSV. Input datasets and fitted-model files are omitted from the ZIP; the included command regenerates the submission from the allowed inputs.
""")
cell("code", "(current_dir / 'CITATIONS.md').write_text(" + repr(citations) + ")\n" + """zip_path = gci.build_bundle(current_dir, manifest, notebook_path=current_dir / 'comp.ipynb')
print('Verified code ZIP:', zip_path)
print('Verified submission CSV:', submission_path)

# Keep tutorial's zipfile approach, with a separate report archive for Obsidian.
import zipfile
report_zip = current_dir / 'obsidian_reports.zip'
report_root = current_dir / 'Obsidian Reports'
with zipfile.ZipFile(report_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in report_root.rglob('*.md'):
        archive.write(path, arcname=path.relative_to(report_root))
print('Obsidian reports:', report_zip)
""")
cell("markdown", """## 9. Download or open files in Drive

Upload `submission.csv` to Omnicampus and the matching `comp.zip` through the competition code-upload flow. No upload is automated. Record the returned public score and timestamp in Obsidian's Submission Tracker.

Download `obsidian_reports.zip` after a Colab run, then import it into the local vault using the Runbook. Drive report export does not automatically synchronize your Mac vault.
""")
cell("code", """DOWNLOAD_FILES = False  # Set True to download the three completed artifacts.
if DOWNLOAD_FILES:
    from google.colab import files
    files.download(str(submission_path))
    files.download(str(zip_path))
    files.download(str(report_zip))
""")
cell("markdown", "## References\n\n" + citations)

notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"name": "comp.ipynb", "provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"},
                         "gci": {"pipeline_version": "gci-setup-v1", "pipeline_sha256": hashlib.sha256(pipeline.encode()).hexdigest(),
                                 "environment_sha256": hashlib.sha256(environment.encode()).hexdigest()}},
            "cells": cells}
backup = ROOT / "research/comp_original.ipynb"
if not backup.exists() and (ROOT / "comp.ipynb").exists():
    shutil.copy2(ROOT / "comp.ipynb", backup)
(ROOT / "comp.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Built comp.ipynb: {len(cells)} cells, self-contained pipeline and bibliography")
