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
default_path = ROOT / "configs/sprint/final.json"
if not default_path.exists():
    default_path = ROOT / "configs/sprint/catboost_gpu_affordability.json"
default_config = json.loads(default_path.read_text())
cells = []


def cell(kind, source):
    cell_id = uuid.uuid4().hex[:12]
    item = {"cell_type": kind, "id": cell_id, "metadata": {"id": cell_id}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """# GCI World: Home Credit submission workflow

This is the new competition notebook. The original tutorial is unchanged.

**Run:** use Run all and authorize your own Drive mount. The frozen CPU ensemble works without a GPU. If you deliberately configure a GPU experiment, choose a T4 in Runtime → Change runtime type first. Setup verifies the packages, then model steps run in fresh Python processes. The notebook session continues without a forced restart. The configuration below contains the current sprint candidate. Development runs reserve an audit set; a frozen final run uses all labelled rows.

Outputs in `MyDrive/GCI_Global/Competition`:
- `output/submission.csv`: the current format-checked candidate.
- `comp.zip`: saved notebook, executed pipeline, frozen configuration, dependencies, citations, and matching CSV.
- `output/runs/<run_id>/`: immutable earlier results, fold predictions and fitted models.
- `Obsidian Reports/Experiments/Runs/`: experiment notes and latest-run summary.

Local AUC is not the public leaderboard score. Upload to Omnicampus manually and record its score. The **last** submission counts. Only supplied input files are allowed.
""")
cell("markdown", "## 1. Install and verify the pinned environment\n\nKeep the tested NumPy 2.2.6 pin on Python 3.12–3.13. Model steps use fresh processes, without forcing a kernel restart. CatBoost GPU needs an assigned GPU and explicit GPU settings; LightGBM remains on CPU.\n")
cell("code", """# Setup and notebook controls use the Python standard library.
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
cell("code", """def pipeline_step(code, **payload):
    return gci_environment.run_step(runtime_dir, code, payload, pins=environment_result['packages'])

input_summary = pipeline_step('''
train, test, sample_sub, input_hashes = gci.load_inputs(payload['input_dir'])
result = {'train_shape': list(train.shape), 'test_shape': list(test.shape),
          'default_rate': float(train.TARGET.mean()), 'input_hashes': input_hashes}
''', input_dir=str(input_dir))
print('Train:', input_summary['train_shape'], 'Test:', input_summary['test_shape'])
print('Default rate:', f"{input_summary['default_rate']:.4%}")
print('Input bytes match the audited dataset.')
""")
cell("markdown", """## 4. Configure one experiment

The default is the saved sprint candidate. For a controlled feature experiment, use a new `label` and `mode='development'`. Available groups: `scores`, `affordability`, `tenure`, `missingness`, `peer`, `categories`, `logs`, `loan_terms`, `peer_profile`, `category_crosses`, `inquiry_patterns`, or `pricing_proxies`. GPU failures stop clearly. To choose CPU deliberately, set CatBoost `task_type='CPU'` and remove `devices`; use at most ten CPU tuning trials.

For an ensemble, `model_options` preserves each member's validated feature groups, iteration limits, early stopping and thread count. Fold and audit seeds stay shared. Keep `evaluate_audit=False` while experimenting. Use a second `seed` when confirming a candidate. Later, after freezing choices, switch to `mode='final'`, `frozen_choices=True`, and supply the selected `blend_weights`. Final mode includes all labeled rows and has no independent audit score.
""")
cell("code", "config = " + repr(default_config) + "\n\n" + """
RUN_OPTUNA = False
OPTUNA_MODEL = 'catboost'
OPTUNA_TRIALS = 20
OPTUNA_TIMEOUT_SECONDS = 7200
""")
cell("code", """backend_info = pipeline_step('''
import platform, subprocess
cfg = gci.Config(**payload['config'])
result = {'python': platform.python_version(), 'models': {}}
if 'catboost' in cfg.models:
    from catboost import CatBoostClassifier
    params = {**gci.model_defaults('catboost', cfg), **cfg.model_params.get('catboost', {})}
    params.update(iterations=5, depth=2)
    params.pop('ignored_features', None)  # The tiny backend probe has two synthetic columns.
    model = CatBoostClassifier(**params)
    model.fit([[float(i), float(i % 3)] for i in range(256)], [i % 2 for i in range(256)])
    backend = model.get_all_params()['task_type']
    if backend != params.get('task_type', 'CPU').upper():
        raise RuntimeError('Requested backend was not used; no automatic CPU fallback')
    result['models']['catboost'] = backend
    if backend == 'GPU':
        result['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                                        '--format=csv'], capture_output=True, text=True, check=True).stdout.strip()
if 'lightgbm' in cfg.models:
    result['models']['lightgbm'] = cfg.model_params.get('lightgbm', {}).get('device_type', 'cpu')
if 'xgboost' in cfg.models:
    import json
    from xgboost import XGBClassifier
    params = {**gci.model_defaults('xgboost', cfg), **cfg.model_params.get('xgboost', {})}
    params.update(n_estimators=5, max_depth=2)
    params.pop('early_stopping_rounds', None)
    model = XGBClassifier(**params)
    model.fit([[float(i), float(i % 3)] for i in range(256)], [i % 2 for i in range(256)])
    device = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if params.get('device', 'cpu').startswith('cuda') and not device.startswith('cuda'):
        raise RuntimeError('Requested XGBoost CUDA backend was not used; no automatic CPU fallback')
    result['models']['xgboost'] = device
gci.save_json(Path(payload['output_dir']) / 'backend_preflight.json', result)
''', config=config, output_dir=str(output_dir))
print('Backend check passed:', backend_info)
""")
cell("markdown", "## 5. Optional Optuna tuning\n\nSearch development rows only with three folds, then confirm on five folds. The trial count is additional trials when resuming. Keep one writer per study. SQLite runs on Colab local disk and is backed up consistently to Drive when the step ends, including interruption. A runtime disconnect can lose trials since the last backup. Do not enable tuning for a frozen final configuration.\n")
cell("code", """if RUN_OPTUNA:
    import shutil
    study_dir = runtime_dir / 'studies'
    study_backup = output_dir / 'studies'
    study_dir.mkdir(exist_ok=True)
    if not (study_dir / 'studies.db').exists() and (study_backup / 'studies.db').exists():
        for path in study_backup.iterdir():
            if path.is_file():
                shutil.copy2(path, study_dir / path.name)
    try:
        best_params = pipeline_step('''
result = gci.tune_model(payload['input_dir'], payload['study_dir'], gci.Config(**payload['config']),
                        model_name=payload['model_name'], n_trials=payload['n_trials'],
                        timeout=payload['timeout'])
''', input_dir=str(input_dir), study_dir=str(study_dir), config=config,
            model_name=OPTUNA_MODEL, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT_SECONDS)
    finally:
        gci_environment.backup_studies(study_dir, study_backup)
    config['model_params'][OPTUNA_MODEL] = {**config['model_params'].get(OPTUNA_MODEL, {}), **best_params}
    print('Parameters selected for confirmation:', best_params)
else:
    print('Optuna disabled: using the configured experiment.')
""")
cell("markdown", "## 6. Train, validate and produce submission.csv\n\nEach fold reports progress. Earlier runs are retained even though `output/submission.csv` points to the most recent completed candidate.\n")
cell("code", """manifest = pipeline_step('''
result = gci.run_experiment(payload['input_dir'], payload['output_dir'],
                            gci.Config(**payload['config']), notes_dir=payload['notes_dir'])
''', input_dir=str(input_dir), output_dir=str(output_dir), notes_dir=str(notes_dir), config=config)
submission_path = output_dir / 'submission.csv'
import csv
from itertools import islice
with submission_path.open(newline='') as stream:
    for row in islice(csv.DictReader(stream), 5):
        print(row)
print('Selected model:', manifest['selected_name'])
print('Local OOF AUC:', manifest['selected_oof_auc'])
print('Submission rows:', manifest['test_rows'])
print('SHA-256:', manifest['submission_sha256'])
""")
cell("markdown", "## 7. Optional blend diagnostic\n\nA run uses explicit frozen blend weights when supplied; otherwise it selects a standalone model. This optional development-only cell proposes coarse blend weights without changing the submission. Review the meta-validation results and confirm any blend on a frozen audit run.\n")
cell("code", """CHECK_BLEND = False
if CHECK_BLEND:
    blend_result = pipeline_step('''
weights, table = gci.select_blend(payload['run_dir'], payload['seed'])
result = {'weights': weights, 'table': table.to_dict(orient='records')}
''', run_dir=manifest['run_dir'], seed=config['seed'])
    for row in blend_result['table']:
        print(row)
    print('Proposed weights:', blend_result['weights'])
    print("To test them, set config['blend_weights'] and use a new experiment label.")
""")
cell("markdown", """## 8. Save the notebook and package the matching code ZIP

Press **Ctrl+S / Cmd+S** in Colab before this cell. The ZIP checks that the saved notebook contains the executed pipeline. It includes the frozen run configuration and matching CSV. Input datasets and fitted-model files are omitted from the ZIP; the included command regenerates the submission from the allowed inputs.
""")
cell("code", "(current_dir / 'CITATIONS.md').write_text(" + repr(citations) + ")\n" + """bundle_result = pipeline_step('''
verification = gci.verify_saved_predictions(payload['input_dir'], payload['manifest']['run_dir'])
bundle = gci.build_bundle(payload['project_dir'], payload['manifest'], notebook_path=payload['notebook_path'])
result = {'path': str(bundle), 'verification': verification}
''', project_dir=str(current_dir), input_dir=str(input_dir), manifest=manifest, notebook_path=str(current_dir / 'comp.ipynb'))
zip_path = Path(bundle_result['path'])
print('Saved-model verification:', bundle_result['verification'])
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
if (default_config.get('model_params', {}).get('catboost', {}).get('task_type') == 'GPU'
        or default_config.get('model_params', {}).get('xgboost', {}).get('device', 'cpu').startswith('cuda')):
    notebook['metadata']['accelerator'] = 'GPU'
    notebook['metadata']['colab']['gpuType'] = 'T4'
backup = ROOT / "research/comp_original.ipynb"
if not backup.exists() and (ROOT / "comp.ipynb").exists():
    shutil.copy2(ROOT / "comp.ipynb", backup)
(ROOT / "comp.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Built comp.ipynb: {len(cells)} cells, self-contained pipeline and bibliography")
