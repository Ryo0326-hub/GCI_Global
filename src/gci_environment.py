"""Stdlib-only setup and fresh-process execution for the competition pipeline."""
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import uuid


def validate_python(pins, python_version):
    minor = tuple(python_version[:2])
    if not (3, 10) <= minor < (3, 14):
        raise RuntimeError("These dependency pins support Python 3.10–3.13. Select a supported Colab runtime.")
    if minor >= (3, 13) and int(pins["numpy"].split(".")[0]) < 2:
        raise RuntimeError("NumPy 1.26 does not support Python 3.13. Keep the tested NumPy 2.2.6 pin.")


def installed_versions(pins):
    values = {}
    for package in pins:
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            values[package] = None
    return values


def check_stack(pins):
    """Exercise the exact failing import path plus both native model libraries."""
    validate_python(pins, sys.version_info)
    versions = installed_versions(pins)
    if versions != pins:
        raise RuntimeError(f"Unexpected installed versions: {versions}; expected {pins}")
    import numpy as np
    import numpy.strings
    import pandas as pd
    import scipy.sparse
    from sklearn.metrics import roc_auc_score
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    if np.__version__ != pins["numpy"]:
        raise RuntimeError("Loaded NumPy differs from installed NumPy. Restart the Colab session.")
    assert np.strings.str_len(np.array(["gci"])).tolist() == [3]
    x = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) % 3})
    y = np.arange(20) % 2
    assert scipy.sparse.csr_matrix(x).shape == (20, 2)
    models = [LGBMClassifier(n_estimators=3, num_leaves=3, min_child_samples=1, n_jobs=1, verbosity=-1),
              CatBoostClassifier(iterations=3, depth=2, thread_count=1, verbose=False,
                                 allow_writing_files=False, random_seed=42)]
    for model in models:
        model.fit(x, y)
        prediction = model.predict_proba(x)[:, 1]
        assert np.isfinite(prediction).all()
        assert 0 <= roc_auc_score(y, prediction) <= 1
    return {"python": sys.version.split()[0], "packages": versions,
            "checks": "NumPy strings, SciPy sparse, sklearn, LightGBM and CatBoost tiny fits passed"}


def prepare_colab(pins):
    import google.colab  # Keep installation scoped to the requested Colab environment.
    validate_python(pins, sys.version_info)
    before = installed_versions(pins)
    if before != pins:
        print("Installing the pinned competition packages...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "--disable-pip-version-check", "--only-binary=:all:",
                               *[f"{name}=={value}" for name, value in pins.items()]])
    # Test the disk installation in a fresh process, without Colab's cached modules.
    script = ("import json,sys; sys.path.insert(0,sys.argv[1]); "
              "from gci_environment import check_stack; "
              "print(json.dumps(check_stack(json.loads(sys.argv[2]))))")
    checked = subprocess.run([sys.executable, "-c", script, str(Path(__file__).parent), json.dumps(pins)],
                             capture_output=True, text=True)
    if checked.returncode:
        raise RuntimeError("Fresh-process environment check failed. Use Runtime → Disconnect and delete runtime, "
                           "then run this notebook again.\n" + checked.stderr[-4000:])
    print(checked.stdout.strip(), flush=True)
    result = json.loads(checked.stdout.strip().splitlines()[-1])
    result["execution_mode"] = "fresh_process"
    print("Environment ready. Run all continues; model steps use fresh Python processes.", flush=True)
    return result


def run_step(runtime_dir, code, payload, pins):
    """Stream a pipeline step from a fresh interpreter without importing models in the notebook kernel."""
    runtime_dir = Path(runtime_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    task_id = uuid.uuid4().hex
    request = runtime_dir / f".gci_step_{task_id}.json"
    response = runtime_dir / f".gci_result_{task_id}.json"
    request.write_text(json.dumps({"payload": payload, "pins": pins}, allow_nan=False))
    bootstrap = """import json, sys
from importlib import metadata
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
request = json.loads(Path(sys.argv[1]).read_text())
actual = {name: metadata.version(name) for name in request['pins']}
if actual != request['pins']:
    raise RuntimeError(f'Worker package versions changed: {actual}')
import gci_pipeline as gci
payload = request['payload']
result = None
"""
    script = bootstrap + code + "\nPath(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))\n"
    process = None
    try:
        process = subprocess.Popen([sys.executable, "-u", "-c", script, str(request), str(response)],
                                   cwd=runtime_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError("Competition step failed. The worker's error is printed above.")
        return json.loads(response.read_text())
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        raise
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        request.unlink(missing_ok=True)
        response.unlink(missing_ok=True)
