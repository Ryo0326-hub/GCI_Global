"""Exercise real saved-model inference and code/package integrity."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gci_pipeline as gci


def test_saved_models_reproduce_and_tampering_is_rejected(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    train = pd.DataFrame({"SK_ID_CURR": range(120), "TARGET": [0, 0, 0, 1] * 30,
                          "DAYS_EMPLOYED": range(-120, 0), "NAME_CONTRACT_TYPE": ["A", "B"] * 60})
    test = pd.DataFrame({"SK_ID_CURR": [200, 201, 202], "DAYS_EMPLOYED": [-20,-10,-30],
                         "NAME_CONTRACT_TYPE": ["A", "UNSEEN", None]})
    sample = pd.DataFrame({"SK_ID_CURR": [202, 200, 201], "TARGET": [.5] * 3})
    for name, frame in (("train", train), ("test", test), ("sample_submission", sample)):
        frame.to_csv(inputs / f"{name}.csv", index=False)
    cfg = gci.Config(models=("catboost",), n_splits=3, max_iterations=5,
                     early_stopping=3, threads=1, verify_input_hashes=False)
    manifest = gci.run_experiment(inputs, tmp_path / "output", cfg)
    run_dir = Path(manifest["run_dir"])
    report = gci.verify_saved_predictions(inputs, run_dir)
    assert report["rows"] == 3 and report["max_absolute_error"] < 1e-12
    assert report["submission_sha256"] == manifest["submission_sha256"]
    assert all(m["backend"] == "CPU" for m in manifest["model_metrics"]["catboost"]["folds"])
    assert (run_dir / "gci_pipeline.py").read_bytes() == Path(gci.__file__).read_bytes()

    # Source/config drift and CSV corruption must not produce a plausible new bundle.
    notebook = tmp_path / "comp.ipynb"
    notebook.write_text(json.dumps({"cells": [{"source": "%%writefile gci_pipeline.py\n" + Path(gci.__file__).read_text()}]}))
    (tmp_path / "CITATIONS.md").write_text("Test fixture; no competition result.\n")
    bundle = gci.build_bundle(tmp_path, manifest, notebook)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert archive.read("submission.csv") == (run_dir / "submission.csv").read_bytes()
    frozen_config = run_dir / "reproduce_config.json"
    original_config = frozen_config.read_bytes()
    changed_config = json.loads(original_config)
    changed_config["seed"] = 9001
    frozen_config.write_text(json.dumps(changed_config))
    with pytest.raises(ValueError, match="artifact"):
        gci.build_bundle(tmp_path, manifest, notebook)
    frozen_config.write_bytes(original_config)
    model_file = run_dir / "catboost_fold_0.joblib"
    model_file.write_bytes(model_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact"):
        gci.verify_saved_predictions(inputs, run_dir)
