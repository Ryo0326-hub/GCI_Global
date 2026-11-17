"""Protect backend selection and audit isolation across the real Optuna loop."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gci_pipeline as gci


@pytest.fixture
def tuning_data(monkeypatch):
    train = pd.DataFrame({"SK_ID_CURR": range(120), "TARGET": [0, 0, 0, 1] * 30,
                          "DAYS_EMPLOYED": range(-120, 0)})
    monkeypatch.setattr(gci, "load_inputs", lambda *a: (train, None, None, {"fixture": "test-only"}))
    return train


@pytest.mark.parametrize("model,fixed", [
    ("catboost", {"task_type": "GPU", "devices": "0", "depth": 2}),
    ("catboost", {"depth": 2}),
    ("lightgbm", {"device_type": "cpu", "max_bin": 127, "subsample_freq": 0}),
])
def test_tuning_preserves_fixed_backend_through_confirmation(tmp_path, monkeypatch, tuning_data, model, fixed):
    cfg = gci.Config(models=(model,), model_params={model: deepcopy(fixed)})
    original = asdict(cfg)
    seen = []
    _, audit = gci.development_split(tuning_data, cfg)

    def capture_fit(x, y, test_x, ids, trial_cfg, **kwargs):
        seen.append(asdict(trial_cfg))
        assert set(ids).isdisjoint(set(tuning_data.SK_ID_CURR.iloc[audit]))
        assert len(ids) == 96 and trial_cfg.n_splits == 3
        assert test_x is None and kwargs.get("audit_x") is None
        return {"metrics": {model: {"oof_auc": 0.73}}}

    monkeypatch.setattr(gci, "fit_cv", capture_fit)
    best = gci.tune_model("unused", tmp_path, cfg, model_name=model, n_trials=1)
    # This is the notebook's public parameter-application contract.
    confirmation = asdict(cfg)
    confirmation["model_params"][model] = best
    confirmed = gci.Config(**confirmation)
    for params in (seen[0]["model_params"][model], best, confirmed.model_params[model]):
        if model == "catboost":
            assert 4 <= params["depth"] <= 8  # sampled keys override fixed search keys
            assert params.get("task_type") == fixed.get("task_type")
            assert params.get("devices") == fixed.get("devices")
            assert "learning_rate" in params and "l2_leaf_reg" in params
        else:
            assert params["device_type"] == "cpu" and params["max_bin"] == 127
            assert params["subsample_freq"] == 1
    assert asdict(cfg) == original  # caller remains reusable


def test_study_identity_separates_executed_source(tmp_path, monkeypatch, tuning_data):
    monkeypatch.setattr(gci, "fit_cv", lambda *a, **k: {"metrics": {"catboost": {"oof_auc": 0.73}}})
    cfg = gci.Config(models=("catboost",))
    for source_hash in ("source-v1", "source-v2"):
        monkeypatch.setattr(gci, "sha256", lambda p, value=source_hash: value)
        gci.tune_model("unused", tmp_path, cfg, model_name="catboost", n_trials=1)
    identities = [json.loads(p.read_text()) for p in tmp_path.glob("*_identity.json")]
    assert len(identities) == 2
    assert {v["pipeline_sha256"] for v in identities} == {"source-v1", "source-v2"}


def test_requested_gpu_failure_is_not_retried_on_cpu(tmp_path, monkeypatch, tuning_data):
    seen = []
    def unavailable(x, y, test_x, ids, cfg, **kwargs):
        seen.append(cfg.model_params["catboost"])
        raise RuntimeError("CUDA device unavailable")
    monkeypatch.setattr(gci, "fit_cv", unavailable)
    cfg = gci.Config(models=("catboost",), model_params={"catboost": {"task_type": "GPU", "devices": "0"}})
    with pytest.raises(RuntimeError, match="CUDA device unavailable"):
        gci.tune_model("unused", tmp_path, cfg, "catboost", n_trials=1)
    assert len(seen) == 1 and seen[0]["task_type"] == "GPU"
    assert not list(tmp_path.glob("*_best.json"))
