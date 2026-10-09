from pathlib import Path
import os
import sys
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gci_environment
from gci_environment import run_step, validate_python

PINS = {"numpy": "2.2.6", "scikit-learn": "1.6.1"}


def test_pipeline_step_uses_fresh_module_and_leaves_parent_alive(tmp_path, monkeypatch, capsys):
    (tmp_path / "gci_pipeline.py").write_text("value = 'from disk'\n")
    stale = ModuleType("gci_pipeline")
    stale.value = "stale cached module"
    monkeypatch.setitem(sys.modules, "gci_pipeline", stale)
    parent_pid = os.getpid()
    result = run_step(tmp_path, "import os\nprint('worker progress', flush=True)\n"
                      "result = {'value': gci.value, 'pid': os.getpid(), 'input': payload['number']}\n",
                      {"number": 7}, PINS)
    assert result["value"] == "from disk"
    assert result["pid"] != parent_pid == os.getpid()
    assert result["input"] == 7
    assert sys.modules["gci_pipeline"].value == "stale cached module"
    assert "worker progress" in capsys.readouterr().out
    assert not list(tmp_path.glob(".gci_*.json"))


def test_worker_failure_is_reported_and_temporary_files_are_removed(tmp_path, capsys):
    (tmp_path / "gci_pipeline.py").write_text("value = 1\n")
    with pytest.raises(RuntimeError, match="worker's error"):
        run_step(tmp_path, "raise ValueError('test worker failure')\n", {}, PINS)
    assert "test worker failure" in capsys.readouterr().out
    assert not list(tmp_path.glob(".gci_*.json"))


def test_colab_setup_checks_fresh_process_without_using_cached_numpy(monkeypatch):
    colab = ModuleType("google.colab")
    google = ModuleType("google")
    google.__path__ = []
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    stale = ModuleType("numpy")
    stale.__version__ = "stale"
    monkeypatch.setitem(sys.modules, "numpy", stale)
    pins = dict(line.split("==") for line in
                (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().splitlines() if line)
    # macOS test workers use the existing OpenMP library; Linux/Colab needs no override.
    if sys.platform == "darwin":
        monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/opt/anaconda3/lib/libomp.dylib")
    parent_pid = os.getpid()
    result = gci_environment.prepare_colab(pins)
    assert result["packages"] == pins
    assert result["execution_mode"] == "fresh_process"
    assert os.getpid() == parent_pid
    assert sys.modules["numpy"].__version__ == "stale"


def test_reject_numpy_1_on_python_313_before_installing():
    with pytest.raises(RuntimeError, match="does not support Python 3.13"):
        validate_python({"numpy": "1.26.4"}, (3, 13))
    validate_python(PINS, (3, 13))
    with pytest.raises(RuntimeError, match="3.10–3.13"):
        validate_python(PINS, (3, 14))
