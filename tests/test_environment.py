from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gci_environment import fingerprint, restart_required, validate_python

PINS = {"numpy": "2.2.6", "scikit-learn": "1.6.1"}


def test_restart_once_and_again_when_packages_or_loaded_modules_change():
    marker = {"fingerprint": fingerprint(PINS, (3, 13)), "install_pid": 100}
    assert restart_required(PINS, PINS, {}, (3, 13), 100, {})
    assert restart_required(PINS, PINS, marker, (3, 13), 100, {})
    assert not restart_required(PINS, PINS, marker, (3, 13), 200, PINS)
    assert restart_required(PINS, {**PINS, "numpy": "2.1.3"}, marker, (3, 13), 200, {})
    assert restart_required(PINS, PINS, marker, (3, 13), 200, {"numpy": "2.1.3"})


def test_reject_numpy_1_on_python_313_before_installing():
    with pytest.raises(RuntimeError, match="does not support Python 3.13"):
        validate_python({"numpy": "1.26.4"}, (3, 13))
    validate_python(PINS, (3, 13))
    with pytest.raises(RuntimeError, match="3.10–3.13"):
        validate_python(PINS, (3, 14))
