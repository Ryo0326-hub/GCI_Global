"""Validate a clean local/CI stack using the same checks as the Colab installer."""
from pathlib import Path
import ctypes
import json
import platform
import sys

root = Path(__file__).resolve().parents[1]
if platform.system() == "Darwin":
    try:
        import lightgbm
    except OSError as exc:
        existing = Path("/opt/anaconda3/lib/libomp.dylib")
        if "libomp" not in str(exc) or not existing.is_file():
            raise
        ctypes.CDLL(str(existing))
sys.path.insert(0, str(root / "src"))
from gci_environment import check_stack

pins = dict(line.split("==") for line in (root / "requirements.txt").read_text().splitlines() if line)
print(json.dumps(check_stack(pins), indent=2))
