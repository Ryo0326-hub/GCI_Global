"""Local CLI wrapper: reuse the Mac's existing OpenMP library if needed."""
from pathlib import Path
import ctypes
import platform
import sys

if platform.system() == "Darwin":
    try:
        import lightgbm
    except OSError as exc:
        existing = Path("/opt/anaconda3/lib/libomp.dylib")
        if "libomp" not in str(exc) or not existing.is_file():
            raise
        ctypes.CDLL(str(existing))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gci_pipeline import main

if __name__ == "__main__":
    main()
