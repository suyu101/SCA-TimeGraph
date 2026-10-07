"""Forwarding entrypoint for regime_ground_truth_tensor.py (moved to src/sca/ground_truth/)."""
import sys
import runpy
from pathlib import Path

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "src"))

from sca.ground_truth.regime_ground_truth_tensor import *  # noqa: F401, F403

if __name__ == "__main__":
    target = repo_root / "src" / "sca" / "ground_truth" / "regime_ground_truth_tensor.py"
    runpy.run_path(str(target), run_name="__main__")
