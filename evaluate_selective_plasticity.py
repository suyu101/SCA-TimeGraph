"""Forwarding entrypoint for evaluate_selective_plasticity.py (moved to scripts/)."""
import sys
import runpy
from pathlib import Path

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "src"))

if __name__ == "__main__":
    target = repo_root / "scripts" / "evaluate_selective_plasticity.py"
    runpy.run_path(str(target), run_name="__main__")
