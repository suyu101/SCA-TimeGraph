"""Forwarding entrypoint for generate_regime_change.py (moved to scripts/)."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).parent / "scripts" / "generate_regime_change.py"
    runpy.run_path(str(target), run_name="__main__")
