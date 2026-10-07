"""
Runner script to execute baseline models (Rolling Regression and PCMCI+).
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument(
        "--model",
        choices=["rolling_regression", "pcmci", "all"],
        default="all",
        help="Which baseline to run (default: all)",
    )
    args = parser.parse_args()

    baselines_dir = Path(__file__).resolve().parent.parent / "src" / "sca" / "baselines"

    if args.model in ("rolling_regression", "all"):
        print("=== Running Rolling Regression Baseline ===")
        import runpy
        runpy.run_path(str(baselines_dir / "rolling_regression_baseline.py"), run_name="__main__")

    if args.model in ("pcmci", "all"):
        print("\n=== Running PCMCI+ Baseline ===")
        import runpy
        runpy.run_path(str(baselines_dir / "pcmci_baseline.py"), run_name="__main__")

if __name__ == "__main__":
    main()
