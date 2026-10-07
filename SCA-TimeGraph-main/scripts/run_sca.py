"""
Runner script to execute SCA models (Full SCA, No Persistence, Global Only).
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="Run SCA model variants")
    parser.add_argument(
        "--variant",
        choices=["full", "no_persistence", "global_only", "all"],
        default="all",
        help="Which SCA model variant to run (default: all)",
    )
    args = parser.parse_args()

    models_dir = Path(__file__).resolve().parent.parent / "src" / "sca" / "models"

    if args.variant in ("full", "all"):
        print("=== Running Full SCA Model ===")
        import runpy
        runpy.run_path(str(models_dir / "sca_model.py"), run_name="__main__")

    if args.variant in ("no_persistence", "all"):
        print("\n=== Running SCA No Persistence Ablation ===")
        import runpy
        runpy.run_path(str(models_dir / "sca_model_no_persistence.py"), run_name="__main__")

    if args.variant in ("global_only", "all"):
        print("\n=== Running SCA Global Only Ablation ===")
        import runpy
        runpy.run_path(str(models_dir / "sca_model_global_only.py"), run_name="__main__")

if __name__ == "__main__":
    main()
