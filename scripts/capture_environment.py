"""Write a machine-readable reproducibility manifest for a completed run."""

from __future__ import annotations

import importlib.metadata
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = ("numpy", "pandas", "scipy", "matplotlib", "tigramite", "causalchamber", "joblib")


def main():
    versions = {}
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
            if not version:
                try:
                    module = importlib.import_module(package)
                    version = getattr(module, "__version__", "unknown")
                except Exception:
                    version = "unknown"
            versions[package] = version
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "processor": platform.processor() or "not reported",
        "packages": versions,
        "commands": [
            "python scripts/run_validation.py --study all --seeds 20",
            "python scripts/evaluate_causal_chambers.py",
            "python scripts/tune_causal_chambers.py",
        ],
    }
    target = Path("results") / "reproducibility_manifest.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
