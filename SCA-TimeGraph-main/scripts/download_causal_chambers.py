"""Download and inventory the selected official Causal Chambers benchmark."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path


DATASET = "wt_walks_v1"
# Version 1 currently contains only 1,016 observations. Version 2 contains
# the 10,000-row random-walk design required by the locked protocol.
EXPERIMENT = "actuators_random_walk_2"
ROOT = Path("Datasets") / "causal_chambers"


def main():
    try:
        from causalchamber.datasets import Dataset
    except ImportError as error:
        raise SystemExit("Install the optional benchmark client first: pip install causalchamber==0.2.8") from error
    # The upstream client does not create its target directory before opening
    # the download archive on Windows.
    ROOT.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(name=DATASET, root=str(ROOT), download=True)
    experiment = dataset.get_experiment(name=EXPERIMENT)
    frame = experiment.as_pandas_dataframe()
    if frame.empty:
        raise RuntimeError("Official experiment downloaded but contains no observations")
    timestamp_candidates = [column for column in frame.columns if column.lower() in {"time", "timestamp", "t"}]
    timestamp = timestamp_candidates[0] if timestamp_candidates else None
    median_interval = None
    if timestamp is not None:
        differences = frame[timestamp].diff().dropna()
        if len(differences):
            median_interval = float(differences.median())
    manifest = {
        "dataset": DATASET,
        "experiment": EXPERIMENT,
        "client_version": importlib.metadata.version("causalchamber"),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "timestamp_column": timestamp,
        "median_sampling_interval": median_interval,
        "selection": "Use all observations if exactly 10,000; otherwise use the first 10,000 chronological observations.",
        "source": "https://github.com/juangamella/causal-chamber",
        "licence": "CC BY 4.0",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
