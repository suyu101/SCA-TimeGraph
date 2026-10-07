"""Loader for the small TimeGraph example without import-time file access."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("Datasets/A1/Gaussian/4 variable/Lag 2/linear_ts_n500_vars4_lag2.csv")
VARIABLES = ("X1", "X2", "X3", "X4")
TRUE_LINKS = {
    ("X1", -2, "X4"): 0.25,
    ("X4", 0, "X3"): 0.35,
    ("X3", -1, "X2"): 0.30,
    ("X2", 0, "X1"): 0.40,
}


def build_ground_truth(variables=VARIABLES, max_lag=2):
    """Return the example's ``source, target, lag`` ground-truth tensor."""
    names = tuple(variables)
    index = {name: position for position, name in enumerate(names)}
    tensor = np.zeros((len(names), len(names), max_lag + 1), dtype=np.float32)
    for (source, signed_lag, target), weight in TRUE_LINKS.items():
        lag = abs(signed_lag)
        if source not in index or target not in index or lag > max_lag:
            continue
        tensor[index[source], index[target], lag] = weight
    return tensor


def load_timegraph_example(path=DATA_PATH, variables=VARIABLES, max_lag=2):
    """Load observations and ground truth on demand.

    The previous module read a project-relative CSV merely by being imported,
    which broke normal package imports outside one working directory.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TimeGraph example dataset not found: {path}")
    frame = pd.read_csv(path)
    required = [*variables, "time"]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"TimeGraph example is missing columns: {missing}")
    return {
        "X": frame.loc[:, variables].to_numpy(dtype=np.float32),
        "time": frame["time"].to_numpy(),
        "A_true": build_ground_truth(variables, max_lag),
        "variables": tuple(variables),
        "true_links": dict(TRUE_LINKS),
    }


if __name__ == "__main__":
    dataset = load_timegraph_example()
    print("X shape:", dataset["X"].shape)
    print("A_true shape:", dataset["A_true"].shape)
