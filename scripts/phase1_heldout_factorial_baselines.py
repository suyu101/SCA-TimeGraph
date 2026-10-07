"""Held-out full-factorial comparison of SCA and matched streaming baselines.

The script accepts only seeds 20--39.  Every condition is checkpointed after
all methods finish, so an interruption cannot silently mix seed partitions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.baselines.streaming_linear import GlobalEMAConfig, GlobalRateEMA, KGLSConfig, KGLSPreviousGraph, RLSConfig, RecursiveLeastSquares
from sca.experiments.synthetic import SYNTHETIC_PROTOCOL_VERSION, generate_synthetic_regimes
from sca.experiments.validation import _evaluate_sequence, _rolling_predictions
from sca.models.selective_adaptation import SCAConfig, SelectiveCausalAdapter


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _selection_by_dimension(path: str, required_keys: tuple[str, ...]) -> dict[tuple[int, int], dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = {}
    for row in rows:
        key = (int(row["n_variables"]), int(row["max_lag"]))
        if any(name not in row for name in required_keys):
            raise ValueError(f"{path} contains an incomplete selection for {key}")
        selected[key] = row
    return selected


def _baseline(method: str, lag: int, params: dict):
    if method == "KGLS ridge-to-previous":
        return KGLSPreviousGraph(lag, KGLSConfig(**params))
    if method == "RLS forgetting":
        return RecursiveLeastSquares(lag, RLSConfig(**params))
    if method == "Global-rate EMA":
        return GlobalRateEMA(lag, GlobalEMAConfig(**params))
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out factorial SCA baseline study")
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[4, 10, 20, 50])
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--distributions", nargs="+", default=["gaussian", "student_t", "uniform"])
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--sca-selection", default="results/phase1/tuning/selected_by_dimension.json")
    parser.add_argument("--baseline-selection", default="results/phase1/baseline_tuning/selected.json")
    parser.add_argument("--output-dir", default="results/phase1/heldout_factorial")
    args = parser.parse_args()
    if args.seed_start < 20 or args.seed_start + args.seeds > 40:
        parser.error("held-out evaluation is restricted to seeds 20--39")
    sca_rows = _selection_by_dimension(args.sca_selection, ("regression_window", "detection_window", "threshold", "persistence_decay", "deviation_scale", "ridge_alpha"))
    baseline_rows = _selection_by_dimension(args.baseline_selection, ("method", "parameters"))
    selected_baselines: dict[tuple[int, int], list[tuple[str, dict]]] = {}
    for key, row in baseline_rows.items():
        selected_baselines.setdefault(key, []).append((row["method"], json.loads(row["parameters"])))
    expected_keys = {(d, lag) for d in args.dimensions for lag in args.lags}
    missing = expected_keys - set(sca_rows)
    if missing:
        raise SystemExit(f"SCA selections missing for {sorted(missing)}")
    # JSON is row-oriented; retain the three independently selected baselines.
    baseline_records = json.loads(Path(args.baseline_selection).read_text(encoding="utf-8"))
    selected_baselines = {}
    for row in baseline_records:
        key = (int(row["n_variables"]), int(row["max_lag"]))
        selected_baselines.setdefault(key, []).append((row["method"], json.loads(row["parameters"])))
    missing = expected_keys - set(selected_baselines)
    if missing:
        raise SystemExit(f"baseline selections missing for {sorted(missing)}")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw.csv"
    columns = ["method", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points", "synthetic_protocol", "parameters", "precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_recall", "per_edge_delay", "per_edge_delay_censor_rate", "auroc", "auprc", "change_detection_rate", "false_alarm_rate"]
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=columns)
    checkpoint_keys = ["method", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points"]
    completed = set(tuple(row) for row in raw[checkpoint_keys].itertuples(index=False, name=None)) if not raw.empty else set()
    rows = raw.to_dict("records")
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for d in args.dimensions:
            for lag in args.lags:
                sca_choice = sca_rows[(d, lag)]
                sca_config = SCAConfig(**{key: sca_choice[key] for key in ("regression_window", "detection_window", "threshold", "persistence_decay", "deviation_scale", "ridge_alpha")})
                for sigma in args.noise_scales:
                    for distribution in args.distributions:
                        condition = (seed, d, lag, sigma, distribution, args.n_points)
                        method_specs = [("SCA dimension-aware ridge", sca_config, None), ("Rolling OLS (W=250, U=10)", None, None)]
                        method_specs.extend((name, None, params) for name, params in selected_baselines[(d, lag)])
                        expected = {(name, *condition) for name, _, _ in method_specs}
                        if expected.issubset(completed):
                            continue
                        data = generate_synthetic_regimes(n_points=args.n_points, n_variables=d, max_lag=lag, noise_scale=sigma, distribution=distribution, seed=seed)
                        sca_prediction, _, sca_flags, _ = SelectiveCausalAdapter(lag, sca_config).predict(data.X)
                        predictions = {
                            "SCA dimension-aware ridge": (sca_prediction, sca_flags, json.dumps({key: sca_choice[key] for key in ("regression_window", "detection_window", "threshold", "persistence_decay", "deviation_scale", "ridge_alpha")}, sort_keys=True)),
                            "Rolling OLS (W=250, U=10)": (_rolling_predictions(data.X, lag), None, json.dumps({"regression_window": 250, "update_interval": 10}, sort_keys=True)),
                        }
                        for name, params in selected_baselines[(d, lag)]:
                            predictions[name] = (_baseline(name, lag, params).predict(data.X), None, json.dumps(params, sort_keys=True))
                        for name, (prediction, flags, encoded) in predictions.items():
                            key = (name, *condition)
                            if key in completed:
                                continue
                            rows.append({"method": name, "seed": seed, "n_variables": d, "max_lag": lag, "noise_scale": sigma, "distribution": distribution, "n_points": args.n_points, "synthetic_protocol": SYNTHETIC_PROTOCOL_VERSION, "parameters": encoded, **_evaluate_sequence(prediction, data.A_true, data.change_points, detected_flags=flags)})
                            completed.add(key)
                        _atomic(pd.DataFrame(rows, columns=columns), raw_path)
                        print(f"checkpoint seed={seed}, d={d}, L={lag}, sigma={sigma}, distribution={distribution}", flush=True)
    frame = pd.DataFrame(rows, columns=columns)
    numeric = [name for name in columns if name.endswith(("precision", "recall", "f1", "shd", "delay", "rate", "auroc", "auprc"))]
    summary = frame.groupby(["method", "n_variables", "max_lag", "noise_scale", "distribution"], dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, map(str, value))).rstrip("_") for value in summary.columns]
    _atomic(summary, output / "mean_sd.csv")


if __name__ == "__main__":
    main()
