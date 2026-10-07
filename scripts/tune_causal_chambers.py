"""Validation-only tuning and held-out reporting for the wind-tunnel study."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.evaluation.evaluation import precision_recall_f1, structural_hamming_distance
from sca.experiments.causal_chambers import load_wt_walks
from sca.models.selective_adaptation import SCAConfig, SelectiveCausalAdapter


def score(prediction, truth, threshold):
    collapsed = np.max(np.abs(prediction[:, :, 1:]), axis=2)
    binary = collapsed[:, :, None]
    target = truth[:, :, None]
    return {**precision_recall_f1(binary, target, threshold), "shd": structural_hamming_distance(binary, target, threshold)}


def run():
    parser = argparse.ArgumentParser(description="Tune Causal Chambers SCA on validation only")
    parser.add_argument("--csv", default="Datasets/causal_chambers/wt_walks_v1/actuators_random_walk_2.csv")
    parser.add_argument("--output-dir", default="results/causal_chambers/wt_walks_v1/tuning")
    parser.add_argument("--max-lag", type=int, default=10)
    args = parser.parse_args()
    X, truth, variables, timestamps = load_wt_walks(args.csv)
    train_end, validation_end = int(.70 * len(X)), int(.85 * len(X))
    # The search is deliberately compact and predeclared.  It never evaluates
    # held-out samples while choosing parameters.
    candidates = [
        {"regression_window": w, "detection_window": d, "threshold": th, "persistence_decay": decay, "deviation_scale": eps}
        for w, d, th, decay, eps in [
            (250, 250, .03, .80, .025),
            (500, 500, .05, .90, .05),
        ]
    ]
    rows = []
    for parameters in candidates:
        config = SCAConfig(**parameters, update_interval=100)
        started = perf_counter()
        final_graph = SelectiveCausalAdapter(args.max_lag, config).final_graph(X[:validation_end])
        metrics = score(final_graph, truth, config.threshold)
        rows.append({**parameters, **metrics, "validation_runtime_seconds": perf_counter() - started})
    table = pd.DataFrame(rows).sort_values(["f1", "shd"], ascending=[False, True]).reset_index(drop=True)
    selected = table.iloc[0].to_dict()
    selected_parameters = {key: selected[key] for key in ("regression_window", "detection_window", "threshold", "persistence_decay", "deviation_scale")}
    selected_parameters["regression_window"] = int(selected_parameters["regression_window"])
    selected_parameters["detection_window"] = int(selected_parameters["detection_window"])
    config = SCAConfig(**selected_parameters, update_interval=100)
    started = perf_counter()
    final_graph = SelectiveCausalAdapter(args.max_lag, config).final_graph(X)
    held_out = score(final_graph, truth, config.threshold)
    held_out["runtime_seconds"] = perf_counter() - started
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "validation_search.csv", index=False)
    pd.DataFrame([{**held_out, **asdict(config)}]).to_csv(output / "held_out_metrics.csv", index=False)
    np.save(output / "held_out_final_graph.npy", final_graph)
    (output / "selection.json").write_text(json.dumps({"selected_on": "chronological validation prefix only", "train_end": train_end, "validation_end": validation_end, "variables": variables, "selected_config": asdict(config), "held_out_metrics": held_out}, indent=2), encoding="utf-8")
    print(f"Selected validation F1={selected['f1']:.4f}; wrote held-out results to {output}")


if __name__ == "__main__":
    run()
