#!/usr/bin/env python
"""Post hoc reproduction check cited in the manuscript's Data and Code Availability statement.

(a) Re-runs three Table VII rows (default generator, seeds 0-19) and prints F1 / SHD / FAR next to the paper values.
(b) Re-runs the frozen d=50, L=4 dimension-aware ridge SCA on held-out seeds 20-39 and compares every metric, seed by
    seed, with results/phase1/heldout_d50_l4/heldout_raw.csv (max absolute difference is printed).
Takes a few minutes on one CPU core.  Usage: python scripts/verify_headline_reproduction.py
"""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from sca.experiments.synthetic import generate_synthetic_regimes
from sca.experiments.validation import _evaluate_sequence
from sca.models.selective_adaptation import SCAConfig, SelectiveCausalAdapter

def table7(**kw):
    cfg, rows = replace(SCAConfig(), **kw), []
    for s in range(20):
        d = generate_synthetic_regimes(n_points=5000, seed=s)
        p, _, f, _ = SelectiveCausalAdapter(2, cfg).predict(d.X)
        m = _evaluate_sequence(p, d.A_true, d.change_points, detected_flags=f); rows.append((m["f1"], m["shd"], m["false_alarm_rate"]))
    return np.mean(rows, axis=0)

print("(a) Table VII rows [paper: F1, SHD, FAR]")
for label, kw, paper in [("base", {}, (0.531, 3.854, 0.094)), ("theta=0.30", {"threshold": 0.30}, (0.567, 3.608, 0.017)), ("W=200", {"regression_window": 200}, (0.609, 2.824, 0.094))]:
    r = table7(**kw); print(f"  {label:10s} got F1={r[0]:.3f} SHD={r[1]:.3f} FAR={r[2]:.3f} | paper {paper}")

sel = json.loads((ROOT / "results/phase1/tuning_d50_l4/selected_by_dimension.json").read_text())[0]
cfg = SCAConfig(regression_window=int(sel["regression_window"]), detection_window=int(sel["detection_window"]), threshold=float(sel["threshold"]),
                persistence_decay=float(sel["persistence_decay"]), deviation_scale=float(sel["deviation_scale"]), ridge_alpha=float(sel["ridge_alpha"]))
saved = pd.read_csv(ROOT / "results/phase1/heldout_d50_l4/heldout_raw.csv")
saved = saved[saved.method == "SCA dimension-aware ridge"].set_index("seed")
keys = ["precision", "recall", "f1", "shd", "auprc", "stable_edge_recall", "per_edge_delay", "change_detection_rate", "false_alarm_rate"]
rows = []
for seed in range(20, 40):
    d = generate_synthetic_regimes(n_points=5000, n_variables=50, max_lag=4, noise_scale=0.1, distribution="gaussian", seed=seed)
    p, _, f, _ = SelectiveCausalAdapter(4, cfg).predict(d.X)
    m = _evaluate_sequence(p, d.A_true, d.change_points, detected_flags=f); rows.append({"seed": seed, **{k: m[k] for k in keys}})
new = pd.DataFrame(rows).set_index("seed")
print("\n(b) d=50, L=4 held-out: max |recomputed - saved| over 20 seeds"); print((new[keys] - saved.loc[new.index, keys]).abs().max().to_string())
print("mean: F1 %.3f AUPRC %.3f SHD %.1f (paper 0.576 / 0.831 / 28.2)" % (new.f1.mean(), new.auprc.mean(), new.shd.mean()))
