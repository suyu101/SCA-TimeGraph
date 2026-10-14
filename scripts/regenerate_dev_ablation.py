#!/usr/bin/env python
"""Regenerate the development-benchmark ablation (Tables 8-9, Fig. 8 inputs) with the
FINAL, strictly past-only (lag-zero-free) estimator.

Why this script exists
----------------------
The arrays originally saved under ``results/sca_*.npy`` were produced by the earlier
R^2-increment estimator that scored lag 0 (current-time sources); they contain non-zero
lag-0 coefficients and lag-0 plasticity values and were never regenerated after the
lag-zero plane was removed.  Those arrays are NOT used by the manuscript any more.

What it does
------------
* regenerates the 4-variable development series (generator default seed 42, T=5000,
  change at t=2500) in memory -- no dataset file is needed;
* runs Full SCA / No Persistence / Global Only through
  ``sca.models.selective_adaptation_ablation.predict_gated`` under two profiles:
  ``defaults``  -- ``SCAConfig()`` defaults (W=D=100, theta=0.08, lambda=0.90, eps=0.05,
                  U=10, no ridge), i.e. the parameter values of the legacy development run;
  ``calibrated_d4_l2`` -- the null-calibrated (d=4, L=2) configuration frozen on tuning
                  seeds 0-19 (read from results/phase1/tuning/selected_by_dimension.json);
  plus the past-only Rolling OLS baseline of the paper (W=250, U=10);
* scores every method on the lag >= 1 planes only (the lag-0 plane is masked in both
  prediction and truth), edge threshold 0.15 (the paper-wide fixed threshold), using
  the repository's own ``_evaluate_sequence`` / ``precision_recall_f1`` /
  ``structural_hamming_distance``;
* writes ``results/dev_ablation_final/<profile>/{table8.csv,table9.csv,summary.json,arrays.npz}``.

Usage:  python scripts/regenerate_dev_ablation.py [output_root]
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sca.evaluation.evaluation import precision_recall_f1, structural_hamming_distance  # noqa: E402
from sca.experiments.validation import _evaluate_sequence, _rolling_predictions  # noqa: E402
from sca.ground_truth.regime_ground_truth_tensor import build_time_indexed_ground_truth  # noqa: E402
from sca.models.selective_adaptation import SCAConfig  # noqa: E402
from sca.models.selective_adaptation_ablation import predict_gated  # noqa: E402

EDGE_THRESHOLD = 0.15          # paper-wide fixed edge-decision threshold
CHANGE_POINT = 2500
MAX_LAG = 2
HORIZON = 500                  # post-change evaluation horizon used by _evaluate_sequence
ADDED_EDGE = (1, 2, 1)         # X2(t-1) -> X3(t): the only changed edge a lag>=1 estimator can represent
PLASTICITY_WINDOW = (CHANGE_POINT + 100, CHANGE_POINT + 300)   # [2600, 2800)
VARIANTS = (("Full SCA", "full"), ("No Persistence", "no_persistence"), ("Global Only", "global_only"))


def load_profiles() -> dict:
    """Configurations evaluated on the development series (nothing is tuned on it)."""
    frozen = json.loads((ROOT / "results" / "phase1" / "tuning" / "selected_by_dimension.json").read_text())
    entry = next(e for e in frozen if e["n_variables"] == 4 and e["max_lag"] == 2)
    calibrated = SCAConfig(
        regression_window=int(entry["regression_window"]), detection_window=int(entry["detection_window"]),
        threshold=float(entry["threshold"]), persistence_decay=float(entry["persistence_decay"]),
        deviation_scale=float(entry["deviation_scale"]), ridge_alpha=float(entry["ridge_alpha"]),
    )
    return {"defaults": SCAConfig(), "calibrated_d4_l2": calibrated}


def load_dev_series() -> np.ndarray:
    spec = importlib.util.spec_from_file_location("gen_rc", ROOT / "scripts" / "generate_regime_change.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = module.generate_regime_change_data()   # defaults: 5000 points, change 2500, noise 0.1, seed 42
    return frame[["X1", "X2", "X3", "X4"]].to_numpy(dtype=np.float32)


def mask_lag0(graph: np.ndarray) -> np.ndarray:
    out = np.array(graph, copy=True)
    out[..., 0] = 0.0
    return out


def score_method(prediction, truth_lagged, flags, static_after=True):
    """Compute the Table 8 row for one method (all on lag >= 1 planes)."""
    prediction = mask_lag0(prediction)
    pre = precision_recall_f1(prediction[CHANGE_POINT - 1], truth_lagged[CHANGE_POINT - 1], threshold=EDGE_THRESHOLD)
    post = precision_recall_f1(prediction[-1], truth_lagged[-1], threshold=EDGE_THRESHOLD)
    sequence = _evaluate_sequence(prediction, truth_lagged, [CHANGE_POINT], detected_flags=flags, edge_threshold=EDGE_THRESHOLD)
    added_present = np.abs(prediction[CHANGE_POINT:CHANGE_POINT + HORIZON, ADDED_EDGE[0], ADDED_EDGE[1], ADDED_EDGE[2]]) > EDGE_THRESHOLD
    return {
        "f1_pre": pre["f1"], "f1_post": post["f1"],
        "shd_pre": structural_hamming_distance(prediction[CHANGE_POINT - 1], truth_lagged[CHANGE_POINT - 1], threshold=EDGE_THRESHOLD),
        "shd_post": structural_hamming_distance(prediction[-1], truth_lagged[-1], threshold=EDGE_THRESHOLD),
        "precision_post": post["precision"], "recall_post": post["recall"],
        "stable_edge_recall": sequence["stable_edge_recall"],
        "new_edge_presence": float(added_present.mean()),
        "new_edge_delay": sequence["per_edge_delay"],
        "new_edge_censored": sequence["per_edge_delay_censor_rate"],
        "whole_graph_delay": sequence["adaptation_delay"],
        "auprc": sequence["auprc"],
    }


def run_profile(profile: str, config: SCAConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X = load_dev_series()
    A_true, _, _ = build_time_indexed_ground_truth()
    truth_lagged = mask_lag0(A_true)          # lag>=1 truth: pre = 2 edges, post = 3 edges

    outputs, rows, arrays = {}, [], {}
    for name, gating in VARIANTS:
        predictions, scores, flags, plasticity = predict_gated(X, MAX_LAG, config, gating)
        assert not np.any(predictions[..., 0]), "lag-0 plane must be identically zero"
        outputs[gating] = (predictions, scores, flags, plasticity)
        rows.append({"method": name, **score_method(predictions, truth_lagged, flags)})
        arrays.update({f"{gating}_pred": predictions, f"{gating}_plasticity": plasticity})
        if gating == "full":
            arrays.update({"scores": scores, "flags": flags})
    rolling = _rolling_predictions(X.astype(float), MAX_LAG)       # W=250, U=10, unregularised
    rows.append({"method": "Rolling OLS (W=250, U=10)", **score_method(rolling, truth_lagged, None)})
    arrays["rolling_ols_pred"] = rolling

    # Static PCMCI+ graph saved earlier by evaluate_pcmci.py (scored on lag>=1 planes only).
    pcmci = np.load(ROOT / "results" / "pcmci_a1_A_pred.npy")
    pcmci_pre = precision_recall_f1(mask_lag0(pcmci), truth_lagged[CHANGE_POINT - 1], threshold=EDGE_THRESHOLD)
    rows.append({
        "method": "PCMCI+ (static)", "f1_pre": pcmci_pre["f1"],
        "shd_pre": structural_hamming_distance(mask_lag0(pcmci), truth_lagged[CHANGE_POINT - 1], threshold=EDGE_THRESHOLD),
    })

    table8 = pd.DataFrame(rows)
    table8.to_csv(out_dir / "table8.csv", index=False)

    # Full-tensor scoring (lag-0 truth edges kept) for the three SCA variants -- reported only
    # to show how much of the gap is the unrepresentable lag-0 truth.
    fulltruth = []
    for name, gating in VARIANTS:
        p = outputs[gating][0]
        pre = precision_recall_f1(p[CHANGE_POINT - 1], A_true[CHANGE_POINT - 1], threshold=EDGE_THRESHOLD)
        post = precision_recall_f1(p[-1], A_true[-1], threshold=EDGE_THRESHOLD)
        fulltruth.append({"method": name, "f1_pre_full_truth": pre["f1"], "f1_post_full_truth": post["f1"],
                          "recall_pre_full_truth": pre["recall"], "recall_post_full_truth": post["recall"]})
    pd.DataFrame(fulltruth).to_csv(out_dir / "table8_full_truth_scoring.csv", index=False)

    # Table 9: realised plasticity at eligible updates (t mod U == 0) inside [2600, 2800), lag>=1 slots only.
    lo, hi = PLASTICITY_WINDOW
    eligible = [t for t in range(lo, hi) if t % config.update_interval == 0]
    slots = [(i, j, l) for i in range(4) for j in range(4) if i != j for l in (1, 2)]
    others = [s for s in slots if s != ADDED_EDGE]
    assert len(slots) == 24 and len(others) == 23
    table9 = []
    for name, gating in VARIANTS:
        P = outputs[gating][3]
        changed = float(np.mean([P[t, ADDED_EDGE[0], ADDED_EDGE[1], ADDED_EDGE[2]] for t in eligible]))
        other = float(np.mean([[P[t, i, j, l] for (i, j, l) in others] for t in eligible]))
        table9.append({"method": name, "changed": changed, "stable_or_other": other, "ratio": changed / max(other, 1e-12),
                       "n_updates_in_window": len(eligible), "n_other_slots": len(others)})
    pd.DataFrame(table9).to_csv(out_dir / "table9.csv", index=False)

    # Detector diagnostics for the text of Sec. VI-E (detector is identical for all gatings).
    scores, flags = outputs["full"][1], outputs["full"][2]
    updates = np.array([t for t in range(MAX_LAG, len(X)) if t % config.update_interval == 0 and t + 1 >= 2 * config.detection_window])
    before = updates[updates < CHANGE_POINT]
    after = updates[updates >= CHANGE_POINT]
    post_horizon = updates[(updates >= CHANGE_POINT) & (updates < CHANGE_POINT + HORIZON)]
    first_flag_after = [int(t) for t in after if flags[t]][:1]
    detector = {
        "eligible_updates_total": int(len(updates)),
        "flagged_share_before_change": float(flags[before].mean()),
        "flagged_share_after_change": float(flags[after].mean()),
        "flagged_share_first_500_after_change": float(flags[post_horizon].mean()),
        "n_before": int(len(before)), "n_after": int(len(after)),
        "first_flag_at_or_after_change": first_flag_after[0] if first_flag_after else None,
        "score_mean_before": float(scores[before].mean()), "score_mean_after": float(scores[after].mean()),
        "threshold": config.threshold,
    }

    X_bytes = np.ascontiguousarray(X).tobytes()
    summary = {
        "script": "scripts/regenerate_dev_ablation.py", "profile": profile,
        "base_repository_snapshot": "github.com/suyu101/SCA-TimeGraph @ be3e159e61b9935fcd683cc2020c021fa60b4cc5 (plus local uncommitted changes)",
        "data": {"generator": "scripts/generate_regime_change.py::generate_regime_change_data()", "seed": 42, "T": int(len(X)),
                 "change_point": CHANGE_POINT, "noise_scale": 0.1, "X_float32_sha256": hashlib.sha256(X_bytes).hexdigest()},
        "config": asdict(config), "max_lag": MAX_LAG, "edge_threshold": EDGE_THRESHOLD, "horizon": HORIZON,
        "scoring": "lag>=1 planes only (lag-0 masked in prediction and truth)",
        "changed_edge_representable": "X2(t-1)->X3(t) (added); removed X4(t)->X3(t) is lag-0 and not representable",
        "stable_edges_scored": "X1(t-2)->X4(t), X3(t-1)->X2(t)",
        "plasticity_window": [lo, hi], "detector": detector,
        "lag0_coefficients_all_zero": bool(all(not np.any(outputs[g][0][..., 0]) for _, g in VARIANTS)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out_dir / "arrays.npz", **arrays)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.4f}")
    print(f"##### profile: {profile} | config: {asdict(config)}")
    print("=== Table 8 (regenerated, lag>=1 scoring, threshold 0.15) ===")
    print(table8.to_string(index=False))
    print("\n=== Table 8, full-tensor scoring of SCA variants (lag-0 truth kept) ===")
    print(pd.DataFrame(fulltruth).to_string(index=False))
    print("\n=== Table 9 (regenerated) ===")
    print(pd.DataFrame(table9).to_string(index=False))
    print("\n=== Detector diagnostics ===")
    print(json.dumps(detector, indent=2))
    print("\nSaved to", out_dir)


def main(out_root: Path) -> None:
    for profile, config in load_profiles().items():
        run_profile(profile, config, out_root / profile)
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "dev_ablation_final")
