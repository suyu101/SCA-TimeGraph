"""Evaluate SCA and causal-discovery baselines on Causal Chambers wt_walks_v1."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.experiments.causal_chambers import load_wt_walks, run_pcmci_wt_walks, run_sca_wt_walks, write_report


def main():
    parser = argparse.ArgumentParser(description="Causal Chambers wind-tunnel evaluation")
    parser.add_argument("--csv", default="Datasets/causal_chambers/wt_walks_v1/actuators_random_walk_2.csv")
    parser.add_argument("--output-dir", default="results/causal_chambers/wt_walks_v1")
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--sca-threshold", type=float, default=0.05)
    parser.add_argument("--pcmci-alpha", type=float, default=0.01)
    args = parser.parse_args()
    X, truth, variables, timestamps = load_wt_walks(args.csv)
    sca_prediction, sca_metrics = run_sca_wt_walks(X, truth, max_lag=args.max_lag, threshold=args.sca_threshold)
    pcmci_prediction, pcmci_metrics = run_pcmci_wt_walks(X, variables, truth, max_lag=args.max_lag, alpha_level=args.pcmci_alpha)
    metrics = [{"method": "SCA", "status": "completed", **sca_metrics}, {"method": "PCMCI+", **pcmci_metrics}]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "sca_prediction.npy", sca_prediction)
    if pcmci_prediction is not None:
        np.save(out / "pcmci_prediction.npy", pcmci_prediction)
    write_report(out, variables=variables, timestamps=timestamps, metrics=metrics, protocol={"graph": "official causalchamber.ground_truth.graph('wt', 'standard')", "scoring": "lag-collapsed directed edge recovery; no claim of instantaneous-edge identification", "max_lag": args.max_lag})
    print(f"Wrote benchmark results to {out}")


if __name__ == "__main__":
    main()
