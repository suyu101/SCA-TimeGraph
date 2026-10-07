"""On-demand plotting utilities for SCA result tensors."""

from pathlib import Path

import numpy as np


def plot_sca_results(prediction_path, plasticity_path, output_dir, change_points=(2500,)):
    """Create edge-strength and plasticity figures from saved result tensors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires the optional matplotlib dependency.") from error
    prediction, plasticity = np.load(prediction_path), np.load(plasticity_path)
    if prediction.shape != plasticity.shape or prediction.ndim != 4:
        raise ValueError("Prediction and plasticity must be same-shaped 4-D tensors.")
    edges = {"X1 → X4 (lag 2)": (0, 3, 2), "X4 → X3 (lag 0)": (3, 2, 0),
             "X3 → X2 (lag 1)": (2, 1, 1), "X2 → X1 (lag 0)": (1, 0, 0)}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, tensor, ylabel in (("causal_edge_trajectories", prediction, "Causal strength"),
                                 ("plasticity_trajectories", plasticity, "Plasticity")):
        figure, axis = plt.subplots(figsize=(12, 6))
        for label, (source, target, lag) in edges.items():
            if source < tensor.shape[1] and target < tensor.shape[2] and lag < tensor.shape[3]:
                axis.plot(tensor[:, source, target, lag], label=label)
        for point in change_points:
            axis.axvline(point, color="black", linestyle="--", alpha=0.6)
        axis.set(xlabel="Time", ylabel=ylabel, title=name.replace("_", " ").title())
        axis.legend(); figure.tight_layout()
        output = output_dir / f"{name}.png"; figure.savefig(output, dpi=200); plt.close(figure)
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    print(*plot_sca_results("results/sca_full_A_pred.npy", "results/sca_full_plasticity_rates.npy", "results/plots"), sep="\n")
