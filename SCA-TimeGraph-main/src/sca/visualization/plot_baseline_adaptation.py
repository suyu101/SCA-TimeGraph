"""Plot a rolling-baseline adaptation trace on demand."""

from pathlib import Path

import numpy as np


def plot_baseline_adaptation(prediction_path, output_path, change_points=(2500,)):
    """Plot two illustrative legacy edges from a prediction tensor.

    Matplotlib is deliberately imported here, rather than at package import
    time, because visualisation is an optional project capability.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires the optional matplotlib dependency.") from error
    prediction = np.load(prediction_path)
    if prediction.ndim != 4 or prediction.shape[1] < 4 or prediction.shape[3] < 2:
        raise ValueError("Expected a [time, variable, variable, lag] prediction tensor.")
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(prediction[:, 3, 2, 0], label="X4 → X3 (lag 0)")
    axis.plot(prediction[:, 1, 2, 1], label="X2 → X3 (lag 1)")
    for point in change_points:
        axis.axvline(point, color="black", linestyle="--", alpha=0.6)
    axis.set(xlabel="Time", ylabel="Estimated edge strength", title="Rolling-regression adaptation")
    axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    print(plot_baseline_adaptation("Datasets/regime_change_A_pred_baseline.npy", "results/baseline_adaptation.png"))
