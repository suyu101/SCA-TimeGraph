"""Backward-compatible, import-safe entry point for SCA visualisation."""

from .visualize_results import plot_sca_results


def main():
    """Render plots from the conventional saved result locations."""
    outputs = plot_sca_results(
        "results/sca_A_pred.npy",
        "results/sca_plasticity_rates.npy",
        "results/plots",
    )
    print(*outputs, sep="\n")


if __name__ == "__main__":
    main()
