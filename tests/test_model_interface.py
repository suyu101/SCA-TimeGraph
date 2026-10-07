import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sca.ground_truth.regime_ground_truth_tensor import (
    build_time_indexed_ground_truth,
)
from sca.evaluation.evaluate_model import evaluate_model




def test_prediction_interface():

    A_true, _, _ = build_time_indexed_ground_truth()

    # Temporary dummy prediction.
    # This represents the shape your model must return.
    A_pred = np.zeros_like(A_true)

    print("A_true shape:", A_true.shape)
    print("A_pred shape:", A_pred.shape)

    assert A_pred.shape == (
        5000,
        4,
        4,
        3,
    )

    results = evaluate_model(
        A_pred,
        A_true,
        change_point=2500,
    )

    print("\nPrediction interface is valid.")

    print(
        "Before-change F1:",
        results["static_before"]["f1"],
    )

    print(
        "After-change F1:",
        results["static_after"]["f1"],
    )


if __name__ == "__main__":
    test_prediction_interface()
