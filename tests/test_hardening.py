"""IEEE Access Hardening Test Suite.

All tests here must pass before the repository is submitted.
They cover the following hardening phases:
  1. Lag-0 non-identifiability documentation
  2. Consistent edge thresholding
  3. Generalized adaptation delay
  4. Null/no-change experiment (false alarm rate)
  5. No future information leakage
  6. Multi-seed reproducibility
  7. Shape consistency
  8. Ground-truth construction
  9. SCAConfig parameter validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.evaluation.adaptation_metrics import (
    PAPER_EDGE_THRESHOLD,
    LEGACY_EVAL_THRESHOLD,
    STABLE_EDGES,
    STABLE_EDGES_LAGGED_ONLY,
    OLD_EDGE,
    NEW_EDGE,
    CHANGED_EDGE_SET,
    CHANGED_EDGES_BEFORE,
    CHANGED_EDGES_AFTER,
    CHANGE_POINT,
    generalized_adaptation_delay,
    adaptation_delay,
    stable_edge_preservation,
    unnecessary_change_rate,
)
from sca.evaluation.evaluation import (
    binary_graph,
    precision_recall_f1,
    structural_hamming_distance,
)
from sca.models.selective_adaptation import (
    SCAConfig,
    SelectiveCausalAdapter,
    estimate_lagged_graph,
)
from sca.experiments.synthetic import generate_synthetic_regimes
from sca.ground_truth.regime_ground_truth_tensor import (
    build_time_indexed_ground_truth,
    REGIME_1_LINKS,
    REGIME_2_LINKS,
)


# ===========================================================
# Phase 1: Lag-0 identifiability
# ===========================================================

class TestLag0Identifiability:
    """Lag-0 limitation is documented and consistent."""

    def test_paper_threshold_value(self):
        # Selected by sensitivity study on tuning seeds 0-19 via select_synthetic_config.py
        assert PAPER_EDGE_THRESHOLD == 0.30

    def test_legacy_threshold_is_lower(self):
        assert LEGACY_EVAL_THRESHOLD < PAPER_EDGE_THRESHOLD

    def test_lag0_plane_is_zero_in_estimator(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 4))
        graph = estimate_lagged_graph(X, max_lag=2)
        assert graph.shape == (4, 4, 3)
        np.testing.assert_array_equal(
            graph[:, :, 0],
            np.zeros((4, 4)),
        )

    def test_stable_edges_contains_lag0(self):
        assert any(lag == 0 for _, _, lag in STABLE_EDGES)

    def test_stable_edges_lagged_only_has_no_lag0(self):
        assert not any(lag == 0 for _, _, lag in STABLE_EDGES_LAGGED_ONLY)

    def test_old_edge_is_lag0(self):
        _, _, lag = OLD_EDGE
        assert lag == 0

    def test_new_edge_is_lagged(self):
        _, _, lag = NEW_EDGE
        assert lag >= 1

    def test_synthetic_regimes_no_lag0(self):
        data = generate_synthetic_regimes(n_points=500, n_variables=4, max_lag=2, seed=0)
        lag0_sum = float(np.abs(data.A_true[:, :, :, 0]).sum())
        assert lag0_sum == 0.0

    def test_a1_ground_truth_has_lag0(self):
        A_true, _, _ = build_time_indexed_ground_truth()
        has_nonzero = np.any(A_true[:, :, :, 0] != 0)
        assert has_nonzero


# ===========================================================
# Phase 2: Consistent edge thresholding
# ===========================================================

class TestEdgeThresholding:

    def test_binary_graph_zero_threshold(self):
        A = np.array([[[0.0, 0.001, -0.5]]])
        result = binary_graph(A, threshold=0.0)
        np.testing.assert_array_equal(result, np.array([[[0, 1, 1]]]))

    def test_binary_graph_positive_threshold(self):
        A = np.array([[[0.0, 0.10, 0.20]]])
        result = binary_graph(A, threshold=0.15)
        np.testing.assert_array_equal(result, np.array([[[0, 0, 1]]]))

    def test_binary_graph_negative_weights(self):
        A = np.array([[[-0.20, 0.0, 0.10]]])
        result = binary_graph(A, threshold=0.15)
        np.testing.assert_array_equal(result, np.array([[[1, 0, 0]]]))

    def test_f1_perfect(self):
        A = np.zeros((4, 4, 3))
        A[0, 1, 1] = 0.30
        A[2, 3, 2] = 0.25
        result = precision_recall_f1(A, A, threshold=0.15)
        assert result["f1"] == 1.0

    def test_f1_empty(self):
        A_true = np.zeros((4, 4, 3))
        A_true[0, 1, 1] = 0.30
        result = precision_recall_f1(np.zeros((4, 4, 3)), A_true, threshold=0.0)
        assert result["f1"] == 0.0

    def test_shd_counts_differences(self):
        A_true = np.zeros((4, 4, 3))
        A_true[0, 1, 1] = 0.30
        A_true[2, 3, 2] = 0.25
        A_pred = np.zeros((4, 4, 3))
        A_pred[0, 1, 1] = 0.30
        A_pred[1, 2, 1] = 0.20  # extra FP
        shd = structural_hamming_distance(A_pred, A_true, threshold=0.15)
        assert shd == 2  # 1 FN + 1 FP


# ===========================================================
# Phase 3: Generalized adaptation delay
# ===========================================================

class TestGeneralizedAdaptationDelay:

    def _make_pred(self, n, change_point, before_weights, after_weights, adapt_at):
        A = np.zeros((n, 4, 4, 3), dtype=np.float32)
        for t in range(adapt_at):
            for (s, tgt, l), w in before_weights.items():
                A[t, s, tgt, l] = w
        for t in range(adapt_at, n):
            for (s, tgt, l), w in after_weights.items():
                A[t, s, tgt, l] = w
        return A

    def test_immediate_adaptation(self):
        before = {(0, 1, 1): 0.30}
        after = {(2, 3, 2): 0.25}
        A = self._make_pred(200, 50, before, after, adapt_at=50)
        delay = generalized_adaptation_delay(
            A, 50,
            edges_removed=[(0, 1, 1)],
            edges_added=[(2, 3, 2)],
            threshold=0.15,
        )
        assert delay == 0

    def test_delayed_adaptation(self):
        before = {(0, 1, 1): 0.30}
        after = {(2, 3, 2): 0.25}
        A = self._make_pred(200, 50, before, after, adapt_at=80)
        delay = generalized_adaptation_delay(
            A, 50,
            edges_removed=[(0, 1, 1)],
            edges_added=[(2, 3, 2)],
            threshold=0.15,
        )
        assert delay == 30

    def test_empty_edge_sets(self):
        A = np.zeros((100, 4, 4, 3), dtype=np.float32)
        delay = generalized_adaptation_delay(A, 50, [], [], threshold=0.0)
        assert delay == 0

    def test_never_adapts(self):
        A = np.zeros((100, 4, 4, 3), dtype=np.float32)
        for t in range(100):
            A[t, 0, 1, 1] = 0.30
        delay = generalized_adaptation_delay(
            A, 50, [(0, 1, 1)], [(2, 3, 2)], threshold=0.15
        )
        assert delay == -1

    def test_partial_recovery_fraction(self):
        A = np.zeros((200, 4, 4, 3), dtype=np.float32)
        for t in range(60, 200):
            A[t, 2, 3, 2] = 0.25
        delay = generalized_adaptation_delay(
            A, 50,
            edges_removed=[(0, 1, 1)],
            edges_added=[(2, 3, 2), (1, 2, 1)],
            threshold=0.15,
            recovery_fraction=0.5,
        )
        assert delay == 10


# ===========================================================
# Phase 4: Null/no-change experiment
# ===========================================================

class TestNullExperiment:

    def test_null_false_alarm_rate(self):
        rng = np.random.default_rng(42)
        n_points = 1000
        n_vars = 4
        max_lag = 2
        X = np.zeros((n_points, n_vars))
        X[:max_lag] = rng.normal(0, 0.1, (max_lag, n_vars))
        for t in range(max_lag, n_points):
            X[t, 0] = 0.3 * X[t - 1, 1] + rng.normal(0, 0.1)
            X[t, 1] = 0.3 * X[t - 1, 2] + rng.normal(0, 0.1)
            X[t, 2] = 0.3 * X[t - 1, 3] + rng.normal(0, 0.1)
            X[t, 3] = 0.3 * X[t - 2, 0] + rng.normal(0, 0.1)
        config = SCAConfig(
            detection_window=50,
            regression_window=100,
            threshold=0.30,
            persistence_decay=0.97,
            deviation_scale=0.10,
        )
        _, _, detected, _ = SelectiveCausalAdapter(max_lag, config).predict(X)
        false_alarm_rate = float(np.mean(detected))
        assert false_alarm_rate < 0.20, (
            f"FAR={false_alarm_rate:.3f} too high for stationary process"
        )

    def test_null_no_unnecessary_adaptation(self):
        data = generate_synthetic_regimes(
            n_points=1000,
            n_variables=4,
            max_lag=2,
            change_points=(),
            noise_scale=0.1,
            seed=1,
        )
        config = SCAConfig(detection_window=50, regression_window=100, threshold=0.30)
        _, _, detected, _ = SelectiveCausalAdapter(2, config).predict(data.X)
        fdr = float(np.mean(detected))
        assert fdr < 0.20, f"FAR={fdr:.3f} too high on null data"


# ===========================================================
# Phase 5: No future information leakage
# ===========================================================

class TestNoLeakage:

    def test_prediction_independent_of_future(self):
        rng = np.random.default_rng(0)
        n = 300
        X = rng.normal(0, 0.1, (n, 4))
        config = SCAConfig(regression_window=50, detection_window=50, update_interval=1)
        adapter = SelectiveCausalAdapter(2, config)
        A_full, _, _, _ = adapter.predict(X)
        t_cut = 150
        X_modified = X.copy()
        X_modified[t_cut:] = 0.0
        A_modified, _, _, _ = adapter.predict(X_modified)
        np.testing.assert_array_almost_equal(
            A_full[:t_cut], A_modified[:t_cut], decimal=5
        )

    def test_standardize_is_local(self):
        rng = np.random.default_rng(1)
        X = rng.normal(10, 5, (100, 4))
        g1 = estimate_lagged_graph(X, max_lag=2)
        g2 = estimate_lagged_graph(X + 1000.0, max_lag=2)
        np.testing.assert_array_almost_equal(g1, g2, decimal=4)


# ===========================================================
# Phase 6: Reproducibility
# ===========================================================

class TestReproducibility:

    def test_same_seed_same_data(self):
        d1 = generate_synthetic_regimes(n_points=200, seed=42)
        d2 = generate_synthetic_regimes(n_points=200, seed=42)
        np.testing.assert_array_equal(d1.X, d2.X)

    def test_different_seeds_differ(self):
        d1 = generate_synthetic_regimes(n_points=200, seed=42)
        d2 = generate_synthetic_regimes(n_points=200, seed=43)
        assert not np.array_equal(d1.X, d2.X)

    def test_adapter_deterministic(self):
        rng = np.random.default_rng(7)
        X = rng.normal(0, 0.1, (200, 4))
        a = SelectiveCausalAdapter(2)
        A1, s1, d1, p1 = a.predict(X)
        A2, s2, d2, p2 = a.predict(X)
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(d1, d2)


# ===========================================================
# Phase 7: Shape consistency
# ===========================================================

class TestShapeConsistency:

    @pytest.mark.parametrize("n_vars,max_lag", [(4, 2), (10, 1), (5, 3)])
    def test_adapter_shapes(self, n_vars, max_lag):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 0.1, (200, n_vars))
        A_pred, scores, detected, plasticity = SelectiveCausalAdapter(max_lag).predict(X)
        assert A_pred.shape == (200, n_vars, n_vars, max_lag + 1)
        assert scores.shape == (200,)
        assert detected.shape == (200,)
        assert plasticity.shape == (200, n_vars, n_vars, max_lag + 1)

    def test_no_self_edges(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0, 0.1, (300, 4))
        A_pred, _, _, _ = SelectiveCausalAdapter(2).predict(X)
        for t in range(len(A_pred)):
            for lag in range(3):
                diag = np.diag(A_pred[t, :, :, lag])
                np.testing.assert_array_equal(diag, np.zeros(4))

    def test_estimate_lagged_graph_shape(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 0.1, (100, 7))
        graph = estimate_lagged_graph(X, max_lag=3)
        assert graph.shape == (7, 7, 4)


# ===========================================================
# Phase 8: Ground-truth construction
# ===========================================================

class TestGroundTruth:

    def test_shape(self):
        A_true, _, _ = build_time_indexed_ground_truth()
        assert A_true.shape == (5000, 4, 4, 3)

    def test_lagged_stable_edges_before(self):
        A_true, _, _ = build_time_indexed_ground_truth()
        before = A_true[CHANGE_POINT - 1]
        assert before[0, 3, 2] != 0.0
        assert before[2, 1, 1] != 0.0

    def test_lagged_stable_edges_after(self):
        A_true, _, _ = build_time_indexed_ground_truth()
        after = A_true[CHANGE_POINT]
        assert after[0, 3, 2] != 0.0
        assert after[2, 1, 1] != 0.0

    def test_new_edge_appears(self):
        A_true, _, _ = build_time_indexed_ground_truth()
        assert A_true[CHANGE_POINT - 1, 1, 2, 1] == 0.0
        assert A_true[CHANGE_POINT, 1, 2, 1] != 0.0

    def test_boundary_graphs(self):
        A_true, g1, g2 = build_time_indexed_ground_truth()
        np.testing.assert_array_equal(A_true[2499], g1)
        np.testing.assert_array_equal(A_true[2500], g2)

    def test_regime1_has_x4_x3(self):
        assert ("X4", 0, "X3") in REGIME_1_LINKS

    def test_regime2_lacks_x4_x3(self):
        assert ("X4", 0, "X3") not in REGIME_2_LINKS

    def test_regime2_has_new_lagged_edge(self):
        assert ("X2", -1, "X3") in REGIME_2_LINKS


# ===========================================================
# Phase 9: SCAConfig validation
# ===========================================================

class TestConfigValidation:

    def test_invalid_regression_window(self):
        with pytest.raises(ValueError):
            SCAConfig(regression_window=2)

    def test_invalid_detection_window(self):
        with pytest.raises(ValueError):
            SCAConfig(detection_window=0)

    def test_invalid_plasticity_bounds(self):
        with pytest.raises(ValueError):
            SCAConfig(min_plasticity=0.8, max_plasticity=0.3)

    def test_invalid_persistence_decay(self):
        with pytest.raises(ValueError):
            SCAConfig(persistence_decay=1.5)

    def test_valid_config(self):
        config = SCAConfig(
            regression_window=100,
            detection_window=50,
            threshold=0.15,
            persistence_decay=0.95,
            min_plasticity=0.01,
            max_plasticity=0.50,
        )
        assert config.threshold == 0.15
