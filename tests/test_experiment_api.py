import numpy as np

from sca.experiments.synthetic import generate_synthetic_regimes
from sca.models.selective_adaptation import SCAConfig, SelectiveCausalAdapter, estimate_lagged_graph


def test_synthetic_generator_has_requested_recurring_regimes():
    data = generate_synthetic_regimes(n_points=120, n_variables=10, max_lag=4, change_points=(30, 60, 90), seed=7)
    assert data.X.shape == (120, 10)
    assert data.A_true.shape == (120, 10, 10, 5)
    assert data.change_points == (30, 60, 90)
    assert np.array_equal(data.A_true[0], data.A_true[60])
    assert np.array_equal(data.A_true[30], data.A_true[90])


def test_noise_scale_changes_observed_snr_without_changing_truth():
    low = generate_synthetic_regimes(n_points=120, noise_scale=.05, seed=11)
    high = generate_synthetic_regimes(n_points=120, noise_scale=.5, seed=11)
    assert np.array_equal(low.A_true, high.A_true)
    assert np.std(high.X - high.clean_X) > 5 * np.std(low.X - low.clean_X)


def test_estimator_is_dimension_agnostic_and_has_no_current_time_plane():
    graph = estimate_lagged_graph(np.random.default_rng(3).normal(size=(80, 10)), max_lag=4)
    assert graph.shape == (10, 10, 5)
    assert np.all(graph[:, :, 0] == 0)
    assert np.all(np.diagonal(graph, axis1=0, axis2=1) == 0)


def test_adapter_returns_complete_online_interface():
    prediction, scores, flags, plasticity = SelectiveCausalAdapter(2, SCAConfig(update_interval=5)).predict(np.random.default_rng(1).normal(size=(60, 4)))
    assert prediction.shape == plasticity.shape == (60, 4, 4, 3)
    assert scores.shape == flags.shape == (60,)
