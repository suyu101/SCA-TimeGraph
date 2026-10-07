import numpy as np

from sca.baselines.streaming_linear import (
    GlobalEMAConfig,
    GlobalRateEMA,
    KGLSConfig,
    KGLSPreviousGraph,
    RLSConfig,
    RecursiveLeastSquares,
)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(80, 3))
    values[1:, 1] += 0.4 * values[:-1, 0]
    return values


def _assert_prediction_shape_and_past_only(factory):
    values = _data()
    prediction = factory().predict(values)
    assert prediction.shape == (80, 3, 3, 3)
    for lag in range(3):
        np.testing.assert_allclose(np.diagonal(prediction[:, :, :, lag], axis1=1, axis2=2), 0.0)
    changed_future = values.copy()
    changed_future[50:] = 0.0
    np.testing.assert_allclose(prediction[:50], factory().predict(changed_future)[:50], atol=1e-6)


def test_kgls_interface_and_no_future_leakage():
    _assert_prediction_shape_and_past_only(lambda: KGLSPreviousGraph(2, KGLSConfig(regression_window=30, prior_strength=1.0)))


def test_rls_interface_and_no_future_leakage():
    _assert_prediction_shape_and_past_only(lambda: RecursiveLeastSquares(2, RLSConfig(forgetting_factor=0.99)))


def test_global_ema_interface_and_no_future_leakage():
    _assert_prediction_shape_and_past_only(lambda: GlobalRateEMA(2, GlobalEMAConfig(regression_window=30, rate=0.2, ridge_alpha=1.0)))
