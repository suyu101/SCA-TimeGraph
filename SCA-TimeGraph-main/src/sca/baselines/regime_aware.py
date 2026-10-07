"""Native Tigramite RPCMCI integration for regime-aware discovery."""

import numpy as np


class RegimeAwareBaselineUnavailable(RuntimeError):
    """Raised when the optional dependencies required by native RPCMCI are absent."""


def run_rpcmci(
    X, *, max_lag=2, num_regimes=4, max_transitions=3, seed=0,
    pc_alpha=0.2, alpha_level=0.05, num_iterations=10, max_anneal=3,
):
    """Run native RPCMCI and return its documented result dictionary.

    The synthetic protocol fixes four regimes and three transitions before
    scoring. This function never falls back to a rolling model.
    """
    try:
        from sklearn.linear_model import LinearRegression
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.rpcmci import RPCMCI
    except ImportError as error:
        raise RegimeAwareBaselineUnavailable(
            "RPCMCI requires tigramite, scikit-learn, joblib, and ortools."
        ) from error
    model = RPCMCI(
        dataframe=pp.DataFrame(np.asarray(X, dtype=float)),
        cond_ind_test=ParCorr(significance="analytic"),
        prediction_model=LinearRegression(), seed=int(seed), verbosity=-1,
    )
    result = model.run_rpcmci(
        num_regimes=int(num_regimes), max_transitions=int(max_transitions),
        tau_min=1, tau_max=int(max_lag), pc_alpha=float(pc_alpha),
        alpha_level=float(alpha_level), num_iterations=int(num_iterations),
        max_anneal=int(max_anneal), n_jobs=1,
    )
    if result is None or not result.get("causal_results"):
        raise RuntimeError("RPCMCI did not converge for this replication")
    return result
