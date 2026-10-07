"""Small dependency-free OLS helper used by regression baselines."""

import numpy as np


class LinearRegression:
    """Subset of scikit-learn's interface backed by ``numpy.linalg.lstsq``."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        design = np.column_stack((np.ones(len(X)), X))
        solution = np.linalg.lstsq(design, y, rcond=None)[0]
        self.intercept_ = solution[0]
        self.coef_ = solution[1:]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
