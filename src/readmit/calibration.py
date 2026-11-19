"""Post-hoc calibrators fitted on a held-out calibration split.

* ``PlattCalibrator``: logistic regression on the logit of the raw score.
* ``IsotonicCalibrator``: monotone step function (scikit-learn).
* ``VennAbersCalibrator``: inductive Venn-Abers predictor (Vovk & Petej 2014). For each test
  score s, two isotonic regressions are fitted on the calibration set augmented with (s, 0) and
  (s, 1), giving a probability interval [p0, p1]; the merged point prediction is
  p1 / (1 - p0 + p1). For speed, p0 and p1 are computed exactly on a grid of score quantiles and
  linearly interpolated between grid points.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def fit(self, s: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(s).reshape(-1, 1), y)
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(_logit(s).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, s: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        return self.model_.predict(np.asarray(s, float))


class VennAbersCalibrator:
    def __init__(self, n_grid: int = 200):
        self.n_grid = n_grid

    def fit(self, s: np.ndarray, y: np.ndarray) -> VennAbersCalibrator:
        s = np.asarray(s, float)
        y = np.asarray(y, float)
        grid = np.unique(np.quantile(s, np.linspace(0, 1, self.n_grid)))
        p0 = np.empty(len(grid))
        p1 = np.empty(len(grid))
        for i, g in enumerate(grid):
            xs = np.append(s, g)
            for label, store in ((0.0, p0), (1.0, p1)):
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(xs, np.append(y, label))
                store[i] = iso.predict([g])[0]
        self.grid_, self.p0_, self.p1_ = grid, p0, p1
        return self

    def predict_interval(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, float)
        return np.interp(s, self.grid_, self.p0_), np.interp(s, self.grid_, self.p1_)

    def predict(self, s: np.ndarray) -> np.ndarray:
        p0, p1 = self.predict_interval(s)
        return p1 / (1.0 - p0 + p1)


CALIBRATORS = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "venn_abers": VennAbersCalibrator,
}
