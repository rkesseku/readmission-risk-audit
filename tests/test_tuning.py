import numpy as np
import pandas as pd
import pytest

from readmit.models import LogisticModel, tune_on_validation


def _data(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (X["x0"] + 0.5 * X["x1"] + rng.normal(size=n) > 0.8).astype(int).to_numpy()
    return X, y


def test_tune_on_validation_scores_every_grid_point_and_picks_the_max():
    Xf, yf = _data(400, 0)
    Xv, yv = _data(200, 1)
    grid = [{"C": c} for c in (1e-6, 1e-2, 1.0)]
    res = tune_on_validation(LogisticModel, grid, Xf, yf, Xv, yv)
    assert [r["params"] for r in res["grid"]] == grid
    assert res["best_val_auroc"] == max(r["val_auroc"] for r in res["grid"])
    assert res["best_params"] in grid


def test_tune_on_validation_ties_go_to_first_and_empty_grid_fails():
    Xf, yf = _data(300, 2)
    Xv, yv = _data(150, 3)
    res = tune_on_validation(LogisticModel, [{"C": 1.0}, {"C": 1.0}], Xf, yf, Xv, yv)
    assert res["best_params"] == {"C": 1.0}
    with pytest.raises(ValueError):
        tune_on_validation(LogisticModel, [], Xf, yf, Xv, yv)
