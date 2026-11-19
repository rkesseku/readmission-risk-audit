"""The three model families compared in the study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _log1p_counts(X: pd.DataFrame) -> np.ndarray:
    X = X.copy()
    for c in ("number_outpatient", "number_emergency", "number_inpatient"):
        if c in X:
            X[c] = np.log1p(X[c])
    return X.to_numpy(dtype=float)


class LogisticModel:
    name = "logistic"

    def __init__(self, seed: int = 0, C: float = 0.1):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, X, y, X_val=None, y_val=None):
        self.pipe.fit(_log1p_counts(X), y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.pipe.predict_proba(_log1p_counts(X))[:, 1]


class MLPModel:
    name = "mlp"

    def __init__(self, seed: int = 0, hidden=(64, 32), alpha: float = 1e-2, max_iter: int = 200):
        self.pipe = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(hidden),
                alpha=alpha,
                batch_size=512,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=max_iter,
                random_state=seed,
            ),
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self.pipe.fit(_log1p_counts(X), y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.pipe.predict_proba(_log1p_counts(X))[:, 1]


# Original hand-set hyperparameters (v0.1.0 headline model).
BASE_XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
}


class XGBModel:
    name = "xgboost"

    def __init__(self, seed: int = 0, n_estimators: int | None = None,
                 params: dict | None = None):
        p = dict(BASE_XGB_PARAMS if params is None else params)
        if n_estimators is not None:
            p["n_estimators"] = n_estimators
        self.model = XGBClassifier(
            **p,
            eval_metric="logloss",
            early_stopping_rounds=100,
            tree_method="hist",
            n_jobs=2,
            random_state=seed,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None:
            raise ValueError("XGBModel needs a validation split for early stopping")
        self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Exact TreeSHAP contributions (log-odds scale) from xgboost's pred_contribs."""
        import xgboost as xgb

        booster = self.model.get_booster()
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        contrib = booster.predict(
            dm, pred_contribs=True, iteration_range=(0, self.model.best_iteration + 1)
        )
        return pd.DataFrame(contrib[:, :-1], columns=X.columns)


MODELS = {"logistic": LogisticModel, "xgboost": XGBModel, "mlp": MLPModel}


def tune_on_validation(cls, grid: list[dict], X_fit, y_fit, X_val, y_val, seed: int = 0,
                       **fixed) -> dict:
    """Grid search: fit each configuration on ``X_fit`` and score AUROC on ``X_val``.

    Only the validation split is used for scoring, so calibration and test data stay untouched.
    Ties go to the earlier grid point. Returns the best configuration and every score.
    """
    from sklearn.metrics import roc_auc_score

    if not grid:
        raise ValueError("empty tuning grid")
    results = []
    for params in grid:
        model = cls(seed=seed, **fixed, **params).fit(X_fit, y_fit, X_val, y_val)
        auc = float(roc_auc_score(y_val, model.predict_proba(X_val)))
        results.append({"params": dict(params), "val_auroc": auc})
    best = max(range(len(results)), key=lambda i: (results[i]["val_auroc"], -i))
    return {"best_params": results[best]["params"], "best_val_auroc": results[best]["val_auroc"],
            "grid": results}
