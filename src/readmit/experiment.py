"""Bounded AUROC-improvement experiment for the XGBoost model.

Steps, applied in order, each building on the best configuration so far:

1. patient-history features from earlier encounters only;
2. richer diagnosis encoding (ICD-9 chapters and comorbidity flags);
3. Optuna tuning of XGBoost hyperparameters.

Selection never touches the test split. A step is kept only if it improves AUROC on BOTH the
early-stopping validation split and the calibration split (pre-specified rule). Test AUROC is
reported for every step, with a paired patient-clustered bootstrap CI against the original
model, purely for transparency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Featurizer, clean, load_raw, patient_split
from .metrics import auroc_stat, paired_cluster_bootstrap_diff
from .models import BASE_XGB_PARAMS, XGBModel


def _fit_eval(splits, history: bool, rich_diag: bool, params: dict, seed: int):
    fit_df, val_df, calib, test = splits
    feat = Featurizer(history=history, rich_diag=rich_diag).fit(fit_df)
    model = XGBModel(seed=seed, params=params).fit(
        feat.transform(fit_df), fit_df["y"].to_numpy(),
        feat.transform(val_df), val_df["y"].to_numpy(),
    )
    preds = {
        name: model.predict_proba(feat.transform(d))
        for name, d in (("val", val_df), ("calib", calib), ("test", test))
    }
    aucs = {
        name: auroc_stat(d["y"].to_numpy(), preds[name])
        for name, d in (("val", val_df), ("calib", calib), ("test", test))
    }
    return aucs, preds, model, len(feat.columns_ or [])


def tune_xgb(splits, history: bool, rich_diag: bool, n_trials: int, seed: int) -> dict:
    """Optuna TPE search maximising validation AUROC (test split never used)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fit_df, val_df, _, _ = splits
    feat = Featurizer(history=history, rich_diag=rich_diag).fit(fit_df)
    Xf, Xv = feat.transform(fit_df), feat.transform(val_df)
    yf, yv = fit_df["y"].to_numpy(), val_df["y"].to_numpy()

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        m = XGBModel(seed=seed, params=params).fit(Xf, yf, Xv, yv)
        return auroc_stat(yv, m.predict_proba(Xv))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return {"best_params": study.best_params, "best_val_auroc": study.best_value,
            "n_trials": n_trials}


def run_experiment(raw: pd.DataFrame, seed: int = 0, n_trials: int = 50, n_boot: int = 500,
                   fast: bool = False) -> dict:
    t0 = time.time()
    df = clean(raw)
    train, calib, test = patient_split(df, (0.6, 0.2, 0.2), seed=seed)
    fit_df, val_df = patient_split(train, (0.85, 0.15, 0.0), seed=seed + 1)[:2]
    splits = (fit_df, val_df, calib, test)
    y_test, g_test = test["y"].to_numpy(), test["patient_nbr"].to_numpy()
    base_params = dict(BASE_XGB_PARAMS)
    if fast:
        base_params["n_estimators"] = 150

    steps = []
    base_auc, base_pred, _, nf = _fit_eval(splits, False, False, base_params, seed)
    steps.append({"step": "0_original", "history": False, "rich_diag": False,
                  "params": "original", "auroc": base_auc, "n_features": nf, "kept": True})
    best = {"history": False, "rich_diag": False, "params": base_params, "auc": base_auc}

    def record(name, history, rich, params, label):
        auc, pred, _, nf = _fit_eval(splits, history, rich, params, seed)
        d, lo, hi = paired_cluster_bootstrap_diff(y_test, pred["test"], base_pred["test"],
                                                  g_test, auroc_stat, n_boot=n_boot, seed=seed)
        kept = auc["val"] > best["auc"]["val"] and auc["calib"] > best["auc"]["calib"]
        steps.append({"step": name, "history": history, "rich_diag": rich, "params": label,
                      "auroc": auc, "n_features": nf,
                      "test_auroc_diff_vs_original": {"diff": d, "ci": [lo, hi]}, "kept": kept})
        if kept:
            best.update(history=history, rich_diag=rich, params=params, auc=auc)

    record("1_history", True, False, base_params, "original")
    record("2_rich_diagnoses", best["history"], True, base_params, "original")
    tuning = tune_xgb(splits, best["history"], best["rich_diag"], n_trials, seed)
    tuned = {**base_params, **tuning["best_params"]}
    record("3_optuna", best["history"], best["rich_diag"], tuned, "optuna")

    return {
        "selection_rule": "keep a step only if AUROC improves on both the validation and the "
                          "calibration split; the test split is never used for selection",
        "seed": seed,
        "steps": steps,
        "optuna": tuning,
        "final": {"history": best["history"], "rich_diag": best["rich_diag"],
                  "xgb_params": best["params"]},
        "runtime_seconds": round(time.time() - t0, 1),
    }


def main_experiment(data_dir, out_dir, seed=0, n_trials=50, n_boot=500, raw=None, fast=False):
    if raw is None:
        raw = load_raw(data_dir)
    res = run_experiment(raw, seed=seed, n_trials=n_trials, n_boot=n_boot, fast=fast)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.json").write_text(json.dumps(res, indent=2, default=_jsonable))
    return res


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))
