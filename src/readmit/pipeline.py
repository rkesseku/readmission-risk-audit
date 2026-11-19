"""End-to-end study: split -> train -> calibrate -> evaluate -> decision curve -> audit."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .calibration import CALIBRATORS
from .config import FINAL_HISTORY, FINAL_RICH_DIAG, FINAL_XGB_PARAMS
from .data import Featurizer, clean, load_raw, patient_split
from .fairness import max_gap, subgroup_audit
from .metrics import (
    auprc_stat,
    auroc_stat,
    brier,
    cluster_bootstrap,
    ece,
    net_benefit,
    net_benefit_treat_all,
    paired_cluster_bootstrap_diff,
    point_metrics,
    rates_at,
)
from .models import MODELS

DEFAULT_THRESHOLD = 0.15
DCA_THRESHOLDS = np.round(np.arange(0.05, 0.41, 0.01), 2)


def run_study(
    raw: pd.DataFrame,
    seed: int = 0,
    n_boot: int = 500,
    threshold: float = DEFAULT_THRESHOLD,
    fast: bool = False,
) -> tuple[dict, dict]:
    """Run the full study on a raw dataframe. Returns (metrics, artefacts-for-plotting)."""
    t0 = time.time()
    df = clean(raw)
    train, calib, test = patient_split(df, (0.6, 0.2, 0.2), seed=seed)
    fit_df, val_df = patient_split(train, (0.85, 0.15, 0.0), seed=seed + 1)[:2]

    feat = Featurizer(min_count=5 if fast else 20, history=FINAL_HISTORY,
                      rich_diag=FINAL_RICH_DIAG).fit(fit_df)
    X_fit, X_val = feat.transform(fit_df), feat.transform(val_df)
    X_cal, X_test = feat.transform(calib), feat.transform(test)
    y_fit, y_val = fit_df["y"].to_numpy(), val_df["y"].to_numpy()
    y_cal, y_test = calib["y"].to_numpy(), test["y"].to_numpy()
    g_test = test["patient_nbr"].to_numpy()

    raw_scores: dict[str, dict[str, np.ndarray]] = {}
    fitted = {}
    for name, cls in MODELS.items():
        kwargs = {"seed": seed}
        if name == "xgboost":
            kwargs["params"] = FINAL_XGB_PARAMS
        if fast and name == "xgboost":
            kwargs["n_estimators"] = 200
        if fast and name == "mlp":
            kwargs["max_iter"] = 30
        model = cls(**kwargs).fit(X_fit, y_fit, X_val, y_val)
        fitted[name] = model
        raw_scores[name] = {"cal": model.predict_proba(X_cal), "test": model.predict_proba(X_test)}

    # Model selection on the calibration split only (never on test).
    cal_auroc = {m: auroc_stat(y_cal, s["cal"]) for m, s in raw_scores.items()}
    best_model = max(cal_auroc, key=cal_auroc.get)

    preds: dict[str, dict[str, np.ndarray]] = {}
    grid: dict[str, dict[str, dict]] = {}
    for m, s in raw_scores.items():
        preds[m] = {"uncalibrated": s["test"]}
        for cname, ccls in CALIBRATORS.items():
            preds[m][cname] = ccls().fit(s["cal"], y_cal).predict(s["test"])
        grid[m] = {}
        for cname, p in preds[m].items():
            row = point_metrics(y_test, p)
            for key, fn in (("auroc", auroc_stat), ("auprc", auprc_stat), ("brier", brier),
                            ("ece", ece)):
                _, lo, hi = cluster_bootstrap(y_test, p, g_test, fn, n_boot=n_boot, seed=seed)
                row[f"{key}_ci"] = [lo, hi]
            grid[m][cname] = row

    headline_cal = "venn_abers"  # pre-specified before looking at test results
    p_head = preds[best_model][headline_cal]

    comparisons = {}
    for other in MODELS:
        if other == best_model:
            continue
        d, lo, hi = paired_cluster_bootstrap_diff(
            y_test, preds[best_model]["uncalibrated"], preds[other]["uncalibrated"], g_test,
            auroc_stat, n_boot=n_boot, seed=seed,
        )
        comparisons[f"auroc_{best_model}_minus_{other}"] = {"diff": d, "ci": [lo, hi]}
    d, lo, hi = paired_cluster_bootstrap_diff(
        y_test, preds[best_model]["uncalibrated"], p_head, g_test, ece, n_boot=n_boot, seed=seed
    )
    comparisons[f"ece_uncalibrated_minus_{headline_cal}"] = {"diff": d, "ci": [lo, hi]}

    dca = {
        "thresholds": DCA_THRESHOLDS.tolist(),
        "treat_all": [net_benefit_treat_all(y_test, t) for t in DCA_THRESHOLDS],
        "treat_none": [0.0] * len(DCA_THRESHOLDS),
    }
    for m in MODELS:
        dca[m] = [net_benefit(y_test, preds[m][headline_cal], t) for t in DCA_THRESHOLDS]
    nb_model, nb_lo, nb_hi = cluster_bootstrap(
        y_test, p_head, g_test, lambda a, b: net_benefit(a, b, threshold), n_boot=n_boot, seed=seed
    )

    audit = subgroup_audit(test, p_head, threshold=threshold, n_boot=max(n_boot // 2, 50),
                           seed=seed, min_n=50 if fast else 200)

    importance = None
    if "xgboost" in fitted:
        sample = X_test.sample(min(5000, len(X_test)), random_state=seed)
        contrib = fitted["xgboost"].contributions(sample).abs().mean()
        parent = contrib.groupby([c.split("=")[0] for c in contrib.index]).sum()
        importance = parent.sort_values(ascending=False).head(15).round(5).to_dict()

    metrics = {
        "version": __version__,
        "seed": seed,
        "n_boot": n_boot,
        "data": {
            "raw_encounters": int(len(raw)),
            "encounters_after_cleaning": int(len(df)),
            "patients_after_cleaning": int(df["patient_nbr"].nunique()),
            "splits": {
                k: {"encounters": int(len(v)), "patients": int(v["patient_nbr"].nunique()),
                    "prevalence": float(v["y"].mean())}
                for k, v in (("train_fit", fit_df), ("train_val", val_df), ("calibration", calib),
                             ("test", test))
            },
            "n_features": len(feat.columns_ or []),
        },
        "calibration_split_auroc": cal_auroc,
        "selected_model": best_model,
        "feature_config": {"history": FINAL_HISTORY, "rich_diag": FINAL_RICH_DIAG,
                           "xgb_params": FINAL_XGB_PARAMS},
        "headline_calibrator": headline_cal,
        "test_metrics": grid,
        "comparisons": comparisons,
        "decision_threshold": threshold,
        "at_threshold": {
            **rates_at(y_test, p_head, threshold),
            "net_benefit": nb_model,
            "net_benefit_ci": [nb_lo, nb_hi],
            "net_benefit_treat_all": net_benefit_treat_all(y_test, threshold),
        },
        "decision_curve": dca,
        "subgroup_audit": audit,
        "subgroup_max_gaps": {
            a: {k: max_gap(audit, a, k) for k in ("tpr", "oe_ratio", "ece")}
            for a in ("age_band", "gender", "race")
        },
        "xgboost_mean_abs_shap_top15": importance,
        "runtime_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
    }
    artefacts = {"y_test": y_test, "preds": preds, "best_model": best_model}
    return metrics, artefacts


def run_all(data_dir: str | Path, out_dir: str | Path, seed: int = 0, n_boot: int = 500,
            threshold: float = DEFAULT_THRESHOLD, fast: bool = False,
            raw: pd.DataFrame | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if raw is None:
        raw = load_raw(data_dir)
    metrics, art = run_study(raw, seed=seed, n_boot=n_boot, threshold=threshold, fast=fast)
    exp_path = out / "experiment.json"
    if exp_path.exists():  # summary of the model-improvement experiment that chose the config
        exp = json.loads(exp_path.read_text())
        metrics["experiment"] = {
            "selection_rule": exp["selection_rule"],
            "steps": [{k: s.get(k) for k in ("step", "auroc", "kept",
                                              "test_auroc_diff_vs_original")}
                      for s in exp["steps"]],
        }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    from .plots import make_all_figures

    make_all_figures(metrics, art, out)
    return metrics
