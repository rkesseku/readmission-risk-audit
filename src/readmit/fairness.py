"""Subgroup audit: is the model equally well calibrated and equally sensitive across groups?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import cluster_bootstrap, ece, rates_at


def _oe_ratio(y, p):
    return float(np.mean(y) / max(np.mean(p), 1e-9))


def subgroup_audit(
    df: pd.DataFrame,
    p: np.ndarray,
    attrs: tuple[str, ...] = ("age_band", "gender", "race"),
    threshold: float = 0.15,
    min_n: int = 200,
    n_boot: int = 300,
    seed: int = 0,
) -> list[dict]:
    """Per-group prevalence, mean prediction, O/E ratio, ECE and TPR/FPR at ``threshold``.

    CIs come from a patient-clustered bootstrap within each group. Groups smaller than
    ``min_n`` encounters are reported with ``insufficient_n`` and no estimates.
    """
    y = df["y"].to_numpy()
    groups = df["patient_nbr"].to_numpy()
    p = np.asarray(p, float)
    rows: list[dict] = []
    for attr in attrs:
        for level, idx in df.groupby(attr).indices.items():
            row = {"attribute": attr, "group": str(level), "n": int(len(idx))}
            if len(idx) < min_n or y[idx].min() == y[idx].max():
                row["insufficient_n"] = True
                rows.append(row)
                continue
            yy, pp, gg = y[idx], p[idx], groups[idx]
            r = rates_at(yy, pp, threshold)

            def tpr(a, b):
                return rates_at(a, b, threshold)["tpr"]

            def fpr(a, b):
                return rates_at(a, b, threshold)["fpr"]

            row.update(
                prevalence=float(yy.mean()),
                mean_pred=float(pp.mean()),
                flag_rate=r["flag_rate"],
                ppv=r["ppv"],
            )
            for name, fn in (("oe_ratio", _oe_ratio), ("ece", ece), ("tpr", tpr), ("fpr", fpr)):
                est, lo, hi = cluster_bootstrap(yy, pp, gg, fn, n_boot=n_boot, seed=seed)
                row[name] = est
                row[f"{name}_ci"] = [lo, hi]
            rows.append(row)
    return rows


def max_gap(rows: list[dict], attr: str, key: str) -> float | None:
    vals = [r[key] for r in rows if r["attribute"] == attr and key in r]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else None
