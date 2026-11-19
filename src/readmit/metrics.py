"""Discrimination, calibration and utility metrics, with patient-clustered bootstrap CIs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-mass (quantile) bins."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, n_bins)
    total = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        total += len(b) * abs(p[b].mean() - y[b].mean())
    return float(total / len(y))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration: regress y on logit(p). Ideal slope 1, intercept 0."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    bins = [b for b in np.array_split(order, n_bins) if len(b)]
    return np.array([p[b].mean() for b in bins]), np.array([y[b].mean() for b in bins])


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """Net benefit of treating patients with predicted risk >= threshold (Vickers & Elkin 2006)."""
    y = np.asarray(y, int)
    flag = np.asarray(p) >= threshold
    n = len(y)
    tp = np.sum(flag & (y == 1)) / n
    fp = np.sum(flag & (y == 0)) / n
    return float(tp - fp * threshold / (1 - threshold))


def net_benefit_treat_all(y: np.ndarray, threshold: float) -> float:
    prev = float(np.mean(y))
    return prev - (1 - prev) * threshold / (1 - threshold)


def rates_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y, int)
    flag = np.asarray(p) >= threshold
    pos = max(int(np.sum(y == 1)), 1)
    neg = max(int(np.sum(y == 0)), 1)
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    return {
        "tpr": tp / pos,
        "fpr": fp / neg,
        "ppv": tp / max(tp + fp, 1),
        "flag_rate": float(np.mean(flag)),
    }


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "cal_slope": slope,
        "cal_intercept": intercept,
        "mean_pred": float(np.mean(p)),
        "prevalence": float(np.mean(y)),
    }


def _resample(rng, members, starts, counts) -> np.ndarray:
    """Indices of one cluster bootstrap draw (vectorised)."""
    pick = rng.integers(0, len(counts), len(counts))
    c = counts[pick]
    base = np.repeat(starts[pick], c)
    off = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
    return members[base + off]


def cluster_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    stat: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling patients (clusters) with replacement."""
    y = np.asarray(y)
    p = np.asarray(p)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = np.argsort(inv, kind="mergesort")
    counts = np.bincount(inv, minlength=len(uniq))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rng = np.random.default_rng(seed)
    est = stat(y, p)
    vals = []
    for _ in range(n_boot):
        idx = _resample(rng, members, starts, counts)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        vals.append(stat(yb, pb))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(est), float(lo), float(hi)


def paired_cluster_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    stat: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI for stat(p_a) - stat(p_b) on the same patients."""
    stacked = np.column_stack([p_a, p_b])

    def diff(yy, pp):
        return stat(yy, pp[:, 0]) - stat(yy, pp[:, 1])

    y = np.asarray(y)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = np.argsort(inv, kind="mergesort")
    counts = np.bincount(inv, minlength=len(uniq))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = _resample(rng, members, starts, counts)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(diff(y[idx], stacked[idx]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(diff(y, stacked)), float(lo), float(hi)


def auroc_stat(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def auprc_stat(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))
