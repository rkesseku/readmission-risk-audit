import numpy as np
import pytest

from readmit.metrics import (
    auroc_stat,
    brier,
    calibration_slope_intercept,
    cluster_bootstrap,
    ece,
    net_benefit,
    net_benefit_treat_all,
    paired_cluster_bootstrap_diff,
    rates_at,
)


def _calibrated(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.6, n)
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_ece_small_when_calibrated_and_large_when_not():
    y, p = _calibrated()
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p + 0.2, 0, 1)) > 0.15


def test_calibration_slope_near_one():
    y, p = _calibrated()
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.1)
    assert intercept == pytest.approx(0.0, abs=0.1)


def test_brier_perfect():
    y = np.array([0, 1, 1, 0])
    assert brier(y, y.astype(float)) == 0.0


def test_net_benefit_formula():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0, 0, 0, 0, 0, 0, 0])
    # flagged at 0.2: one true positive, one false positive
    assert net_benefit(y, p, 0.2) == pytest.approx(0.1 - 0.1 * 0.25)
    assert net_benefit_treat_all(y, 0.2) == pytest.approx(0.2 - 0.8 * 0.25)
    assert net_benefit(y, p, 0.95) == 0.0


def test_rates_at():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.0])
    r = rates_at(y, p, 0.5)
    assert r == {"tpr": 0.5, "fpr": 0.5, "ppv": 0.5, "flag_rate": 0.5}


def test_cluster_bootstrap_ci_contains_estimate():
    y, p = _calibrated(4000)
    groups = np.arange(len(y)) // 2
    est, lo, hi = cluster_bootstrap(y, p, groups, auroc_stat, n_boot=100)
    assert lo <= est <= hi
    assert hi - lo < 0.1


def test_paired_diff_zero_for_identical_models():
    y, p = _calibrated(2000)
    d, lo, hi = paired_cluster_bootstrap_diff(y, p, p, np.arange(len(y)), auroc_stat, n_boot=50)
    assert d == 0.0 and lo == 0.0 and hi == 0.0
