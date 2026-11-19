import numpy as np
import pytest

from readmit.calibration import CALIBRATORS, VennAbersCalibrator
from readmit.data import clean
from readmit.fairness import max_gap, subgroup_audit
from readmit.metrics import ece


def _miscalibrated(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.02, 0.5, n)
    y = (rng.random(n) < true_p).astype(int)
    score = true_p**0.4  # monotone but badly calibrated
    return score, y


@pytest.mark.parametrize("name", list(CALIBRATORS))
def test_calibrators_reduce_ece(name):
    s, y = _miscalibrated()
    cal = CALIBRATORS[name]().fit(s[:3000], y[:3000])
    p = cal.predict(s[3000:])
    assert np.all((p >= 0) & (p <= 1))
    assert ece(y[3000:], p) < 0.5 * ece(y[3000:], s[3000:])


def test_venn_abers_interval_is_ordered():
    s, y = _miscalibrated()
    va = VennAbersCalibrator(n_grid=50).fit(s[:3000], y[:3000])
    p0, p1 = va.predict_interval(s[3000:])
    assert np.all(p0 <= p1 + 1e-12)
    p = va.predict(s[3000:])
    assert np.all((p >= p0 - 1e-9) & (p <= p1 + 1e-9))


def test_subgroup_audit_structure(raw):
    df = clean(raw)
    rng = np.random.default_rng(0)
    p = np.clip(0.1 + 0.05 * df["number_inpatient"].to_numpy() + rng.normal(0, 0.02, len(df)),
                0.001, 0.999)
    rows = subgroup_audit(df, p, n_boot=30, min_n=100)
    assert {r["attribute"] for r in rows} == {"age_band", "gender", "race"}
    full = [r for r in rows if "tpr" in r]
    assert full
    for r in full:
        assert r["tpr_ci"][0] <= r["tpr"] <= r["tpr_ci"][1]
        assert 0 <= r["fpr"] <= 1
    assert all(r.get("insufficient_n") for r in rows if r["n"] < 100)
    assert max_gap(rows, "gender", "tpr") is not None
