import numpy as np
import pandas as pd
import pytest

from readmit.data import (
    HISTORY,
    RICH_DIAG,
    Featurizer,
    add_history_features,
    clean,
    comorbidity_flags,
    icd9_chapter,
    patient_split,
)
from readmit.experiment import run_experiment


def _toy():
    # patient 1 has three encounters (ids 10, 20, 30); patient 2 has one
    return pd.DataFrame(
        {
            "encounter_id": [30, 10, 20, 15],
            "patient_nbr": [1, 1, 1, 2],
            "readmitted": ["NO", "<30", ">30", "<30"],
            "time_in_hospital": [4, 2, 3, 7],
            "number_inpatient": [5, 0, 2, 1],
            "number_emergency": [0, 1, 3, 0],
        }
    )


def test_history_values_on_toy():
    h = add_history_features(_toy()).set_index("encounter_id")
    assert h.loc[10, "prior_encounters"] == 0 and h.loc[10, "is_first_encounter"] == 1
    assert h.loc[20, "prior_encounters"] == 1 and h.loc[30, "prior_encounters"] == 2
    assert h.loc[10, "prior_readmit30"] == 0
    assert h.loc[20, "prior_readmit30"] == 1 and h.loc[30, "prior_readmit30"] == 1
    assert h.loc[30, "prior_readmit_any"] == 2
    assert h.loc[30, "prior_days"] == 5 and h.loc[20, "prior_days"] == 2
    assert h.loc[30, "prior_inpatient_visits_max"] == 2
    assert h.loc[30, "prior_emergency_max"] == 3
    assert (h.loc[15, HISTORY].drop("is_first_encounter") == 0).all()


def test_history_keeps_row_order():
    df = _toy()
    h = add_history_features(df)
    assert list(h["encounter_id"]) == list(df["encounter_id"])


def test_history_never_uses_later_encounters(raw):
    """Changing every later encounter of a patient must not change any earlier feature."""
    base = add_history_features(raw)
    pid = raw["patient_nbr"].value_counts().index[0]
    rows = raw[raw["patient_nbr"] == pid].sort_values("encounter_id")
    assert len(rows) >= 2
    cutoff = rows["encounter_id"].iloc[0]
    tampered = raw.copy()
    later = (tampered["patient_nbr"] == pid) & (tampered["encounter_id"] > cutoff)
    tampered.loc[later, "readmitted"] = "<30"
    tampered.loc[later, "time_in_hospital"] = 99
    tampered.loc[later, "number_inpatient"] = 50
    tampered.loc[later, "number_emergency"] = 50
    after = add_history_features(tampered)
    keep = ~later
    pd.testing.assert_frame_equal(base.loc[keep, HISTORY], after.loc[keep, HISTORY])


def test_history_matches_brute_force(raw):
    h = add_history_features(raw)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(raw), 50, replace=False):
        r = raw.iloc[i]
        earlier = raw[(raw["patient_nbr"] == r["patient_nbr"])
                      & (raw["encounter_id"] < r["encounter_id"])]
        assert h.iloc[i]["prior_encounters"] == len(earlier)
        assert h.iloc[i]["prior_readmit30"] == (earlier["readmitted"] == "<30").sum()
        assert h.iloc[i]["prior_days"] == earlier["time_in_hospital"].sum()


@pytest.mark.parametrize(
    "code,chapter",
    [("428", "circulatory"), ("250.83", "endocrine"), ("038", "infectious"), ("V45", "V"),
     ("E878", "E"), ("?", "Missing"), ("996", "injury"), ("780", "symptoms")],
)
def test_icd9_chapter(code, chapter):
    assert icd9_chapter(code) == chapter


def test_comorbidity_flags():
    df = pd.DataFrame({"diag_1": ["428", "250.43", "V57"], "diag_2": ["585", "491", "?"],
                       "diag_3": ["427", "250.01", "E888"]})
    f = comorbidity_flags(df)
    assert list(f.columns) == RICH_DIAG
    assert f.loc[0, ["cm_chf", "cm_ckd", "cm_arrhythmia"]].tolist() == [1, 1, 1]
    assert f.loc[1, "cm_dm_complication"] == 1 and f.loc[1, "cm_copd"] == 1
    assert f.loc[2].drop("n_chapters").sum() == 0
    assert f["n_chapters"].tolist() == [2, 2, 2]  # V and E codes count as chapters


def test_featurizer_optional_blocks(raw):
    tr, _, te = patient_split(clean(raw), seed=0)
    plain = Featurizer(min_count=5).fit(tr)
    full = Featurizer(min_count=5, history=True, rich_diag=True).fit(tr)
    cols = set(full.transform(te).columns)
    assert set(HISTORY) <= cols and set(RICH_DIAG) <= cols
    assert not set(HISTORY) & set(plain.transform(te).columns)
    assert any(c.startswith("diag_1_chapter=") for c in cols)


def test_experiment_runs_and_never_selects_on_test(raw):
    res = run_experiment(raw, n_trials=2, n_boot=10, fast=True)
    assert [s["step"] for s in res["steps"]] == [
        "0_original", "1_history", "2_rich_diagnoses", "3_optuna"]
    assert "test split is never used for selection" in res["selection_rule"]
    for s in res["steps"][1:]:
        a = s["auroc"]
        prev = [p for p in res["steps"][: res["steps"].index(s)] if p["kept"]][-1]["auroc"]
        b = a["val"] > prev["val"] and a["calib"] > prev["calib"]
        assert s["kept"] == b
