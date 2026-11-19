import pandas as pd
import pytest

from readmit.data import Featurizer, age_band, clean, icd9_group, patient_split


@pytest.mark.parametrize(
    "code,group",
    [("428", "Circulatory"), ("785", "Circulatory"), ("486", "Respiratory"),
     ("250.83", "Diabetes"), ("820", "Injury"), ("715", "Musculoskeletal"),
     ("599", "Genitourinary"), ("174", "Neoplasms"), ("V57", "Other"), ("E888", "Other"),
     ("?", "Missing"), (None, "Missing"), ("530", "Digestive"), ("300", "Other")],
)
def test_icd9_group(code, group):
    assert icd9_group(code) == group


def test_age_band():
    assert age_band("[40-50)") == "<50"
    assert age_band("[60-70)") == "50-69"
    assert age_band("[90-100)") == "70+"


def test_clean_drops_expired_and_builds_target(raw):
    df = clean(raw)
    assert not df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21]).any()
    assert set(df["y"].unique()) <= {0, 1}
    assert (df["y"] == (df["readmitted"] == "<30")).all()
    assert "?" not in set(df["race"])


def test_patient_split_has_no_leakage(raw):
    df = clean(raw)
    tr, ca, te = patient_split(df, seed=3)
    assert len(tr) + len(ca) + len(te) == len(df)
    s = [set(x["patient_nbr"]) for x in (tr, ca, te)]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])


def test_patient_split_rejects_bad_fractions(raw):
    with pytest.raises(ValueError):
        patient_split(clean(raw), (0.5, 0.2, 0.2))


def test_featurizer_fixed_columns_and_safe_names(raw):
    df = clean(raw)
    tr, _, te = patient_split(df, seed=0)
    f = Featurizer(min_count=5).fit(tr)
    Xtr, Xte = f.transform(tr), f.transform(te)
    assert list(Xtr.columns) == list(Xte.columns)
    bad = set("[]<")
    assert not any(bad & set(c) for c in Xtr.columns)
    unseen = te.head(3).copy()
    unseen["medical_specialty"] = "NeverSeenBefore"
    assert list(f.transform(unseen).columns) == list(Xtr.columns)
    assert isinstance(Xte, pd.DataFrame) and Xte.notna().all().all()


def test_featurizer_requires_fit(raw):
    with pytest.raises(RuntimeError):
        Featurizer().transform(clean(raw))
