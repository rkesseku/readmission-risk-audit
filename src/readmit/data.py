"""Loading, cleaning and featurising the UCI Diabetes 130-US Hospitals data.

Cleaning follows the conventions of Strack et al. (2014): encounters that ended in death or
hospice are removed (the patient cannot be readmitted), and the primary/secondary diagnoses are
mapped from ICD-9 codes to nine broad disease groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# discharge_disposition_id values meaning death or hospice (see IDS_mapping.csv)
EXPIRED_OR_HOSPICE = {11, 13, 14, 19, 20, 21}

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]

CATEGORICAL = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "payer_code", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed", "diag_1_group", "diag_2_group", "diag_3_group",
]

AGE_ORDER = [f"[{10 * i}-{10 * (i + 1)})" for i in range(10)]


def icd9_group(code: object) -> str:
    """Map an ICD-9 code string to one of the nine Strack et al. (2014) disease groups."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    s = str(code).strip()
    if s in {"", "?"}:
        return "Missing"
    if s[0] in {"V", "E"}:
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def age_band(age: str) -> str:
    """Collapse the ten-year age brackets into three bands used for the fairness audit."""
    lo = int(str(age).strip("[").split("-")[0])
    if lo < 50:
        return "<50"
    if lo < 70:
        return "50-69"
    return "70+"


def load_raw(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "diabetic_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the UCI 'Diabetes 130-US Hospitals' zip and unzip it into "
            "data/raw/ (see README)."
        )
    return pd.read_csv(path, low_memory=False)


HISTORY = [
    "prior_encounters", "prior_readmit30", "prior_readmit_any", "prior_days",
    "prior_inpatient_visits_max", "prior_emergency_max", "is_first_encounter",
]

ICD9_CHAPTERS = [
    (1, 139, "infectious"), (140, 239, "neoplasms"), (240, 279, "endocrine"),
    (280, 289, "blood"), (290, 319, "mental"), (320, 389, "nervous"), (390, 459, "circulatory"),
    (460, 519, "respiratory"), (520, 579, "digestive"), (580, 629, "genitourinary"),
    (630, 679, "pregnancy"), (680, 709, "skin"), (710, 739, "musculoskeletal"),
    (740, 759, "congenital"), (760, 779, "perinatal"), (780, 799, "symptoms"),
    (800, 999, "injury"),
]

# comorbidity flags over diag_1..3: name -> predicate on the numeric ICD-9 code
COMORBIDITIES = {
    "chf": lambda v: int(v) == 428,
    "ckd": lambda v: int(v) in (403, 404, 585, 586),
    "acute_renal": lambda v: int(v) == 584,
    "copd": lambda v: 490 <= v < 497,
    "pneumonia": lambda v: 480 <= v < 487,
    "sepsis": lambda v: int(v) in (38, 995),
    "ihd": lambda v: 410 <= v < 415,
    "arrhythmia": lambda v: int(v) == 427,
    "stroke": lambda v: 430 <= v < 439,
    "dm_complication": lambda v: int(v) == 250 and round((v - 250) * 100) >= 10,
    "mental": lambda v: 290 <= v < 320,
    "cancer": lambda v: 140 <= v < 209,
    "anemia": lambda v: 280 <= v < 286,
}

RICH_DIAG = [f"cm_{k}" for k in COMORBIDITIES] + ["n_chapters"]


def icd9_chapter(code: object) -> str:
    """Map an ICD-9 code to its chapter (17 numeric chapters plus V, E and Missing)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    s = str(code).strip()
    if s in {"", "?"}:
        return "Missing"
    if s[0] in {"V", "E"}:
        return s[0]
    try:
        v = float(s)
    except ValueError:
        return "Missing"
    for lo, hi, name in ICD9_CHAPTERS:
        if lo <= v < hi + 1:
            return name
    return "Missing"


def _numeric_code(code: object) -> float | None:
    s = str(code).strip()
    if not s or s[0] in "VE?" or s == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def comorbidity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary comorbidity flags over the three diagnosis codes, plus a distinct-chapter count."""
    codes = [df[f"diag_{k}"].map(_numeric_code) for k in (1, 2, 3)]
    out = {}
    for name, pred in COMORBIDITIES.items():
        hit = np.zeros(len(df), dtype=bool)
        for c in codes:
            hit |= c.map(lambda v, f=pred: v is not None and not pd.isna(v) and f(v)).to_numpy()
        out[f"cm_{name}"] = hit.astype(float)
    chapters = pd.concat([df[f"diag_{k}"].map(icd9_chapter) for k in (1, 2, 3)], axis=1)
    out["n_chapters"] = chapters.apply(lambda r: len(set(r) - {"Missing"}), axis=1).astype(float)
    return pd.DataFrame(out, index=df.index)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient history built ONLY from the patient's earlier encounters.

    ``encounter_id`` order is used as a time proxy (the public data has no dates). For each
    encounter, every feature is an aggregate over encounters of the same patient with a smaller
    ``encounter_id``; the encounter itself and later ones never contribute. The earlier
    encounters' readmission labels are known by the time of the current discharge (a prior
    "<30" label means the patient was already readmitted, possibly by this very admission).
    """
    out = df.sort_values(["patient_nbr", "encounter_id"]).copy()
    g = out.groupby("patient_nbr", sort=False)
    r30 = (out["readmitted"] == "<30").astype(float)
    rany = (out["readmitted"] != "NO").astype(float)
    out["prior_encounters"] = g.cumcount().astype(float)
    out["prior_readmit30"] = r30.groupby(out["patient_nbr"]).cumsum() - r30
    out["prior_readmit_any"] = rany.groupby(out["patient_nbr"]).cumsum() - rany
    days = out["time_in_hospital"].astype(float)
    out["prior_days"] = days.groupby(out["patient_nbr"]).cumsum() - days
    for src, dst in (("number_inpatient", "prior_inpatient_visits_max"),
                     ("number_emergency", "prior_emergency_max")):
        shifted = g[src].shift(1)
        out[dst] = shifted.groupby(out["patient_nbr"]).cummax().fillna(0.0).astype(float)
    out["is_first_encounter"] = (out["prior_encounters"] == 0).astype(float)
    return out.sort_index()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with a binary ``y`` (readmitted within 30 days)."""
    out = add_history_features(df)
    out = out[~out["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)]
    out = out[out["gender"].isin(["Female", "Male"])]
    for k in (1, 2, 3):
        out[f"diag_{k}_group"] = out[f"diag_{k}"].map(icd9_group)
        out[f"diag_{k}_chapter"] = out[f"diag_{k}"].map(icd9_chapter)
    out = pd.concat([out, comorbidity_flags(out)], axis=1)
    out["race"] = out["race"].replace("?", "Unknown")
    out["medical_specialty"] = out["medical_specialty"].replace("?", "Unknown")
    out["payer_code"] = out["payer_code"].replace("?", "Unknown")
    for col in ("max_glu_serum", "A1Cresult"):
        out[col] = out[col].fillna("None")
    out["age_band"] = out["age"].map(age_band)
    out["y"] = (out["readmitted"] == "<30").astype(int)
    return out.reset_index(drop=True)


def _safe(name: str) -> str:
    """xgboost rejects feature names containing '[', ']' or '<'."""
    return (str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt"))


@dataclass
class Featurizer:
    """One-hot encoder whose columns are fixed on the training split."""

    min_count: int = 20
    history: bool = False
    rich_diag: bool = False
    columns_: list[str] | None = None
    keep_levels_: dict[str, set] | None = None

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [df[NUMERIC].astype(float)]
        for m in MEDICATIONS:
            if m in df:
                parts.append((df[m] != "No").astype(float).rename(f"med_{m}"))
        parts.append(
            df["age"].map({a: i for i, a in enumerate(AGE_ORDER)}).astype(float).rename("age_ord")
        )
        cats = {}
        for c in CATEGORICAL:
            vals = df[c].astype(str)
            if self.keep_levels_ is not None:
                vals = vals.where(vals.isin(self.keep_levels_[c]), "Rare")
            cats[c] = vals
        if self.history:
            parts.append(df[HISTORY].astype(float))
        if self.rich_diag:
            parts.append(df[RICH_DIAG].astype(float))
            for k in (1, 2, 3):
                cats[f"diag_{k}_chapter"] = df[f"diag_{k}_chapter"].astype(str)
        dummies = pd.get_dummies(pd.DataFrame(cats), prefix_sep="=", dtype=float)
        out = pd.concat(parts + [dummies], axis=1)
        out.columns = [_safe(c) for c in out.columns]
        return out

    def fit(self, df: pd.DataFrame) -> Featurizer:
        self.keep_levels_ = {}
        for c in CATEGORICAL:
            vc = df[c].astype(str).value_counts()
            self.keep_levels_[c] = set(vc[vc >= self.min_count].index)
        self.columns_ = list(self._frame(df).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("Featurizer is not fitted")
        return self._frame(df).reindex(columns=self.columns_, fill_value=0.0)


def patient_split(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    group_col: str = "patient_nbr",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / calibration / test so that no patient appears in two splits."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    patients = df[group_col].unique()
    rng.shuffle(patients)
    n = len(patients)
    a = int(round(fractions[0] * n))
    b = a + int(round(fractions[1] * n))
    groups = [set(patients[:a]), set(patients[a:b]), set(patients[b:])]
    return tuple(df[df[group_col].isin(g)].reset_index(drop=True) for g in groups)  # type: ignore[return-value]
