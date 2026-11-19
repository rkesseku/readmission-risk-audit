"""A small synthetic table with the same schema as diabetic_data.csv, for tests and demos.

The values are random and carry no clinical meaning; a weak signal is planted so that models
can learn something and the pipeline can be exercised end to end without the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AGE_ORDER, MEDICATIONS


def make_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_patients = max(n * 3 // 4, 1)
    patient = rng.integers(0, n_patients, n) + 1000
    inpatient = rng.poisson(0.6, n)
    emergency = rng.poisson(0.2, n)
    los = rng.integers(1, 15, n)
    age = rng.choice(AGE_ORDER[3:], n)
    diag_pool = np.array(["428", "250.02", "486", "414", "V57", "820", "715", "599", "174", "?"])
    logit = -2.4 + 0.45 * inpatient + 0.2 * emergency + 0.03 * los
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    readmitted = np.where(y, "<30", rng.choice(["NO", ">30"], n))
    df = pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": patient,
            "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "?"], n,
                               p=[0.7, 0.2, 0.05, 0.05]),
            "gender": rng.choice(["Female", "Male"], n),
            "age": age,
            "weight": "?",
            "admission_type_id": rng.choice([1, 2, 3], n),
            "discharge_disposition_id": rng.choice([1, 3, 6, 11], n, p=[0.6, 0.2, 0.17, 0.03]),
            "admission_source_id": rng.choice([1, 7], n),
            "time_in_hospital": los,
            "payer_code": rng.choice(["MC", "?"], n),
            "medical_specialty": rng.choice(["InternalMedicine", "?", "Cardiology"], n),
            "num_lab_procedures": rng.integers(1, 100, n),
            "num_procedures": rng.integers(0, 6, n),
            "num_medications": rng.integers(1, 40, n),
            "number_outpatient": rng.poisson(0.3, n),
            "number_emergency": emergency,
            "number_inpatient": inpatient,
            "diag_1": rng.choice(diag_pool, n),
            "diag_2": rng.choice(diag_pool, n),
            "diag_3": rng.choice(diag_pool, n),
            "number_diagnoses": rng.integers(1, 16, n),
            "max_glu_serum": rng.choice(["None", ">300", "Norm"], n),
            "A1Cresult": rng.choice(["None", ">8", "Norm"], n),
            "change": rng.choice(["No", "Ch"], n),
            "diabetesMed": rng.choice(["Yes", "No"], n),
            "readmitted": readmitted,
        }
    )
    for m in MEDICATIONS:
        df[m] = rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.85, 0.1, 0.03, 0.02])
    return df
