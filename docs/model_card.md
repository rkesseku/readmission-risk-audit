# Model card: 30-day readmission risk (XGBoost)

Format adapted from Mitchell et al., 2019, "Model Cards for Model Reporting". All figures are from
`docs/results/metrics.json` and `docs/results/experiment.json` (seed 0).

## Model details

* **Type:** gradient-boosted trees (XGBoost, `hist`, early stopping on a held-out 15% of training
  patients). Hyperparameters come from a 50-trial Optuna search on that validation split (depth 7,
  learning rate 0.037, subsample 0.60, column sample 0.48, gamma 3.76; full set in `src/readmit/config.py`).
  The model was selected over L2 logistic regression and a 2-layer MLP by AUROC on the calibration split.
* **Inputs:** 257 features covering encounter utilisation, medications, demographics, admission and discharge
  categories, diagnosis groups, ICD-9 chapters and comorbidity flags, and patient-history features built only
  from the patient's earlier encounters.
* **Output:** probability of readmission within 30 days of discharge. The raw output is the recommended score
  (see "Calibration" below); Venn-Abers-calibrated output is also produced.
* **Version:** 0.2.0 (v0.1.0 had no history or comorbidity features and hand-set hyperparameters).
  Author: Reagan Kesseku. Licence: MIT.

## Intended use

* **Primary use:** a portfolio demonstration of how a readmission model should be evaluated: ranking
  discharged inpatients with diabetes for a **low-cost, low-harm** follow-up (a phone call or an early clinic visit).
* **Out of scope:** any real clinical decision; denying or rationing care; use outside adult inpatients
  with diabetes; use at any hospital without local validation.

## Training data

UCI Diabetes 130-US Hospitals 1999–2008 (Strack et al., 2014), CC BY 4.0. After removing death and hospice
discharges: 99,340 encounters from 69,987 patients. Split by patient into train 59,521 (of which 9,079 are held
out for early stopping and tuning), calibration 19,875 and test 19,944 encounters.

**History features and the time proxy.** The public data has no dates, so `encounter_id` order is used as the
time order within each patient. Each history feature aggregates only encounters of the same patient with a
smaller id, and a unit test checks that changing later encounters never changes earlier features. In deployment
these features must be computed from the patient's actual EHR history as of the discharge.

## Metrics (test split, 95% patient-clustered bootstrap CIs)

| Metric | Value |
|---|---|
| AUROC | 0.688 (0.675–0.703); v0.1.0 0.681, paired Δ +0.008 (+0.003 to +0.014) |
| AUPRC (prevalence 0.111) | 0.241 |
| Brier score | 0.0926 |
| ECE, raw / Venn-Abers | 0.0050 / 0.0076 |
| Calibration slope / intercept, raw | 1.05 / 0.06 |
| At threshold 0.15 (Venn-Abers): flag rate, TPR, PPV | 24.0%, 47.0%, 21.6% |
| Net benefit at 0.15 vs. flag everyone | 0.0187 (0.0147–0.0229) vs. −0.046 |

Of the v0.2.0 improvement, history features account for nearly all the test gain (+0.008). Comorbidity coding and
tuning improved validation and calibration AUROC slightly but not test AUROC.

## Calibration

The raw model is well calibrated. Post-hoc recalibration on the calibration split, which has a slightly higher
readmission rate (11.7% vs. 11.1% in test), did not improve ECE (paired Δ −0.0026, CI −0.0056 to +0.0024).
A deployment should monitor calibration-in-the-large locally and recalibrate only when drift is detected.

## Subgroup performance

At threshold 0.15: TPR is 0.51 for women vs. 0.42 for men, and 0.54 for patients under 50 vs. 0.45–0.46 for older
bands. Patients with race recorded as *unknown* (n = 459) or *Other* (n = 355) are flagged rarely (6.5%, 12.4%)
and have TPR 0.12 and 0.33, with wide CIs. Asian patients (n = 138) are too few to estimate. These gaps must be
reviewed before any use, for example with group-specific thresholds or by examining why these records carry
fewer risk signals.

## Ethical considerations

* **Proxy variables:** `payer_code` ranks fourth by mean |SHAP|. It can encode socioeconomic status. It is kept
  here so that its influence is visible; a deployment should test whether removing it changes the subgroup gaps.
* **Label bias:** the label records readmission to the same health system only. Patients readmitted elsewhere
  appear as negatives, and this may differ systematically by group. The same limitation applies to the history features.
* **Feedback loops:** if flagged patients receive interventions, future labels and history features will be affected.

## Caveats

One dataset and one random split from data now 15+ years old, with no temporal or external validation.
Encounters are clustered within patients; the CIs account for this.
