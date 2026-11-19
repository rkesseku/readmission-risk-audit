# readmission-risk-audit

[![CI](https://github.com/rkesseku/readmission-risk-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/rkesseku/readmission-risk-audit/actions/workflows/ci.yml)

**30-day hospital readmission risk, evaluated the way a clinical deployment review would evaluate it:**
patient-level splits, leak-free patient-history features, calibration (Platt, isotonic, Venn-Abers),
decision-curve analysis, a subgroup fairness audit, and patient-clustered bootstrap confidence intervals
on every headline number.

## Why this exists

Most readmission-model write-ups stop at AUROC. A hospital deciding whether to use a model needs
answers to different questions:

* **Are the probabilities trustworthy?** If the model says 20%, do about 20% of those patients come back?
* **Is acting on the model better than the obvious alternatives** ("call every discharged patient" or "call nobody")?
* **Does it work equally well for different patient groups**, or does it miss more readmissions in some?
* **Is the evaluation free of leakage**, given that the same patient can have many encounters?

This repo answers each of those on a public dataset, with uncertainty attached.

## Data

[UCI *Diabetes 130-US Hospitals for years 1999–2008*](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(101,766 inpatient encounters of patients with diabetes, 130 US hospitals; Strack et al., 2014,
*BioMed Research International*, "Impact of HbA1c measurement on hospital readmission rates").
Licence: CC BY 4.0. The data is **not** included in this repo.

```bash
mkdir -p data/raw && cd data/raw
curl -L -o diabetes130.zip "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
unzip diabetes130.zip   # gives diabetic_data.csv and IDS_mapping.csv
```

Cleaning, following Strack et al.: encounters ending in death or hospice are removed (they cannot be
readmitted); the three diagnosis codes are mapped from ICD-9 to nine disease groups; the target is
`readmitted == "<30"`. After cleaning: 99,340 encounters from 69,987 patients, 11.3% positive.

## Method

| Step | What is done |
|---|---|
| Split | 60 / 20 / 20 train / calibration / test **by patient** (no patient in two splits); 15% of training patients held out for early stopping and tuning |
| Features | 8 utilisation counts, 23 diabetes-medication flags, ordinal age, one-hot categoricals (levels with < 20 training rows pooled as `Rare`), plus the blocks kept by the experiment below: 257 features |
| History | built from each patient's **earlier** encounters only: number of earlier encounters, earlier <30-day and any-time readmissions, earlier inpatient days, maximum earlier inpatient and emergency counts, a first-encounter flag. `encounter_id` order is the time proxy because the public data has no dates |
| Diagnoses | nine Strack groups, plus ICD-9 chapters for each of the three codes, 13 comorbidity flags (for example heart failure, chronic kidney disease, COPD, diabetic complications) and a distinct-chapter count |
| Models | L2 logistic regression, XGBoost (early-stopped; hyperparameters from a 50-trial Optuna search on the validation split), a 2-layer MLP (scikit-learn) |
| Selection | best AUROC on the **calibration** split; the test split is touched only for the final report |
| Calibration | Platt scaling, isotonic regression, and inductive Venn-Abers (Vovk & Petej, 2014), all fitted on the calibration split; Venn-Abers was pre-specified as the headline calibrator |
| Metrics | AUROC, AUPRC, Brier, ECE (10 equal-mass bins), calibration slope and intercept; 95% CIs from a 500-draw bootstrap that resamples **patients**, not encounters |
| Utility | decision curve: net benefit vs. "flag everyone" / "flag no one" for thresholds 0.05–0.40 (Vickers & Elkin, 2006) |
| Fairness | by age band, sex and race: O/E ratio, ECE, TPR and FPR at a 0.15 threshold, each with a clustered-bootstrap CI; groups under 200 test encounters are reported as insufficient |
| Explanation | exact TreeSHAP from XGBoost's `pred_contribs`, one-hot levels summed back to the source variable |

The 0.15 threshold is illustrative. It corresponds to judging a post-discharge intervention worthwhile
if it prevents one readmission for every ~5.7 patients flagged needlessly (odds 0.15/0.85). A real
deployment would set it with the clinical team.

## Results

All numbers below come from `readmit experiment` (seed 0, 50 Optuna trials) followed by `readmit run-all`
(seed 0, 500 bootstrap draws), recorded in [`docs/results/experiment.json`](docs/results/experiment.json)
and [`docs/results/metrics.json`](docs/results/metrics.json). Test split: 19,944 encounters from
13,998 patients, prevalence 11.1%.

**Model-improvement experiment (XGBoost).** Each step builds on the previous one and is kept only if
AUROC rises on *both* the validation and the calibration split; the test split was never used for selection.
Test deltas are paired, patient-clustered bootstrap 95% CIs against the original model.

| Step | Validation AUROC | Calibration AUROC | Test AUROC | Test Δ vs. original (95% CI) | Kept |
|---|---|---|---|---|---|
| 0. Original (v0.1.0, 180 features) | 0.677 | 0.673 | 0.6805 | – | – |
| 1. + patient-history features | 0.689 | 0.684 | 0.6883 | +0.0079 (+0.0041 to +0.0119) | yes |
| 2. + ICD-9 chapters and comorbidity flags | 0.690 | 0.685 | 0.6885 | +0.0081 (+0.0030 to +0.0135) | yes |
| 3. + Optuna tuning (50 trials) | 0.693 | 0.687 | 0.6883 | +0.0078 (+0.0027 to +0.0135) | yes |

Almost all of the gain came from the history features. Richer diagnosis coding and tuning each passed the
selection rule by small margins on validation and calibration (+0.001 and +0.002 to +0.003) but added nothing
measurable on test. Step 3's validation figure is optimistic because Optuna tuned on that split. The final
configuration (all three steps) is stored in [`src/readmit/config.py`](src/readmit/config.py), and `run-all` uses it.

**Discrimination and calibration on the test split (final configuration)**

| Model | AUROC (95% CI) | AUPRC | Brier | ECE uncalibrated | ECE Venn-Abers |
|---|---|---|---|---|---|
| XGBoost (selected) | 0.688 (0.675–0.703) | 0.241 | 0.0926 | 0.0050 | 0.0076 |
| Logistic regression | 0.676 (0.663–0.691) | 0.222 | 0.0938 | 0.0080 | 0.0080 |
| MLP | 0.657 (0.643–0.671) | 0.204 | 0.0952 | 0.0155 | 0.0076 |

* XGBoost beats logistic regression by **+0.012 AUROC** (paired CI +0.006 to +0.017) and the MLP by
  +0.031 (+0.023 to +0.040). The new features lifted the other two models as well (logistic 0.667 → 0.676,
  MLP 0.629 → 0.657).
* **Post-hoc calibration did not help the selected model.** Raw XGBoost (trained on log-loss) was already
  well calibrated (ECE 0.0050, slope 1.05). Venn-Abers gave ECE 0.0076; the paired difference is
  −0.0026 (95% CI −0.0056 to +0.0024), so the two are statistically indistinguishable. The calibration split
  happens to have a slightly higher readmission rate (11.7%) than test (11.1%), and the calibrators carry
  that offset over. Calibration clearly helped the MLP (ECE 0.0155 → 0.0076, raw slope 0.73).
  The lesson: recalibrate when there is evidence of miscalibration, not by reflex.

**Clinical utility.** At the 0.15 threshold the selected model (Venn-Abers output) flags 24.0% of
discharges, catches 47.0% of readmissions (TPR) with PPV 21.6%, and has net benefit
**0.0187 (95% CI 0.0147–0.0229)** against −0.046 for flagging everyone (the original model scored 0.0155).
Its net benefit is positive and above both default strategies at every threshold from 0.05 to 0.40
([decision curve](docs/results/decision_curve.png)).

**Fairness audit** (selected model, Venn-Abers, threshold 0.15; full table in `metrics.json`)

| Group | n | Flag rate | TPR (95% CI) | O/E (95% CI) |
|---|---|---|---|---|
| Female | 10,877 | 24.9% | 0.508 (0.475–0.537) | 0.96 (0.91–1.01) |
| Male | 9,067 | 23.0% | 0.419 (0.375–0.456) | 0.91 (0.85–0.96) |
| Age < 50 | 3,276 | 22.9% | 0.544 (0.479–0.623) | 0.94 (0.85–1.01) |
| Age 50–69 | 7,816 | 21.2% | 0.458 (0.418–0.502) | 0.98 (0.91–1.04) |
| Age 70+ | 8,852 | 26.9% | 0.454 (0.424–0.487) | 0.91 (0.86–0.96) |
| African American | 3,775 | 26.1% | 0.469 (0.413–0.531) | 0.96 (0.88–1.03) |
| Caucasian | 14,802 | 24.3% | 0.478 (0.447–0.508) | 0.93 (0.89–0.97) |
| Race unknown | 459 | 6.5% | 0.119 (0.000–0.231) | 1.12 (0.83–1.47) |

* Sensitivity is lower for men than for women (gap 0.09) and for patients aged 50 and over than for those under 50.
* Patients with race recorded as *unknown* (and *Other*: flag rate 12.4%, TPR 0.33) are flagged far less often
  and have much lower sensitivity. The groups are small, so the CIs are wide, but the direction is a red flag
  that a deployment review would have to investigate.
* O/E below 1 in most groups is the same calibration-split offset described above (slight over-prediction).
* Asian patients (138 test encounters) are below the 200-encounter minimum and are not estimated.

**What drives predictions.** By mean |SHAP|, the top variables are discharge disposition, inpatient visits
in the prior year, the history feature *maximum earlier inpatient count*, payer code, and the number of earlier
encounters followed by a readmission ([figure](docs/results/importance.png)). Payer code is a plausible
socioeconomic proxy; see the [model card](docs/model_card.md).

Figures: [reliability](docs/results/reliability.png) · [ROC](docs/results/roc.png) ·
[decision curve](docs/results/decision_curve.png) · [subgroup audit](docs/results/subgroup_audit.png) ·
[importance](docs/results/importance.png). A short summary is in [`docs/brief/brief.pdf`](docs/brief/brief.pdf).

## Limitations

* **History features rely on a proxy.** `encounter_id` order stands in for time, which is reasonable but
  unverified for this dataset. Because splits are by patient, a patient's earlier encounters are always in
  the same split, so no information crosses splits. At deployment these features must come from the patient's
  real EHR history, and the definition must match what the hospital actually knows at discharge.
* One dataset from 1999–2008, one random patient-level split, one seed. There is no temporal or external
  validation (the public data has no dates or hospital identifiers), so performance at a new hospital is unknown.
* An AUROC around 0.69 is typical of this dataset; it supports ranking patients for a low-cost
  intervention, not individual-level decisions.
* The cohort is patients with diabetes only; race, sex and payer are as recorded administratively.
* The fairness audit measures disparities; it does not establish their cause or fix them.

## How to run

```bash
pip install -e ".[dev]"
readmit experiment              # history -> diagnoses -> Optuna; writes docs/results/experiment.json (~35 min CPU)
readmit run-all                 # full study with the selected config -> docs/results/ (a few minutes)
readmit run-all --synthetic --out-dir /tmp/smoke   # schema-identical synthetic data; numbers are meaningless
pytest && ruff check src tests  # tests use synthetic data only, no downloads
```

`experiment` additionally needs `optuna` (`pip install ".[experiment]"`). Rebuild the brief with
`python docs/brief/make_figures.py`, then `quarto render docs/brief/brief.qmd --to typst`.

## Layout

```
src/readmit/   data.py (cleaning, ICD-9 groups/chapters, comorbidities, history features, featuriser,
               patient split) · experiment.py · config.py · models.py · calibration.py
               metrics.py (ECE, net benefit, clustered bootstrap) · fairness.py · pipeline.py · plots.py · cli.py
tests/         tests on synthetic data, including a no-leakage test for the history features
docs/          results/ (metrics.json, experiment.json, figures) · model_card.md · brief/
```

## License

MIT © 2026 Reagan Kesseku
