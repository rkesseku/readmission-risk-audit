# readmission-risk-audit

[![CI](https://github.com/rkesseku/readmission-risk-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/rkesseku/readmission-risk-audit/actions/workflows/ci.yml)

**30-day hospital readmission risk, evaluated as a clinical deployment review would evaluate it:**
patient-level splits, leak-free patient-history features, calibration (Platt, isotonic, Venn-Abers),
decision-curve analysis, a subgroup fairness audit, and patient-clustered bootstrap confidence intervals
on every headline number.

## Why this exists

Most readmission-model write-ups stop at AUROC. A hospital deciding whether to use a model needs
answers to other questions:

* **Are the probabilities trustworthy?** If the model says 20%, do about 20% of those patients come back?
* **Is acting on the model better than the obvious alternatives** ("call every discharged patient" or "call nobody")?
* **Does it work equally well for different patient groups**, or does it miss more readmissions in some?
* **Is the evaluation free of leakage**, given that the same patient can have many encounters?

This repo answers each of these questions on a public dataset, with uncertainty attached.

## Data

[UCI *Diabetes 130-US Hospitals for years 1999–2008*](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(101,766 inpatient encounters for patients with diabetes at 130 US hospitals, from Strack et al., 2014,
*BioMed Research International*, "Impact of HbA1c measurement on hospital readmission rates").
Licence: CC BY 4.0. The data is **not** included in this repo.

```bash
mkdir -p data/raw && cd data/raw
curl -L -o diabetes130.zip "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
unzip diabetes130.zip   # gives diabetic_data.csv and IDS_mapping.csv
```

Cleaning follows Strack et al. Encounters ending in death or hospice are removed, since those patients
cannot be readmitted. The three diagnosis codes are mapped from ICD-9 to nine disease groups, and the
target is `readmitted == "<30"`. After cleaning, 99,340 encounters from 69,987 patients remain, 11.3% of them positive.

## Method

| Step | What is done |
|---|---|
| Split | 60 / 20 / 20 train / calibration / test **by patient** (no patient in two splits); 15% of training patients held out for early stopping and tuning |
| Features | 8 utilisation counts, 23 diabetes-medication flags, ordinal age, one-hot categoricals (levels with < 20 training rows pooled as `Rare`), plus the blocks kept by the experiment below: 257 features |
| History | built from each patient's **earlier** encounters only: number of earlier encounters, earlier <30-day and any-time readmissions, earlier inpatient days, maximum earlier inpatient and emergency counts, a first-encounter flag. `encounter_id` order is the time proxy because the public data has no dates |
| Diagnoses | nine Strack groups, plus ICD-9 chapters for each of the three codes, 13 comorbidity flags (for example heart failure, chronic kidney disease, COPD, diabetic complications) and a distinct-chapter count |
| Models | L2 logistic regression, XGBoost (early-stopped; hyperparameters from a 50-trial Optuna search on the validation split), an MLP (scikit-learn). The two baselines are tuned on the **same validation split**: logistic `C` over a log grid 1e-4 to 1e2 (7 values), MLP width {(32), (64, 32), (128, 64)} × L2 `alpha` {1e-4, 1e-3, 1e-2, 1e-1}; calibration and test data are not used for tuning |
| Selection | best AUROC on the **calibration** split; the test split is touched only for the final report |
| Calibration | Platt scaling, isotonic regression, and inductive Venn-Abers (Vovk & Petej, 2014), all fitted on the calibration split; Venn-Abers was pre-specified as the headline calibrator |
| Metrics | AUROC, AUPRC, Brier, ECE (10 equal-mass bins), calibration slope and intercept; 95% CIs from a 500-draw bootstrap that resamples **patients**, not encounters |
| Utility | decision curve: net benefit vs. "flag everyone" / "flag no one" for thresholds 0.05–0.40 (Vickers & Elkin, 2006) |
| Fairness | by age band, sex and race: O/E ratio, ECE, TPR and FPR at a 0.15 threshold, each with a clustered-bootstrap CI; groups under 200 test encounters are reported as insufficient |
| Explanation | exact TreeSHAP from XGBoost's `pred_contribs`, one-hot levels summed back to the source variable |

The 0.15 threshold is illustrative. It corresponds to judging that a post-discharge intervention is worthwhile
if preventing one readmission justifies flagging about 5.7 patients needlessly (odds 0.15/0.85). A real
deployment would set it with the clinical team.

## Results

All numbers below come from `readmit experiment` (seed 0, 50 Optuna trials) followed by `readmit run-all`
(seed 0, 500 bootstrap draws), recorded in [`docs/results/experiment.json`](docs/results/experiment.json)
and [`docs/results/metrics.json`](docs/results/metrics.json). The test split has 19,944 encounters from
13,998 patients, with a prevalence of 11.1%.

**Model-improvement experiment (XGBoost).** Each step builds on the previous one and is kept only if
AUROC rises on *both* the validation and the calibration split. The test split was never used for selection.
Test deltas are measured against the original model, with paired, patient-clustered bootstrap 95% CIs.

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
`run-all` now fits XGBoost with 2 threads instead of 4. Histogram sums are then accumulated in a different order,
which moves its test AUROC from the experiment's 0.6883 to 0.6880. The experiment table was not re-run.

**Discrimination and calibration on the test split (final configuration, tuned baselines)**

The validation-split grid search chose logistic `C` = 1.0, with validation AUROC 0.683. The previous fixed
`C` = 0.1 scored 0.683 as well, and every value from 1e-4 to 1e2 lay within 0.0035. For the MLP it chose
hidden layers (128, 64) and `alpha` = 0.1 (validation AUROC 0.671 vs. 0.654 for the previous fixed (64, 32),
`alpha` = 0.01). XGBoost scored 0.690 on the same split. Selection on the calibration split still picks
XGBoost (calibration AUROC 0.686 vs. 0.669 for logistic regression and 0.657 for the MLP).

| Model | AUROC (95% CI) | AUPRC | Brier | ECE uncalibrated | ECE Venn-Abers |
|---|---|---|---|---|---|
| XGBoost (selected) | 0.688 (0.675–0.702) | 0.242 | 0.0926 | 0.0044 | 0.0089 |
| Logistic regression (`C` = 1.0) | 0.677 (0.663–0.692) | 0.226 | 0.0936 | 0.0070 | 0.0078 |
| MLP (128, 64), `alpha` = 0.1 | 0.667 (0.653–0.682) | 0.214 | 0.0949 | 0.0201 | 0.0086 |

* With both baselines tuned on the same validation split as XGBoost, XGBoost still beats logistic regression,
  by **+0.011 AUROC** (paired CI +0.006 to +0.017). Against the untuned `C` = 0.1 baseline the gap was
  +0.012 (+0.006 to +0.017). Tuning barely moved logistic regression (test AUROC 0.676 → 0.677), so the earlier
  comparison was not materially inflated. The gap is real but small. Tuning helped the MLP more
  (0.657 → 0.667), cutting XGBoost's lead over it from +0.031 to **+0.022** (+0.014 to +0.029).
* **Post-hoc calibration did not help the selected model.** Raw XGBoost (trained on log-loss) was already
  well calibrated (ECE 0.0044, slope 1.04). Venn-Abers gave ECE 0.0089. The paired difference is
  −0.0045 (95% CI −0.0052 to +0.0016), so the interval includes zero. The calibration split
  happens to have a slightly higher readmission rate (11.7%) than the test split (11.1%), and the calibrators
  carry that offset into their test predictions. Calibration clearly helped the MLP (ECE 0.0201 → 0.0086, raw slope 0.69).
  The lesson is to recalibrate when there is evidence of miscalibration rather than by reflex.

**Clinical utility.** At the 0.15 threshold the selected model (Venn-Abers output) flags 22.9% of
discharges, catches 45.6% of readmissions (TPR) with PPV 22.1%, and has a net benefit of
**0.0190 (95% CI 0.0148–0.0233)** against −0.046 for flagging everyone (the original model scored 0.0155).
Its net benefit is positive and above both default strategies at every threshold from 0.05 to 0.40
([decision curve](docs/results/decision_curve.png)).

**Fairness audit** (selected model, Venn-Abers, threshold 0.15, with the full table in `metrics.json`)

| Group | n | Flag rate | TPR (95% CI) | O/E (95% CI) |
|---|---|---|---|---|
| Female | 10,877 | 23.6% | 0.489 (0.458–0.515) | 0.96 (0.91–1.01) |
| Male | 9,067 | 22.0% | 0.413 (0.370–0.455) | 0.90 (0.85–0.96) |
| Age < 50 | 3,276 | 22.4% | 0.524 (0.453–0.598) | 0.94 (0.85–1.01) |
| Age 50–69 | 7,816 | 20.1% | 0.449 (0.408–0.487) | 0.98 (0.92–1.04) |
| Age 70+ | 8,852 | 25.5% | 0.439 (0.409–0.471) | 0.90 (0.86–0.95) |
| African American | 3,775 | 25.0% | 0.462 (0.406–0.520) | 0.96 (0.87–1.03) |
| Caucasian | 14,802 | 23.1% | 0.464 (0.438–0.492) | 0.93 (0.89–0.97) |
| Race unknown | 459 | 6.5% | 0.095 (0.000–0.181) | 1.11 (0.82–1.45) |

* Sensitivity is lower for men than for women (gap 0.08) and for patients aged 50 and over than for those under 50.
* Patients with race recorded as *unknown* are flagged far less often and have much lower sensitivity, and the
  same holds for those recorded as *Other* (flag rate 11.5%, TPR 0.31). The groups are small, so the CIs are wide, but the direction is a red flag
  that a deployment review would have to investigate.
* O/E below 1 in most groups reflects the calibration-split offset described above (slight over-prediction).
* Asian patients (138 test encounters) are below the 200-encounter minimum and are not estimated.

**What drives predictions.** By mean |SHAP|, the top variables are discharge disposition, inpatient visits
in the prior year, the history feature *maximum earlier inpatient count*, medical specialty and payer code
([figure](docs/results/importance.png)). Payer code is a plausible
socioeconomic proxy (see the [model card](docs/model_card.md)).

Figures: [reliability](docs/results/reliability.png) · [ROC](docs/results/roc.png) ·
[decision curve](docs/results/decision_curve.png) · [subgroup audit](docs/results/subgroup_audit.png) ·
[importance](docs/results/importance.png). A short summary is in [`docs/brief/brief.pdf`](docs/brief/brief.pdf).

## Limitations

* **History features rely on a proxy.** `encounter_id` order stands in for time, an assumption that is reasonable but
  unverified for this dataset. Because splits are by patient, a patient's earlier encounters are always in
  the same split, so no information crosses splits. At deployment these features must come from the patient's
  real EHR history, and the definition must match what the hospital actually knows at discharge.
* One dataset from 1999–2008, one random patient-level split, one seed. There is no temporal or external
  validation (the public data has no dates or hospital identifiers), so performance at a new hospital is unknown.
* An AUROC around 0.69 is typical of this dataset. That is enough to rank patients for a low-cost
  intervention but not to make individual-level decisions.
* The cohort is patients with diabetes only. Race, sex and payer are as recorded administratively.
* The fairness audit measures disparities but does not establish their cause or fix them.

## How to run

```bash
pip install -e ".[dev]"
readmit experiment              # history -> diagnoses -> Optuna; writes docs/results/experiment.json (~35 min CPU)
readmit run-all                 # baseline tuning + full study -> docs/results/ (~12 min on 2 threads)
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
