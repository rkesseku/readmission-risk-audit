"""Configuration of the headline model, as selected by ``readmit experiment``.

Values are copied from ``docs/results/experiment.json`` (seed 0, 50 Optuna trials, selection on
the validation and calibration splits only) so that ``readmit run-all`` is reproducible without
re-running the search.
"""

FINAL_HISTORY = True
FINAL_RICH_DIAG = True
FINAL_XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.0368526674151172,
    "max_depth": 7,
    "min_child_weight": 2.0994440997888977,
    "subsample": 0.5954855839717457,
    "colsample_bytree": 0.47510840690144446,
    "reg_lambda": 0.1093806167958321,
    "reg_alpha": 0.026034044986388206,
    "gamma": 3.764574733103346,
}

# Baseline tuning grids. ``run-all`` fits every grid point on the training-fit split and keeps
# the one with the highest AUROC on the same validation split that the XGBoost Optuna search used
# (never the calibration or test split), so all three model families are tuned on equal footing.
LOGISTIC_C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
MLP_GRID = [
    {"hidden": h, "alpha": a}
    for h in ((32,), (64, 32), (128, 64))
    for a in (1e-4, 1e-3, 1e-2, 1e-1)
]
