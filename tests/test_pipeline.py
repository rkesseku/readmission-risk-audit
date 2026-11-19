import json

from readmit.cli import main
from readmit.pipeline import run_study
from readmit.synthetic import make_synthetic


def test_run_study_end_to_end(raw):
    m, _ = run_study(raw, n_boot=20, fast=True)
    assert m["selected_model"] in {"logistic", "xgboost", "mlp"}
    for model in ("logistic", "xgboost", "mlp"):
        for cal in ("uncalibrated", "platt", "isotonic", "venn_abers"):
            row = m["test_metrics"][model][cal]
            assert 0 <= row["auroc"] <= 1
            assert row["auroc_ci"][0] <= row["auroc_ci"][1]
    assert len(m["decision_curve"]["thresholds"]) == len(m["decision_curve"]["treat_all"])
    assert m["xgboost_mean_abs_shap_top15"]
    splits = m["data"]["splits"]
    assert sum(v["encounters"] for v in splits.values()) == m["data"]["encounters_after_cleaning"]


def test_cli_synthetic_writes_results(tmp_path, capsys):
    main(["run-all", "--synthetic", "--out-dir", str(tmp_path), "--n-boot", "20"])
    data = json.loads((tmp_path / "metrics.json").read_text())
    assert "subgroup_audit" in data
    for f in ("reliability.png", "decision_curve.png", "subgroup_audit.png", "importance.png",
              "roc.png"):
        assert (tmp_path / f).stat().st_size > 1000
    assert "selected_model" in capsys.readouterr().out


def test_synthetic_schema_matches_real_columns():
    df = make_synthetic(50)
    for col in ("encounter_id", "patient_nbr", "readmitted", "diag_1", "insulin"):
        assert col in df
