"""Result figures written next to metrics.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .metrics import reliability_curve  # noqa: E402

COLORS = {"logistic": "#1f4e9e", "xgboost": "#c0392b", "mlp": "#2e8b57"}
CAL_COLORS = {"uncalibrated": "#888888", "platt": "#1f4e9e", "isotonic": "#2e8b57",
              "venn_abers": "#c0392b"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reliability(metrics: dict, art: dict, out: Path) -> None:
    m = art["best_model"]
    y = art["y_test"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    top = 0.0
    for cname, p in art["preds"][m].items():
        mp, fo = reliability_curve(y, p, n_bins=10)
        top = max(top, mp.max(), fo.max())
        e = metrics["test_metrics"][m][cname]["ece"]
        ax.plot(mp, fo, "o-", ms=3.5, color=CAL_COLORS[cname], label=f"{cname} (ECE {e:.3f})")
    ax.plot([0, top * 1.05], [0, top * 1.05], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted risk (decile)")
    ax.set_ylabel("Observed 30-day readmission rate")
    ax.set_title(f"Reliability on test split: {m}")
    ax.legend(fontsize=8)
    _save(fig, out / "reliability.png")


def decision_curve(metrics: dict, out: Path) -> None:
    d = metrics["decision_curve"]
    t = np.array(d["thresholds"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for m in COLORS:
        if m in d:
            ax.plot(t, d[m], color=COLORS[m], label=f"{m} (Venn-Abers)")
    ax.plot(t, d["treat_all"], color="#555555", ls=":", label="flag everyone")
    ax.plot(t, d["treat_none"], color="black", lw=0.8, label="flag no one")
    ymax = max(max(d[m]) for m in COLORS if m in d)
    ax.set_ylim(-0.01, ymax * 1.3 + 1e-3)
    ax.axvline(metrics["decision_threshold"], color="#999999", lw=0.6, ls="--")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve (test split)")
    ax.legend(fontsize=8)
    _save(fig, out / "decision_curve.png")


def subgroups(metrics: dict, out: Path) -> None:
    rows = [r for r in metrics["subgroup_audit"] if "tpr" in r]
    labels = [f"{r['attribute'].replace('_band', '')}: {r['group']} (n={r['n']})" for r in rows]
    yy = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 0.32 * len(rows) + 1.4), sharey=True)
    for ax, key, ref, title in ((axes[0], "tpr", None, "Sensitivity (TPR) at threshold"),
                                (axes[1], "oe_ratio", 1.0, "Observed / expected readmissions")):
        est = np.array([r[key] for r in rows])
        ci = np.array([r[f"{key}_ci"] for r in rows])
        ax.errorbar(est, yy, xerr=[est - ci[:, 0], ci[:, 1] - est], fmt="o", color="#1f4e9e",
                    ms=4, capsize=2)
        if ref is not None:
            ax.axvline(ref, color="black", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(labels, fontsize=8)
    _save(fig, out / "subgroup_audit.png")


def importance(metrics: dict, out: Path) -> None:
    imp = metrics.get("xgboost_mean_abs_shap_top15")
    if not imp:
        return
    names = list(imp)[::-1]
    vals = [imp[k] for k in names]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.barh(names, vals, color="#1f4e9e")
    ax.set_xlabel("Mean |SHAP| (log-odds), one-hot levels summed")
    ax.set_title("XGBoost feature importance (test sample)")
    ax.tick_params(axis="y", labelsize=8)
    _save(fig, out / "importance.png")


def roc(metrics: dict, art: dict, out: Path) -> None:
    y = art["y_test"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m, d in art["preds"].items():
        fpr, tpr, _ = roc_curve(y, d["uncalibrated"])
        a = metrics["test_metrics"][m]["uncalibrated"]["auroc"]
        ax.plot(fpr, tpr, color=COLORS.get(m, "k"), label=f"{m} (AUROC {a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8)
    ax.set_title("ROC on test split")
    _save(fig, out / "roc.png")


def make_all_figures(metrics: dict, art: dict, out: Path) -> None:
    out = Path(out)
    reliability(metrics, art, out)
    decision_curve(metrics, out)
    subgroups(metrics, out)
    importance(metrics, out)
    roc(metrics, art, out)
