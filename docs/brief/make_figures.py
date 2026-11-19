"""Build the brief's figures: an architecture diagram plus copies of the run's result plots."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / "figures"
RESULTS = HERE.parent / "results"


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#1f4e9e",
                                lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1})


def architecture():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    light, mid = "#eef3fb", "#dbe6f7"
    box(ax, 0.1, 1.5, 1.6, 1.0, "UCI Diabetes 130\n101,766 encounters", light)
    box(ax, 2.0, 1.5, 1.7, 1.0, "Clean + featurise\n(drop death/hospice,\nICD-9 groups)", light)
    box(ax, 4.0, 2.7, 1.7, 0.9, "Train 60%\n(patients)", mid)
    box(ax, 4.0, 1.55, 1.7, 0.9, "Calibration 20%\n(patients)", mid)
    box(ax, 4.0, 0.4, 1.7, 0.9, "Test 20%\n(patients)", mid)
    box(ax, 6.1, 2.7, 1.7, 0.9, "LogReg / XGBoost\n/ MLP", light)
    box(ax, 6.1, 1.55, 1.7, 0.9, "Select model;\nPlatt / isotonic /\nVenn-Abers", light)
    box(ax, 8.2, 2.3, 1.7, 1.3, "Discrimination +\ncalibration\n(clustered\nbootstrap CIs)", light)
    box(ax, 8.2, 0.4, 1.7, 1.6, "Decision curve,\nsubgroup audit,\nSHAP, model card", light)
    arrow(ax, 1.7, 2.0, 2.0, 2.0)
    for y in (3.15, 2.0, 0.85):
        arrow(ax, 3.7, 2.0, 4.0, y)
    arrow(ax, 5.7, 3.15, 6.1, 3.15)
    arrow(ax, 5.7, 2.0, 6.1, 2.0)
    arrow(ax, 6.95, 2.7, 6.95, 2.45)
    arrow(ax, 7.8, 2.0, 8.2, 2.9)
    arrow(ax, 5.7, 0.85, 8.2, 1.1)
    arrow(ax, 7.8, 2.0, 8.2, 1.4)
    fig.tight_layout()
    fig.savefig(FIG / "architecture.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    architecture()
    for name in ("reliability.png", "decision_curve.png", "subgroup_audit.png", "importance.png",
                 "roc.png"):
        src = RESULTS / name
        if src.exists():
            shutil.copy(src, FIG / name)


if __name__ == "__main__":
    main()
