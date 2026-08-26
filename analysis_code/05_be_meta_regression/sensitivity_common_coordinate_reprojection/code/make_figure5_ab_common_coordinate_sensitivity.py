from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PACKAGE_ROOT / "output"
FIGURE_DIR = PACKAGE_ROOT / "output"

ACTIVITY_ORDER = ["all", "ride", "run", "walk"]
ACTIVITY_LABELS = {
    "all": "All activity",
    "ride": "Cycling",
    "run": "Running",
    "walk": "Walking",
}
ACTIVITY_COLORS = {
    "all": "#4D4D4D",
    "ride": "#A02D2B",
    "run": "#D99066",
    "walk": "#4C8EBA",
}
VARIABLE_ORDER = [
    "Urbanization Rate",
    "Population (20-55)",
    "Building Density",
    "Unemployment",
    "NDVI",
    "GDP",
    "Street Intersection Density",
    "Walkability Index",
]


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["legend.frameon"] = False


def forest_xlim(data: pd.DataFrame) -> tuple[float, float]:
    values = data[["coefficient", "ci_low", "ci_high"]].to_numpy(float).ravel()
    values = values[np.isfinite(values)]
    lo = min(float(values.min()), 0.0)
    hi = max(float(values.max()), 0.0)
    span = max(hi - lo, 0.02)
    pad = max(span * 0.13, 0.004)
    return lo - pad, hi + pad


def plot_panel(metric: str, panel_letter: str, title_label: str) -> None:
    path = RESULT_DIR / f"fig5_forest_plot_data_{metric}.csv"
    data = pd.read_csv(path)
    data = data[data["activity_type"].isin(ACTIVITY_ORDER)].copy()
    data = data[data["variable_only"].isin(VARIABLE_ORDER)].copy()
    if len(data) != 32:
        raise RuntimeError(f"Expected 32 forest rows for {metric}; found {len(data)}")

    n_variables = len(VARIABLE_ORDER)
    offsets = {"all": -0.30, "ride": -0.10, "run": 0.10, "walk": 0.30}
    y_base = {variable: i for i, variable in enumerate(VARIABLE_ORDER[::-1])}
    xlim = forest_xlim(data)

    fig = plt.figure(figsize=(8.6, 7.0))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.48, 4.10],
        left=0.035,
        right=0.985,
        top=0.82,
        bottom=0.12,
        wspace=0.035,
    )
    ax_label = fig.add_subplot(grid[0, 0])
    ax_forest = fig.add_subplot(grid[0, 1], sharey=ax_label)

    for axis in (ax_label, ax_forest):
        axis.set_ylim(-0.62, n_variables + 0.18)
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        for y in range(n_variables):
            if y % 2 == 0:
                axis.axhspan(y - 0.5, y + 0.5, color="#F7F7F7", zorder=-5)
            if y > 0:
                axis.axhline(y - 0.5, color="#CFCFCF", lw=0.65, zorder=-1)

    for variable in VARIABLE_ORDER[::-1]:
        ax_label.text(
            0.98,
            y_base[variable],
            variable,
            ha="right",
            va="center",
            fontsize=10.2,
            color="#1F1F1F",
        )
    ax_label.set_xlim(0, 1)
    ax_label.set_xticks([])
    ax_label.axhspan(0.935, 1.0, transform=ax_label.transAxes, color="#F2F2F2", zorder=-3)
    ax_label.text(0.98, 0.966, "Predictor", transform=ax_label.transAxes, ha="right", va="center", fontsize=10, fontweight="bold")

    ax_forest.spines["bottom"].set_visible(True)
    ax_forest.spines["left"].set_visible(True)
    ax_forest.axvline(0, color="#777777", lw=0.85, ls=(0, (4, 3)), zorder=0)
    ax_forest.grid(axis="x", color="#E9E4DE", lw=0.65)
    ax_forest.set_xlim(*xlim)
    ax_forest.set_xlabel("Average meta-regression coefficient", fontsize=10.5, labelpad=6)
    ax_forest.axhspan(0.935, 1.0, transform=ax_forest.transAxes, color="#F2F2F2", zorder=-3)
    ax_forest.text(
        0.02,
        0.966,
        "Coefficient (95% CI)",
        transform=ax_forest.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    for row in data.itertuples(index=False):
        activity = str(row.activity_type)
        y = y_base[str(row.variable_only)] + offsets[activity]
        color = ACTIVITY_COLORS[activity]
        ax_forest.hlines(y, row.ci_low, row.ci_high, color=color, lw=1.8, alpha=0.88, zorder=2, clip_on=False)
        ax_forest.scatter(
            row.coefficient,
            y,
            s=38,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
            clip_on=False,
        )
        star = str(row.star_label) if pd.notna(row.star_label) else ""
        if star:
            ax_forest.text(
                row.ci_high + (xlim[1] - xlim[0]) * 0.012,
                y,
                star,
                color=color,
                fontsize=7.2,
                fontweight="bold",
                ha="left",
                va="center",
                clip_on=False,
            )

    handles = [
        Line2D([0], [0], marker="o", lw=1.5, color=ACTIVITY_COLORS[a], markersize=5.2, label=ACTIVITY_LABELS[a])
        for a in ACTIVITY_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.885),
        ncol=4,
        fontsize=8.8,
        handlelength=1.8,
        columnspacing=1.0,
    )

    fig.text(0.035, 0.958, f"{panel_letter}) National | CEHWI | Composite | {title_label}", ha="left", va="top", fontsize=15.2, fontweight="bold")
    fig.text(
        0.035,
        0.912,
        "Common CEHWI coordinates (2, 4 and 6 versus 0); stars denote the joint 3-df Wald test.",
        ha="left",
        va="top",
        fontsize=8.8,
        color="#555555",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / f"fig5_{panel_letter}_{metric}_common_exposure_coordinate_forest"
    fig.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    apply_style()
    plot_panel("mean", "a", "Mean predictors")
    plot_panel("gini", "b", "Gini predictors")
    print(f"Saved Fig. 5 common-coordinate forests to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
