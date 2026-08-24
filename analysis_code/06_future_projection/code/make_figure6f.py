"""
Create Fig. 6f scope-comparison forest plots from the packaged future-projection
summary table.

This script is intentionally self-contained for figure reproduction. It reads
only the clean Fig. 6 data table:

    analysis_code/06_future_projection/data/all_scopes_yearly_projection_all_rows.csv

and writes a standalone Fig. 6f folder containing plot-ready CSV, PNG and SVG.
The final version summarizes the full 2025-2050 projection window and exports
all three SSP scenarios.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
DATA_PATH = MODULE_DIR / "data" / "all_scopes_yearly_projection_all_rows.csv"
OUT_ROOT = MODULE_DIR / "output" / "figure6f_generated"

SCENARIOS = ["ssp245", "ssp370", "ssp585"]
SCENARIO_LABELS = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
YEARS = list(range(2025, 2051))
PERIOD_LABEL = "2025-2050"
DENOMINATOR = "annual_all_days_primary"
METRIC_VALUE = "pa_loss_fraction_percent_all_days"
METRIC_LOW = "pa_loss_fraction_low_percent_all_days"
METRIC_HIGH = "pa_loss_fraction_high_percent_all_days"

SCOPE_ORDER = [
    "primary_national",
    "sensitivity_dtw4lag12",
    "control_zone",
    "control_region",
    "sensitivity_city_specific",
]

SCOPE_LABELS = {
    "primary_national": "National\npooled",
    "sensitivity_dtw4lag12": "DTW4lag12\nphenotype",
    "control_zone": "Climate-zone\ncontrol",
    "control_region": "Region\ncontrol",
    "sensitivity_city_specific": "City-specific\nsensitivity",
}

HEATWAVE_ORDER = ["composite", "day", "night"]
HEATWAVE_LABELS = {
    "composite": "Compound",
    "day": "Daytime",
    "night": "Nighttime",
}

# Updated red-yellow-blue scheme used for the final Fig. 6 family.
HEATWAVE_COLORS = {
    "composite": "#A12D32",
    "day": "#E3A000",
    "night": "#4C8EBA",
}

INDICATOR_LABELS = {
    "cehwi": "CEHWI",
    "exceeded_quantity": "Exceeded cumulative intensity",
}


def set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 240,
            "savefig.dpi": 360,
        }
    )


def load_plot_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input table: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = {
        "scope",
        "scenario",
        "period",
        "denominator_package",
        "indicator",
        "heatwave_type",
        METRIC_VALUE,
        METRIC_LOW,
        METRIC_HIGH,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")
    sub = df[
        (df["denominator_package"] == DENOMINATOR)
        & (df["scenario"].isin(SCENARIOS))
        & (df["year"].isin(YEARS))
        & (df["activity_type"] == "all")
        & (df["scope"].isin(SCOPE_ORDER))
        & (df["heatwave_type"].isin(HEATWAVE_ORDER))
        & (df["indicator"].isin(["cehwi", "exceeded_quantity"]))
    ].copy()
    sub["scope_label_plot"] = sub["scope"].map(SCOPE_LABELS)
    sub["heatwave_label_plot"] = sub["heatwave_type"].map(HEATWAVE_LABELS)
    sub["indicator_label_plot"] = sub["indicator"].map(INDICATOR_LABELS)
    sub["scenario_label_plot"] = sub["scenario"].map(SCENARIO_LABELS)
    sub["period_plot"] = PERIOD_LABEL
    group_cols = [
        "scope",
        "scope_label",
        "denominator_package",
        "scenario",
        "indicator",
        "heatwave_type",
        "activity_type",
        "curve_scope",
        "partition_family",
        "partition_name",
        "denominator_id",
        "denominator_label",
        "denominator_months",
        "scope_label_plot",
        "heatwave_label_plot",
        "indicator_label_plot",
        "scenario_label_plot",
        "period_plot",
    ]
    numeric_cols = [
        "heatwave_days",
        METRIC_VALUE,
        METRIC_LOW,
        METRIC_HIGH,
        "annualized_asri_percent",
        "annualized_asri_low_percent",
        "annualized_asri_high_percent",
        "benefit_adjusted_asri_percent",
        "benefit_adjusted_asri_low_percent",
        "benefit_adjusted_asri_high_percent",
        "exposure_clipped_share",
    ]
    numeric_cols = [c for c in numeric_cols if c in sub.columns]
    # The source table is annual. Fig. 6f is a long-horizon scope check, so each
    # point is the 2025-2050 mean across annual rows. This keeps the figure
    # aligned with the full projection window rather than a single late-century
    # slice.
    sub = (
        sub.groupby(group_cols, dropna=False)[numeric_cols]
        .mean()
        .reset_index()
    )
    sub = sub.sort_values(
        by=["indicator", "scope", "heatwave_type"],
        key=lambda s: s.map({v: i for i, v in enumerate(SCOPE_ORDER + HEATWAVE_ORDER)}).fillna(s),
    )
    return sub


def plot_indicator(data: pd.DataFrame, indicator: str, scenario: str, out_dir: Path) -> None:
    sub = data[(data["indicator"] == indicator) & (data["scenario"] == scenario)].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6.7, 3.65))
    x_base = np.arange(len(SCOPE_ORDER), dtype=float)
    offsets = {"composite": -0.18, "day": 0.0, "night": 0.18}

    ymax = 0.0
    ymin = 0.0
    for heatwave in HEATWAVE_ORDER:
        hdf = sub[sub["heatwave_type"] == heatwave].set_index("scope").reindex(SCOPE_ORDER)
        x = x_base + offsets[heatwave]
        y = hdf[METRIC_VALUE].astype(float).to_numpy()
        low = hdf[METRIC_LOW].astype(float).to_numpy()
        high = hdf[METRIC_HIGH].astype(float).to_numpy()
        color = HEATWAVE_COLORS[heatwave]
        for xi, lo, hi in zip(x, low, high):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.vlines(xi, lo, hi, color=color, lw=1.35, alpha=0.38, zorder=1)
        ax.scatter(
            x,
            y,
            s=31,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
            label=HEATWAVE_LABELS[heatwave],
        )
        ymax = max(ymax, np.nanmax(high))
        ymin = min(ymin, np.nanmin(low))

    pad = max(0.15, 0.10 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(min(-0.15, ymin - pad), ymax + pad)
    ax.axhline(0, color="#8E8E8E", lw=0.8, ls=(0, (4, 2)), zorder=0)
    ax.set_xlim(-0.55, len(SCOPE_ORDER) - 0.45)
    ax.set_xticks(x_base)
    ax.set_xticklabels([SCOPE_LABELS[s] for s in SCOPE_ORDER])
    ax.set_ylabel("PA loss fraction (%)")
    ax.set_title(f"Scope check: {INDICATOR_LABELS[indicator]}, {SCENARIO_LABELS[scenario]}, {PERIOD_LABEL}", pad=10)
    ax.grid(axis="y", color="#E6E6E6", lw=0.65)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=3, width=0.8)
    ax.legend(
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        handletextpad=0.35,
        columnspacing=1.45,
    )
    fig.tight_layout(pad=0.8)

    stem = f"fig6f_scope_forest_{indicator}_{scenario}_2025_2050_pa_loss_fraction"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_combined_for_scenario(data: pd.DataFrame, scenario: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.75), sharey=False)
    x_base = np.arange(len(SCOPE_ORDER), dtype=float)
    offsets = {"composite": -0.18, "day": 0.0, "night": 0.18}
    for ax, indicator in zip(axes, ["cehwi", "exceeded_quantity"]):
        sub = data[(data["indicator"] == indicator) & (data["scenario"] == scenario)].copy()
        ymax = 0.0
        ymin = 0.0
        for heatwave in HEATWAVE_ORDER:
            hdf = sub[sub["heatwave_type"] == heatwave].set_index("scope").reindex(SCOPE_ORDER)
            x = x_base + offsets[heatwave]
            y = hdf[METRIC_VALUE].astype(float).to_numpy()
            low = hdf[METRIC_LOW].astype(float).to_numpy()
            high = hdf[METRIC_HIGH].astype(float).to_numpy()
            color = HEATWAVE_COLORS[heatwave]
            for xi, lo, hi in zip(x, low, high):
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.vlines(xi, lo, hi, color=color, lw=1.25, alpha=0.38, zorder=1)
            ax.scatter(
                x,
                y,
                s=28,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label=HEATWAVE_LABELS[heatwave],
            )
            ymax = max(ymax, np.nanmax(high))
            ymin = min(ymin, np.nanmin(low))
        pad = max(0.15, 0.10 * (ymax - ymin if ymax > ymin else 1.0))
        ax.set_ylim(min(-0.15, ymin - pad), ymax + pad)
        ax.axhline(0, color="#8E8E8E", lw=0.8, ls=(0, (4, 2)), zorder=0)
        ax.set_xlim(-0.55, len(SCOPE_ORDER) - 0.45)
        ax.set_xticks(x_base)
        ax.set_xticklabels([SCOPE_LABELS[s] for s in SCOPE_ORDER], rotation=22, ha="right")
        ax.set_title(INDICATOR_LABELS[indicator], pad=8)
        ax.grid(axis="y", color="#E6E6E6", lw=0.65)
        ax.tick_params(axis="x", length=0, pad=5)
    axes[0].set_ylabel("PA loss fraction (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle(f"Fig. 6f | Scope robustness of projected PA loss, {SCENARIO_LABELS[scenario]}, {PERIOD_LABEL}", y=1.16, fontsize=12.5)
    fig.tight_layout(pad=0.9, w_pad=1.5)
    stem = f"fig6f_scope_forest_combined_cehwi_eq_{scenario}_2025_2050_pa_loss_fraction"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_all_scenarios(data: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 8.8), sharey=False)
    x_base = np.arange(len(SCOPE_ORDER), dtype=float)
    offsets = {"composite": -0.18, "day": 0.0, "night": 0.18}
    for r, scenario in enumerate(SCENARIOS):
        for c, indicator in enumerate(["cehwi", "exceeded_quantity"]):
            ax = axes[r, c]
            sub = data[(data["scenario"] == scenario) & (data["indicator"] == indicator)].copy()
            ymax = 0.0
            ymin = 0.0
            for heatwave in HEATWAVE_ORDER:
                hdf = sub[sub["heatwave_type"] == heatwave].set_index("scope").reindex(SCOPE_ORDER)
                x = x_base + offsets[heatwave]
                y = hdf[METRIC_VALUE].astype(float).to_numpy()
                low = hdf[METRIC_LOW].astype(float).to_numpy()
                high = hdf[METRIC_HIGH].astype(float).to_numpy()
                color = HEATWAVE_COLORS[heatwave]
                for xi, lo, hi in zip(x, low, high):
                    if np.isfinite(lo) and np.isfinite(hi):
                        ax.vlines(xi, lo, hi, color=color, lw=1.15, alpha=0.36, zorder=1)
                ax.scatter(
                    x,
                    y,
                    s=23,
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                    label=HEATWAVE_LABELS[heatwave],
                )
                if np.isfinite(high).any():
                    ymax = max(ymax, np.nanmax(high))
                if np.isfinite(low).any():
                    ymin = min(ymin, np.nanmin(low))
            pad = max(0.15, 0.10 * (ymax - ymin if ymax > ymin else 1.0))
            ax.set_ylim(min(-0.15, ymin - pad), ymax + pad)
            ax.axhline(0, color="#8E8E8E", lw=0.75, ls=(0, (4, 2)), zorder=0)
            ax.set_xlim(-0.55, len(SCOPE_ORDER) - 0.45)
            ax.set_xticks(x_base)
            if r == len(SCENARIOS) - 1:
                ax.set_xticklabels([SCOPE_LABELS[s] for s in SCOPE_ORDER], rotation=22, ha="right")
            else:
                ax.set_xticklabels([])
            ax.grid(axis="y", color="#E6E6E6", lw=0.6)
            if r == 0:
                ax.set_title(INDICATOR_LABELS[indicator], pad=8)
            if c == 0:
                ax.set_ylabel(f"{SCENARIO_LABELS[scenario]}\nPA loss (%)")
            ax.tick_params(axis="x", length=0, pad=4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.015), frameon=False)
    fig.suptitle(f"Fig. 6f | Scope robustness across SSPs, {PERIOD_LABEL}", y=1.055, fontsize=12.5)
    fig.tight_layout(pad=0.9, h_pad=1.0, w_pad=1.5)
    stem = "fig6f_scope_forest_all_scenarios_cehwi_eq_2025_2050_pa_loss_fraction"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def write_note(out_dir: Path) -> None:
    note = f"""# Fig. 6f scope forest plot

Input data:
`{DATA_PATH}`

Filter:
- denominator package: `{DENOMINATOR}`
- scenarios: `{', '.join(SCENARIOS)}`
- period: `{PERIOD_LABEL}` (annual rows 2025-2050 averaged)
- activity type: all
- metric: PA loss fraction (%), with lower/upper response-curve bounds

Interpretation:
Each point shows the projected 2025-2050 annual heatwave-attributable PA loss
fraction under one SSP for one response-curve scope. Vertical lines show the
corresponding lower/upper bounds exported by the projection model. Colours
separate compound, daytime and nighttime heatwaves using the final red-yellow-blue
Fig. 6 palette.
"""
    (out_dir / "README_fig6f.md").write_text(note, encoding="utf-8")


def main() -> None:
    set_nature_style()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_plot_data()
    data.to_csv(OUT_ROOT / "fig6f_scope_forest_plot_data_2025_2050_all_ssps.csv", index=False)
    for scenario in SCENARIOS:
        for indicator in ["cehwi", "exceeded_quantity"]:
            plot_indicator(data, indicator, scenario, OUT_ROOT)
        plot_combined_for_scenario(data, scenario, OUT_ROOT)
    plot_all_scenarios(data, OUT_ROOT)
    write_note(OUT_ROOT)
    print(f"Fig. 6f exported to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
