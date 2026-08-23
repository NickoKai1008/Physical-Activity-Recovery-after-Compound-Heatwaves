from __future__ import annotations

from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_ROOT = Path(__file__).resolve().parents[1]
INDICATOR = os.environ.get("HEATPA_INDICATOR", "cehwi").lower()
if INDICATOR not in {"cehwi", "exceeded_quantity"}:
    raise ValueError("HEATPA_INDICATOR must be cehwi or exceeded_quantity")
ROOT = BASE_ROOT
INDICATOR_DIR = "cehwi" if INDICATOR == "cehwi" else "exceeded_quantity"
SOURCE_ROOT = BASE_ROOT / "data" / "common_basis" / INDICATOR_DIR / "model_summaries"
DATA = BASE_ROOT / "data" / "lag_window" / INDICATOR_DIR
RESULTS = DATA / "results"
FIGURES = BASE_ROOT / "output" / "lag_window" / INDICATOR_DIR
EXPOSURE_LABEL = "CEHWI" if INDICATOR == "cehwi" else "Exceeded quantity"

SCENARIOS = ["fixed_7_day", "fixed_12_day", "dynamic_8_12_day"]
SCENARIO_LABELS = {
    "fixed_7_day": "Fixed lag 0-6",
    "fixed_12_day": "Fixed lag 0-11",
    "dynamic_8_12_day": "Phenotype-specific lag 0-7/0-11",
}
COLORS = {
    "fixed_7_day": "#4C8EBA",
    "fixed_12_day": "#A9363E",
    "dynamic_8_12_day": "#E2A11A",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    for directory in (DATA, RESULTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves_path = SOURCE_ROOT / "national_pooled_curves_exact_common_basis.csv"
    af_path = SOURCE_ROOT / "exact_common_basis_af_mc.csv"
    coverage_path = SOURCE_ROOT / "exact_common_basis_scenario_coverage.csv"
    lag_path = BASE_ROOT / "data" / "lag_assignment.csv"
    missing = [
        path for path in (curves_path, af_path, coverage_path, lag_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required deposited lag-window source tables are missing: "
            + ", ".join(str(path) for path in missing)
        )
    curves = pd.read_csv(curves_path)
    af = pd.read_csv(af_path)
    coverage = pd.read_csv(coverage_path)
    lag = pd.read_csv(lag_path)
    curves.to_csv(DATA / "lag_window_pooled_curve_source.csv", index=False)
    af.to_csv(DATA / "lag_window_af_source.csv", index=False)
    coverage.to_csv(DATA / "lag_window_model_coverage.csv", index=False)
    lag.to_csv(DATA / "lag_window_city_assignments.csv", index=False)
    return curves, af, coverage, lag


def draw_comparison(curves: pd.DataFrame, af: pd.DataFrame) -> None:
    percentile = af.loc[af["estimand"].eq("positive_exposure_percentile_af")].copy()
    percentile["percentile_number"] = (
        percentile["exposure_percentile"].str.replace("p", "", regex=False).astype(int)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.75),
        gridspec_kw={"width_ratios": [1.25, 1]},
        constrained_layout=True,
    )
    curve_ax, af_ax = axes

    for scenario in SCENARIOS:
        line = curves.loc[
            curves["scenario"].eq(scenario)
            & curves["support_segment"].eq("common_empirical_support")
        ].sort_values("exposure")
        curve_ax.fill_between(
            line["exposure"].to_numpy(float),
            line["log_rr_low"].to_numpy(float),
            line["log_rr_high"].to_numpy(float),
            color=COLORS[scenario],
            alpha=0.10,
            linewidth=0,
        )
        curve_ax.plot(
            line["exposure"].to_numpy(float),
            line["log_rr"].to_numpy(float),
            color=COLORS[scenario],
            linewidth=1.45 if scenario == "fixed_12_day" else 1.1,
            label=SCENARIO_LABELS[scenario],
        )
    curve_ax.axhline(0, color="#8B949A", linestyle=(0, (3, 2)), linewidth=0.7)
    curve_ax.set_xlabel(EXPOSURE_LABEL)
    curve_ax.set_ylabel("Cumulative log-RR")
    curve_ax.set_title("a  National pooled response", loc="left", fontsize=8.6, weight="bold")
    curve_ax.grid(axis="y", color="#E7EAEC", linewidth=0.45)

    x_base = np.arange(5)
    offsets = {"fixed_7_day": -0.18, "fixed_12_day": 0, "dynamic_8_12_day": 0.18}
    for scenario in SCENARIOS:
        points = percentile.loc[percentile["scenario"].eq(scenario)].sort_values(
            "percentile_number"
        )
        x = x_base + offsets[scenario]
        af_ax.vlines(
            x,
            points["ci_low"],
            points["ci_high"],
            color=COLORS[scenario],
            linewidth=0.85,
            alpha=0.75,
        )
        within = points["support_rule"].eq("within_common_empirical_support").to_numpy()
        af_ax.scatter(
            x[within],
            points.loc[within, "estimate"],
            s=19,
            color=COLORS[scenario],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        af_ax.scatter(
            x[~within],
            points.loc[~within, "estimate"],
            s=19,
            facecolor="white",
            edgecolor=COLORS[scenario],
            linewidth=0.8,
            zorder=3,
        )
    af_ax.axhline(0, color="#8B949A", linestyle=(0, (3, 2)), linewidth=0.7)
    af_ax.set_xticks(x_base)
    af_ax.set_xticklabels(["25th", "50th", "75th", "90th", "95th"])
    af_ax.set_xlabel(f"Positive {EXPOSURE_LABEL} percentile")
    af_ax.set_ylabel("Signed attributable fraction (%)")
    af_ax.set_title("b  Percentile-specific AF", loc="left", fontsize=8.6, weight="bold")
    af_ax.grid(axis="y", color="#E7EAEC", linewidth=0.45)
    af_ax.text(
        0.5,
        -0.27,
        "Filled symbols: within shared empirical support; open symbols: linear-tail sensitivity.",
        transform=af_ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        color="#5A5A5A",
    )

    handles, labels = curve_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.05),
        ncol=3,
        handlelength=2.2,
    )
    fig.suptitle(
        f"Lag-window sensitivity under an identical shared {EXPOSURE_LABEL} basis",
        x=0.01,
        y=1.12,
        ha="left",
        fontsize=9.5,
        weight="bold",
    )
    save_figure(fig, f"lag_window_comparison_{INDICATOR}_composite")


def write_contrasts(af: pd.DataFrame, coverage: pd.DataFrame, lag: pd.DataFrame) -> None:
    selected = af.loc[
        af["estimand"].eq("positive_exposure_percentile_af")
        & af["exposure_percentile"].isin(["p50", "p90"])
    ].copy()
    wide = selected.pivot(
        index=["estimand", "exposure_percentile", "support_rule"],
        columns="scenario",
        values="estimate",
    ).reset_index()
    wide["dynamic_minus_fixed12_af_pp"] = (
        wide["dynamic_8_12_day"] - wide["fixed_12_day"]
    )
    wide["fixed7_minus_fixed12_af_pp"] = wide["fixed_7_day"] - wide["fixed_12_day"]
    wide.to_csv(RESULTS / "lag_window_af_contrasts.csv", index=False)

    lag_summary = (
        lag.groupby(["scenario", "lag_days"], as_index=False)
        .agg(n_cities=("city", "nunique"))
        .sort_values(["scenario", "lag_days"])
    )
    lag_summary.to_csv(RESULTS / "lag_window_city_count_summary.csv", index=False)
    coverage.to_csv(RESULTS / "lag_window_model_coverage.csv", index=False)


def main() -> None:
    configure_style()
    curves, af, coverage, lag = load_sources()
    draw_comparison(curves, af)
    write_contrasts(af, coverage, lag)


if __name__ == "__main__":
    main()
