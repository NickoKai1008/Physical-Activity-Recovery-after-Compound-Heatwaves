"""Create the corrected Figure 4 from exact common-basis phenotype results."""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_MODULE_ROOT = Path(__file__).resolve().parents[1]
INDICATOR = os.environ.get("HEATPA_INDICATOR", "cehwi").lower()
if INDICATOR not in {"cehwi", "exceeded_quantity"}:
    raise ValueError("HEATPA_INDICATOR must be cehwi or exceeded_quantity")
MODULE_ROOT = BASE_MODULE_ROOT
INDICATOR_DIR = "cehwi" if INDICATOR == "cehwi" else "exceeded_quantity"
RESULTS = MODULE_ROOT / "data" / "common_basis" / INDICATOR_DIR / "model_summaries"
DATA = MODULE_ROOT / "data" / "common_basis" / INDICATOR_DIR
FIGURES = MODULE_ROOT / "output" / "common_basis" / INDICATOR_DIR
EXPOSURE_LABEL = "CEHWI" if INDICATOR == "cehwi" else "Exceeded quantity"
FILE_SUFFIX = "" if INDICATOR == "cehwi" else "_exceeded_quantity"

CLUSTER_COLORS = {
    1: "#B73A3A",
    2: "#E39A12",
    3: "#5A9FC7",
    4: "#244F87",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def padded_limits(values: np.ndarray, reference: float = 0.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    finite = np.r_[finite, reference]
    lower, upper = float(finite.min()), float(finite.max())
    span = max(upper - lower, 0.08)
    return lower - 0.08 * span, upper + 0.08 * span


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def main() -> None:
    configure()
    curves = pd.read_csv(RESULTS / "exact_common_basis_partition_curves.csv")
    histograms = pd.read_csv(RESULTS / "exact_common_basis_partition_histograms.csv")
    af = pd.read_csv(RESULTS / "exact_common_basis_partition_af_percentiles_mc.csv")
    coverage = pd.read_csv(RESULTS / "exact_common_basis_partition_coverage.csv")
    specification = pd.read_csv(RESULTS / "submission_common_basis_specification.csv").iloc[0]
    common_support_high = float(specification["selected_boundary_high"])

    selector = (
        curves["analysis"].eq("phenotype_specific_dynamic_lag")
        & curves["specification"].eq("archived_DTW_Ward")
    )
    curves = curves.loc[selector].copy()
    histograms = histograms.loc[
        histograms["analysis"].eq("phenotype_specific_dynamic_lag")
        & histograms["specification"].eq("archived_DTW_Ward")
    ].copy()
    af = af.loc[
        af["analysis"].eq("phenotype_specific_dynamic_lag")
        & af["specification"].eq("archived_DTW_Ward")
    ].copy()
    coverage = coverage.loc[
        coverage["analysis"].eq("phenotype_specific_dynamic_lag")
        & coverage["specification"].eq("archived_DTW_Ward")
    ].copy()

    DATA.mkdir(parents=True, exist_ok=True)
    curves.to_csv(DATA / "figure4_exact_common_basis_phenotype_curves.csv", index=False)
    histograms.to_csv(DATA / "figure4_exact_common_basis_exposure_histograms.csv", index=False)
    af.to_csv(DATA / "figure4_exact_common_basis_af_percentiles_mc.csv", index=False)
    coverage.to_csv(DATA / "figure4_exact_common_basis_cluster_coverage.csv", index=False)

    fig = plt.figure(figsize=(7.25, 8.3))
    outer = fig.add_gridspec(
        4,
        2,
        width_ratios=[1.75, 1.0],
        hspace=0.43,
        wspace=0.31,
        left=0.085,
        right=0.985,
        top=0.95,
        bottom=0.085,
    )
    percentile_order = ["p25", "p50", "p75", "p90", "p95"]

    for row, cluster in enumerate([1, 2, 3, 4]):
        color = CLUSTER_COLORS[cluster]
        left = outer[row, 0].subgridspec(2, 1, height_ratios=[2.35, 0.9], hspace=0.04)
        ax_curve = fig.add_subplot(left[0])
        ax_hist = fig.add_subplot(left[1], sharex=ax_curve)
        ax_af = fig.add_subplot(outer[row, 1])

        curve_all = curves.loc[curves["cluster"].eq(cluster)].sort_values("exposure")
        curve = curve_all.loc[
            curve_all["support_segment"].eq("common_empirical_support")
        ].copy()
        histogram = histograms.loc[histograms["cluster"].eq(cluster)].sort_values("bin_left")
        af_part = (
            af.loc[af["cluster"].eq(cluster)]
            .assign(exposure_percentile=lambda frame: pd.Categorical(
                frame["exposure_percentile"], categories=percentile_order, ordered=True
            ))
            .sort_values("exposure_percentile")
        )
        if curve.empty or histogram.empty or af_part.empty:
            raise RuntimeError(f"Incomplete corrected Figure 4 data for cluster {cluster}")

        x = curve["exposure"].to_numpy(float)
        estimate = curve["log_rr"].to_numpy(float)
        lower = curve["log_rr_low"].to_numpy(float)
        upper = curve["log_rr_high"].to_numpy(float)
        ax_curve.fill_between(x, lower, upper, color="#9AA4A8", alpha=0.22, linewidth=0)
        ax_curve.plot(x, estimate, color="#26383F", lw=1.45)
        ax_curve.axhline(0, color="#8A8A8A", lw=0.65, ls=(0, (3, 2)))
        ax_curve.set_ylim(*padded_limits(np.r_[estimate, lower, upper]))
        ax_curve.set_xlim(0, float(curve["common_support_high"].iloc[0]))
        ax_curve.tick_params(axis="x", labelbottom=False, length=2.3)
        ax_curve.grid(axis="y", color="#E9E9E9", lw=0.5)
        n_cities = int(curve["n_cities"].iloc[0])
        lag_min = int(curve["lag_min"].iloc[0]) + 1
        lag_max = int(curve["lag_max"].iloc[0]) + 1
        lag_label = f"{lag_min} d" if lag_min == lag_max else f"{lag_min}-{lag_max} d"
        ax_curve.set_title(
            f"C{cluster}  n={n_cities}  phenotype lag={lag_label}",
            loc="left",
            color=color,
            weight="bold",
            pad=3,
        )
        ax_curve.set_ylabel("Cumulative log-RR")

        widths = histogram["bin_right"].to_numpy(float) - histogram["bin_left"].to_numpy(float)
        ax_hist.bar(
            histogram["bin_left"].to_numpy(float),
            histogram["count"].to_numpy(float),
            width=widths,
            align="edge",
            color=color,
            alpha=0.60,
            edgecolor=color,
            linewidth=0.22,
        )
        ax_hist.set_ylabel("Grid-days")
        ax_hist.set_xlabel(EXPOSURE_LABEL)
        ax_hist.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_hist.spines["top"].set_visible(False)
        ax_hist.tick_params(length=2.3)

        positions = np.arange(len(af_part))
        point = af_part["af_percent"].to_numpy(float)
        low = af_part["ci_low"].to_numpy(float)
        high = af_part["ci_high"].to_numpy(float)
        ax_af.vlines(positions, low, high, color=color, lw=0.9, alpha=0.78)
        within_support = af_part["exposure_value"].to_numpy(float) <= float(
            curve["common_support_high"].iloc[0]
        )
        ax_af.scatter(
            positions[within_support],
            point[within_support],
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax_af.scatter(
            positions[~within_support],
            point[~within_support],
            s=22,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=3,
        )
        ax_af.axhline(0, color="#8A8A8A", lw=0.65, ls=(0, (3, 2)))
        ax_af.set_xticks(positions)
        ax_af.set_xticklabels(["25th", "50th", "75th", "90th", "95th"])
        ax_af.set_ylim(*padded_limits(np.r_[point, low, high]))
        ax_af.set_ylabel("Signed AF (%)")
        ax_af.set_xlabel("Positive-exposure percentile")
        ax_af.grid(axis="y", color="#E9E9E9", lw=0.5)
        ax_af.set_title("Dose-specific attributable fraction", loc="left", weight="bold", pad=3)
        ax_af.tick_params(length=2.3)

    fig.text(0.02, 0.985, "a", fontsize=10, weight="bold", va="top")
    fig.text(0.69, 0.985, "b", fontsize=10, weight="bold", va="top")
    fig.suptitle(
        "Common-support phenotype-specific response and attributable fraction",
        x=0.085,
        y=0.988,
        ha="left",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.085,
        0.018,
        "Open AF symbols lie beyond the shared empirical-support boundary and use the linear-tail sensitivity. "
        "The histogram is restricted to shared empirical support; the integrated summary retains the full distribution through each partition's p99.",
        fontsize=6.6,
        color="#555555",
    )
    save(fig, f"figure4_exact_common_basis_phenotype_rr_af{FILE_SUFFIX}")
    plt.close(fig)

    tail_fig, tail_axes = plt.subplots(2, 2, figsize=(7.2, 4.9), sharex=False)
    for axis, cluster in zip(tail_axes.flat, [1, 2, 3, 4]):
        curve = curves.loc[curves["cluster"].eq(cluster)].sort_values("exposure")
        common = curve["support_segment"].eq("common_empirical_support").to_numpy()
        x = curve["exposure"].to_numpy(float)
        estimate = curve["log_rr"].to_numpy(float)
        lower = curve["log_rr_low"].to_numpy(float)
        upper = curve["log_rr_high"].to_numpy(float)
        axis.fill_between(x, lower, upper, color="#9AA4A8", alpha=0.17, linewidth=0)
        axis.plot(x[common], estimate[common], color="#26383F", lw=1.35)
        axis.plot(
            x[~common],
            estimate[~common],
            color="#26383F",
            lw=1.1,
            ls=(0, (3, 2)),
        )
        axis.axvline(common_support_high, color="#777777", lw=0.7, ls=(0, (2, 2)))
        axis.axhline(0, color="#999999", lw=0.6, ls=(0, (3, 2)))
        axis.set_ylim(*padded_limits(np.r_[estimate, lower, upper]))
        axis.set_title(f"C{cluster}", loc="left", color=CLUSTER_COLORS[cluster], weight="bold")
        axis.set_xlabel(EXPOSURE_LABEL)
        axis.set_ylabel("Cumulative log-RR")
        axis.grid(axis="y", color="#E9E9E9", lw=0.5)
    tail_fig.suptitle(
        "Linear-tail sensitivity beyond the shared empirical support boundary",
        x=0.08,
        ha="left",
        fontsize=9,
        weight="bold",
    )
    tail_fig.tight_layout()
    save(tail_fig, f"figure4_exact_common_basis_linear_tail_sensitivity{FILE_SUFFIX}")
    plt.close(tail_fig)


if __name__ == "__main__":
    main()
