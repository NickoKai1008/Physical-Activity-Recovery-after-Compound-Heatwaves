"""Build one integrated common-basis phenotype, AF and robustness figure."""

from __future__ import annotations

import os
from pathlib import Path

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
METHOD_ORDER = [
    "archived_DTW_Ward",
    "path_normalized_DTW_Ward",
    "normalized_DTW_average",
    "normalized_DTW_complete",
    "normalized_DTW_kmedoids",
]
METHOD_LABELS = [
    "Archived\nWard",
    "Path-normalized\nWard",
    "Normalized\naverage",
    "Normalized\ncomplete",
    "Normalized\nmedoids",
]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.8,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def padded_limits(values: np.ndarray, reference: float = 0.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    finite = np.r_[finite, reference]
    lower, upper = float(finite.min()), float(finite.max())
    span = max(upper - lower, 0.06)
    return lower - 0.08 * span, upper + 0.08 * span


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def interpolate_contrasts(
    fixed: pd.DataFrame, exposure_value: float, exposure_percentile: str
) -> pd.DataFrame:
    rows: list[dict] = []
    for (specification, cluster), part in fixed.groupby(["specification", "cluster"]):
        part = part.sort_values("exposure")
        rows.append(
            {
                "specification": specification,
                "cluster": int(cluster),
                "exposure_percentile": exposure_percentile,
                "exposure_value": exposure_value,
                "log_rr": float(np.interp(exposure_value, part["exposure"], part["log_rr"])),
                "ci_low": float(np.interp(exposure_value, part["exposure"], part["log_rr_low"])),
                "ci_high": float(np.interp(exposure_value, part["exposure"], part["log_rr_high"])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    configure()
    curves = pd.read_csv(RESULTS / "exact_common_basis_partition_curves.csv")
    histograms = pd.read_csv(RESULTS / "exact_common_basis_partition_histograms.csv")
    af = pd.read_csv(RESULTS / "exact_common_basis_partition_af_percentiles_mc.csv")
    specification = pd.read_csv(RESULTS / "submission_common_basis_specification.csv").iloc[0]

    dynamic = curves.loc[
        curves["analysis"].eq("phenotype_specific_dynamic_lag")
        & curves["specification"].eq("archived_DTW_Ward")
    ].copy()
    dynamic_hist = histograms.loc[
        histograms["analysis"].eq("phenotype_specific_dynamic_lag")
        & histograms["specification"].eq("archived_DTW_Ward")
    ].copy()
    dynamic_af = af.loc[
        af["analysis"].eq("phenotype_specific_dynamic_lag")
        & af["specification"].eq("archived_DTW_Ward")
    ].copy()
    fixed = curves.loc[curves["analysis"].eq("fixed_12_day_cluster_sensitivity")].copy()

    common_high = float(specification["selected_boundary_high"])
    p25 = float(specification["pooled_positive_p25"])
    p90 = float(specification["pooled_positive_p90"])
    method_p25 = interpolate_contrasts(fixed, p25, "p25")
    method_p90 = interpolate_contrasts(fixed, p90, "p90")

    DATA.mkdir(parents=True, exist_ok=True)
    dynamic.to_csv(DATA / "integrated_common_basis_phenotype_curves.csv", index=False)
    dynamic_hist.to_csv(DATA / "integrated_common_basis_histograms.csv", index=False)
    dynamic_af.to_csv(DATA / "integrated_common_basis_af_percentiles.csv", index=False)
    pd.concat([method_p25, method_p90], ignore_index=True).to_csv(
        DATA / "integrated_common_basis_clustering_method_contrasts.csv", index=False
    )

    fig = plt.figure(figsize=(8.0, 7.95))
    outer = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.70, 0.92, 1.30],
        hspace=0.40,
        wspace=0.32,
        left=0.075,
        right=0.985,
        top=0.955,
        bottom=0.075,
    )
    percentile_order = ["p25", "p50", "p75", "p90", "p95"]

    for column, cluster in enumerate([1, 2, 3, 4]):
        color = CLUSTER_COLORS[cluster]
        top = outer[0, column].subgridspec(2, 1, height_ratios=[2.7, 1.0], hspace=0.04)
        ax_curve = fig.add_subplot(top[0])
        ax_hist = fig.add_subplot(top[1], sharex=ax_curve)
        ax_af = fig.add_subplot(outer[1, column])

        curve = dynamic.loc[dynamic["cluster"].eq(cluster)].sort_values("exposure")
        histogram = dynamic_hist.loc[dynamic_hist["cluster"].eq(cluster)].sort_values("bin_left")
        af_part = (
            dynamic_af.loc[dynamic_af["cluster"].eq(cluster)]
            .assign(
                exposure_percentile=lambda frame: pd.Categorical(
                    frame["exposure_percentile"], categories=percentile_order, ordered=True
                )
            )
            .sort_values("exposure_percentile")
        )
        if curve.empty or histogram.empty or af_part.empty:
            raise RuntimeError(f"Incomplete integrated figure data for cluster {cluster}")

        x = curve["exposure"].to_numpy(float)
        estimate = curve["log_rr"].to_numpy(float)
        lower = curve["log_rr_low"].to_numpy(float)
        upper = curve["log_rr_high"].to_numpy(float)
        common = curve["support_segment"].eq("common_empirical_support").to_numpy()
        ax_curve.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
        ax_curve.plot(x[common], estimate[common], color=color, lw=1.55)
        ax_curve.plot(x[~common], estimate[~common], color=color, lw=1.25, ls=(0, (3, 2)))
        ax_curve.axvline(common_high, color="#737C80", lw=0.75, ls=(0, (2, 2)))
        ax_curve.axhline(0, color="#8A8A8A", lw=0.65, ls=(0, (3, 2)))
        ax_curve.set_ylim(*padded_limits(np.r_[estimate, lower, upper]))
        ax_curve.set_xlim(float(x.min()), float(x.max()))
        ax_curve.tick_params(axis="x", labelbottom=False, length=2.2)
        ax_curve.grid(axis="y", color="#E9E9E9", lw=0.48)
        n_cities = int(curve["n_cities"].iloc[0])
        lag_min = int(curve["lag_min"].iloc[0]) + 1
        lag_max = int(curve["lag_max"].iloc[0]) + 1
        lag_label = f"{lag_min} d" if lag_min == lag_max else f"{lag_min}-{lag_max} d"
        ax_curve.set_title(
            f"C{cluster}  n={n_cities}  lag={lag_label}",
            loc="left",
            color=color,
            weight="bold",
            pad=3,
        )
        if column == 0:
            ax_curve.set_ylabel("Cumulative log-RR")

        widths = histogram["bin_right"].to_numpy(float) - histogram["bin_left"].to_numpy(float)
        histogram_common = histogram["bin_midpoint"].to_numpy(float) <= common_high
        for mask, fill, alpha in (
            (histogram_common, color, 0.50),
            (~histogram_common, "#B8BEC1", 0.72),
        ):
            ax_hist.bar(
                histogram.loc[mask, "bin_left"].to_numpy(float),
                histogram.loc[mask, "count"].to_numpy(float),
                width=widths[mask],
                align="edge",
                color=fill,
                alpha=alpha,
                edgecolor=fill,
                linewidth=0.16,
            )
        ax_hist.axvline(common_high, color="#737C80", lw=0.75, ls=(0, (2, 2)))
        ax_hist.set_xlabel(EXPOSURE_LABEL)
        ax_hist.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_hist.tick_params(length=2.2)
        if column == 0:
            ax_hist.set_ylabel("Grid-days")

        positions = np.arange(len(af_part))
        point = af_part["af_percent"].to_numpy(float)
        low = af_part["ci_low"].to_numpy(float)
        high = af_part["ci_high"].to_numpy(float)
        within = af_part["exposure_value"].to_numpy(float) <= common_high
        ax_af.vlines(positions, low, high, color=color, lw=0.85, alpha=0.78)
        ax_af.scatter(
            positions[within], point[within], s=20, color=color, edgecolor="white", linewidth=0.4, zorder=3
        )
        ax_af.scatter(
            positions[~within],
            point[~within],
            s=20,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=3,
        )
        ax_af.axhline(0, color="#8A8A8A", lw=0.65, ls=(0, (3, 2)))
        ax_af.set_xticks(positions)
        ax_af.set_xticklabels(["25", "50", "75", "90", "95"])
        ax_af.set_ylim(*padded_limits(np.r_[point, low, high]))
        ax_af.set_title(f"C{cluster} dose-specific AF", loc="left", color=color, weight="bold", pad=3)
        ax_af.set_xlabel("Positive-exposure percentile")
        if column == 0:
            ax_af.set_ylabel("Signed AF (%)")
        ax_af.grid(axis="y", color="#E9E9E9", lw=0.48)
        ax_af.tick_params(length=2.2)

    ax_method = fig.add_subplot(outer[2, :])
    positions = np.arange(len(METHOD_ORDER))
    offsets = {1: -0.24, 2: -0.08, 3: 0.08, 4: 0.24}
    for cluster in [1, 2, 3, 4]:
        values = method_p25.loc[method_p25["cluster"].eq(cluster)].set_index("specification").reindex(METHOD_ORDER)
        x = positions + offsets[cluster]
        ax_method.vlines(x, values["ci_low"], values["ci_high"], color=CLUSTER_COLORS[cluster], lw=0.85, alpha=0.74)
        ax_method.scatter(
            x,
            values["log_rr"],
            s=20,
            color=CLUSTER_COLORS[cluster],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
            label=f"C{cluster}",
        )
    ax_method.axhline(0, color="#777777", lw=0.7, ls=(0, (3, 2)))
    ax_method.set_xticks(positions)
    ax_method.set_xticklabels(METHOD_LABELS)
    ax_method.set_ylabel(f"Cumulative log-RR at {EXPOSURE_LABEL} p25")
    ax_method.set_title(
        "Clustering-method robustness within shared empirical support (fixed lag 0-11)",
        loc="left",
        weight="bold",
        pad=4,
    )
    ax_method.legend(ncol=4, loc="upper right", handletextpad=0.3, columnspacing=0.9)
    ax_method.grid(axis="y", color="#E8E8E8", lw=0.52)
    ax_method.tick_params(length=2.3)

    fig.text(0.015, 0.982, "a", fontsize=9.5, weight="bold", va="top")
    fig.text(0.006, 0.505, "b", fontsize=9.5, weight="bold", va="top")
    fig.text(0.006, 0.270, "c", fontsize=9.5, weight="bold", va="top")
    fig.suptitle(
        f"Phenotype-specific {EXPOSURE_LABEL} response, attributable fraction and robustness",
        x=0.075,
        y=0.988,
        ha="left",
        fontsize=9.8,
        weight="bold",
    )
    fig.text(
        0.075,
        0.018,
        "Solid curves and filled AF symbols are supported by all included cities; dashed curves and open symbols are linear-tail sensitivities. "
        "Histograms retain the observed positive-exposure distribution through each partition's p99; bins beyond shared support are gray and values above p99 are not displayed. "
        "Error bars are empirical 95% CIs from 500 coefficient draws.",
        fontsize=6.2,
        color="#555555",
    )
    save(fig, f"figure4_integrated_common_basis_phenotype_af_robustness{FILE_SUFFIX}")
    plt.close(fig)


if __name__ == "__main__":
    main()
