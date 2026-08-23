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
            "font.size": 7,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def main() -> None:
    configure()
    curves = pd.read_csv(RESULTS / "exact_common_basis_partition_curves.csv")
    coverage = pd.read_csv(RESULTS / "exact_common_basis_partition_coverage.csv")
    specification = pd.read_csv(RESULTS / "submission_common_basis_specification.csv").iloc[0]
    DATA.mkdir(parents=True, exist_ok=True)
    curves.to_csv(DATA / "exact_common_basis_partition_curves.csv", index=False)
    coverage.to_csv(DATA / "exact_common_basis_partition_coverage.csv", index=False)

    dynamic = curves.loc[
        curves["analysis"].eq("phenotype_specific_dynamic_lag")
    ].copy()
    fixed = curves.loc[
        curves["analysis"].eq("fixed_12_day_cluster_sensitivity")
    ].copy()

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=0.48, wspace=0.32)

    for index, cluster in enumerate([1, 2, 3, 4]):
        ax = fig.add_subplot(gs[0, index])
        part = dynamic.loc[dynamic["cluster"].eq(cluster)].sort_values("exposure")
        color = CLUSTER_COLORS[cluster]
        ax.fill_between(
            part["exposure"].to_numpy(float),
            part["log_rr_low"].to_numpy(float),
            part["log_rr_high"].to_numpy(float),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            part["exposure"].to_numpy(float),
            part["log_rr"].to_numpy(float),
            color=color,
            lw=1.6,
        )
        ax.axhline(0, color="#777777", lw=0.7, ls=(0, (3, 2)))
        n = int(part["n_cities"].iloc[0])
        lag_min = int(part["lag_min"].iloc[0]) + 1
        lag_max = int(part["lag_max"].iloc[0]) + 1
        lag_text = f"{lag_min} d" if lag_min == lag_max else f"{lag_min}-{lag_max} d"
        ax.set_title(f"C{cluster}  n={n}  lag={lag_text}", color=color, weight="bold", pad=4)
        ax.set_xlabel(EXPOSURE_LABEL)
        if index == 0:
            ax.set_ylabel("Cumulative log-RR")
        ax.tick_params(length=2.5, width=0.7)

    p90 = float(specification["pooled_positive_p90"])
    rows: list[dict] = []
    for (specification, cluster), part in fixed.groupby(["specification", "cluster"]):
        part = part.sort_values("exposure")
        rows.append(
            {
                "specification": specification,
                "cluster": int(cluster),
                "log_rr_p90": float(np.interp(p90, part["exposure"], part["log_rr"])),
                "low_p90": float(np.interp(p90, part["exposure"], part["log_rr_low"])),
                "high_p90": float(np.interp(p90, part["exposure"], part["log_rr_high"])),
            }
        )
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(DATA / "cluster_method_p90_logrr_contrasts.csv", index=False)

    ax = fig.add_subplot(gs[1, :])
    method_order = [
        "archived_DTW_Ward",
        "path_normalized_DTW_Ward",
        "normalized_DTW_average",
        "normalized_DTW_complete",
        "normalized_DTW_kmedoids",
    ]
    method_labels = [
        "Archived Ward",
        "Path-normalized Ward",
        "Normalized average",
        "Normalized complete",
        "Normalized medoids",
    ]
    positions = np.arange(len(method_order))
    offsets = {1: -0.24, 2: -0.08, 3: 0.08, 4: 0.24}
    for cluster in [1, 2, 3, 4]:
        part = contrasts.loc[contrasts["cluster"].eq(cluster)].set_index("specification")
        values = part.reindex(method_order)
        x = positions + offsets[cluster]
        ax.vlines(
            x,
            values["low_p90"],
            values["high_p90"],
            color=CLUSTER_COLORS[cluster],
            lw=0.85,
            alpha=0.72,
        )
        ax.scatter(
            x,
            values["log_rr_p90"],
            s=19,
            color=CLUSTER_COLORS[cluster],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
            label=f"C{cluster}",
        )
    ax.axhline(0, color="#777777", lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(positions)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel(f"Cumulative log-RR at {EXPOSURE_LABEL} p90")
    ax.set_title("Fixed lag 0-11: phenotype estimates across DTW and clustering specifications", loc="left", weight="bold")
    ax.legend(ncol=4, loc="upper right", handletextpad=0.35, columnspacing=1.0)
    ax.grid(axis="y", color="#E8E8E8", lw=0.55)
    ax.tick_params(axis="x", rotation=0)

    fig.text(0.01, 0.985, "a", fontsize=9, weight="bold", va="top")
    fig.text(0.01, 0.455, "b", fontsize=9, weight="bold", va="top")
    fig.suptitle(
        f"Phenotype-specific {EXPOSURE_LABEL} response and clustering-method robustness",
        x=0.04,
        y=1.025,
        ha="left",
        fontsize=9,
        weight="bold",
    )
    save(fig, f"exact_common_basis_partition_robustness{FILE_SUFFIX}")
    plt.close(fig)


if __name__ == "__main__":
    main()
