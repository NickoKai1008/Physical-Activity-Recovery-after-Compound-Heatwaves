"""
Create Fig. 6e: future city-level heatwave-attributable PA loss forests.

The figure presents future city-level heatwave-attributable PA loss fractions
calculated by applying the national pooled historical DLNM response function
to projected 2025-2050 heatwave exposures. The input is the archived city-year
projection table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
DATA_PATH = MODULE_DIR / "data" / "figure6e" / "fig6e_future_city_pa_loss_forest_plot_data.csv"
OUT_ROOT = MODULE_DIR / "output" / "figure6e_generated"
DTW4LAG12_CLUSTER_SOURCE = MODULE_DIR / "data" / "figure6e" / "dtw4lag12_city_cluster_map.csv"

YEARS = list(range(2025, 2051))
SCENARIOS = ["ssp245", "ssp370", "ssp585"]
SCENARIO_LABELS = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
INDICATORS = ["cehwi", "exceeded_quantity"]
INDICATOR_LABELS = {
    "cehwi": "CEHWI",
    "exceeded_quantity": "Exceeded cumulative intensity",
}
HEATWAVES = ["composite", "day", "night"]
HEATWAVE_LABELS = {
    "composite": "Compound",
    "day": "Daytime",
    "night": "Nighttime",
}
HEATWAVE_COLORS = {
    "composite": "#A12D32",
    "day": "#E3A000",
    "night": "#4C8EBA",
}
DTW4LAG12_CLUSTER_COLORS = {
    1: "#B83A34",
    2: "#E39D00",
    3: "#5EA4CE",
    4: "#1F4F8C",
    5: "#BDBDBD",
}
DTW4LAG12_CLUSTER_LABELS = {
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
    5: "Outside estimable phenotype subset",
}


def standardize_city_name(name: str) -> str:
    return str(name).replace(".", "").replace(" ", "_")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.85,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 8.2,
            "legend.frameon": False,
            "figure.dpi": 260,
            "savefig.dpi": 420,
        }
    )


def _pct_from_sums(g: pd.DataFrame, sum_col: str) -> float:
    n_days = pd.to_numeric(g["n_days"], errors="coerce")
    denom = n_days.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return 100.0 * pd.to_numeric(g[sum_col], errors="coerce").sum() / denom


def load_and_aggregate() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    if "year" not in df.columns:
        return df.copy()
    df = df[
        df["year"].isin(YEARS)
        & df["activity_type"].eq("all")
        & df["scenario"].isin(SCENARIOS)
        & df["indicator"].isin(INDICATORS)
        & df["heatwave_type"].isin(HEATWAVES)
    ].copy()
    group_cols = ["scenario", "city_standard", "indicator", "heatwave_type", "activity_type"]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["city_label"] = row["city_standard"].replace("_", " ")
        row["period"] = "2025-2050"
        row["n_days"] = pd.to_numeric(g["n_days"], errors="coerce").sum()
        row["heatwave_days"] = pd.to_numeric(g["heatwave_days"], errors="coerce").sum()
        row["pa_loss_fraction_percent"] = _pct_from_sums(g, "pa_loss_sum_days")
        row["pa_loss_fraction_low_percent"] = _pct_from_sums(g, "pa_loss_low_sum_days")
        row["pa_loss_fraction_high_percent"] = _pct_from_sums(g, "pa_loss_high_sum_days")
        row["annualized_asri_percent"] = _pct_from_sums(g, "asri_sum_days")
        row["annualized_asri_low_percent"] = _pct_from_sums(g, "asri_low_sum_days")
        row["annualized_asri_high_percent"] = _pct_from_sums(g, "asri_high_sum_days")
        rows.append(row)
    out = pd.DataFrame(rows)
    out["scenario_label"] = out["scenario"].map(SCENARIO_LABELS)
    out["indicator_label"] = out["indicator"].map(INDICATOR_LABELS)
    out["heatwave_label"] = out["heatwave_type"].map(HEATWAVE_LABELS)
    return out


def load_dtw4lag12_cluster_map(out_dir: Path) -> pd.DataFrame:
    archive_path = out_dir / "dtw4lag12_city_cluster_map.csv"
    if archive_path.exists():
        cluster_df = pd.read_csv(archive_path)
        if "city_label" in cluster_df.columns:
            cluster_df["city_standard"] = cluster_df["city_label"].map(standardize_city_name)
    elif DTW4LAG12_CLUSTER_SOURCE.exists():
        src = pd.read_csv(DTW4LAG12_CLUSTER_SOURCE)
        if {"city_label", "dtw4lag12_cluster"}.issubset(src.columns):
            cluster_df = src.copy()
            if "city_standard" not in cluster_df.columns:
                cluster_df["city_standard"] = cluster_df["city_label"].map(standardize_city_name)
        else:
            cluster_df = src[["City", "Group"]].rename(
                columns={"City": "city_label", "Group": "dtw4lag12_cluster"}
            )
            cluster_df["city_standard"] = cluster_df["city_label"].map(standardize_city_name)
    else:
        raise FileNotFoundError(
            f"Could not find DTW4lag12 cluster map at {archive_path} or {DTW4LAG12_CLUSTER_SOURCE}"
        )
    cluster_df["dtw4lag12_cluster"] = pd.to_numeric(
        cluster_df["dtw4lag12_cluster"], errors="coerce"
    ).astype("Int64")
    cluster_df["dtw4lag12_cluster_label"] = cluster_df["dtw4lag12_cluster"].map(
        DTW4LAG12_CLUSTER_LABELS
    )
    cluster_df["dtw4lag12_cluster_color"] = cluster_df["dtw4lag12_cluster"].map(
        DTW4LAG12_CLUSTER_COLORS
    )
    cluster_df = cluster_df[
        [
            "city_standard",
            "city_label",
            "dtw4lag12_cluster",
            "dtw4lag12_cluster_label",
            "dtw4lag12_cluster_color",
        ]
    ].drop_duplicates()
    cluster_df.to_csv(archive_path, index=False)
    return cluster_df


def city_order_for_indicator(data: pd.DataFrame, indicator: str) -> list[str]:
    ref = data[
        data["indicator"].eq(indicator)
        & data["scenario"].eq("ssp585")
        & data["heatwave_type"].eq("composite")
    ].copy()
    ref = ref.sort_values("pa_loss_fraction_percent", ascending=False)
    return ref["city_standard"].tolist()


def plot_indicator_all_ssps(data: pd.DataFrame, indicator: str, out_dir: Path) -> None:
    sub = data[data["indicator"].eq(indicator)].copy()
    if sub.empty:
        return
    city_order = city_order_for_indicator(data, indicator)
    city_labels = [c.replace("_", " ") for c in city_order]
    n_city = len(city_order)
    y_base = np.arange(n_city)
    offsets = {"composite": -0.22, "day": 0.0, "night": 0.22}

    fig, axes = plt.subplots(1, 3, figsize=(10.8, max(10.5, n_city * 0.145)), sharey=True)
    all_high = pd.to_numeric(sub["pa_loss_fraction_high_percent"], errors="coerce")
    all_val = pd.to_numeric(sub["pa_loss_fraction_percent"], errors="coerce")
    xmax = np.nanpercentile(pd.concat([all_high, all_val]).to_numpy(float), 99)
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    xmax *= 1.08

    for ax, scenario in zip(axes, SCENARIOS):
        ssub = sub[sub["scenario"].eq(scenario)].copy()
        for heatwave in HEATWAVES:
            hdf = ssub[ssub["heatwave_type"].eq(heatwave)].set_index("city_standard").reindex(city_order)
            y = y_base + offsets[heatwave]
            x = pd.to_numeric(hdf["pa_loss_fraction_percent"], errors="coerce").to_numpy(float)
            lo = pd.to_numeric(hdf["pa_loss_fraction_low_percent"], errors="coerce").to_numpy(float)
            hi = pd.to_numeric(hdf["pa_loss_fraction_high_percent"], errors="coerce").to_numpy(float)
            color = HEATWAVE_COLORS[heatwave]
            for yi, xlo, xhi in zip(y, lo, hi):
                if np.isfinite(xlo) and np.isfinite(xhi):
                    ax.hlines(yi, xlo, xhi, color=color, lw=0.8, alpha=0.33, zorder=1)
            ax.scatter(
                x,
                y,
                s=12,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                label=HEATWAVE_LABELS[heatwave],
                zorder=3,
            )
        ax.axvline(0, color="#888888", lw=0.7, ls=(0, (4, 2)), zorder=0)
        ax.set_xlim(-0.03 * xmax, xmax)
        ax.set_ylim(n_city - 0.5, -0.5)
        ax.grid(axis="x", color="#E7E7E7", lw=0.55)
        ax.set_title(SCENARIO_LABELS[scenario], pad=8)
        ax.set_xlabel("Future PA loss fraction (%)")
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(city_labels)
    axes[0].set_ylabel("City")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate labels while preserving order.
    seen = {}
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen[lab] = h
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.008),
        handletextpad=0.35,
        columnspacing=1.25,
    )
    fig.suptitle(
        f"Fig. 6e | City-level future heatwave-attributable PA loss, {INDICATOR_LABELS[indicator]}, 2025-2050",
        x=0.01,
        y=1.035,
        ha="left",
        fontsize=11.2,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.998,
        "National pooled response curve; points show city-level 2025-2050 means and lines show response-curve uncertainty bounds.",
        ha="left",
        fontsize=7.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=0.9)
    stem = f"fig6e_future_city_pa_loss_forest_{indicator}_all_ssps_2025_2050"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_scenario_indicator(data: pd.DataFrame, indicator: str, scenario: str, out_dir: Path) -> None:
    sub = data[(data["indicator"].eq(indicator)) & (data["scenario"].eq(scenario))].copy()
    city_order = city_order_for_indicator(data, indicator)
    city_labels = [c.replace("_", " ") for c in city_order]
    n_city = len(city_order)
    y_base = np.arange(n_city)
    offsets = {"composite": -0.22, "day": 0.0, "night": 0.22}
    fig, ax = plt.subplots(figsize=(5.2, max(10.5, n_city * 0.145)))
    vals = []
    for heatwave in HEATWAVES:
        hdf = sub[sub["heatwave_type"].eq(heatwave)].set_index("city_standard").reindex(city_order)
        y = y_base + offsets[heatwave]
        x = pd.to_numeric(hdf["pa_loss_fraction_percent"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(hdf["pa_loss_fraction_low_percent"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(hdf["pa_loss_fraction_high_percent"], errors="coerce").to_numpy(float)
        vals.extend(list(hi))
        color = HEATWAVE_COLORS[heatwave]
        for yi, xlo, xhi in zip(y, lo, hi):
            if np.isfinite(xlo) and np.isfinite(xhi):
                ax.hlines(yi, xlo, xhi, color=color, lw=0.8, alpha=0.33, zorder=1)
        ax.scatter(
            x,
            y,
            s=12,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            label=HEATWAVE_LABELS[heatwave],
            zorder=3,
        )
    vals = np.asarray(vals, dtype=float)
    xmax = np.nanpercentile(vals[np.isfinite(vals)], 99) if np.isfinite(vals).any() else 1.0
    xmax = max(xmax * 1.08, 0.25)
    ax.axvline(0, color="#888888", lw=0.7, ls=(0, (4, 2)), zorder=0)
    ax.set_xlim(-0.03 * xmax, xmax)
    ax.set_ylim(n_city - 0.5, -0.5)
    ax.set_yticks(y_base)
    ax.set_yticklabels(city_labels)
    ax.grid(axis="x", color="#E7E7E7", lw=0.55)
    ax.set_xlabel("Future PA loss fraction (%)")
    ax.set_ylabel("City")
    ax.set_title(f"{INDICATOR_LABELS[indicator]}, {SCENARIO_LABELS[scenario]}, 2025-2050")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.035), handletextpad=0.3)
    fig.tight_layout()
    stem = f"fig6e_future_city_pa_loss_forest_{indicator}_{scenario}_2025_2050"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_compound_only_dtw4lag12_clustered(
    data: pd.DataFrame,
    indicator: str,
    cluster_map: pd.DataFrame,
    out_dir: Path,
    clustered_only: bool = True,
) -> None:
    sub = data[
        data["indicator"].eq(indicator) & data["heatwave_type"].eq("composite")
    ].copy()
    sub = sub.merge(
        cluster_map.drop(columns=["city_label"], errors="ignore"),
        on="city_standard",
        how="left",
    )
    if clustered_only:
        sub = sub[sub["dtw4lag12_cluster"].isin([1, 2, 3, 4])].copy()
        suffix = "dtw4lag12_clusters_clustered63"
        legend_clusters = [1, 2, 3, 4]
    else:
        suffix = "dtw4lag12_clusters_all_cities"
        legend_clusters = [1, 2, 3, 4, 5]
    if sub.empty:
        return

    ref = sub[sub["scenario"].eq("ssp585")].sort_values(
        "pa_loss_fraction_percent", ascending=False
    )
    city_order = ref["city_standard"].drop_duplicates().tolist()
    city_labels = [c.replace("_", " ") for c in city_order]
    n_city = len(city_order)
    y_base = np.arange(n_city)

    plot_data = sub[sub["city_standard"].isin(city_order)].copy()
    plot_data.to_csv(
        out_dir / f"fig6e_future_city_pa_loss_forest_{indicator}_compound_only_{suffix}_plot_data.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, max(8.8, n_city * 0.155)), sharey=True)
    xmax_values = pd.to_numeric(plot_data["pa_loss_fraction_high_percent"], errors="coerce")
    xmax_values = xmax_values[np.isfinite(xmax_values)]
    xmax = xmax_values.max() if len(xmax_values) else np.nan
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    xmax *= 1.05

    for ax, scenario in zip(axes, SCENARIOS):
        ssub = (
            plot_data[plot_data["scenario"].eq(scenario)]
            .set_index("city_standard")
            .reindex(city_order)
        )
        x = pd.to_numeric(ssub["pa_loss_fraction_percent"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(ssub["pa_loss_fraction_low_percent"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(ssub["pa_loss_fraction_high_percent"], errors="coerce").to_numpy(float)
        colors = [
            DTW4LAG12_CLUSTER_COLORS.get(int(v), "#BDBDBD")
            if pd.notna(v)
            else "#BDBDBD"
            for v in ssub["dtw4lag12_cluster"].to_list()
        ]
        for yi, xlo, xhi, color in zip(y_base, lo, hi, colors):
            if np.isfinite(xlo) and np.isfinite(xhi):
                ax.hlines(yi, xlo, xhi, color=color, lw=0.85, alpha=0.36, zorder=1)
        ax.scatter(
            x,
            y_base,
            s=13,
            color=colors,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        ax.axvline(0, color="#8A8A8A", lw=0.7, ls=(0, (4, 2)), zorder=0)
        ax.set_xlim(-0.03 * xmax, xmax)
        ax.set_ylim(n_city - 0.5, -0.5)
        ax.grid(axis="x", color="#E7E7E7", lw=0.55)
        ax.set_title(SCENARIO_LABELS[scenario], pad=8)
        ax.set_xlabel("Future PA loss fraction (%)")
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(city_labels)
    axes[0].set_ylabel("City")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles = []
    labels = []
    for cluster_id in legend_clusters:
        count = int(
            cluster_map.loc[
                cluster_map["dtw4lag12_cluster"].eq(cluster_id), "city_standard"
            ].nunique()
        )
        label = f"{DTW4LAG12_CLUSTER_LABELS[cluster_id]} (n={count})"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=DTW4LAG12_CLUSTER_COLORS[cluster_id],
                markeredgecolor="none",
                markersize=5.4,
            )
        )
        labels.append(label)
    fig.legend(
        handles,
        labels,
        ncol=4 if clustered_only else 3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.008),
        handletextpad=0.35,
        columnspacing=1.15,
    )
    display_cluster_scope = "DTW4lag12 clustered cities" if clustered_only else "all cities"
    fig.suptitle(
        f"Fig. 6e | Compound future PA loss by DTW4lag12 phenotype, {INDICATOR_LABELS[indicator]}, 2025-2050",
        x=0.01,
        y=1.035,
        ha="left",
        fontsize=11.2,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.998,
        f"National pooled response curve; cities sorted by SSP5-8.5 compound estimate; colours denote {display_cluster_scope}.",
        ha="left",
        fontsize=7.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=0.9)
    stem = f"fig6e_future_city_pa_loss_forest_{indicator}_compound_only_{suffix}_2025_2050"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def write_readme(data: pd.DataFrame, out_dir: Path) -> None:
    top = (
        data[
            data["scenario"].eq("ssp585")
            & data["indicator"].eq("cehwi")
            & data["heatwave_type"].eq("composite")
        ]
        .sort_values("pa_loss_fraction_percent", ascending=False)
        .head(8)
    )
    lines = [
        "# Fig. 6e future city PA-loss forest",
        "",
        f"Input: `{DATA_PATH}`",
        "",
        "The figure reports future heatwave-attributable PA loss fractions.",
        "Each city value is calculated by applying the national pooled historical DLNM response curve to projected 2025-2050 heatwave exposure.",
        "Non-heatwave days contribute zero, and annual all-day denominator is used.",
        "",
        "Top CEHWI compound cities under SSP5-8.5:",
    ]
    for _, row in top.iterrows():
        lines.append(f"- {row['city_label']}: {row['pa_loss_fraction_percent']:.3f}%")
    lines.extend(
        [
            "",
            "Additional compound-only DTW4lag12 cluster-coloured panels:",
            "- `*_compound_only_dtw4lag12_clusters_clustered63_2025_2050.*`: the 63 cities assigned to C1-C4.",
            "- `*_compound_only_dtw4lag12_clusters_all_cities_2025_2050.*`: the prespecified 75-city frame, with cities outside the estimable phenotype subset shown in grey.",
        ]
    )
    (out_dir / "README_fig6e.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_style()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_and_aggregate()
    cluster_map = load_dtw4lag12_cluster_map(OUT_ROOT)
    data.to_csv(OUT_ROOT / "fig6e_future_city_pa_loss_forest_plot_data.csv", index=False)
    for indicator in INDICATORS:
        plot_indicator_all_ssps(data, indicator, OUT_ROOT)
        plot_compound_only_dtw4lag12_clustered(
            data, indicator, cluster_map, OUT_ROOT, clustered_only=True
        )
        plot_compound_only_dtw4lag12_clustered(
            data, indicator, cluster_map, OUT_ROOT, clustered_only=False
        )
        for scenario in SCENARIOS:
            plot_scenario_indicator(data, indicator, scenario, OUT_ROOT)
    write_readme(data, OUT_ROOT)
    print(f"Fig. 6e exported to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
