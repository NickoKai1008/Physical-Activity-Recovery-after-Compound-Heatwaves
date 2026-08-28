# -*- coding: utf-8 -*-
"""
Publication-grade visualization of city-level heatwave exposure -> PA
lag response (PPML results).

This independent version generates the formal 12-day Main Figure 2 source
panels and the 7-day sensitivity figure set.

Inputs:
    data/city_lag_estimates_7day.csv
    data/city_lag_estimates_12day.csv

Outputs:
    output/lag7_panels/*.png and *.svg
    output/lag12_panels/*.png and *.svg

SVG output keeps text editable for PowerPoint/Illustrator workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 0. Global aesthetic setup (Nature-family inspired)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Red / blue / yellow anchors tuned to the supplied reference figures.
# The main triad is intentionally restrained: red for PA increase,
# blue for PA decrease and warm yellow for non-significant estimates.
PALETTE = {
    "harm_deep": "#9F3032",
    "harm": "#A64D51",
    "harm_soft": "#C38673",
    "harm_pale": "#E2B7A6",
    "ns": "#E7C862",
    "ns_soft": "#F6DEA0",
    "ns_pale": "#F9E9BB",
    "sand": "#D9BA97",
    "teal_soft": "#BBD4D0",
    "prot_pale": "#C9DDE6",
    "prot_soft": "#86B9CF",
    "prot": "#4C8EBA",
    "prot_deep": "#234F8C",
    "ink": "#1A1A1A",
    "subink": "#5F6368",
    "grid": "#D9D9D9",
    "panel": "#F7F7F5",
    "paper": "#FFFFFF",
}

DIV_CMAP = LinearSegmentedColormap.from_list(
    "reference_rby_diverging",
    [
        (0.00, PALETTE["prot_deep"]),
        (0.18, PALETTE["prot"]),
        (0.34, PALETTE["prot_soft"]),
        (0.50, PALETTE["ns_soft"]),
        (0.66, PALETTE["harm_soft"]),
        (0.84, PALETTE["harm"]),
        (1.00, PALETTE["harm_deep"]),
    ],
    N=256,
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": PALETTE["ink"],
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": PALETTE["ink"],
    "xtick.color": PALETTE["subink"],
    "ytick.color": PALETTE["subink"],
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


@dataclass(frozen=True)
class RunConfig:
    label: str
    csv_name: str
    output_dir: str


RUNS = [
    RunConfig(
        label="7-day",
        csv_name="city_lag_estimates_7day.csv",
        output_dir="output/lag7_panels",
    ),
    RunConfig(
        label="12-day",
        csv_name="city_lag_estimates_12day.csv",
        output_dir="output/lag12_panels",
    ),
]


RUN_MANIFEST_ROWS: list[dict[str, str | int]] = []

FIG1_DISPLAY_FROM_POOLED_CI = {"7-day", "12-day"}

# The submitted Main Figure 2 uses only these source panels. Other plots made
# during exploration are retained as functions but are not release outputs.
FORMAL_12DAY_STEMS = {
    "fig2_pooled_lag_response",
    "fig3_heatmap_city_lag",
    "fig5_lag_tally_distribution",
}



def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def pooled_ci_ylim(pooled_by_lag: pd.DataFrame, margin: float = 0.05) -> tuple[float, float]:
    """Match the master-overview A panel by scaling to pooled estimates and 95% CI."""
    y_min = pooled_by_lag[["pct_lo", "pct"]].min().min()
    y_max = pooled_by_lag[["pct_hi", "pct"]].max().max()
    pad = (y_max - y_min) * margin
    return float(y_min - pad), float(y_max + pad)


def pool_random_effects(sub: pd.DataFrame) -> pd.Series:
    y = sub["estimate"].values
    v = sub["std.error"].values ** 2
    w_fe = 1.0 / v
    mu_fe = np.sum(w_fe * y) / np.sum(w_fe)
    q_stat = np.sum(w_fe * (y - mu_fe) ** 2)
    k = len(y)
    c_term = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = max(0.0, (q_stat - (k - 1)) / c_term) if c_term > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    mu = np.sum(w_re * y) / np.sum(w_re)
    se = np.sqrt(1.0 / np.sum(w_re))
    lo, hi = mu - 1.96 * se, mu + 1.96 * se
    z_val = mu / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
    return pd.Series({
        "mu": mu,
        "se": se,
        "lo": lo,
        "hi": hi,
        "p": p_val,
        "tau2": tau2,
        "k": k,
        "Q": q_stat,
    })


def city_overall(sub: pd.DataFrame) -> pd.Series:
    y = sub["estimate"].values
    v = sub["std.error"].values ** 2
    w = 1.0 / v
    mu = np.sum(w * y) / np.sum(w)
    se = np.sqrt(1.0 / np.sum(w))
    lo, hi = mu - 1.96 * se, mu + 1.96 * se
    z_val = mu / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
    return pd.Series({
        "mu": mu,
        "se": se,
        "lo": lo,
        "hi": hi,
        "p": p_val,
        "n_obs": sub["n_obs"].iloc[0],
    })


def prepare_data(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    df = pd.read_csv(csv_path)
    df["sig05"] = df["p.value"] < 0.05
    df["sig01"] = df["p.value"] < 0.01
    df["sig001"] = df["p.value"] < 0.001
    df["stars"] = df["p.value"].apply(sig_stars)
    df["w"] = 1.0 / (df["std.error"] ** 2)

    pooled_by_lag = df.groupby("lag").apply(pool_random_effects).reset_index()
    pooled_by_lag["pct"] = (np.exp(pooled_by_lag["mu"]) - 1) * 100
    pooled_by_lag["pct_lo"] = (np.exp(pooled_by_lag["lo"]) - 1) * 100
    pooled_by_lag["pct_hi"] = (np.exp(pooled_by_lag["hi"]) - 1) * 100

    city_overall_df = (
        df.groupby("city").apply(city_overall).reset_index().sort_values("mu")
    )
    city_overall_df["pct"] = (np.exp(city_overall_df["mu"]) - 1) * 100
    city_overall_df["pct_lo"] = (np.exp(city_overall_df["lo"]) - 1) * 100
    city_overall_df["pct_hi"] = (np.exp(city_overall_df["hi"]) - 1) * 100

    lag_values = sorted(int(x) for x in df["lag"].dropna().unique())
    return df, pooled_by_lag, city_overall_df, lag_values


def add_caption(fig: mpl.figure.Figure, text: str, y: float = 0.015, fontsize: float = 8) -> None:
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        color=PALETTE["subink"],
        fontsize=fontsize,
        wrap=True,
        style="italic",
    )


def save(fig: mpl.figure.Figure, out_dir: Path, name: str) -> None:
    saved_paths = []
    for ext in ("png", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        saved_paths.append(path)
    print(f"  saved -> {out_dir / name}.png / .svg")
    for path in saved_paths:
        RUN_MANIFEST_ROWS.append({
            "kind": "figure",
            "run": out_dir.name,
            "name": name,
            "format": path.suffix.lstrip("."),
            "path": str(path.relative_to(BASE_DIR)),
            "bytes": path.stat().st_size if path.exists() else 0,
        })


def write_manifest() -> None:
    rows: list[dict[str, str | int]] = [
        {
            "kind": "environment",
            "run": "all",
            "name": "python_version",
            "format": "version",
            "path": platform.python_version(),
            "bytes": 0,
        },
        {
            "kind": "environment",
            "run": "all",
            "name": "platform",
            "format": "",
            "path": platform.platform(),
            "bytes": 0,
        },
    ]
    for cfg in RUNS:
        csv_path = DATA_DIR / cfg.csv_name
        rows.append({
            "kind": "input_csv",
            "run": cfg.output_dir,
            "name": cfg.csv_name,
            "format": "csv",
            "path": str(csv_path.relative_to(BASE_DIR)),
            "bytes": csv_path.stat().st_size if csv_path.exists() else 0,
        })
    for key in ["harm", "ns", "prot", "harm_deep", "ns_soft", "prot_deep"]:
        rows.append({
            "kind": "palette",
            "run": "all",
            "name": key,
            "format": "hex",
            "path": PALETTE[key],
            "bytes": 0,
        })
    rows.extend(RUN_MANIFEST_ROWS)
    pd.DataFrame(rows).to_csv(BASE_DIR / "run_manifest.csv", index=False)


def write_fig2_package(
    config: RunConfig | None = None,
    package_name: str = "fig1",
    csv_suffix: str = "12day",
) -> None:
    """Collect a full Fig. 2 family into a self-contained author handoff folder."""
    config = config or RUNS[1]
    package_dir = BASE_DIR / package_name
    code_dir = package_dir / "code"
    data_dir = package_dir / "data"
    csv_dir = package_dir / "csv"
    output_dir = package_dir / "outputs"
    for folder in (code_dir, data_dir, csv_dir, output_dir):
        folder.mkdir(parents=True, exist_ok=True)

    source_csv = DATA_DIR / config.csv_name

    df, pooled_by_lag, city_overall_df, lag_values = prepare_data(source_csv)
    n_cities = df["city"].nunique()
    n_sig05 = (
        df[df["p.value"] < 0.05]
        .groupby("lag")["city"]
        .nunique()
        .reindex(lag_values, fill_value=0)
    )
    y_min, y_max = pooled_ci_ylim(pooled_by_lag)

    pooled_summary = pooled_by_lag.copy()
    pooled_summary["n_cities"] = n_cities
    pooled_summary["n_sig05_cities"] = pooled_summary["lag"].map(n_sig05).astype(int)
    pooled_summary["plot_ymin_pct"] = y_min
    pooled_summary["plot_ymax_pct"] = y_max
    pooled_summary["axis_scaled_from"] = "pooled random-effects estimate and 95% CI"
    pooled_summary.to_csv(csv_dir / f"fig2_pooled_lag_response_{csv_suffix}_summary.csv", index=False)

    city_points = df[
        ["city", "lag", "estimate", "std.error", "p.value", "n_obs", "pct", "sig05", "sig01", "sig001", "stars"]
    ].copy()
    city_points["direction"] = np.select(
        [
            (city_points["p.value"] < 0.05) & (city_points["estimate"] > 0),
            (city_points["p.value"] < 0.05) & (city_points["estimate"] < 0),
        ],
        ["PA_up", "PA_down"],
        default="non_significant",
    )
    city_points["inside_plot_range"] = city_points["pct"].between(y_min, y_max)
    city_points["plot_ymin_pct"] = y_min
    city_points["plot_ymax_pct"] = y_max
    city_points.to_csv(csv_dir / f"fig2_city_lag_points_{csv_suffix}.csv", index=False)

    lag_tally = (
        city_points.groupby(["lag", "direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=lag_values, columns=["PA_down", "non_significant", "PA_up"], fill_value=0)
        .reset_index()
    )
    lag_tally["n_cities"] = n_cities
    lag_tally.to_csv(csv_dir / f"fig2_lag_direction_tally_{csv_suffix}.csv", index=False)

    city_overall_export = city_overall_df.copy().reset_index(drop=True)
    city_overall_export["rank_by_overall_effect"] = np.arange(1, len(city_overall_export) + 1)
    city_overall_export.to_csv(csv_dir / f"fig2_city_overall_{csv_suffix}.csv", index=False)

    significant_overall = city_overall_df[city_overall_df["p"] < 0.05].copy()
    top_movers = pd.concat([
        significant_overall.nlargest(15, "pct").assign(panel="largest_PA_increase"),
        significant_overall.nsmallest(15, "pct").assign(panel="largest_PA_decrease"),
    ])
    top_movers.to_csv(csv_dir / f"fig2_top_movers_{csv_suffix}.csv", index=False)

    pooled_min = pooled_summary["pct"].min()
    pooled_max = pooled_summary["pct"].max()
    pooled_min_lag = int(pooled_summary.loc[pooled_summary["pct"].idxmin(), "lag"])
    pooled_max_lag = int(pooled_summary.loc[pooled_summary["pct"].idxmax(), "lag"])
    p_min = pooled_summary["p"].min()
    p_max = pooled_summary["p"].max()
    direction_counts = city_points["direction"].value_counts()
    overall_sig = city_overall_df[city_overall_df["p"] < 0.05]
    overall_sig_up = int((overall_sig["pct"] > 0).sum())
    overall_sig_down = int((overall_sig["pct"] < 0).sum())
    top_increase = top_movers[top_movers["panel"].eq("largest_PA_increase")]
    top_decrease = top_movers[top_movers["panel"].eq("largest_PA_decrease")]

    def city_series(names: pd.Series) -> str:
        vals = [str(x) for x in names.head(3).tolist()]
        if len(vals) == 1:
            return vals[0]
        if len(vals) == 2:
            return f"{vals[0]} and {vals[1]}"
        return f"{vals[0]}, {vals[1]} and {vals[2]}"

    ready_text = package_dir / "fig2_results_methods_ready_text.md"
    ready_text.write_text(
        "\n".join([
            "Results",
            "",
            f"Heatwave exposure was followed by a negative pooled physical-activity response across the {config.label} lag window.",
            f"Fig. 2A showed random-effects estimates from {pooled_min:.1f}% to {pooled_max:.1f}%.",
            f"The strongest negative estimate occurred at lag {pooled_min_lag}.",
            f"The least negative estimate occurred at lag {pooled_max_lag}.",
            "The pooled confidence intervals crossed zero at every lag.",
            f"Pooled p values ranged from {p_min:.3f} to {p_max:.3f}.",
            "Fig. 2B showed that nominally significant city-lag associations were common.",
            f"At each lag, {int(n_sig05.min())}-{int(n_sig05.max())} of {n_cities} cities showed nominal significance.",
            (
                f"Across {len(city_points)} city-lag estimates, "
                f"{int(direction_counts.get('PA_up', 0))} indicated increased activity and "
                f"{int(direction_counts.get('PA_down', 0))} indicated decreased activity."
            ),
            "The heatmap and small multiples showed marked city-specific temporal structure.",
            f"The city-level forest plot identified {len(overall_sig)} significant city-average responses.",
            f"These included {overall_sig_up} positive and {overall_sig_down} negative city-average responses.",
            f"The largest increases occurred in {city_series(top_increase['city'])}.",
            f"The largest decreases occurred in {city_series(top_decrease['city'])}.",
            "The master overview therefore supports a negative average tendency with strong spatial heterogeneity.",
            "",
            "Methods",
            "",
            "For each city and lag, PPML coefficients were converted to percentage change as 100 x [exp(beta)-1].",
            "Fig. 2A plotted all city-lag estimates behind a lag-specific random-effects summary.",
            "The pooled summary used a DerSimonian-Laird random-effects model.",
            "The ribbon shows the pooled 95% confidence interval.",
            "The display range follows the master overview panel.",
            "It was scaled to the pooled estimate and 95% confidence interval.",
            "Fig. 2B counted positive, negative and non-significant city-lag estimates.",
            "The heatmap displayed percentage change by city and lag.",
            "Asterisks marked p<0.001 in the master overview heatmap.",
            "Small multiples plotted city-specific lag curves with 95% confidence intervals.",
            "City-average estimates were inverse-variance weighted across available lags.",
            "The forest plot ordered cities by these city-average estimates.",
            "Top movers retained significant city-average effects from both distribution tails.",
            "The volcano plot displayed PPML coefficients against -log10 p values.",
            "All raw estimates and unclipped values were retained in the CSV files.",
        ]),
        encoding="utf-8",
    )

    wrapper_path = code_dir / "fig2_figure_set.py"
    wrapper_path.write_text(
        f'''"""Regenerate the {config.label} Fig. 2 family from the packaged data.

Run from this folder with:
    python fig2_figure_set.py

The script expects:
    ../data/{config.csv_name}
    ./viz_eventlag_publication_rby.py
"""

from __future__ import annotations

from pathlib import Path

import viz_eventlag_publication_rby as viz


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "regenerated_outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    viz.BASE_DIR = PACKAGE_DIR
    viz.DATA_DIR = DATA_DIR
    viz.RUN_MANIFEST_ROWS = []

    csv_path = DATA_DIR / "{config.csv_name}"
    df, pooled_by_lag, city_overall_df, lag_values = viz.prepare_data(csv_path)

    viz.figure1(df, pooled_by_lag, lag_values, OUTPUT_DIR, "{config.label}")
    viz.figure2(city_overall_df, lag_values, OUTPUT_DIR, "{config.label}")
    viz.figure5(df, lag_values, OUTPUT_DIR, "{config.label}")
    viz.figure6(df, city_overall_df, lag_values, OUTPUT_DIR, "{config.label}")
    viz.figure7(city_overall_df, OUTPUT_DIR, "{config.label}")
    viz.figure_master(df, pooled_by_lag, city_overall_df, lag_values, OUTPUT_DIR, "{config.label}")

    print(f"Regenerated Fig. 2 family -> {{OUTPUT_DIR}}")


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )

    package_rows = []

    def copy_asset(
        src: Path,
        dst: Path,
        kind: str,
        name: str,
        role: str = "",
        source_figure: str = "",
    ) -> None:
        shutil.copy2(src, dst)
        package_rows.append({
            "kind": kind,
            "name": name,
            "role": role,
            "source_figure": source_figure,
            "path": str(dst.relative_to(package_dir)),
            "bytes": dst.stat().st_size,
        })

    copy_asset(Path(__file__), code_dir / "viz_eventlag_publication_rby.py", "code", "figure_script", "source")
    copy_asset(source_csv, data_dir / source_csv.name, "input_data", source_csv.name, "source")
    copy_asset(BASE_DIR / "run_manifest.csv", csv_dir / "run_manifest.csv", "manifest", "run_manifest.csv", "source")

    fig2_family = [
        ("fig2_pooled_lag_response", "fig2_pooled_lag_response", "source_alias"),
        ("fig1a_pooled_lag_response", "fig2_pooled_lag_response", "main_panel"),
        ("master_overview", "master_overview", "source_alias"),
        ("fig2_master_overview", "master_overview", "main_composite"),
        ("fig2_forest_city_overall", "fig2_forest_city_overall", "source_alias"),
        ("fig1e_city_overall_forest", "fig2_forest_city_overall", "extended_panel"),
        ("fig5_lag_tally_distribution", "fig5_lag_tally_distribution", "source_alias"),
        ("fig1b_lag_tally_distribution", "fig5_lag_tally_distribution", "supporting_panel"),
        ("fig6_smallmultiples_cities", "fig6_smallmultiples_cities", "source_alias"),
        ("fig1c_city_smallmultiples", "fig6_smallmultiples_cities", "extended_panel"),
        ("fig7_top_movers", "fig7_top_movers", "source_alias"),
        ("fig1d_top_movers", "fig7_top_movers", "extended_panel"),
    ]
    for packaged_name, source_name, role in fig2_family:
        for ext in ("png", "svg"):
            src = BASE_DIR / config.output_dir / f"{source_name}.{ext}"
            if src.exists():
                copy_asset(
                    src,
                    output_dir / f"{packaged_name}.{ext}",
                    "figure",
                    f"{packaged_name}.{ext}",
                    role,
                    source_name,
                )

    if ready_text.exists():
        package_rows.append({
            "kind": "ready_text",
            "name": ready_text.name,
            "role": "manuscript_text",
            "source_figure": "fig2_family",
            "path": str(ready_text.relative_to(package_dir)),
            "bytes": ready_text.stat().st_size,
        })

    for code_path in sorted(code_dir.glob("*.py")):
        if code_path.name == "viz_eventlag_publication_rby.py":
            continue
        package_rows.append({
            "kind": "code",
            "name": code_path.name,
            "role": "fig2_family_wrapper",
            "source_figure": "fig2_family",
            "path": str(code_path.relative_to(package_dir)),
            "bytes": code_path.stat().st_size,
        })

    for csv_path in sorted(csv_dir.glob("fig2_*.csv")):
        package_rows.append({
            "kind": "derived_csv",
            "name": csv_path.name,
            "role": "figure_data",
            "source_figure": "fig2_family",
            "path": str(csv_path.relative_to(package_dir)),
            "bytes": csv_path.stat().st_size,
        })

    pd.DataFrame(package_rows).to_csv(csv_dir / "fig2_package_manifest.csv", index=False)
    print(f"{config.label} Fig. 2 package written ->", package_dir)


def write_fig2_packages() -> None:
    write_fig2_package(RUNS[1], "fig1", "12day")
    write_fig2_package(RUNS[0], "fig2_lag7", "7day")


def direction_colors(estimates, pvals):
    estimates = np.asarray(estimates)
    pvals = np.asarray(pvals)
    return np.where(
        (pvals < 0.05) & (estimates > 0),
        PALETTE["harm"],
        np.where((pvals < 0.05) & (estimates < 0), PALETTE["prot"], PALETTE["ns"]),
    )


def add_volcano_callouts(ax, points: pd.DataFrame, xlim: tuple[float, float]) -> None:
    """Draw non-overlapping two-column labels with leader lines for volcano plots."""
    if points.empty:
        return

    ymin, ymax = ax.get_ylim()
    y_top = ymax - 7
    y_bottom = max(12, ymax * 0.42)
    min_gap = max(8, ymax * 0.035)

    highlight_colors = direction_colors(points["estimate"], points["p.value"])
    ax.scatter(
        points["est_vis"],
        points["logp"],
        s=np.clip(points["n_obs"] / 2100, 24, 96),
        facecolors="none",
        edgecolors=highlight_colors,
        linewidths=1.0,
        zorder=6,
    )

    for side, side_df in (
        ("left", points[points["est_vis"] < 0]),
        ("right", points[points["est_vis"] >= 0]),
    ):
        if side_df.empty:
            continue
        side_df = side_df.sort_values(["logp", "estimate"], ascending=[False, True]).reset_index(drop=True)
        n = len(side_df)
        if n <= 4:
            y_labels = np.clip(side_df["logp"].to_numpy(float), y_bottom, y_top)
            order = np.argsort(-y_labels)
            y_sorted = y_labels[order].copy()
            for i in range(1, len(y_sorted)):
                y_sorted[i] = min(y_sorted[i], y_sorted[i - 1] - min_gap)
            if len(y_sorted) and y_sorted[-1] < y_bottom:
                y_sorted = np.linspace(y_top, y_bottom, n)
            y_labels[order] = y_sorted
        else:
            y_labels = np.linspace(y_top, y_bottom, n)

        x_text = xlim[0] + 0.18 if side == "left" else xlim[1] - 0.18
        ha = "left" if side == "left" else "right"
        for (_, r), y_text in zip(side_df.iterrows(), y_labels):
            color = PALETTE["harm_deep"] if r["estimate"] > 0 else PALETTE["prot_deep"]
            ax.annotate(
                f"{r['city']} L{int(r['lag'])}",
                xy=(r["est_vis"], r["logp"]),
                xytext=(x_text, y_text),
                textcoords="data",
                ha=ha,
                va="center",
                fontsize=7.0,
                color=color,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88),
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.55,
                    color=color,
                    alpha=0.78,
                    shrinkA=2,
                    shrinkB=2,
                    connectionstyle="arc3,rad=0.05" if side == "left" else "arc3,rad=-0.05",
                ),
                zorder=7,
            )


def figure1(df, pooled_by_lag, lag_values, out_dir, label):
    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.24)

    jitter = (np.random.RandomState(0).rand(len(df)) - 0.5) * 0.22
    colors = direction_colors(df["estimate"], df["p.value"])
    ax.scatter(
        df["lag"] + jitter,
        df["pct"],
        s=np.clip(df["n_obs"] / 2500, 4, 40),
        c=colors,
        alpha=0.58,
        linewidths=0,
        zorder=2,
    )

    x = pooled_by_lag["lag"].values
    ax.fill_between(
        x,
        pooled_by_lag["pct_lo"].to_numpy(),
        pooled_by_lag["pct_hi"].to_numpy(),
        color=PALETTE["ink"],
        alpha=0.12,
        zorder=3,
        label="Pooled 95% CI (random-effects)",
    )
    ax.plot(
        x,
        pooled_by_lag["pct"].to_numpy(),
        "-",
        color=PALETTE["ink"],
        lw=2.4,
        zorder=4,
        label="Pooled % change in PA",
    )

    for _, r in pooled_by_lag.iterrows():
        z_val = r["mu"] / r["se"]
        col = PALETTE["harm_deep"] if r["mu"] > 0 else PALETTE["prot_deep"]
        ax.scatter(
            r["lag"],
            r["pct"],
            s=180 + 35 * abs(z_val),
            facecolor=col,
            edgecolor="white",
            linewidth=1.8,
            zorder=6,
        )
        star = sig_stars(r["p"])
        if star:
            ax.text(
                r["lag"],
                r["pct"] + (3 if r["pct"] >= 0 else -5),
                star,
                ha="center",
                va="center",
                color=col,
                fontsize=13,
                fontweight="bold",
                zorder=7,
            )

    if label in FIG1_DISPLAY_FROM_POOLED_CI:
        ax.set_ylim(*pooled_ci_ylim(pooled_by_lag))

    ax.axhline(0, color=PALETTE["subink"], lw=0.7, ls="--")
    ax.set_xlabel("Lag after heatwave event (days)")
    ax.set_ylabel("Change in physical activity (%)")
    ax.set_xticks(lag_values)
    ax.set_title(
        f"Pooled {label} lag response",
        loc="left",
        pad=8,
    )

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["harm"], markersize=7, label="PA increase"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["prot"], markersize=7, label="PA decrease"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["ns"], markersize=7, label="Non-significant"),
        Line2D([0], [0], color=PALETTE["ink"], lw=2.2, label="Pooled estimate"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.14))
    save(fig, out_dir, "fig2_pooled_lag_response")
    plt.close(fig)


def figure2(city_overall_df, lag_values, out_dir, label):
    d = city_overall_df.copy().sort_values("mu").reset_index(drop=True)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.8, 12.5))
    fig.subplots_adjust(left=0.25, right=0.92, top=0.95, bottom=0.07)

    colors = direction_colors(d["mu"], d["p"])
    ax.hlines(y, d["pct_lo"], d["pct_hi"], color=colors, lw=1.6, alpha=0.92)
    ax.scatter(
        d["pct"],
        y,
        s=np.clip(d["n_obs"] / 2500, 10, 60),
        c=colors,
        edgecolor="white",
        lw=0.6,
        zorder=4,
    )

    for yi in y[::2]:
        ax.axhspan(yi - 0.5, yi + 0.5, color=PALETTE["panel"], zorder=0)

    ax.axvline(0, color=PALETTE["ink"], lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["city"], fontsize=7.5)
    ax.set_xlabel(
        f"Overall change in PA (%, averaged across lag {min(lag_values)}-{max(lag_values)}, 95% CI)"
    )
    ax.set_title(
        f"City-level overall PA response to heatwave exposure ({label})",
        loc="left",
        pad=10,
    )

    xmin = np.percentile(d["pct_lo"], 2)
    xmax = np.percentile(d["pct_hi"], 98)
    pad = (xmax - xmin) * 0.08
    ax.set_xlim(xmin - pad, xmax + pad)

    for i, r in d.iterrows():
        txt = sig_stars(r["p"])
        if txt:
            ax.text(
                r["pct_hi"] + (xmax - xmin) * 0.015,
                i,
                txt,
                va="center",
                fontsize=7.5,
                color=colors[i],
            )

    add_caption(
        fig,
        f"Figure 2 | Forest plot of city-level overall heatwave effects on PA ({label}). "
        "Each city contributes an inverse-variance weighted estimate across post-event "
        "lag days. Red = PA increased, blue = PA decreased, yellow = non-significant.",
        y=0.005,
    )
    save(fig, out_dir, "fig2_forest_city_overall")
    plt.close(fig)


def figure3(df, city_overall_df, lag_values, out_dir, label):
    order = city_overall_df["city"].values
    mat = df.pivot(index="city", columns="lag", values="pct").loc[order, lag_values]
    pvals = df.pivot(index="city", columns="lag", values="p.value").loc[order, lag_values]
    vmax = np.nanpercentile(np.abs(mat.values), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    width = max(7.8, 5.8 + 0.22 * len(lag_values))
    fig, ax = plt.subplots(figsize=(width, 13.5))
    fig.subplots_adjust(left=0.22, right=0.90, top=0.95, bottom=0.08)

    im = ax.imshow(mat.values, aspect="auto", cmap=DIV_CMAP, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(lag_values)))
    ax.set_xticklabels([f"Lag {i}" for i in lag_values])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("Days after heatwave event")
    ax.set_title(f"Heatmap of PA response (%) by city x lag ({label})", loc="left", pad=10)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            s = sig_stars(pvals.values[i, j])
            if s:
                ax.text(
                    j,
                    i,
                    s,
                    ha="center",
                    va="center",
                    color="white" if abs(mat.values[i, j]) > vmax * 0.55 else PALETTE["ink"],
                    fontsize=6.3,
                    fontweight="bold",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("% change in PA", rotation=90, labelpad=8)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)

    add_caption(
        fig,
        "Figure 3 | Heatmap of PA response by city x lag. The colour scale is "
        "anchored by blue, pale yellow and red, and is symmetric around zero. "
        "Stars denote significance; cities are ordered by overall effect size.",
        y=0.005,
    )
    save(fig, out_dir, "fig3_heatmap_city_lag")
    plt.close(fig)


def figure4(df, lag_values, out_dir, label):
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.22)

    d = df.copy()
    d["logp"] = -np.log10(np.clip(d["p.value"], 1e-300, 1))
    d["est_vis"] = d["estimate"].clip(-3.5, 3.5)
    colors = direction_colors(d["estimate"], d["p.value"])

    ax.scatter(
        d["est_vis"],
        d["logp"],
        s=np.clip(d["n_obs"] / 2500, 6, 70),
        c=colors,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.4,
    )

    ax.axhline(-np.log10(0.05), ls="--", lw=0.8, color=PALETTE["subink"])
    ax.axhline(-np.log10(0.001), ls=":", lw=0.8, color=PALETTE["subink"])
    ax.axvline(0, ls="--", lw=0.8, color=PALETTE["subink"])
    ax.text(
        3.45,
        -np.log10(0.05) + 0.6,
        "p = 0.05",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["subink"],
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8),
    )
    ax.text(
        3.45,
        -np.log10(0.001) + 0.9,
        "p = 0.001",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["subink"],
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8),
    )

    xlim = (-3.6, 3.6)
    ax.set_xlim(*xlim)
    y_upper = max(10, min(320, float(d["logp"].max()) * 1.06))
    ax.set_ylim(-15, y_upper)
    top = d.reindex(d["estimate"].abs().sort_values(ascending=False).index).head(12)
    add_volcano_callouts(ax, top, xlim)

    ax.set_xlabel("Log-rate coefficient (PPML estimate, clipped to +/-3.5)")
    ax.set_ylabel("-log10 p-value")
    ax.set_title(
        f"Volcano plot of {df['city'].nunique()} cities x {len(lag_values)} lags "
        f"({len(df)} estimates, {label})",
        loc="left",
        pad=10,
    )

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["harm"], markersize=7, label="PA increased & p<0.05"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["prot"], markersize=7, label="PA decreased & p<0.05"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["ns"], markersize=7, label="Non-significant"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )

    add_caption(
        fig,
        f"Figure 4 | Volcano plot ({label}). Each dot is one city x lag estimate; "
        "dashed and dotted lines mark p=0.05 and p=0.001. The top-12 largest "
        "absolute effects are labelled.",
        y=0.04,
    )
    save(fig, out_dir, "fig4_volcano")
    plt.close(fig)


def figure5(df, lag_values, out_dir, label):
    tally = (
        df.assign(
            direction=np.where(
                df["p.value"] < 0.05,
                np.where(df["estimate"] > 0, "PA up (p<0.05)", "PA down (p<0.05)"),
                "Non-significant",
            )
        )
        .groupby(["lag", "direction"])
        .size()
        .unstack(fill_value=0)
    )
    columns = ["PA down (p<0.05)", "Non-significant", "PA up (p<0.05)"]
    tally = tally.reindex(index=lag_values, columns=columns, fill_value=0)

    fig_width = max(11.5, 8.5 + 0.32 * len(lag_values))
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(fig_width, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1]},
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.25, wspace=0.28)

    bottom = np.zeros(len(lag_values))
    col_map = {
        "PA down (p<0.05)": PALETTE["prot"],
        "Non-significant": PALETTE["ns"],
        "PA up (p<0.05)": PALETTE["harm"],
    }
    for direction in columns:
        values = tally[direction].values
        ax1.bar(
            lag_values,
            values,
            bottom=bottom,
            color=col_map[direction],
            width=0.72,
            label=direction,
            edgecolor="white",
            lw=0.8,
        )
        for i, val in enumerate(values):
            if val > 0:
                ax1.text(
                    lag_values[i],
                    bottom[i] + val / 2,
                    str(int(val)),
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color="white" if direction != "Non-significant" else PALETTE["ink"],
                    fontweight="bold",
                )
        bottom += values

    ax1.set_xticks(lag_values)
    ax1.set_xlabel("Lag day")
    ax1.set_ylabel(f"Number of cities (of {df['city'].nunique()})")
    ax1.set_title("A  Directional tally by lag", loc="left", pad=6)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)

    data = [df.loc[df["lag"] == lag, "pct"].clip(-100, 250).values for lag in lag_values]
    parts = ax2.violinplot(data, positions=lag_values, widths=0.85, showextrema=False, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(PALETTE["ns_soft"])
        pc.set_alpha(0.45)
        pc.set_edgecolor("none")

    for lag in lag_values:
        sub = df.loc[df["lag"] == lag]
        y = sub["pct"].clip(-100, 250).values
        x = np.full_like(y, lag, dtype=float) + (np.random.RandomState(lag).rand(len(y)) - 0.5) * 0.18
        colors = direction_colors(sub["estimate"], sub["p.value"])
        ax2.scatter(x, y, s=14, c=colors, alpha=0.78, linewidths=0)
        ax2.hlines(np.median(y), lag - 0.28, lag + 0.28, color=PALETTE["ink"], lw=1.6)

    ax2.axhline(0, color=PALETTE["subink"], lw=0.7, ls="--")
    ax2.set_xticks(lag_values)
    ax2.set_xlabel("Lag day")
    ax2.set_ylabel("% change in PA (clipped)")
    ax2.set_title("B  Distribution of city effects per lag", loc="left", pad=6)

    add_caption(
        fig,
        f"Figure 5 | Lag-wise summary across {df['city'].nunique()} cities ({label}). "
        "The stacked bars count significantly reduced, non-significant and "
        "significantly increased PA at each lag. Raincloud points retain the "
        "same red/yellow/blue direction coding.",
        y=0.04,
    )
    save(fig, out_dir, "fig5_lag_tally_distribution")
    plt.close(fig)


def figure6(df, city_overall_df, lag_values, out_dir, label):
    order = city_overall_df["city"].values
    ncol = 9
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 1.4), sharex=True)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.10, hspace=0.55, wspace=0.25)

    yrng = np.nanpercentile(df["pct"].clip(-100, 300), [2, 98])
    tick_middle = lag_values[len(lag_values) // 2]
    x_ticks = sorted(set([min(lag_values), tick_middle, max(lag_values)]))

    for ax, city in zip(axes.flat, order):
        sub = df[df["city"] == city].sort_values("lag")
        colors = direction_colors(sub["estimate"], sub["p.value"])
        ax.fill_between(
            sub["lag"].to_numpy(),
            sub["pct_low"].clip(-100, 400).to_numpy(),
            sub["pct_high"].clip(-100, 400).to_numpy(),
            color=PALETTE["grid"],
            alpha=0.24,
        )
        ax.plot(
            sub["lag"].to_numpy(),
            sub["pct"].clip(-100, 400).to_numpy(),
            "-",
            color=PALETTE["ink"],
            lw=1.0,
            alpha=0.85,
        )
        ax.scatter(sub["lag"], sub["pct"].clip(-100, 400), c=colors, s=18, edgecolor="white", lw=0.4, zorder=3)
        ax.axhline(0, color=PALETTE["subink"], lw=0.5, ls="--")
        ax.set_title(city, fontsize=7.2, pad=2, color=PALETTE["ink"])
        ax.set_ylim(yrng[0], yrng[1])
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="both", labelsize=6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes.flat[len(order):]:
        ax.axis("off")

    fig.text(0.5, 0.04, "Lag day after heatwave event", ha="center", fontsize=10, color=PALETTE["ink"])
    fig.text(0.005, 0.5, "% change in PA (95% CI)", va="center", rotation=90, fontsize=10, color=PALETTE["ink"])
    fig.suptitle(
        f"City-specific {label} lag-response curves (ordered by overall effect)",
        x=0.04,
        ha="left",
        y=0.985,
        fontsize=12,
        fontweight="bold",
    )

    add_caption(
        fig,
        "Figure 6 | Small-multiples gallery of city-specific lag-response curves. "
        "Each panel shows estimated PA % change with 95% CI ribbon; dot colour "
        "flags significance and direction.",
        y=0.005,
        fontsize=7.5,
    )
    save(fig, out_dir, "fig6_smallmultiples_cities")
    plt.close(fig)


def figure7(city_overall_df, out_dir, label):
    d = city_overall_df.copy()
    d = d[d["p"] < 0.05]
    top_pos = d.nlargest(15, "pct")
    top_neg = d.nsmallest(15, "pct")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.3))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.22, wspace=0.55)

    y = np.arange(len(top_pos))
    ax1.barh(
        y,
        top_pos["pct"],
        xerr=[top_pos["pct"] - top_pos["pct_lo"], top_pos["pct_hi"] - top_pos["pct"]],
        color=PALETTE["harm"],
        ecolor=PALETTE["harm_deep"],
        height=0.7,
        capsize=2.5,
        alpha=0.92,
        error_kw={"lw": 0.9},
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(top_pos["city"], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.axvline(0, color=PALETTE["ink"], lw=0.7, ls="--")
    ax1.set_xlabel("% change in PA (95% CI)")
    ax1.set_title("A  Top-15 cities with largest PA increase", loc="left", pad=8, color=PALETTE["harm_deep"])

    y = np.arange(len(top_neg))
    ax2.barh(
        y,
        top_neg["pct"],
        xerr=[top_neg["pct"] - top_neg["pct_lo"], top_neg["pct_hi"] - top_neg["pct"]],
        color=PALETTE["prot"],
        ecolor=PALETTE["prot_deep"],
        height=0.7,
        capsize=2.5,
        alpha=0.92,
        error_kw={"lw": 0.9},
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels(top_neg["city"], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.axvline(0, color=PALETTE["ink"], lw=0.7, ls="--")
    ax2.set_xlabel("% change in PA (95% CI)")
    ax2.set_title("B  Top-15 cities with largest PA decrease", loc="left", pad=8, color=PALETTE["prot_deep"])

    add_caption(
        fig,
        f"Figure 7 | Ranked extremes ({label}). Top-15 cities in each direction among "
        "those with a significant overall effect. Error bars show 95% CI on the % change.",
        y=0.02,
        fontsize=8,
    )
    save(fig, out_dir, "fig7_top_movers")
    plt.close(fig)


def figure_master(df, pooled_by_lag, city_overall_df, lag_values, out_dir, label):
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1, 1, 1.1],
        width_ratios=[1.1, 1, 1.1],
        hspace=0.55,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.93,
        bottom=0.09,
    )

    axA = fig.add_subplot(gs[0, :2])
    x = pooled_by_lag["lag"].values
    axA.fill_between(
        x,
        pooled_by_lag["pct_lo"].to_numpy(),
        pooled_by_lag["pct_hi"].to_numpy(),
        color=PALETTE["ink"],
        alpha=0.12,
    )
    axA.plot(x, pooled_by_lag["pct"].to_numpy(), "-", color=PALETTE["ink"], lw=2.2)
    for _, r in pooled_by_lag.iterrows():
        c = PALETTE["harm_deep"] if r["mu"] > 0 else PALETTE["prot_deep"]
        axA.scatter(r["lag"], r["pct"], s=160, c=c, edgecolor="white", linewidth=1.6, zorder=5)
    axA.axhline(0, color=PALETTE["subink"], lw=0.6, ls="--")
    axA.set_xticks(lag_values)
    axA.set_xlabel("Lag day")
    axA.set_ylabel("% change in PA")
    axA.set_title("A  Pooled lag response (random-effects)", loc="left", pad=6)

    axB = fig.add_subplot(gs[0, 2])
    tally = (
        df.assign(direction=np.where(df["p.value"] < 0.05, np.where(df["estimate"] > 0, "up", "down"), "ns"))
        .groupby(["lag", "direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=lag_values, columns=["down", "ns", "up"], fill_value=0)
    )
    bottom = np.zeros(len(lag_values))
    for direction, color in zip(["down", "ns", "up"], [PALETTE["prot"], PALETTE["ns"], PALETTE["harm"]]):
        axB.bar(tally.index, tally[direction], bottom=bottom, color=color, width=0.72, edgecolor="white", lw=0.6)
        bottom += tally[direction].values
    axB.set_xticks(lag_values)
    axB.set_xlabel("Lag day")
    axB.set_ylabel(f"# cities (of {df['city'].nunique()})")
    axB.set_title("B  Directional tally by lag", loc="left", pad=6)

    axC = fig.add_subplot(gs[1:, 0])
    order = city_overall_df["city"].values
    mat = df.pivot(index="city", columns="lag", values="pct").loc[order, lag_values]
    pvals = df.pivot(index="city", columns="lag", values="p.value").loc[order, lag_values]
    vmax = np.nanpercentile(np.abs(mat.values), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = axC.imshow(mat.values, aspect="auto", cmap=DIV_CMAP, norm=norm, interpolation="nearest")
    axC.set_xticks(range(len(lag_values)))
    axC.set_xticklabels([f"L{i}" for i in lag_values], fontsize=7, rotation=45 if len(lag_values) > 9 else 0)
    axC.set_yticks(range(len(order)))
    axC.set_yticklabels(order, fontsize=5.2)
    axC.set_title("C  City x lag heatmap (% change)", loc="left", pad=6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if sig_stars(pvals.values[i, j]) == "***":
                axC.text(
                    j,
                    i,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(mat.values[i, j]) > vmax * 0.5 else PALETTE["ink"],
                )
    cbar = fig.colorbar(im, ax=axC, fraction=0.035, pad=0.02)
    cbar.set_label("% change", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    axD = fig.add_subplot(gs[1, 1:])
    d = city_overall_df.sort_values("mu")
    picks = pd.concat([d.head(12), d.tail(12)]).reset_index(drop=True)
    y = np.arange(len(picks))
    colors = direction_colors(picks["mu"], picks["p"])
    axD.hlines(y, picks["pct_lo"], picks["pct_hi"], color=colors, lw=1.5)
    axD.scatter(picks["pct"], y, s=35, c=colors, edgecolor="white", lw=0.6)
    axD.axvline(0, color=PALETTE["ink"], lw=0.7, ls="--")
    axD.set_yticks(y)
    axD.set_yticklabels(picks["city"], fontsize=7)
    axD.invert_yaxis()
    axD.set_xlabel("% change in PA (95% CI)")
    axD.set_title("D  Forest plot: 12 most protective & 12 most harmful cities", loc="left", pad=6)
    xmin = np.percentile(picks["pct_lo"], 2)
    xmax = np.percentile(picks["pct_hi"], 98)
    pad = (xmax - xmin) * 0.1
    axD.set_xlim(xmin - pad, xmax + pad)

    axE = fig.add_subplot(gs[2, 1:])
    dd = df.copy()
    dd["logp"] = -np.log10(np.clip(dd["p.value"], 1e-300, 1))
    dd["est_vis"] = dd["estimate"].clip(-3.5, 3.5)
    colors = direction_colors(dd["estimate"], dd["p.value"])
    axE.scatter(
        dd["est_vis"],
        dd["logp"],
        s=np.clip(dd["n_obs"] / 3500, 4, 50),
        c=colors,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.3,
    )
    axE.axhline(-np.log10(0.05), ls="--", lw=0.6, color=PALETTE["subink"])
    axE.axvline(0, ls="--", lw=0.6, color=PALETTE["subink"])
    axE.set_xlabel("PPML log-rate coefficient (clipped +/-3.5)")
    axE.set_ylabel("-log10 p")
    axE.set_title(f"E  Volcano plot of {len(df)} city x lag estimates", loc="left", pad=6)

    fig.suptitle(
        f"Heatwave exposure and physical activity - {label} lag response across "
        f"{df['city'].nunique()} U.S. cities",
        x=0.06,
        ha="left",
        y=0.985,
        fontsize=14.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )

    add_caption(
        fig,
        "Master overview | (A) Pooled random-effects lag response; (B) per-lag "
        "directional tally; (C) city x lag heatmap (asterisks mark p<0.001); "
        "(D) forest plot of the 12 most protective and 12 most harmful cities; "
        "(E) volcano of all estimates. Red = PA increased, blue = PA decreased, "
        "yellow = non-significant.",
        y=0.015,
        fontsize=8.5,
    )
    save(fig, out_dir, "master_overview")
    plt.close(fig)


def run_one(config: RunConfig) -> None:
    csv_path = DATA_DIR / config.csv_name
    out_dir = BASE_DIR / config.output_dir
    out_dir.mkdir(exist_ok=True)

    df, pooled_by_lag, city_overall_df, lag_values = prepare_data(csv_path)
    print(
        f"\nLoaded {config.label}: {len(df)} rows, {df['city'].nunique()} cities, "
        f"{len(lag_values)} lag days from {csv_path.name}"
    )
    print(f"Producing {config.label} figures ...")

    if config.label == "12-day":
        for path in out_dir.iterdir():
            if path.suffix.lower() in {".png", ".svg"} and path.stem not in FORMAL_12DAY_STEMS:
                path.unlink()

        figure1(df, pooled_by_lag, lag_values, out_dir, config.label)
        figure3(df, city_overall_df, lag_values, out_dir, config.label)
        figure5(df, lag_values, out_dir, config.label)
        print(f"Done: {out_dir.resolve()} (formal Main Figure 2 source panels only)")
        return

    figure1(df, pooled_by_lag, lag_values, out_dir, config.label)
    figure2(city_overall_df, lag_values, out_dir, config.label)
    figure3(df, city_overall_df, lag_values, out_dir, config.label)
    figure4(df, lag_values, out_dir, config.label)
    figure5(df, lag_values, out_dir, config.label)
    figure6(df, city_overall_df, lag_values, out_dir, config.label)
    figure7(city_overall_df, out_dir, config.label)
    figure_master(df, pooled_by_lag, city_overall_df, lag_values, out_dir, config.label)
    print(f"Done: {out_dir.resolve()}")


def main() -> None:
    print("Red/blue/yellow publication figure batch")
    print("Base folder:", BASE_DIR)
    for config in RUNS:
        run_one(config)
    write_manifest()
    print("Manifest written ->", BASE_DIR / "run_manifest.csv")
    

if __name__ == "__main__":
    main()
