from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


INK = "#343434"
MID = "#8C8C8C"
LIGHT = "#D7D7D7"
ZERO = "#AFAFAF"
EMPTY = "#F7F7F5"


def city_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def tick_format(value: float, _position: int) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def city_limits(group: pd.DataFrame) -> tuple[float, float]:
    finite = np.concatenate(
        [
            group["pct_low"].to_numpy(dtype=float),
            group["pct"].to_numpy(dtype=float),
            group["pct_high"].to_numpy(dtype=float),
            np.array([0.0]),
        ]
    )
    finite = finite[np.isfinite(finite)]
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = max(high - low, 10.0)
    return low - 0.09 * span, high + 0.09 * span


def build_source_data(raw: pd.DataFrame, city_order: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "city",
        "lag",
        "estimate",
        "std.error",
        "p.value",
        "conf.low",
        "conf.high",
        "pct",
        "pct_low",
        "pct_high",
        "n_obs",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"PPML source table is missing required columns: {missing}")

    if len(city_order) != 75 or city_order["city"].nunique() != 75:
        raise ValueError("The city-order table must contain exactly 75 unique cities.")

    raw = raw.copy()
    raw["city_key"] = raw["city"].map(city_key)
    order = city_order.copy().sort_values("city").reset_index(drop=True)
    order["city_key"] = order["city"].map(city_key)
    order["city_order"] = np.arange(1, 76)
    order["panel_row"] = (order["city_order"] - 1) // 5 + 1
    order["panel_col"] = (order["city_order"] - 1) % 5 + 1

    observed_keys = set(raw["city_key"])
    order["model_available"] = order["city_key"].isin(observed_keys)

    key_to_display = dict(zip(order["city_key"], order["city"]))
    unknown = sorted(set(raw["city_key"]) - set(key_to_display))
    if unknown:
        raise ValueError(f"PPML source contains cities outside the 75-city order: {unknown}")

    raw["city"] = raw["city_key"].map(key_to_display)
    raw = raw.merge(
        order[["city_key", "city_order", "panel_row", "panel_col"]],
        on="city_key",
        how="left",
        validate="many_to_one",
    )
    raw["lag"] = pd.to_numeric(raw["lag"], errors="raise").astype(int)
    raw["significant_05"] = pd.to_numeric(raw["p.value"], errors="coerce") < 0.05
    raw["effect_direction"] = np.select(
        [raw["significant_05"] & (raw["estimate"] < 0), raw["significant_05"] & (raw["estimate"] > 0)],
        ["significant_suppression", "significant_enhancement"],
        default="not_significant",
    )
    raw = raw.sort_values(["city_order", "lag"]).reset_index(drop=True)

    counts = raw.groupby("city_key")["lag"].agg(["count", "nunique", "min", "max"])
    bad = counts[(counts["count"] != 12) | (counts["nunique"] != 12) | (counts["min"] != 1) | (counts["max"] != 12)]
    if not bad.empty:
        raise ValueError(f"Each available city must have one estimate for every lag 1-12:\n{bad}")

    complete = pd.MultiIndex.from_product(
        [order["city_key"], np.arange(1, 13)], names=["city_key", "lag"]
    ).to_frame(index=False)
    complete = complete.merge(order, on="city_key", how="left", validate="many_to_one", suffixes=("", "_order"))
    observed_columns = [
        "city_key",
        "lag",
        "estimate",
        "std.error",
        "p.value",
        "conf.low",
        "conf.high",
        "pct",
        "pct_low",
        "pct_high",
        "n_obs",
        "significant_05",
        "effect_direction",
    ]
    complete = complete.merge(raw[observed_columns], on=["city_key", "lag"], how="left", validate="one_to_one")
    complete = complete.sort_values(["city_order", "lag"]).reset_index(drop=True)

    summary_rows = []
    for row in order.itertuples(index=False):
        group = raw[raw["city_key"] == row.city_key]
        if group.empty:
            summary_rows.append(
                {
                    "city": row.city,
                    "city_order": row.city_order,
                    "model_available": False,
                    "n_obs": np.nan,
                    "n_lags": 0,
                    "n_significant_suppression": 0,
                    "n_significant_enhancement": 0,
                    "latest_significant_lag": np.nan,
                    "strongest_suppression_lag": np.nan,
                    "strongest_suppression_pct": np.nan,
                    "strongest_suppression_pct_low": np.nan,
                    "strongest_suppression_pct_high": np.nan,
                }
            )
            continue
        significant = group[group["significant_05"]]
        strongest = group.loc[group["pct"].idxmin()]
        summary_rows.append(
            {
                "city": row.city,
                "city_order": row.city_order,
                "model_available": True,
                "n_obs": int(group["n_obs"].iloc[0]),
                "n_lags": len(group),
                "n_significant_suppression": int(((group["significant_05"]) & (group["estimate"] < 0)).sum()),
                "n_significant_enhancement": int(((group["significant_05"]) & (group["estimate"] > 0)).sum()),
                "latest_significant_lag": int(significant["lag"].max()) if not significant.empty else np.nan,
                "strongest_suppression_lag": int(strongest["lag"]),
                "strongest_suppression_pct": strongest["pct"],
                "strongest_suppression_pct_low": strongest["pct_low"],
                "strongest_suppression_pct_high": strongest["pct_high"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    coverage = order[["city", "city_order", "panel_row", "panel_col", "model_available"]].copy()
    coverage["status"] = np.where(coverage["model_available"], "12 lag estimates available", "No composite heatwave")
    return raw, complete, summary.merge(coverage[["city", "status"]], on="city", how="left")


def draw_atlas(observed: pd.DataFrame, complete: pd.DataFrame, output_stem: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.55,
            "axes.unicode_minus": True,
        }
    )

    fig, axes = plt.subplots(15, 5, figsize=(11.7, 18.2), squeeze=False)
    fig.patch.set_facecolor("white")

    city_rows = complete.drop_duplicates("city_key").sort_values("city_order")
    for axis, city_row in zip(axes.flat, city_rows.itertuples(index=False)):
        group = observed[observed["city_key"] == city_row.city_key].sort_values("lag")
        axis.set_facecolor("white" if not group.empty else EMPTY)
        axis.set_title(city_row.city, loc="left", fontsize=5.9, fontweight="bold", color=INK, pad=3.0)

        if group.empty:
            axis.text(
                0.5,
                0.50,
                "No composite\nheatwave",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=5.2,
                color=MID,
                linespacing=1.25,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            continue

        x = group["lag"].to_numpy(dtype=float)
        y = group["pct"].to_numpy(dtype=float)
        low = group["pct_low"].to_numpy(dtype=float)
        high = group["pct_high"].to_numpy(dtype=float)
        sig = group["significant_05"].to_numpy(dtype=bool)

        axis.axhline(0, color=ZERO, linewidth=0.55, linestyle=(0, (2.3, 2.3)), zorder=0)
        axis.vlines(x, low, high, color=MID, linewidth=0.62, alpha=0.82, zorder=1)
        axis.plot(x, y, color=INK, linewidth=0.88, zorder=2)
        axis.scatter(x[~sig], y[~sig], s=9.0, facecolors="white", edgecolors=INK, linewidths=0.55, zorder=3)
        axis.scatter(x[sig], y[sig], s=10.5, facecolors=INK, edgecolors=INK, linewidths=0.4, zorder=4)

        axis.set_xlim(0.5, 12.5)
        axis.set_ylim(*city_limits(group))
        axis.set_xticks([1, 6, 12])
        axis.tick_params(axis="x", labelsize=5.0, length=1.8, width=0.45, pad=1.4, colors=INK)
        axis.tick_params(axis="y", labelsize=5.0, length=1.8, width=0.45, pad=1.2, colors=INK)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        axis.yaxis.set_major_formatter(FuncFormatter(tick_format))
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            axis.spines[side].set_color("#666666")
            axis.spines[side].set_linewidth(0.5)
        axis.text(
            0.99,
            1.03,
            f"n={int(group['n_obs'].iloc[0]):,}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.0,
            color="#767676",
        )

    fig.suptitle(
        "City-specific post-heatwave physical-activity responses",
        x=0.055,
        y=0.988,
        ha="left",
        fontsize=13.0,
        fontweight="bold",
        color="#202020",
    )
    fig.text(
        0.055,
        0.971,
        "PPML estimates for calendar days 1-12 after composite heatwave event end; city-specific y-axis scales",
        ha="left",
        va="top",
        fontsize=7.1,
        color="#5A5A5A",
    )
    legend_handles = [
        Line2D([0], [0], color=INK, marker="o", markersize=3.7, markerfacecolor=INK, linewidth=0.9, label="P < 0.05"),
        Line2D([0], [0], color=INK, marker="o", markersize=3.7, markerfacecolor="white", linewidth=0.9, label="P >= 0.05"),
        Line2D([0], [0], color=MID, marker="|", markersize=8, linewidth=0, markeredgewidth=0.9, label="95% CI"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.988, 0.979),
        ncol=3,
        frameon=False,
        fontsize=6.0,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.text(0.50, 0.018, "Lag day after composite heatwave event end", ha="center", fontsize=7.2, color=INK)
    fig.text(
        0.014,
        0.50,
        "Change in physical activity (%)",
        va="center",
        ha="center",
        rotation=90,
        fontsize=7.2,
        color=INK,
    )
    fig.text(
        0.985,
        0.018,
        "Filled points denote two-sided Wald P < 0.05; vertical lines are 95% confidence intervals.",
        ha="right",
        fontsize=5.2,
        color="#6A6A6A",
    )
    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.045, top=0.947, wspace=0.30, hspace=0.68)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def write_report(root: Path, observed: pd.DataFrame, summary: pd.DataFrame) -> None:
    (root / "report").mkdir(parents=True, exist_ok=True)
    available = int(summary["model_available"].sum())
    missing = summary.loc[~summary["model_available"], "city"].tolist()
    text = f"""# PPML city lag-response atlas

## Figure contract

- Core conclusion: city-specific physical-activity responses can persist for up to 12 calendar days after a composite heatwave event ends, with substantial heterogeneity in direction, magnitude and precision.
- Evidence: each panel reports one city's 12 post-event PPML coefficients as percentage changes with complete 95% confidence intervals.
- Layout: 75 fixed alphabetical city positions in a 5 x 15 quantitative grid; no DTW grouping or phenotype colour is used.
- Statistical encoding: filled points indicate two-sided Wald P < 0.05; open points indicate P >= 0.05; vertical lines are 95% confidence intervals.

## Coverage

- Available city models: {available}/75.
- PPML estimates: {len(observed)} rows ({available} cities x 12 lag days).
- Cities outside the estimable compound-heatwave subset: {', '.join(missing)}.

## Interpretation

Each panel uses a city-specific y-axis so every point estimate and confidence interval remains visible. Cross-city comparisons use the exported source table. Percentage changes are calculated as `100 * (exp(beta) - 1)`. The figure covers calendar days 1-12 after heatwave-event termination.

## Reproduction

Run `python code/make_ppml_city_lag_atlas.py --root <package_directory>`. All required inputs are archived under `data/input`, and all plotted values are exported under `data/derived`.
"""
    (root / "report" / "README.md").write_text(text, encoding="utf-8")

    legend = """**Extended Data Fig. X | City-specific post-heatwave physical-activity response trajectories.** City-specific percentage changes in outdoor physical activity during calendar days 1-12 after the end of a composite heatwave event were estimated using Poisson pseudo-maximum-likelihood models. Points show `100[exp(beta)-1]`; vertical lines denote 95% confidence intervals obtained by transforming the coefficient-scale Wald limits. Filled points indicate two-sided Wald P < 0.05 and open points indicate P >= 0.05. Panels use city-specific y-axis scales to retain complete estimates and intervals. The prespecified 75-city frame displays the 12 cities outside the estimable compound-heatwave subset as empty panels. `n` denotes city-grid-day observations. Source data are provided in `data/derived/ppml_city_lag_source_data_75grid.csv`."""
    (root / "report" / "FIGURE_LEGEND.md").write_text(legend, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 75-city PPML lag-response atlas.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    raw = pd.read_csv(root / "data" / "city_lag_estimates_12day.csv")
    city_order = pd.read_csv(root / "data" / "city_order_75.csv")
    observed, complete, summary = build_source_data(raw, city_order)

    derived = root / "data" / "city_atlas"
    derived.mkdir(parents=True, exist_ok=True)
    observed.to_csv(derived / "ppml_city_lag_observed_63city.csv", index=False)
    complete.to_csv(derived / "ppml_city_lag_source_data_75grid.csv", index=False)
    summary.to_csv(derived / "ppml_city_summary_75city.csv", index=False)
    summary[["city", "city_order", "model_available", "status"]].to_csv(
        derived / "ppml_city_coverage_75city.csv", index=False
    )

    draw_atlas(observed, complete, root / "output" / "city_atlas" / "fig2_ppml_city_lag_atlas_5x15")
    write_report(root, observed, summary)
    print(f"Completed PPML city atlas: {root}")


if __name__ == "__main__":
    main()
