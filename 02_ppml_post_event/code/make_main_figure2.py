"""Reproduce the submitted Main Figure 2 panel structure from compact PPML data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import make_figure2_panels as base


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "city_lag_estimates_12day.csv"
OUTPUT = ROOT / "output" / "main_figure"


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="top",
    )


def plot_pooled(ax, df, pooled, lags) -> None:
    rng = np.random.RandomState(0)
    jitter = (rng.rand(len(df)) - 0.5) * 0.20
    colors = base.direction_colors(df["estimate"], df["p.value"])
    ax.scatter(
        df["lag"].to_numpy() + jitter,
        df["pct"].to_numpy(),
        s=np.clip(df["n_obs"].to_numpy() / 2500, 5, 34),
        c=colors,
        alpha=0.56,
        linewidths=0,
        zorder=2,
    )

    x = pooled["lag"].to_numpy()
    ax.fill_between(
        x,
        pooled["pct_lo"].to_numpy(),
        pooled["pct_hi"].to_numpy(),
        color=base.PALETTE["ink"],
        alpha=0.12,
        linewidth=0,
        zorder=3,
    )
    ax.plot(x, pooled["pct"].to_numpy(), color=base.PALETTE["ink"], lw=2.2, zorder=4)
    ax.scatter(
        x,
        pooled["pct"].to_numpy(),
        s=120,
        color=base.PALETTE["prot_deep"],
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
    )

    n_cities = df["city"].nunique()
    counts = (
        df.loc[df["p.value"] < 0.05]
        .groupby("lag")["city"]
        .nunique()
        .reindex(lags, fill_value=0)
    )
    y_min, y_max = base.pooled_ci_ylim(pooled)
    ax.set_ylim(y_min, y_max)
    for lag in lags:
        ax.text(
            lag,
            y_max - 0.04 * (y_max - y_min),
            f"k={counts[lag]}/{n_cities}",
            ha="center",
            va="top",
            fontsize=6.5,
            color=base.PALETTE["subink"],
        )

    ax.axhline(0, color=base.PALETTE["subink"], lw=0.7, ls="--")
    ax.set_xticks(lags)
    ax.set_xlabel("Lag after compound heatwave event (days)")
    ax.set_ylabel("Change in physical activity (%)")
    ax.set_title(
        "Pooled 12-day lag response of physical activity to compound heatwave exposure",
        loc="left",
        pad=8,
        fontsize=11,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=base.PALETTE["harm"], markeredgecolor="none", label="PA up (P<0.05)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=base.PALETTE["prot"], markeredgecolor="none", label="PA down (P<0.05)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=base.PALETTE["ns"], markeredgecolor="none", label="Non-significant"),
            Line2D([0], [0], color=base.PALETTE["ink"], lw=2.2, label="Random-effects pooled estimate"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.58, -0.18),
        ncol=2,
        fontsize=7,
    )
    add_panel_label(ax, "a")


def lag_tally(df, lags):
    return (
        df.assign(
            direction=np.where(
                df["p.value"] < 0.05,
                np.where(df["estimate"] > 0, "PA up", "PA down"),
                "Non-significant",
            )
        )
        .groupby(["lag", "direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=lags, columns=["PA down", "Non-significant", "PA up"], fill_value=0)
    )


def plot_tally(ax, df, lags) -> None:
    tally = lag_tally(df, lags)
    colors = {
        "PA down": base.PALETTE["prot"],
        "Non-significant": base.PALETTE["ns"],
        "PA up": base.PALETTE["harm"],
    }
    bottom = np.zeros(len(lags))
    for direction in ["PA down", "Non-significant", "PA up"]:
        values = tally[direction].to_numpy()
        ax.bar(
            lags,
            values,
            bottom=bottom,
            width=0.72,
            color=colors[direction],
            edgecolor="white",
            linewidth=0.6,
        )
        for i, value in enumerate(values):
            if value:
                ax.text(
                    lags[i],
                    bottom[i] + value / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    fontweight="bold",
                    color="white" if direction != "Non-significant" else base.PALETTE["ink"],
                )
        bottom += values
    ax.set_xticks(lags)
    ax.set_xlabel("Lag day")
    ax.set_ylabel(f"Number of cities (of {df['city'].nunique()})")
    ax.set_title("Directional tally by lag", loc="left", pad=6, fontsize=10.5)
    ax.legend(
        handles=[Patch(facecolor=colors[key], label=key) for key in ["PA down", "Non-significant", "PA up"]],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=6.6,
    )
    add_panel_label(ax, "b")


def plot_distribution(ax, df, lags) -> None:
    values = [df.loc[df["lag"] == lag, "pct"].clip(-100, 250).to_numpy() for lag in lags]
    violins = ax.violinplot(values, positions=lags, widths=0.82, showextrema=False, showmedians=False)
    for body in violins["bodies"]:
        body.set_facecolor(base.PALETTE["ink"])
        body.set_alpha(0.10)
        body.set_edgecolor("none")
    for lag in lags:
        subset = df.loc[df["lag"] == lag]
        y = subset["pct"].clip(-100, 250).to_numpy()
        x = np.full(len(y), lag) + (np.random.RandomState(lag).rand(len(y)) - 0.5) * 0.18
        ax.scatter(
            x,
            y,
            s=11,
            c=base.direction_colors(subset["estimate"], subset["p.value"]),
            alpha=0.72,
            linewidths=0,
        )
        ax.hlines(np.median(y), lag - 0.25, lag + 0.25, color=base.PALETTE["ink"], lw=1.4)
    ax.axhline(0, color=base.PALETTE["subink"], lw=0.7, ls="--")
    ax.set_xticks(lags)
    ax.set_xlabel("Lag day")
    ax.set_ylabel("Change in physical activity (%)")
    ax.set_title("Distribution of city effects per lag", loc="left", pad=6, fontsize=10.5)
    add_panel_label(ax, "c")


def plot_heatmap(ax, fig, df, city_summary, lags) -> None:
    order = city_summary["city"].to_numpy()
    matrix = df.pivot(index="city", columns="lag", values="pct").loc[order, lags]
    p_values = df.pivot(index="city", columns="lag", values="p.value").loc[order, lags]
    vmax = float(np.nanpercentile(np.abs(matrix.to_numpy()), 95))
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap=base.DIV_CMAP,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f"L{lag}" for lag in lags], rotation=45, ha="right", fontsize=6.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=4.6)
    ax.set_title("City × lag heatmap (% change)", loc="left", pad=6, fontsize=10.5)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            stars = base.sig_stars(float(p_values.iat[row, col]))
            if stars:
                colour = "white" if abs(float(matrix.iat[row, col])) > 0.55 * vmax else base.PALETTE["ink"]
                ax.text(col, row, stars, ha="center", va="center", fontsize=3.0, color=colour)
    colour_bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colour_bar.set_label("Change in physical activity (%)", fontsize=7)
    colour_bar.ax.tick_params(labelsize=6)
    colour_bar.outline.set_visible(False)
    add_panel_label(ax, "d")


def main() -> None:
    df, pooled, city_summary, lags = base.prepare_data(INPUT)
    if len(lags) != 12 or df["city"].nunique() != 63:
        raise ValueError("Main Figure 2 requires 63 cities with complete lag days 1-12.")

    fig = plt.figure(figsize=(17.2, 9.6), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.00, 1.12],
        height_ratios=[0.49, 0.51],
        left=0.065,
        right=0.985,
        bottom=0.10,
        top=0.965,
        wspace=0.25,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(grid[0, :2])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[:, 2])

    plot_pooled(ax_a, df, pooled, lags)
    plot_tally(ax_b, df, lags)
    plot_distribution(ax_c, df, lags)
    plot_heatmap(ax_d, fig, df, city_summary, lags)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "main_figure_02_reproduced"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"Main Figure 2 reproduced -> {stem.with_suffix('.png')} / .svg")


if __name__ == "__main__":
    main()
