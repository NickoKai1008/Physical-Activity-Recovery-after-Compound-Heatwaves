"""Render exactly two complete CEHWI Fig. 4a+b figures from frozen plotting nodes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "main_figure_input"
OUTPUT = ROOT / "data" / "main_figure_output"
FIGURES = ROOT / "output" / "main_panels"

TYPE_ORDER = ("compound", "day", "night")
TYPE_LABELS = {"compound": "Compound", "day": "Daytime", "night": "Nighttime"}
PHENOTYPE_LABELS = {
    1: "Delayed Deterioration",
    2: "Weak Oscillatory Response",
    3: "Early Suppression Followed by Attenuation",
    4: "Low-Amplitude Persistent Disturbance",
}
AF_COLORS = {"compound": "#A02D2B", "day": "#E9B858", "night": "#4C98C9"}
FORMAL_TITLE_COLORS = {"compound": "#9E3D36", "day": "#E4B447", "night": "#3F78A8"}
AF_FONT = "Times New Roman"
AF_Y_RULES = {
    1: ((-52.0, 20.0), np.arange(-50.0, 21.0, 10.0)),
    2: ((-102.0, 60.0), np.arange(-100.0, 51.0, 25.0)),
    3: ((-102.0, 40.0), np.arange(-100.0, 41.0, 20.0)),
    4: ((-110.0, 45.0), np.arange(-100.0, 41.0, 20.0)),
}
MODE_ORDER = ("ride", "run", "walk")
MODE_LABELS = {"ride": "Ride", "run": "Run", "walk": "Walk"}
MODE_COLORS = {"ride": "#B15351", "run": "#E0A482", "walk": "#4C8EBA"}
MODE_ALPHA = 1.0
CURVE_COLOR = "#27343A"
BAND_COLOR = "#CDD2D4"
PERCENTILE_ORDER = ("p25", "p50", "p75", "p90", "p95")
PERCENTILE_LABELS = ("25th", "50th", "75th", "90th", "95th")


class VerticalIntervalHandler(HandlerBase):
    """Draw a point with a vertical interval in compact legends."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        x = xdescent + 0.5 * width
        y = ydescent + 0.5 * height
        color = orig_handle.get_color()
        interval = Line2D(
            [x, x],
            [ydescent + 0.08 * height, ydescent + 0.92 * height],
            color=color,
            lw=0.9,
            transform=trans,
        )
        point = Line2D(
            [x],
            [y],
            marker="o",
            markersize=5.3,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            linestyle="None",
            transform=trans,
        )
        return [interval, point]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "heatpa-fig4-twofigure-20260809",
            "pdf.fonttype": 42,
        }
    )


def validate_inputs(
    curves: pd.DataFrame, histograms: pd.DataFrame, metadata: pd.DataFrame, version: str
) -> None:
    curve_columns = {
        "cluster",
        "heatwave_type",
        "exposure",
        "log_rr",
        "ci_low",
        "ci_high",
    }
    hist_columns = {
        "cluster",
        "heatwave_type",
        "activity_type",
        "bin_left",
        "bin_right",
        "bar_height",
        "histogram_scale",
    }
    metadata_columns = {
        "cluster",
        "heatwave_type",
        "display_high",
        "common_support_high",
        "marker_25",
        "marker_50",
        "marker_75",
        "marker_90",
        "y_min",
        "y_max",
        "n_cities",
        "lag_max",
        "show_u",
    }
    if not curve_columns.issubset(curves.columns):
        raise RuntimeError(f"{version}: incomplete curve columns")
    if not hist_columns.issubset(histograms.columns):
        raise RuntimeError(f"{version}: incomplete histogram columns")
    if not metadata_columns.issubset(metadata.columns):
        raise RuntimeError(f"{version}: incomplete metadata columns")
    keys = ["cluster", "heatwave_type"]
    if curves.groupby(keys).ngroups != 12 or histograms.groupby(keys).ngroups != 12:
        raise RuntimeError(f"{version}: expected 12 curve and histogram panels")
    if len(metadata) != 12 or metadata.groupby(keys).ngroups != 12:
        raise RuntimeError(f"{version}: expected one metadata row per panel")
    if not np.isfinite(curves[["exposure", "log_rr", "ci_low", "ci_high"]]).all().all():
        raise RuntimeError(f"{version}: non-finite curve node")
    if (curves["ci_low"] > curves["log_rr"]).any() or (
        curves["log_rr"] > curves["ci_high"]
    ).any():
        raise RuntimeError(f"{version}: estimate lies outside its confidence interval")
    if not set(curves["heatwave_type"]).issubset(TYPE_ORDER):
        raise RuntimeError(f"{version}: unexpected heatwave type")
    if not set(histograms["activity_type"]).issubset(MODE_ORDER):
        raise RuntimeError(f"{version}: unexpected activity type")
    if version == "p80":
        annotation_columns = {
            "fraction_trips_above_common_support",
            "total_positive_exposure_trips",
            "trip_share_5pct_count",
        }
        if not annotation_columns.issubset(metadata.columns):
            raise RuntimeError("p80: missing trip-above-u annotation metadata")
        if not np.isfinite(metadata[list(annotation_columns)]).all().all():
            raise RuntimeError("p80: non-finite trip-above-u annotation metadata")
    else:
        annotation_columns = {
            "total_positive_exposure_trips",
            "trip_share_5pct_count",
        }
        if not annotation_columns.issubset(metadata.columns):
            raise RuntimeError("p99: missing formal Fig. 4 trip-count annotations")
        if not np.isfinite(metadata[list(annotation_columns)]).all().all():
            raise RuntimeError("p99: non-finite formal Fig. 4 trip-count annotations")


def validate_af(af: pd.DataFrame, version: str) -> None:
    required = {
        "cluster",
        "heatwave_type",
        "exposure_percentile",
        "af_percent",
        "ci_low",
        "ci_high",
    }
    if not required.issubset(af.columns) or len(af) != 60:
        raise RuntimeError(f"{version}: expected 60 complete AF rows")
    if af.groupby(["cluster", "heatwave_type"]).ngroups != 12:
        raise RuntimeError(f"{version}: AF data do not contain 12 panels")
    if set(af["exposure_percentile"]) != set(PERCENTILE_ORDER):
        raise RuntimeError(f"{version}: incomplete AF percentiles")
    if not np.isfinite(af[["af_percent", "ci_low", "ci_high"]]).all().all():
        raise RuntimeError(f"{version}: non-finite AF value")
    if (af["ci_low"] > af["af_percent"]).any() or (
        af["af_percent"] > af["ci_high"]
    ).any():
        raise RuntimeError(f"{version}: AF estimate lies outside its interval")
    if version == "p99" and "significance_symbol" not in af.columns:
        raise RuntimeError("p99: missing final manuscript significance symbols")
    if version == "p80" and "within_common_support" not in af.columns:
        raise RuntimeError("p80: missing support flag")


def panel_row(frame: pd.DataFrame, cluster: int, heatwave_type: str) -> pd.DataFrame:
    return frame.loc[
        frame["cluster"].eq(cluster) & frame["heatwave_type"].eq(heatwave_type)
    ].copy()


def ordered_af(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(
            exposure_percentile=lambda x: pd.Categorical(
                x["exposure_percentile"], categories=PERCENTILE_ORDER, ordered=True
            )
        )
        .sort_values("exposure_percentile")
        .reset_index(drop=True)
    )


def format_exposure(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def padded_limits(
    values: np.ndarray, reference: float = 0.0, fraction: float = 0.10
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    finite = np.r_[finite, reference]
    lower, upper = float(finite.min()), float(finite.max())
    span = max(upper - lower, 1.0)
    return lower - fraction * span, upper + fraction * span


def plot_histogram(
    ax: plt.Axes, histogram: pd.DataFrame, meta: pd.Series, version: str
) -> None:
    reference = histogram.loc[histogram["activity_type"].eq("ride")].sort_values(
        "bin_left"
    )
    if reference.empty:
        raise RuntimeError("Ride histogram nodes are missing")
    left = reference["bin_left"].to_numpy(float)
    widths = reference["bin_right"].to_numpy(float) - left
    bottom = np.zeros(len(reference), dtype=float)
    for mode in MODE_ORDER:
        part = histogram.loc[histogram["activity_type"].eq(mode)].sort_values("bin_left")
        if len(part) != len(reference):
            raise RuntimeError(f"Incomplete {mode} histogram")
        values = part["bar_height"].to_numpy(float)
        ax.bar(
            left,
            values,
            bottom=bottom,
            width=widths,
            align="edge",
            color=MODE_COLORS[mode],
            edgecolor="white",
            linewidth=0.30 if version == "p99" else 0.16,
            alpha=MODE_ALPHA,
        )
        bottom += values
    scale = str(histogram["histogram_scale"].iloc[0])
    if version == "p99":
        ymax = max(float(bottom.max()) * 1.18, 0.058)
        ax.set_ylim(0, ymax)
        max_tick = max(0.05, np.floor(max(ymax - 0.008, 0.05) / 0.05) * 0.05)
        share_ticks = [0.0, 0.05]
        if max_tick > 0.05:
            share_ticks.append(max_tick)
        ax.set_yticks(np.unique(share_ticks))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
        )
        five_percent_count = float(meta["trip_share_5pct_count"])
        ax.text(
            0.985,
            0.91,
            f"5% = {five_percent_count:,.0f} trips",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#555555",
        )
        ax.tick_params(axis="both", labelsize=8.3, length=2.5, width=0.6, pad=1.5)
        ax.set_xlabel("CEHWI", fontsize=8.8, labelpad=1.5)
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_linewidth(0.75)
    else:
        ax.set_ylim(0, max(float(bottom.max()) * 1.12, 0.01))
        if scale == "share":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
        )
        fraction_above = float(meta["fraction_trips_above_common_support"])
        five_percent_count = float(meta["trip_share_5pct_count"])
        text_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5}
        ax.text(
            0.98,
            0.90,
            f"{100 * fraction_above:.0f}% trips > u",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.6,
            color="#505659",
            bbox=text_box,
        )
        ax.text(
            0.98,
            0.58,
            f"5% = {five_percent_count:,.0f} trips",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#5A6063",
            bbox=text_box,
        )
        ax.tick_params(axis="both", labelsize=8.3, length=2.5, width=0.6, pad=1.5)
        ax.set_xlabel("CEHWI", fontsize=8.8, labelpad=1.5)
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_linewidth(0.75)


def formal_colored_panel_title(
    ax: plt.Axes, cluster: int, heatwave_type: str
) -> None:
    prefix = TextArea(
        f"DTW4lag12 Cluster {cluster} |",
        textprops={
            "fontsize": 9.3,
            "fontweight": "bold",
            "fontfamily": AF_FONT,
            "color": "#111111",
        },
    )
    suffix = TextArea(
        TYPE_LABELS[heatwave_type],
        textprops={
            "fontsize": 9.3,
            "fontweight": "bold",
            "fontfamily": AF_FONT,
            "color": FORMAL_TITLE_COLORS[heatwave_type],
        },
    )
    title = HPacker(children=[prefix, suffix], align="baseline", pad=0, sep=2.0)
    anchored = AnchoredOffsetbox(
        loc="lower left",
        child=title,
        frameon=False,
        pad=0,
        borderpad=0,
        bbox_to_anchor=(0.0, 1.055),
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(anchored)


def draw_formal_p99_markers(
    ax_curve: plt.Axes, ax_hist: plt.Axes, meta: pd.Series
) -> None:
    for column, label in (
        ("marker_25", "25th"),
        ("marker_75", "75th"),
        ("marker_90", "90th"),
    ):
        value = float(meta[column])
        if not np.isfinite(value) or value > float(meta["display_high"]) + 1e-10:
            continue
        for ax in (ax_curve, ax_hist):
            ax.axvline(value, color="#8D8D8D", lw=0.68, ls=(0, (3, 3)), alpha=0.85)
        ax_curve.annotate(
            label,
            xy=(value, 0.965),
            xycoords=ax_curve.get_xaxis_transform(),
            xytext=(3.0, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.0,
            color="#7F7F7F",
        )


def draw_formal_p80_markers(
    ax_curve: plt.Axes, ax_hist: plt.Axes, meta: pd.Series
) -> None:
    for column, label in (
        ("marker_25", "25th"),
        ("marker_50", "50th"),
        ("marker_75", "75th"),
    ):
        value = float(meta[column])
        if not np.isfinite(value) or value > float(meta["display_high"]) + 1e-10:
            continue
        for ax in (ax_curve, ax_hist):
            ax.axvline(value, color="#8D8D8D", lw=0.68, ls=(0, (3, 3)), alpha=0.85)
        ax_curve.annotate(
            label,
            xy=(value, 0.965),
            xycoords=ax_curve.get_xaxis_transform(),
            xytext=(3.0, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.0,
            color="#7F7F7F",
        )

    support_high = float(meta["common_support_high"])
    for ax in (ax_curve, ax_hist):
        ax.axvline(support_high, color="#656B6E", lw=0.68, ls=(0, (2, 2)))
    ax_curve.annotate(
        f"u={format_exposure(support_high)}",
        xy=(support_high, 0.77),
        xycoords=ax_curve.get_xaxis_transform(),
        xytext=(-3.0, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555B5E",
    )


def expanded_p99_axis(panel: pd.DataFrame) -> tuple[float, float, float]:
    """Expand the locked p99 axis by a deterministic 1.55-2.30 factor.

    The shared rule targets a median CI width of about 45% of the displayed
    response-axis span. Panels with greater uncertainty receive more headroom.
    This changes only the plotted axis, never estimates or confidence limits.
    """
    lower = float(panel["y_min"].iloc[0])
    upper = float(panel["y_max"].iloc[0])
    ci_width = panel["ci_high"].to_numpy(float) - panel["ci_low"].to_numpy(float)
    base_axis_span = upper - lower
    uncertainty_ratio = float(np.nanmedian(ci_width) / base_axis_span)
    factor = float(np.clip(max(1.55, uncertainty_ratio / 0.45), 1.55, 2.30))
    center = 0.5 * (lower + upper)
    half_span = 0.5 * (upper - lower) * factor
    return center - half_span, center + half_span, factor


def prepare_plot_output(
    curves: pd.DataFrame, metadata: pd.DataFrame, version: str, output_name: str
) -> pd.DataFrame:
    merged = curves.merge(
        metadata[
            [
                "cluster",
                "heatwave_type",
                "display_high",
                "common_support_high",
                "y_min",
                "y_max",
            ]
        ],
        on=["cluster", "heatwave_type"],
        how="left",
        validate="many_to_one",
    )
    merged["phenotype_name"] = merged["cluster"].map(PHENOTYPE_LABELS)
    merged = merged.loc[merged["exposure"].le(merged["display_high"] + 1e-10)].copy()
    if version == "p99":
        merged["line_segment"] = "continuous_full_curve"
        merged["axis_expansion_factor"] = np.nan
        for _, index in merged.groupby(["cluster", "heatwave_type"]).groups.items():
            panel = merged.loc[index]
            y_min, y_max, factor = expanded_p99_axis(panel)
            merged.loc[index, "y_min"] = y_min
            merged.loc[index, "y_max"] = y_max
            merged.loc[index, "axis_expansion_factor"] = factor
    else:
        merged["axis_expansion_factor"] = 1.0
        merged["line_segment"] = np.where(
            merged["exposure"].le(merged["common_support_high"] + 1e-10),
            "solid_common_support",
            "dashed_p80_tail",
        )
    merged.to_csv(OUTPUT / output_name, index=False)
    return merged


def prepare_af_output(af: pd.DataFrame, version: str, output_name: str) -> pd.DataFrame:
    plotted = af.copy()
    plotted["phenotype_name"] = plotted["cluster"].map(PHENOTYPE_LABELS)
    if version == "p99":
        plotted["marker_style"] = "filled"
        plotted["star_display"] = plotted["significance_symbol"].fillna("")
    else:
        plotted["marker_style"] = np.where(
            plotted["within_common_support"].astype(bool), "filled", "open"
        )
        plotted["star_display"] = ""
    plotted.to_csv(OUTPUT / output_name, index=False)
    return plotted


def render(
    curves: pd.DataFrame,
    histograms: pd.DataFrame,
    metadata: pd.DataFrame,
    af: pd.DataFrame,
    *,
    version: str,
    output_stem: str,
    output_csv: str,
    af_output_csv: str,
) -> None:
    curves = prepare_plot_output(curves, metadata, version, output_csv)
    af = prepare_af_output(af, version, af_output_csv)
    # Both specifications share the locked formal Fig. 4 canvas and geometry.
    fig = plt.figure(figsize=(12.0, 9.76))
    outer = fig.add_gridspec(
        4,
        4,
        width_ratios=[1.0, 1.0, 1.0, 1.25],
        hspace=0.34,
        wspace=0.23,
        left=0.052,
        right=0.991,
        top=0.957,
        bottom=0.062,
    )

    for row, cluster in enumerate(range(1, 5)):
        for column, heatwave_type in enumerate(TYPE_ORDER):
            meta_frame = panel_row(metadata, cluster, heatwave_type)
            if len(meta_frame) != 1:
                raise RuntimeError(f"Missing metadata C{cluster} {heatwave_type}")
            meta = meta_frame.iloc[0]
            display_high = float(meta["display_high"])
            curve = panel_row(curves, cluster, heatwave_type).sort_values("exposure")
            curve = curve.loc[curve["exposure"].le(display_high + 1e-10)]
            histogram = panel_row(histograms, cluster, heatwave_type)

            cell = outer[row, column].subgridspec(
                2, 1, height_ratios=[2.4, 1.65], hspace=0.0
            )
            ax_curve = fig.add_subplot(cell[0])
            ax_hist = fig.add_subplot(cell[1], sharex=ax_curve)

            x = curve["exposure"].to_numpy(float)
            estimate = curve["log_rr"].to_numpy(float)
            lower = curve["ci_low"].to_numpy(float)
            upper = curve["ci_high"].to_numpy(float)

            if version == "p99":
                ax_curve.fill_between(
                    x, lower, upper, color="#D4D9DB", alpha=0.58, linewidth=0
                )
                ax_curve.plot(
                    x, estimate, color="#35434A", lw=1.35, solid_capstyle="round"
                )
            else:
                support_high = float(meta["common_support_high"])
                support = curve.loc[curve["exposure"].le(support_high + 1e-10)]
                tail = curve.loc[curve["exposure"].gt(support_high + 1e-10)]
                sx = support["exposure"].to_numpy(float)
                ax_curve.fill_between(
                    sx,
                    support["ci_low"].to_numpy(float),
                    support["ci_high"].to_numpy(float),
                    color=BAND_COLOR,
                    alpha=0.52,
                    linewidth=0,
                )
                ax_curve.plot(
                    sx,
                    support["log_rr"].to_numpy(float),
                    color=CURVE_COLOR,
                    lw=1.18,
                    solid_capstyle="round",
                )
                if not tail.empty:
                    tail = pd.concat([support.tail(1), tail], ignore_index=True)
                    tx = tail["exposure"].to_numpy(float)
                    ax_curve.fill_between(
                        tx,
                        tail["ci_low"].to_numpy(float),
                        tail["ci_high"].to_numpy(float),
                        color=BAND_COLOR,
                        alpha=0.30,
                        linewidth=0,
                    )
                    ax_curve.plot(
                        tx,
                        tail["log_rr"].to_numpy(float),
                        color=CURVE_COLOR,
                        lw=1.02,
                        ls=(0, (3, 2)),
                    )

            ax_curve.axhline(0, color="#777D80", lw=0.52, ls=(0, (3, 2)))
            ax_curve.set_xlim(0, display_high)
            ax_curve.set_ylim(
                float(curve["y_min"].iloc[0]), float(curve["y_max"].iloc[0])
            )
            ax_curve.grid(axis="y", color="#E6E8E9", lw=0.50)
            ax_curve.tick_params(
                axis="x", labelbottom=False, length=2.5, width=0.60, pad=1.5
            )
            ax_curve.tick_params(axis="y", labelsize=8.3, length=2.5, width=0.60, pad=1.5)
            ax_curve.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=4,
                    min_n_ticks=3,
                    prune="lower",
                    steps=[1, 2, 2.5, 5, 10],
                )
            )
            ax_curve.set_ylabel("log-RR", fontsize=8.8, labelpad=2.5)
            formal_colored_panel_title(ax_curve, cluster, heatwave_type)
            plot_histogram(ax_hist, histogram, meta, version)
            if version == "p99":
                draw_formal_p99_markers(ax_curve, ax_hist, meta)
            else:
                draw_formal_p80_markers(ax_curve, ax_hist, meta)
            ax_hist.set_ylabel("Trip share", fontsize=8.8, labelpad=2.5)
            if row == 3 and heatwave_type == "night":
                ax_hist.text(
                    0.995,
                    0.43,
                    "Observed PA composition",
                    transform=ax_hist.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6.5,
                    color="#333333",
                )
                mode_legend = ax_hist.legend(
                    handles=[
                        Patch(
                            facecolor=MODE_COLORS[mode],
                            edgecolor="white",
                            linewidth=0.2,
                            alpha=MODE_ALPHA,
                            label=MODE_LABELS[mode],
                        )
                        for mode in MODE_ORDER
                    ],
                    loc="center right",
                    bbox_to_anchor=(0.995, 0.28),
                    ncol=3,
                    fontsize=6.4,
                    handlelength=0.75,
                    handletextpad=0.24,
                    columnspacing=0.48,
                    labelspacing=0.0,
                    borderaxespad=0.0,
                )
                mode_legend._legend_box.align = "left"

        ax_af = fig.add_subplot(outer[row, 3])
        cluster_af = af.loc[af["cluster"].eq(cluster)].copy()
        y_values = cluster_af[["af_percent", "ci_low", "ci_high"]].to_numpy(float).ravel()
        if version == "p99":
            limits, ticks = AF_Y_RULES[cluster]
            ax_af.set_ylim(*limits)
            ax_af.yaxis.set_major_locator(FixedLocator(ticks))
        else:
            ax_af.set_ylim(*padded_limits(y_values))
            ax_af.yaxis.set_major_locator(
                MaxNLocator(nbins=5, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
            )
        positions = np.arange(len(PERCENTILE_ORDER), dtype=float)
        offsets = {"compound": -0.17, "day": 0.0, "night": 0.17}
        for heatwave_type in TYPE_ORDER:
            part = ordered_af(
                cluster_af.loc[cluster_af["heatwave_type"].eq(heatwave_type)]
            )
            if len(part) != 5:
                raise RuntimeError(f"Incomplete AF panel C{cluster} {heatwave_type}")
            color = AF_COLORS[heatwave_type]
            point = part["af_percent"].to_numpy(float)
            low = part["ci_low"].to_numpy(float)
            high = part["ci_high"].to_numpy(float)
            xpos = positions + offsets[heatwave_type]
            ax_af.vlines(xpos, low, high, color=color, lw=0.90, alpha=1.0, zorder=1)
            if version == "p99":
                ax_af.scatter(
                    xpos,
                    point,
                    s=23,
                    color=color,
                    edgecolor="white",
                    linewidth=0.38,
                    zorder=3,
                )
                symbols = part["significance_symbol"].fillna("").astype(str).to_numpy()
                for star_x, star_y, symbol in zip(xpos, point, symbols):
                    if symbol:
                        ax_af.text(
                            star_x + 0.055,
                            star_y,
                            symbol,
                            color=color,
                            fontsize=6.6,
                            weight="bold",
                            fontname=AF_FONT,
                            ha="left",
                            va="center",
                            zorder=4,
                        )
            else:
                within = part["within_common_support"].astype(bool).to_numpy()
                ax_af.scatter(
                    xpos[within],
                    point[within],
                    s=23,
                    color=color,
                    edgecolor="white",
                    linewidth=0.38,
                    zorder=3,
                )
                ax_af.scatter(
                    xpos[~within],
                    point[~within],
                    s=23,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.82,
                    zorder=3,
                )
        ax_af.axhline(0, color="#8D8D8D", lw=0.65, zorder=1)
        ax_af.set_xticks(positions)
        ax_af.set_xticklabels(PERCENTILE_LABELS)
        ax_af.set_ylabel("Pooled AF (%)", fontsize=9.9, fontname=AF_FONT, labelpad=4.0)
        ax_af.set_xlabel("Exposure percentile", fontsize=9.9, fontname=AF_FONT, labelpad=0.8)
        ax_af.text(
            0.0,
            1.045,
            f"C{cluster} | {PHENOTYPE_LABELS[cluster]}",
            transform=ax_af.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.2,
            weight="bold",
            fontname=AF_FONT,
            clip_on=False,
        )
        ax_af.grid(axis="y", color="#E7E2DC", lw=0.48)
        ax_af.tick_params(axis="both", labelsize=8.5, length=2.7, width=0.60, pad=1.5)
        for label in [*ax_af.get_xticklabels(), *ax_af.get_yticklabels()]:
            label.set_fontname(AF_FONT)
            label.set_fontsize(7.8)
        if cluster == 1:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=AF_COLORS[heatwave_type],
                    markersize=5.3,
                    lw=0,
                    label=TYPE_LABELS[heatwave_type],
                )
                for heatwave_type in TYPE_ORDER
            ]
            legend = ax_af.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.52, 1.015),
                ncol=3,
                frameon=False,
                fontsize=7.0,
                handlelength=1.0,
                handletextpad=0.20,
                columnspacing=0.60,
                borderaxespad=0.0,
                handleheight=1.45,
                handler_map={Line2D: VerticalIntervalHandler()},
            )
            for text, heatwave_type in zip(legend.get_texts(), TYPE_ORDER):
                text.set_color(AF_COLORS[heatwave_type])
                text.set_fontweight("bold")
                text.set_fontname(AF_FONT)

    fig.text(0.006, 0.991, "a", fontsize=14.2, weight="bold", va="top")
    fig.text(0.720, 0.991, "b", fontsize=14.2, weight="bold", va="top")
    for suffix in ("png", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {"metadata": {"Date": "2026-08-09"}}
        fig.savefig(
            FIGURES / f"{output_stem}.{suffix}", facecolor="white", **kwargs
        )
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_checks() -> None:
    expected_inputs = {
        "p99_curve_nodes.csv",
        "p99_af_nodes.csv",
        "p99_histogram_nodes.csv",
        "p99_panel_metadata.csv",
        "p80_curve_nodes.csv",
        "p80_af_nodes.csv",
        "p80_histogram_nodes.csv",
        "p80_panel_metadata.csv",
    }
    expected_outputs = {
        "Fig4a_CEHWI_p99_curve_plot_data.csv",
        "Fig4b_CEHWI_p99_AF_plot_data.csv",
        "Extended_Fig4a_CEHWI_p80_curve_plot_data.csv",
        "Extended_Fig4b_CEHWI_p80_support_AF_plot_data.csv",
    }
    expected_figures = {
        "Fig4_CEHWI_p99.png",
        "Fig4_CEHWI_p99.svg",
        "Extended_Fig_CEHWI_p80_support_coded_AF.png",
        "Extended_Fig_CEHWI_p80_support_coded_AF.svg",
    }
    if {p.name for p in INPUT.glob("*.csv")} != expected_inputs:
        raise RuntimeError("Input folder is not the locked eight-CSV set")
    if {p.name for p in OUTPUT.glob("*.csv")} != expected_outputs:
        raise RuntimeError("Output folder must contain exactly four plotting CSVs")
    # Windows Explorer may create Thumbs.db in this folder. Validate only the
    # intended publication figure formats so that OS metadata cannot break an
    # otherwise deterministic reproduction run.
    figure_files = {
        p.name for p in FIGURES.iterdir() if p.suffix.lower() in {".png", ".svg"}
    }
    if figure_files != expected_figures:
        raise RuntimeError("Figures folder must contain exactly two PNG/SVG figure pairs")
    for path in FIGURES.iterdir():
        if path.suffix == ".png" and path.stat().st_size < 300_000:
            raise RuntimeError(f"Undersized PNG: {path.name}")
        if path.suffix == ".svg" and path.stat().st_size < 40_000:
            raise RuntimeError(f"Undersized SVG: {path.name}")
    p99 = pd.read_csv(OUTPUT / "Fig4a_CEHWI_p99_curve_plot_data.csv")
    if p99.groupby(["cluster", "heatwave_type"]).ngroups != 12:
        raise RuntimeError("p99 output does not contain all 12 finalized panels")
    if set(p99["heatwave_type"]) != set(TYPE_ORDER):
        raise RuntimeError("p99 output does not use the locked Compound/Daytime/Nighttime keys")
    if p99.groupby("cluster")["phenotype_name"].first().to_dict() != PHENOTYPE_LABELS:
        raise RuntimeError("p99 phenotype labels are inconsistent")
    if not p99["line_segment"].eq("continuous_full_curve").all():
        raise RuntimeError("p99 contains a support split")
    p99_factors = p99.groupby(["cluster", "heatwave_type"])[
        "axis_expansion_factor"
    ].first()
    if not p99_factors.between(1.55, 2.30, inclusive="both").all():
        raise RuntimeError("p99 axis expansion falls outside the locked 1.55-2.30 rule")
    p99_source = pd.read_csv(INPUT / "p99_curve_nodes.csv")
    p99_identity = p99_source.merge(
        p99,
        on=["cluster", "heatwave_type", "exposure"],
        suffixes=("_input", "_output"),
        validate="one_to_one",
    )
    for field in ("log_rr", "ci_low", "ci_high"):
        if not np.allclose(
            p99_identity[f"{field}_input"],
            p99_identity[f"{field}_output"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"p99 {field} changed between frozen input and plot output")
    p99_af = pd.read_csv(OUTPUT / "Fig4b_CEHWI_p99_AF_plot_data.csv")
    if p99_af.groupby("cluster")["phenotype_name"].first().to_dict() != PHENOTYPE_LABELS:
        raise RuntimeError("p99 AF phenotype labels are inconsistent")
    if not p99_af["marker_style"].eq("filled").all():
        raise RuntimeError("p99 AF panel contains a non-filled marker")
    expected_stars = p99_af["significance_symbol"].fillna("")
    actual_stars = p99_af["star_display"].fillna("")
    if not expected_stars.equals(actual_stars):
        raise RuntimeError("p99 significance-star encoding is inconsistent")
    p99_af_source = pd.read_csv(INPUT / "p99_af_nodes.csv")
    p99_af_identity = p99_af_source.merge(
        p99_af,
        on=["cluster", "heatwave_type", "exposure_percentile"],
        suffixes=("_input", "_output"),
        validate="one_to_one",
    )
    for field in ("af_percent", "ci_low", "ci_high"):
        if not np.allclose(
            p99_af_identity[f"{field}_input"],
            p99_af_identity[f"{field}_output"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"p99 AF {field} changed between input and output")
    p80 = pd.read_csv(OUTPUT / "Extended_Fig4a_CEHWI_p80_curve_plot_data.csv")
    if set(p80["heatwave_type"]) != set(TYPE_ORDER):
        raise RuntimeError("p80 output does not use the locked Compound/Daytime/Nighttime keys")
    if p80.groupby("cluster")["phenotype_name"].first().to_dict() != PHENOTYPE_LABELS:
        raise RuntimeError("p80 phenotype labels are inconsistent")
    if set(p80["line_segment"]) != {"solid_common_support", "dashed_p80_tail"}:
        raise RuntimeError("p80 support/tail segmentation is incomplete")
    if not p80["axis_expansion_factor"].eq(1.0).all():
        raise RuntimeError("p80 response-axis limits were altered")
    p80_source = pd.read_csv(INPUT / "p80_curve_nodes.csv")
    p80_identity = p80_source.merge(
        p80,
        on=["cluster", "heatwave_type", "exposure"],
        suffixes=("_input", "_output"),
        validate="one_to_one",
    )
    for field in ("log_rr", "ci_low", "ci_high"):
        if not np.allclose(
            p80_identity[f"{field}_input"],
            p80_identity[f"{field}_output"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"p80 {field} changed between frozen input and plot output")
    p80_af = pd.read_csv(OUTPUT / "Extended_Fig4b_CEHWI_p80_support_AF_plot_data.csv")
    if p80_af.groupby("cluster")["phenotype_name"].first().to_dict() != PHENOTYPE_LABELS:
        raise RuntimeError("p80 AF phenotype labels are inconsistent")
    if set(p80_af["marker_style"]) != {"filled", "open"}:
        raise RuntimeError("p80 AF panel lacks filled/open support coding")
    p80_af_source = pd.read_csv(INPUT / "p80_af_nodes.csv")
    p80_af_identity = p80_af_source.merge(
        p80_af,
        on=["cluster", "heatwave_type", "exposure_percentile"],
        suffixes=("_input", "_output"),
        validate="one_to_one",
    )
    for field in ("af_percent", "ci_low", "ci_high"):
        if not np.allclose(
            p80_af_identity[f"{field}_input"],
            p80_af_identity[f"{field}_output"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"p80 AF {field} changed between input and output")
    for path in FIGURES.glob("*.svg"):
        svg = path.read_text(encoding="utf-8")
        if "Composite" in svg or "Compound" not in svg or "<text" not in svg:
            raise RuntimeError(f"SVG label/editable-text audit failed: {path.name}")
        if "Times New Roman" not in svg or "Arial" in svg:
            raise RuntimeError(f"SVG Times-font audit failed: {path.name}")
        if svg.count("Observed PA composition") != 1:
            raise RuntimeError(f"Embedded PA legend audit failed: {path.name}")
        if "CI extends beyond axis" in svg or "95% CI" in svg:
            raise RuntimeError(f"Obsolete CI-overflow annotation remains: {path.name}")
        if "attributable fraction by exposure intensity" in svg:
            raise RuntimeError(f"Redundant AF title remains: {path.name}")
        if any(name not in svg for name in PHENOTYPE_LABELS.values()):
            raise RuntimeError(f"Phenotype-title audit failed: {path.name}")
        if any(color.lower() not in svg.lower() for color in MODE_COLORS.values()):
            raise RuntimeError(f"Original Fig. 4 histogram-palette audit failed: {path.name}")
    print("Two-figure package checks passed")
    publication_figures = sorted(
        path for path in FIGURES.iterdir() if path.suffix.lower() in {".png", ".svg"}
    )
    for path in publication_figures:
        print(f"{path.name}: {sha256(path)}")


def main() -> None:
    configure()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    curve_value_columns = [
        "cluster",
        "heatwave_type",
        "exposure",
        "log_rr",
        "ci_low",
        "ci_high",
    ]
    p99_curves = pd.read_csv(
        INPUT / "p99_curve_nodes.csv", usecols=curve_value_columns
    )
    p99_histograms = pd.read_csv(INPUT / "p99_histogram_nodes.csv")
    p99_metadata = pd.read_csv(INPUT / "p99_panel_metadata.csv")
    p99_af = pd.read_csv(INPUT / "p99_af_nodes.csv")
    validate_inputs(p99_curves, p99_histograms, p99_metadata, "p99")
    validate_af(p99_af, "p99")
    render(
        p99_curves,
        p99_histograms,
        p99_metadata,
        p99_af,
        version="p99",
        output_stem="Fig4_CEHWI_p99",
        output_csv="Fig4a_CEHWI_p99_curve_plot_data.csv",
        af_output_csv="Fig4b_CEHWI_p99_AF_plot_data.csv",
    )

    p80_curves = pd.read_csv(
        INPUT / "p80_curve_nodes.csv", usecols=curve_value_columns
    )
    p80_histograms = pd.read_csv(INPUT / "p80_histogram_nodes.csv")
    p80_metadata = pd.read_csv(INPUT / "p80_panel_metadata.csv")
    p80_af = pd.read_csv(INPUT / "p80_af_nodes.csv")
    validate_inputs(p80_curves, p80_histograms, p80_metadata, "p80")
    validate_af(p80_af, "p80")
    render(
        p80_curves,
        p80_histograms,
        p80_metadata,
        p80_af,
        version="p80",
        output_stem="Extended_Fig_CEHWI_p80_support_coded_AF",
        output_csv="Extended_Fig4a_CEHWI_p80_curve_plot_data.csv",
        af_output_csv="Extended_Fig4b_CEHWI_p80_support_AF_plot_data.csv",
    )
    final_checks()


if __name__ == "__main__":
    main()
