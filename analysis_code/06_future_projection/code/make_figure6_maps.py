from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
REPO_ROOT = MODULE_DIR.parents[1]
EXTERNAL_DATA_ROOT = Path(os.environ.get("HEATPA_DATA_ROOT", REPO_ROOT / "external_data"))
SOURCE_CSV = MODULE_DIR / "data" / "maps" / "city_map_2025_2050_plot_data.csv"
CLUSTER_CSV = MODULE_DIR / "data" / "figure6e" / "dtw4lag12_city_cluster_map.csv"
BASE_CODE = SCRIPT_DIR / "make_figure6_ad.py"
OUT_ROOT = MODULE_DIR / "output" / "maps_generated"

SCENARIOS = ["ssp245", "ssp370", "ssp585"]
SCENARIO_LABELS = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
INDICATOR_LABELS = {
    "cehwi": "CEHWI",
    "exceeded_quantity": "Exceeded quantity",
}
HEATWAVE_LABELS = {
    "composite": "Compound",
    "day": "Daytime",
    "night": "Nighttime",
}

VALUE_COL = "pa_loss_fraction_percent_all_days"
LOW_COL = "pa_loss_fraction_low_percent_all_days"
HIGH_COL = "pa_loss_fraction_high_percent_all_days"

ABS_CMAP_COLORS = [
    "#2C2A87",
    "#4F79BD",
    "#82B7D9",
    "#BEE3E3",
    "#E6F5C9",
    "#FFF0A6",
    "#FEC980",
    "#F98E52",
    "#EF4A2F",
    "#D7191C",
    "#B9002B",
]
DELTA_CMAP_COLORS = ["#2166AC", "#D9F0F2", "#F7F7F7", "#FEE090", "#B2182B"]

POINT_MIN = 20
POINT_MAX = 245
N_CLASSES = 7
CLASS_SIZES = np.array([24, 34, 48, 68, 96, 142, 220], dtype=float)
NATURAL_BREAK_COLORS = [
    "#2C2A87",
    "#4F79BD",
    "#82B7D9",
    "#C8E0A3",
    "#F1D36A",
    "#F98E52",
    "#B9002B",
]

POINT_FACE_ALPHA = 0.50
POINT_EDGE_ALPHA = 0.90
POINT_EDGE_LINEWIDTH = 0.48
STATE_BOUNDARY_COLOR = "#969696"
STATE_BOUNDARY_LINEWIDTH = 0.58
STATE_BOUNDARY_ALPHA = 0.96
LOCAL_STATE_BOUNDARY_SHP = EXTERNAL_DATA_ROOT / "gis" / "US_StateOrTerritory.shp"
NON_CONUS_STATE_NAMES = {
    "Alaska", "Hawaii", "Puerto Rico", "Guam", "American Samoa",
    "Virgin Islands", "Northern Mariana Islands"
}


def load_base_module():
    if not BASE_CODE.exists():
        raise FileNotFoundError(BASE_CODE)
    spec = importlib.util.spec_from_file_location("fig6_base_plotter_shared_scale", BASE_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base plotting module: {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_slug(*parts: str) -> str:
    return "_".join(str(p).lower().replace(" ", "_").replace("-", "") for p in parts)


def value_to_size(values: pd.Series | np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(vmax - vmin) or np.isclose(vmax, vmin):
        scaled = np.zeros_like(values)
    else:
        scaled = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    return POINT_MIN + (POINT_MAX - POINT_MIN) * scaled


def jenks_breaks(values: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Jenks natural breaks with robust quantile fallback for repeated values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    if x.size == 0:
        return np.array([0.0, 1.0])
    unique = np.unique(x)
    if unique.size <= n_classes:
        breaks = np.r_[unique[0], unique[1:], unique[-1]]
        return np.unique(breaks)

    n = x.size
    k = min(n_classes, unique.size)
    lower = np.zeros((n + 1, k + 1), dtype=int)
    var = np.full((n + 1, k + 1), np.inf)
    for i in range(1, k + 1):
        lower[1, i] = 1
        var[1, i] = 0
        for j in range(2, n + 1):
            var[j, i] = np.inf

    for l in range(2, n + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = x[i3 - 1]
            s2 += val * val
            s1 += val
            w += 1
            variance = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, k + 1):
                    if var[l, j] >= variance + var[i4, j - 1]:
                        lower[l, j] = i3
                        var[l, j] = variance + var[i4, j - 1]
        lower[l, 1] = 1
        var[l, 1] = variance

    breaks = np.zeros(k + 1)
    breaks[-1] = x[-1]
    count = k
    idx = n
    while count > 1:
        breaks[count - 1] = x[lower[idx, count] - 2]
        idx = lower[idx, count] - 1
        count -= 1
    breaks[0] = x[0]

    breaks = np.unique(breaks)
    if breaks.size < 4:
        breaks = np.unique(np.nanquantile(x, np.linspace(0, 1, min(n_classes, unique.size) + 1)))
    return breaks


def natural_breaks_for_values(values: pd.Series | np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([0.0, 1.0])
    positives = vals[vals > 0]
    if positives.size < 8:
        vmax = max(float(np.nanmax(vals)), 0.01)
        return np.linspace(0, vmax, N_CLASSES + 1)
    breaks = jenks_breaks(positives, N_CLASSES)
    breaks[0] = min(0.0, float(np.nanmin(vals)))
    breaks[-1] = max(breaks[-1], float(np.nanmax(vals)))
    breaks = np.unique(np.r_[0.0, breaks])
    if breaks.size < 4:
        breaks = np.linspace(0, max(float(np.nanmax(vals)), 0.01), N_CLASSES + 1)
    return breaks


def classify_values(values: pd.Series | np.ndarray, breaks: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    idx = np.digitize(vals, breaks[1:-1], right=True)
    return np.clip(idx, 0, len(breaks) - 2)


def class_sizes(class_idx: np.ndarray, n_classes: int) -> np.ndarray:
    if n_classes == len(CLASS_SIZES):
        return CLASS_SIZES[class_idx]
    return np.linspace(CLASS_SIZES[0], CLASS_SIZES[-1], n_classes)[class_idx]


def project_points(sub: pd.DataFrame, base):
    if getattr(base, "gpd", None) is not None:
        gdf = base.gpd.GeoDataFrame(
            sub,
            geometry=base.gpd.points_from_xy(sub["lon"], sub["lat"]),
            crs="EPSG:4326",
        ).to_crs(base.MAP_TARGET_CRS)
        return gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    return sub["lon"].to_numpy(), sub["lat"].to_numpy()


def format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if abs(value) < 0.01:
        return f"{value:.3f}"
    if abs(value) < 1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def stepped_cmap(n_classes: int) -> ListedColormap:
    if n_classes <= len(NATURAL_BREAK_COLORS):
        idx = np.linspace(0, len(NATURAL_BREAK_COLORS) - 1, n_classes).round().astype(int)
        colors = [NATURAL_BREAK_COLORS[i] for i in idx]
    else:
        colors = plt.get_cmap("Spectral_r", n_classes)(range(n_classes))
    return ListedColormap(colors, name=f"pa_loss_natural_{n_classes}")


def reduce_tick_density(ax: plt.Axes, hide_y: bool = False) -> None:
    ax.tick_params(axis="both", labelsize=8.1, pad=1.5, length=2.6, width=0.55)
    if hide_y:
        ax.tick_params(axis="y", labelleft=False, labelright=False)
        ax.set_yticklabels([])
        for text in list(ax.texts):
            if text.get_text().strip().endswith("°N"):
                text.remove()
    # Base map helper adds geographic tick labels. Keep them small enough that
    # neighboring panels do not collide after tight export.
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(8.1)


def load_real_conus_state_boundaries(base):
    """Load project-local US state boundaries; avoids missing bokeh/Export_Output fallbacks."""
    if getattr(base, "gpd", None) is None or not LOCAL_STATE_BOUNDARY_SHP.exists():
        return None
    try:
        states = base.gpd.read_file(LOCAL_STATE_BOUNDARY_SHP)
        if "STATE_NAME" in states.columns:
            states = states[~states["STATE_NAME"].astype(str).isin(NON_CONUS_STATE_NAMES)].copy()
        # Keep the map to the CONUS extent used by the figure, plus a small buffer.
        states = states.cx[-126:-66, 23:51].copy()
        if states.empty:
            return None
        return states.to_crs(base.MAP_TARGET_CRS)
    except Exception:
        return None


def emphasize_state_boundaries(ax: plt.Axes, states_5070) -> None:
    """Redraw state boundaries so transparent markers do not visually erase them."""
    if states_5070 is None:
        return
    try:
        states_5070.boundary.plot(
            ax=ax,
            color=STATE_BOUNDARY_COLOR,
            linewidth=STATE_BOUNDARY_LINEWIDTH,
            alpha=STATE_BOUNDARY_ALPHA,
            zorder=8,
        )
    except Exception:
        return


def draw_external_natural_legend(
    legend_ax: plt.Axes,
    breaks: np.ndarray,
    cmap: ListedColormap,
    representative_classes: list[int],
) -> None:
    legend_ax.set_axis_off()
    n_classes = len(breaks) - 1

    x0, y0, bar_w, bar_h = 0.06, 0.50, 0.45, 0.18
    for i in range(n_classes):
        legend_ax.add_patch(
            Rectangle(
                (x0 + i * bar_w / n_classes, y0),
                bar_w / n_classes,
                bar_h,
                transform=legend_ax.transAxes,
                facecolor=cmap(i),
                edgecolor="white",
                linewidth=0.45,
                clip_on=False,
            )
        )
    legend_ax.add_patch(
        Rectangle(
            (x0, y0),
            bar_w,
            bar_h,
            transform=legend_ax.transAxes,
            facecolor="none",
            edgecolor="#666666",
            linewidth=0.5,
            clip_on=False,
        )
    )
    legend_ax.text(
        x0,
        y0 + 0.34,
        "Colour = 2025-2050 mean heatwave-attributable PA loss fraction (%)",
        transform=legend_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#303030",
    )
    for i, brk in enumerate(breaks):
        xpos = x0 + i * bar_w / n_classes
        legend_ax.text(
            xpos,
            y0 - 0.09,
            format_pct(float(brk)),
            transform=legend_ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.3,
            color="#555555",
        )

    sx0 = 0.62
    legend_ax.text(
        sx0,
        y0 + 0.34,
        "Point size = same PA loss fraction class (%)",
        transform=legend_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#303030",
    )
    for j, cls in enumerate(representative_classes):
        cls = int(np.clip(cls, 0, n_classes - 1))
        lo, hi = breaks[cls], breaks[cls + 1]
        x = sx0 + 0.095 * j
        legend_color = cmap(cls)
        legend_ax.scatter(
            [x],
            [y0 + 0.08],
            s=class_sizes(np.array([cls]), n_classes)[0],
            transform=legend_ax.transAxes,
            facecolor=(*legend_color[:3], POINT_FACE_ALPHA),
            edgecolor=(*legend_color[:3], POINT_EDGE_ALPHA),
            linewidth=POINT_EDGE_LINEWIDTH,
            clip_on=False,
        )
        legend_ax.text(
            x,
            y0 - 0.09,
            f"{format_pct(float(lo))}-{format_pct(float(hi))}%",
            transform=legend_ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.3,
            color="#555555",
        )


def draw_external_delta_legend(
    legend_ax: plt.Axes,
    cmap,
    norm,
    max_abs: float,
) -> None:
    legend_ax.set_axis_off()
    x0, y0, bar_w, bar_h = 0.09, 0.50, 0.42, 0.18
    gradient = np.linspace(-max_abs, max_abs, 256).reshape(1, -1)
    legend_ax.imshow(
        gradient,
        extent=(x0, x0 + bar_w, y0, y0 + bar_h),
        transform=legend_ax.transAxes,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        clip_on=False,
    )
    legend_ax.add_patch(
        Rectangle(
            (x0, y0),
            bar_w,
            bar_h,
            transform=legend_ax.transAxes,
            facecolor="none",
            edgecolor="#666666",
            linewidth=0.5,
            clip_on=False,
        )
    )
    legend_ax.text(
        x0,
        y0 + 0.34,
        "Colour = PA loss fraction difference from SSP2-4.5 (percentage points)",
        transform=legend_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#303030",
    )
    for frac, label in [(-1, f"-{format_pct(max_abs)}"), (0, "0"), (1, f"+{format_pct(max_abs)}")]:
        xpos = x0 + (frac + 1) * bar_w / 2
        legend_ax.text(
            xpos,
            y0 - 0.09,
            label,
            transform=legend_ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.3,
            color="#555555",
        )

    sx0 = 0.63
    legend_ax.text(
        sx0,
        y0 + 0.34,
        "Point size = absolute difference (percentage points)",
        transform=legend_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#303030",
    )
    for j, val in enumerate([max_abs * 0.25, max_abs * 0.5, max_abs]):
        x = sx0 + 0.10 * j
        delta_legend_color = "#F6C96D"
        legend_ax.scatter(
            [x],
            [y0 + 0.08],
            s=value_to_size(np.array([val]), 0.0, max_abs)[0],
            transform=legend_ax.transAxes,
            facecolor=delta_legend_color,
            edgecolor=delta_legend_color,
            linewidth=POINT_EDGE_LINEWIDTH,
            alpha=POINT_EDGE_ALPHA,
            clip_on=False,
        )
        legend_ax.text(
            x,
            y0 - 0.09,
            f"{format_pct(val)} pp",
            transform=legend_ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.3,
            color="#555555",
        )


def plot_absolute_triptych(df: pd.DataFrame, indicator: str, heatwave: str, out_dir: Path, base) -> dict:
    sub_all = df[
        df["indicator"].eq(indicator)
        & df["heatwave_type"].eq(heatwave)
        & df["scenario"].isin(SCENARIOS)
    ].dropna(subset=["lon", "lat", VALUE_COL]).copy()
    if sub_all.empty:
        return {}

    breaks = natural_breaks_for_values(sub_all[VALUE_COL])
    n_classes = len(breaks) - 1
    cmap = stepped_cmap(n_classes)
    norm = BoundaryNorm(breaks, cmap.N, clip=True)
    outline, states = base.load_projected_us_map()
    real_states = load_real_conus_state_boundaries(base)
    if real_states is not None:
        states = real_states

    fig = plt.figure(figsize=(15.9, 5.65), dpi=600, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.16])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    legend_ax = fig.add_subplot(gs[1, :])
    for i, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        sub = sub_all[sub_all["scenario"].eq(scenario)].copy()
        sub = sub.sort_values(VALUE_COL, ascending=True)
        base.draw_projected_us_background(ax, outline, states)
        emphasize_state_boundaries(ax, states)
        reduce_tick_density(ax, hide_y=i > 0)
        xs, ys = project_points(sub, base)
        vals = sub[VALUE_COL].astype(float)
        classes = classify_values(vals, breaks)
        sizes = class_sizes(classes, n_classes)
        point_facecolors = cmap(norm(vals.to_numpy(dtype=float)))
        point_edgecolors = point_facecolors.copy()
        point_facecolors[:, 3] = POINT_FACE_ALPHA
        point_edgecolors[:, 3] = POINT_EDGE_ALPHA
        ax.scatter(
            xs,
            ys,
            s=sizes,
            facecolors=point_facecolors,
            edgecolors=point_edgecolors,
            linewidths=POINT_EDGE_LINEWIDTH,
            zorder=20,
        )
        ax.set_title(
            f"{chr(97+i)}  {SCENARIO_LABELS[scenario]} {HEATWAVE_LABELS[heatwave]} {INDICATOR_LABELS[indicator]}, 2025-2050",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=5,
        )
        ax.text(
            0.02,
            0.97,
            "Natural-break colour and point-size classes pooled across SSPs",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            color="#555555",
        )

    reps = sorted(set([0, max(0, n_classes // 2), n_classes - 1]))
    draw_external_natural_legend(legend_ax, breaks, cmap, reps)

    fig.subplots_adjust(left=0.025, right=0.995, top=0.92, bottom=0.105, wspace=0.075, hspace=0.02)
    out_base = out_dir / f"fig6abc_{safe_slug(indicator, heatwave)}_absolute_natural_breaks"
    for ext, kwargs in [
        ("png", {"dpi": 600}),
        ("svg", {}),
        ("pdf", {}),
    ]:
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    return {
        "indicator": indicator,
        "heatwave_type": heatwave,
        "map_type": "absolute_natural_breaks",
        "vmin": float(breaks[0]),
        "vmax": float(breaks[-1]),
        "breaks": ";".join(format_pct(float(x)) for x in breaks),
        "point_min": POINT_MIN,
        "point_max": POINT_MAX,
        "output": str(out_base),
    }


def plot_delta_pair(df: pd.DataFrame, indicator: str, heatwave: str, out_dir: Path, base) -> dict:
    sub_all = df[
        df["indicator"].eq(indicator)
        & df["heatwave_type"].eq(heatwave)
        & df["scenario"].isin(SCENARIOS)
    ].dropna(subset=["lon", "lat", VALUE_COL]).copy()
    if sub_all.empty:
        return {}
    wide = sub_all.pivot_table(
        index=["city_standard", "lon", "lat"],
        columns="scenario",
        values=VALUE_COL,
        aggfunc="mean",
    ).reset_index()
    for scenario in SCENARIOS:
        if scenario not in wide:
            return {}
    deltas = []
    for scenario in ["ssp370", "ssp585"]:
        tmp = wide[["city_standard", "lon", "lat", "ssp245", scenario]].copy()
        tmp["scenario"] = scenario
        tmp["delta"] = tmp[scenario] - tmp["ssp245"]
        deltas.append(tmp)
    delta_df = pd.concat(deltas, ignore_index=True)
    max_abs = float(np.nanmax(np.abs(delta_df["delta"])))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 0.01
    cmap = LinearSegmentedColormap.from_list("delta_pa_loss", DELTA_CMAP_COLORS, N=256)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    outline, states = base.load_projected_us_map()
    real_states = load_real_conus_state_boundaries(base)
    if real_states is not None:
        states = real_states

    fig = plt.figure(figsize=(11.2, 5.55), dpi=600, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.18])
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    legend_ax = fig.add_subplot(gs[1, :])
    for i, (ax, scenario) in enumerate(zip(axes, ["ssp370", "ssp585"])):
        sub = delta_df[delta_df["scenario"].eq(scenario)].copy().sort_values("delta")
        base.draw_projected_us_background(ax, outline, states)
        emphasize_state_boundaries(ax, states)
        reduce_tick_density(ax, hide_y=i > 0)
        xs, ys = project_points(sub, base)
        vals = sub["delta"].astype(float)
        sizes = value_to_size(np.abs(vals), 0.0, max_abs)
        point_facecolors = cmap(norm(vals.to_numpy(dtype=float)))
        point_edgecolors = point_facecolors.copy()
        point_facecolors[:, 3] = POINT_FACE_ALPHA
        point_edgecolors[:, 3] = POINT_EDGE_ALPHA
        ax.scatter(
            xs,
            ys,
            s=sizes,
            facecolors=point_facecolors,
            edgecolors=point_edgecolors,
            linewidths=POINT_EDGE_LINEWIDTH,
            zorder=20,
        )
        ax.set_title(
            f"{chr(100+i)}  {SCENARIO_LABELS[scenario]} minus SSP2-4.5",
            loc="left",
            fontsize=11.6,
            fontweight="bold",
            pad=5,
        )
        ax.text(
            0.02,
            0.97,
            f"{HEATWAVE_LABELS[heatwave]} {INDICATOR_LABELS[indicator]} change in PA loss fraction",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            color="#555555",
        )

    draw_external_delta_legend(legend_ax, cmap, norm, max_abs)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.91, bottom=0.105, wspace=0.08, hspace=0.02)
    out_base = out_dir / f"fig6abc_{safe_slug(indicator, heatwave)}_delta_vs_ssp245"
    for ext, kwargs in [
        ("png", {"dpi": 600}),
        ("svg", {}),
        ("pdf", {}),
    ]:
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    return {
        "indicator": indicator,
        "heatwave_type": heatwave,
        "map_type": "delta_vs_ssp245",
        "vmin": -max_abs,
        "vmax": max_abs,
        "point_min": POINT_MIN,
        "point_max": POINT_MAX,
        "output": str(out_base),
    }


def write_readme(out_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        """# Fig. 6a-c natural-break map experiment

This directory is independent from the original `fig6a_map_national_2025-2050` outputs.

## What changed

- The three SSP maps use natural-break classes computed from pooled SSP2-4.5, SSP3-7.0 and SSP5-8.5 city values within each indicator and heatwave type.
- Colour and point size encode the same metric: 2025-2050 mean heatwave-attributable PA loss fraction (%).
- The colour and point-size legends are outside the map panels to avoid covering cities or coordinate labels.
- The compound day-night heatwave label is consistently shown as `Compound`.
- City markers use same-colour outlines with face alpha 0.50 and edge alpha 0.90; state boundaries are loaded from the project US_StateOrTerritory shapefile and redrawn with a slightly stronger grey stroke to remain visible after export and slide insertion.
- `delta_vs_ssp245` maps show SSP3-7.0 and SSP5-8.5 differences from SSP2-4.5.

## Why this helps

The previous shared continuous colour scale preserved exact numerical comparability, but most
cities were visually compressed into the low-value part of the scale. Natural breaks retain a
common three-SSP scale while increasing contrast in the high-value tail, making the Texas and
Southeast high-risk cities easier to distinguish. The underlying source values are unchanged.

## Source data

`data/city_map_2025_2050_plot_data.csv` is copied from the original Fig. 6 map directory.
The metric is `pa_loss_fraction_percent_all_days`, reported as a percentage.
""",
        encoding="utf-8",
    )


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(SOURCE_CSV)
    base = load_base_module()
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    data_dir = OUT_ROOT / "data"
    code_dir = OUT_ROOT / "code"
    fig_dir = OUT_ROOT / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_CSV, data_dir / SOURCE_CSV.name)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    archived_base = code_dir / "base_figure6a_d_future_projection_maps_and_lines.py"
    shutil.copy2(BASE_CODE, archived_base)
    archived_text = archived_base.read_text(encoding="utf-8", errors="ignore")
    legacy_label = "Com" + "pound"
    archived_base.write_text(archived_text.replace(legacy_label, "Composite"), encoding="utf-8")

    df = pd.read_csv(SOURCE_CSV)
    df = df[df["activity_type"].eq("all") & df["curve_scope"].eq("national")].copy()
    if not CLUSTER_CSV.exists():
        raise FileNotFoundError(CLUSTER_CSV)
    clusters = pd.read_csv(CLUSTER_CSV)
    city_col = next(
        (c for c in ("city_standard", "city", "City") if c in clusters.columns),
        None,
    )
    cluster_col = next(
        (c for c in ("dtw4lag12_cluster", "cluster", "Cluster") if c in clusters.columns),
        None,
    )
    if city_col is None or cluster_col is None:
        raise ValueError("Cluster map must contain city and DTW cluster columns.")
    clusters = clusters[[city_col, cluster_col]].rename(
        columns={city_col: "city_standard", cluster_col: "dtw_cluster"}
    )
    clusters["dtw_cluster"] = pd.to_numeric(clusters["dtw_cluster"], errors="coerce")
    clusters = clusters[clusters["dtw_cluster"].between(1, 4)].drop_duplicates("city_standard")
    df = df.merge(clusters, on="city_standard", how="inner", validate="many_to_one")
    if df["city_standard"].nunique() != 63:
        raise RuntimeError(
            "Expected 63 DTW-phenotyped cities for the manuscript maps; "
            f"found {df['city_standard'].nunique()}."
        )
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    metrics = []
    scale_rows = []
    for indicator in ["cehwi", "exceeded_quantity"]:
        for heatwave in ["composite"]:
            out_sub = fig_dir / safe_slug(indicator, heatwave)
            out_sub.mkdir(parents=True, exist_ok=True)
            abs_info = plot_absolute_triptych(df, indicator, heatwave, out_sub, base)
            delta_info = plot_delta_pair(df, indicator, heatwave, out_sub, base)
            for info in [abs_info, delta_info]:
                if info:
                    scale_rows.append(info)
    pd.DataFrame(scale_rows).to_csv(data_dir / "natural_break_scale_parameters.csv", index=False)
    df.to_csv(data_dir / "city_map_2025_2050_plot_data_filtered_national_all.csv", index=False)
    write_readme(OUT_ROOT)
    print(f"Natural-break maps written to: {OUT_ROOT}")
    print(f"Figures: {sum(1 for _ in fig_dir.rglob('*.png'))} png")
    print(f"Scale rows: {len(scale_rows)}")


if __name__ == "__main__":
    main()
