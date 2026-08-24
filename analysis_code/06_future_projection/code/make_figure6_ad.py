from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - plotting fallback only
    gpd = None
    LineString = None
    Polygon = None
    box = None
    unary_union = None


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
REPO_ROOT = MODULE_DIR.parents[1]
EXTERNAL_DATA_ROOT = Path(os.environ.get("HEATPA_DATA_ROOT", REPO_ROOT / "external_data"))
PACKAGE_ROOT = Path(os.environ.get(
    "FUTURE_PA_REPLOT_PACKAGE_ROOT",
    MODULE_DIR / "output" / "figure6_ad_generated",
)).resolve()
HEATWAVE_PA_ROOT = EXTERNAL_DATA_ROOT
ASRI_CI_RERUN_ROOT = Path(os.environ.get("FUTURE_PA_ASRI_CI_RERUN_ROOT", MODULE_DIR / "data"))

FUTURE_DIR = Path(os.environ.get(
    "FUTURE_PA_REPLOT_DIR",
    str(ASRI_CI_RERUN_ROOT / "primary_national"),
))
HIST_DIR = Path(os.environ.get(
    "FUTURE_PA_HISTORICAL_RISK_DIR",
    str(HEATWAVE_PA_ROOT / "historical_future_pa_projection_full_2010_2024_future_2025_2050_primary_national_warmseason_allactivity_20260512"),
))
STAGE2_ROOT = HEATWAVE_PA_ROOT / "R_output_DLNM_two_stage_2010_2025" / "lag_group_median_overall"
GRID10_PATH = HEATWAVE_PA_ROOT / "grid10_environment_predictors.csv"
REFERENCE_GIS_ROOT = Path(os.environ.get("FUTURE_PA_GIS_ROOT", EXTERNAL_DATA_ROOT / "gis"))
REFERENCE_FISHNET_SHP = REFERENCE_GIS_ROOT / "fishnet_reference.shp"
REFERENCE_GRID_RASTER = REFERENCE_GIS_ROOT / "USA_fish_reference_10km_label.tif"
REFERENCE_MAP_GDB = REFERENCE_GIS_ROOT / "MyProject1.gdb"
REFERENCE_MAP_GDB_LAYER = "Outline"
REFERENCE_STATES_SHP = REFERENCE_GIS_ROOT / "Export_Output.shp"
MAP_CACHE_ROOT = Path(os.environ.get(
    "FUTURE_PA_MAP_CACHE_ROOT",
    str(MODULE_DIR / "output" / "map_cache"),
))
VALIDATION_SOURCE_DIR = Path(os.environ.get(
    "FUTURE_PA_VALIDATION_SOURCE_DIR",
    str(MODULE_DIR / "data"),
))
VALIDATION_ALL_SCOPES = Path(os.environ.get(
    "FUTURE_PA_VALIDATION_ALL_SCOPES",
    str(HEATWAVE_PA_ROOT / "future_projection_validation_holdout_all_scopes_20260512.csv"),
))
VALIDATION_PRIMARY_RAW = Path(os.environ.get(
    "FUTURE_PA_VALIDATION_PRIMARY_RAW",
    str(VALIDATION_SOURCE_DIR / "validation_holdout_city_year_predictions.csv"),
))
VALIDATION_PRIMARY_SUMMARY = Path(os.environ.get(
    "FUTURE_PA_VALIDATION_PRIMARY_SUMMARY",
    str(VALIDATION_SOURCE_DIR / "validation_holdout_summary.csv"),
))
SCOPE_MANIFEST = HEATWAVE_PA_ROOT / "future_projection_scope_manifest_20260512.csv"
FUTURE_R_SOURCE = HEATWAVE_PA_ROOT / "future_pa_heat_risk_projection.R"
REFERENCE_NOTEBOOK = Path(os.environ.get(
    "FUTURE_PA_REFERENCE_NOTEBOOK",
    EXTERNAL_DATA_ROOT / "reference_future_projection.ipynb",
))


SCENARIOS = {
    "ssp245": {"label": "SSP2-4.5", "color": "#679dbf"},
    "ssp370": {"label": "SSP3-7.0", "color": "#da9c15"},
    "ssp585": {"label": "SSP5-8.5", "color": "#a84238"},
}
SCENARIO_ORDER = ["ssp245", "ssp370", "ssp585"]

HEATWAVE_LABELS = {
    "composite": "Compound",
    "day": "Daytime",
    "night": "Nighttime",
}
HEATWAVE_ORDER = ["composite", "day", "night"]

INDICATOR_LABELS = {
    "cehwi": "CEHWI",
    "exceeded_quantity": "Exceeded cumulative intensity",
}
INDICATOR_ORDER = ["cehwi", "exceeded_quantity"]

METRICS = [
    {
        "column": "pa_loss_fraction_percent_all_days",
        "low": "pa_loss_fraction_low_percent_all_days",
        "high": "pa_loss_fraction_high_percent_all_days",
        "label": "PA loss fraction (%)",
        "short": "PA loss",
        "stem": "pa_loss_fraction",
    },
    {
        "column": "annualized_asri_percent",
        "low": "annualized_asri_low_percent",
        "high": "annualized_asri_high_percent",
        "label": "ASRI (%)",
        "short": "ASRI",
        "stem": "asri",
    },
    {
        "column": "benefit_adjusted_asri_percent",
        "low": "benefit_adjusted_asri_low_percent",
        "high": "benefit_adjusted_asri_high_percent",
        "label": "Benefit-adjusted ASRI (%)",
        "short": "Net ASRI",
        "stem": "benefit_adjusted_asri",
    },
]

MAP_TARGET_CRS = "EPSG:5070"
MAP_FIG_WIDTH = 7.0
MAP_FIG_HEIGHT_RATIO = 0.68
MAP_FIG_HEIGHT = MAP_FIG_WIDTH * MAP_FIG_HEIGHT_RATIO
MAP_DPI = 600
MAP_FONT_FAMILY = "Times New Roman"
MAP_US_FILL = "#F1F1F1"
MAP_OUTLINE_COLOR = "#666666"
MAP_OUTLINE_WIDTH = 1.2
MAP_STATE_LINE_COLOR = "#8A8A8A"
MAP_STATE_LINE_WIDTH = 0.50
MAP_GRID_LONS = [-120, -110, -100, -90, -80]
MAP_GRID_LATS = [20, 30, 40, 50]
MAP_LAT_LABELS = [30, 40, 50]
MAP_GRID_EXTEND_LON_MIN = -132
MAP_GRID_EXTEND_LON_MAX = -58
MAP_GRID_EXTEND_LAT_MIN = 12
MAP_GRID_EXTEND_LAT_MAX = 60
MAP_GRID_COLOR = "#BFBFBF"
MAP_GRID_WIDTH = 0.45
MAP_GRID_ALPHA = 0.90
MAP_TICK_SIZE = 10
MAP_LABEL_SIZE = 12
MAP_CBAR_TICK_SIZE = 9
MAP_SCALE_TEXT_SIZE = 9
MAP_SPINE_WIDTH = 0.8
MAP_CBAR_WIDTH = "30%"
MAP_CBAR_HEIGHT = "4%"
MAP_CBAR_X_FRAC = 0.070
MAP_CBAR_Y_FRAC = 0.070
MAP_SCALE_LENGTH_M = 100_000
MAP_SCALE_HEIGHT_M = 70_000
MAP_SCALE_X_FRAC = 0.065
MAP_SCALE_Y_FRAC = 0.205
MAP_SCALE_TEXT_OFFSET_M = 18_000
MAP_PAD_X_RATIO = 0.015
MAP_PAD_Y_RATIO = 0.020
MAP_SHRINK_SCALE = 1.08
MAP_DATA_FRAME_ASPECT_RATIO = 0.70
MAP_POINT_SIZE_MIN = 26
MAP_POINT_SIZE_MAX = 215
MAP_POINT_ALPHA = 0.88
MAP_POINT_EDGE_COLOR = "black"
MAP_POINT_EDGE_WIDTH = 0.35
MAP_COLOR_LIST = [
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


def _connected_future_series(hist: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
    """Connect each future SSP line to the last observed historical point."""
    if hist.empty or fut.empty:
        return fut
    anchor = hist.sort_values("year").tail(1).copy()
    if anchor.empty:
        return fut
    anchor_value = pd.to_numeric(anchor["value"], errors="coerce").iloc[0]
    if not np.isfinite(anchor_value):
        return fut
    anchor = anchor.reindex(columns=fut.columns)
    anchor.loc[:, "scenario"] = fut["scenario"].iloc[0]
    anchor.loc[:, "scenario_label"] = fut["scenario_label"].iloc[0]
    anchor.loc[:, "period_source"] = "historical_anchor"
    anchor.loc[:, "low"] = np.nan
    anchor.loc[:, "high"] = np.nan
    return pd.concat([anchor, fut], ignore_index=True).sort_values("year")


def target_labels_for_validation(target: str) -> str:
    return {
        "pa_loss_fraction_percent_all_days": "PA loss",
        "annualized_asri_percent": "ASRI",
        "benefit_adjusted_asri_percent": "Net ASRI",
    }.get(target, str(target).replace("_", " "))


@dataclass(frozen=True)
class PackagePaths:
    data_raw: Path
    data_plot: Path
    data_model: Path
    code: Path
    figures_timeseries: Path
    figures_maps: Path
    figures_validation: Path
    docs: Path


def ensure_package() -> PackagePaths:
    module_root = MODULE_DIR.resolve()
    if PACKAGE_ROOT == module_root or PACKAGE_ROOT in module_root.parents:
        raise RuntimeError(
            "Refusing to overwrite the repository module. Set "
            "FUTURE_PA_REPLOT_PACKAGE_ROOT to a dedicated output directory."
        )
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    paths = PackagePaths(
        data_raw=PACKAGE_ROOT / "data" / "raw_source_csv",
        data_plot=PACKAGE_ROOT / "data" / "plot_ready_csv",
        data_model=PACKAGE_ROOT / "data" / "model_relationships",
        code=PACKAGE_ROOT / "code",
        figures_timeseries=PACKAGE_ROOT / "figures" / "timeseries_modewise",
        figures_maps=PACKAGE_ROOT / "figures" / "maps_city_change",
        figures_validation=PACKAGE_ROOT / "figures" / "validation",
        docs=PACKAGE_ROOT / "docs",
    )
    for p in paths.__dict__.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )


def copy_file(src: Path, dst_dir: Path, manifest_rows: list[dict], label: str) -> Path | None:
    if not src.exists():
        manifest_rows.append({"label": label, "source": str(src), "copied_to": "", "status": "missing"})
        return None
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    manifest_rows.append({"label": label, "source": str(src), "copied_to": str(dst), "status": "copied"})
    return dst


def write_patched_future_r_script(paths: PackagePaths, manifest_rows: list[dict]) -> Path | None:
    """Archive a complete R calculation script that exports annualized ASRI CI columns."""
    if not FUTURE_R_SOURCE.exists():
        manifest_rows.append(
            {
                "label": "patched future R script",
                "source": str(FUTURE_R_SOURCE),
                "copied_to": "",
                "status": "missing",
            }
        )
        return None

    text = FUTURE_R_SOURCE.read_text(encoding="utf-8", errors="replace")
    replacements = [
        (
            "      asri <- pmax(0, -log(rr))\n",
            "      asri <- pmax(0, -log(rr))\n"
            "      # ASRI is monotone decreasing in RR; lower CI uses RR_high and upper CI uses RR_low.\n"
            "      asri_low <- pmax(0, -log(pmax(rr_high, 1e-12)))\n"
            "      asri_high <- pmax(0, -log(pmax(rr_low, 1e-12)))\n",
        ),
        (
            "          asri_sum_days = sum(asri[idy], na.rm = TRUE),\n",
            "          asri_sum_days = sum(asri[idy], na.rm = TRUE),\n"
            "          asri_low_sum_days = sum(asri_low[idy], na.rm = TRUE),\n"
            "          asri_high_sum_days = sum(asri_high[idy], na.rm = TRUE),\n",
        ),
        (
            "    asri_sum_days,\n",
            "    asri_sum_days,\n"
            "    asri_low_sum_days,\n"
            "    asri_high_sum_days,\n",
        ),
        (
            "city_year$annualized_asri_percent <- city_year$asri_mean_all_days * 100\n",
            "city_year$annualized_asri_percent <- city_year$asri_mean_all_days * 100\n"
            "city_year$annualized_asri_low_percent <- city_year$asri_low_sum_days / city_year$n_days * 100\n"
            "city_year$annualized_asri_high_percent <- city_year$asri_high_sum_days / city_year$n_days * 100\n",
        ),
        (
            "    annualized_asri_percent,\n",
            "    annualized_asri_percent,\n"
            "    annualized_asri_low_percent,\n"
            "    annualized_asri_high_percent,\n",
        ),
        (
            "city_period$annualized_asri_percent <- city_period$asri_mean_all_days * 100\n",
            "city_period$annualized_asri_percent <- city_period$asri_mean_all_days * 100\n"
            "city_period$annualized_asri_low_percent <- city_period$asri_low_sum_days / city_period$n_days * 100\n"
            "city_period$annualized_asri_high_percent <- city_period$asri_high_sum_days / city_period$n_days * 100\n",
        ),
        (
            "  \"  Annualized ASRI percent = mean daily ASRI across warm-season days * 100.\",\n",
            "  \"  Annualized ASRI percent = mean daily ASRI across warm-season days * 100.\",\n"
            "  \"  Annualized ASRI CI = mean(max(0, -log(RR_high))) and mean(max(0, -log(RR_low))) across warm-season days * 100.\",\n",
        ),
        (
            "  \"  Benefit-adjusted ASRI CI uses lower = max(0, mean -log(RR_high)); upper = max(0, mean -log(RR_low)).\",\n",
            "  \"  Annualized ASRI CI uses lower = mean(max(0, -log(RR_high))); upper = mean(max(0, -log(RR_low))).\",\n"
            "  \"  Benefit-adjusted ASRI CI uses lower = max(0, mean -log(RR_high)); upper = max(0, mean -log(RR_low)).\",\n",
        ),
    ]
    patched = text
    for old, new in replacements:
        if old not in patched:
            manifest_rows.append(
                {
                    "label": "patched future R script replacement warning",
                    "source": str(FUTURE_R_SOURCE),
                    "copied_to": old.strip()[:120],
                    "status": "pattern_missing",
                }
            )
        patched = patched.replace(old, new)

    header = (
        "# Patched by make_future_projection_modewise_replot_20260618.py\n"
        "# Purpose: export annualized_asri_low_percent and annualized_asri_high_percent\n"
        "# from daily RR low/high before annual/period aggregation. The full projection\n"
        "# script computes ASRI intervals at the daily stage before annual aggregation.\n\n"
    )
    dst = paths.code / "future_pa_heat_risk_projection_with_annualized_asri_ci.R"
    dst.write_text(header + patched, encoding="utf-8")
    manifest_rows.append(
        {
            "label": "patched future R script",
            "source": str(FUTURE_R_SOURCE),
            "copied_to": str(dst),
            "status": "generated",
        }
    )
    return dst


def copy_source_material(paths: PackagePaths) -> pd.DataFrame:
    manifest_rows: list[dict] = []
    raw_files = [
        (FUTURE_DIR / "future_heat_pa_risk_national_year_trial.csv", "future national year"),
        (FUTURE_DIR / "future_heat_pa_risk_national_period_trial.csv", "future national period"),
        (FUTURE_DIR / "future_heat_pa_risk_city_year_trial.csv", "future city year"),
        (FUTURE_DIR / "future_heat_pa_risk_city_period_trial.csv", "future city period"),
        (FUTURE_DIR / "future_primary_pa_loss_fraction_summary.csv", "future primary summary"),
        (FUTURE_DIR / "future_exposure_clipping_diagnostics_city_period.csv", "future clipping diagnostics"),
        (FUTURE_DIR / "future_projection_run_notes.txt", "future run notes"),
        (FUTURE_DIR / "historical_pooled_curve_inventory.csv", "pooled curve inventory"),
        (HIST_DIR / "historical_heat_pa_risk_national_year.csv", "historical national year"),
        (HIST_DIR / "historical_heat_pa_risk_city_year.csv", "historical city year"),
        (HIST_DIR / "historical_heat_pa_risk_city_period.csv", "historical city period"),
        (HIST_DIR / "historical_future_indicator_similarity_summary.csv", "indicator similarity summary"),
        (VALIDATION_ALL_SCOPES, "hold-out validation all scopes"),
        (VALIDATION_PRIMARY_RAW, "primary national hold-out raw city-year predictions"),
        (VALIDATION_PRIMARY_SUMMARY, "primary national hold-out summary"),
        (SCOPE_MANIFEST, "projection scope manifest"),
        (GRID10_PATH, "grid10 city coordinates"),
        (REFERENCE_NOTEBOOK, "reference notebook style"),
    ]
    for src, label in raw_files:
        copy_file(src, paths.data_raw, manifest_rows, label)

    code_files = [
        HEATWAVE_PA_ROOT / "historical_future_pa_projection_nature.py",
        HEATWAVE_PA_ROOT / "historical_future_pa_projection_publication_v2.py",
        HEATWAVE_PA_ROOT / "plot_future_pa_projection_nature.py",
        HEATWAVE_PA_ROOT / "compare_future_projection_scopes_20260512.py",
        HEATWAVE_PA_ROOT / "run_future_projection_primary_controls_sensitivity_20260512.ps1",
        FUTURE_R_SOURCE,
        Path(__file__).with_name("future_pa_heat_risk_projection_with_annualized_asri_ci_new_cmip.R"),
        Path(__file__).with_name("future_pa_projection_fast_new_cmip_all_scopes_20260619.py"),
        Path(__file__).with_name("run_full_future_projection_new_cmip_all_scopes_20260619.ps1"),
        Path(__file__),
    ]
    for src in code_files:
        copy_file(src, paths.code, manifest_rows, "method/source code")

    write_patched_future_r_script(paths, manifest_rows)

    inventory = FUTURE_DIR / "historical_pooled_curve_inventory.csv"
    if inventory.exists():
        inv = pd.read_csv(inventory)
        inv.to_csv(paths.data_model / "historical_pooled_curve_inventory.csv", index=False)
        for curve_file in inv.get("curve_file", pd.Series(dtype=str)).dropna().astype(str).unique():
            src = Path(curve_file)
            if src.exists():
                model_key = src.parent.name
                dst_dir = paths.data_model / model_key
                dst_dir.mkdir(parents=True, exist_ok=True)
                copy_file(src, dst_dir, manifest_rows, f"pooled RR curve {model_key}")
                for extra in ["conditional_RR_curves.csv", "meta_model_summary.csv", "AF_meta_summary.csv"]:
                    copy_file(src.parent / extra, dst_dir, manifest_rows, f"model relationship {model_key} {extra}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(paths.data_raw.parent / "source_file_manifest.csv", index=False)
    return manifest


def load_projection_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    future_national = pd.read_csv(FUTURE_DIR / "future_heat_pa_risk_national_year_trial.csv")
    historical_national = pd.read_csv(HIST_DIR / "historical_heat_pa_risk_national_year.csv")
    future_city_period = pd.read_csv(FUTURE_DIR / "future_heat_pa_risk_city_period_trial.csv")
    future_city_year = pd.read_csv(FUTURE_DIR / "future_heat_pa_risk_city_year_trial.csv")
    return future_national, historical_national, future_city_period, future_city_year


def _metric_ci_columns(df: pd.DataFrame, metric: dict) -> tuple[pd.Series | None, pd.Series | None]:
    low_col = metric["low"]
    high_col = metric["high"]
    if low_col and high_col and low_col in df.columns and high_col in df.columns:
        return pd.to_numeric(df[low_col], errors="coerce"), pd.to_numeric(df[high_col], errors="coerce")
    return None, None


def build_plot_ready_timeseries(
    future: pd.DataFrame, historical: pd.DataFrame, paths: PackagePaths
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    hist = historical.copy()
    hist["scenario"] = "historical"
    hist["scenario_label"] = "Historical"
    hist["period_source"] = "historical"
    hist["denominator_id"] = hist.get("denominator_id", "historical_observed_reference")
    hist["denominator_label"] = hist.get(
        "denominator_label",
        "Observed historical reference table; denominator follows the historical source directory",
    )
    hist["denominator_months"] = hist.get("denominator_months", "source_defined")
    fut = future.copy()
    fut["scenario_label"] = fut["scenario"].map(lambda x: SCENARIOS.get(str(x), {}).get("label", str(x)))
    fut["period_source"] = "future"

    for source_df in [hist, fut]:
        for metric in METRICS:
            keep_cols = [
                "period_source",
                "scenario",
                "scenario_label",
                "year",
                "indicator",
                "heatwave_type",
                "activity_type",
                "model_key",
                metric["column"],
            ]
            keep_cols.extend([c for c in ["denominator_id", "denominator_label", "denominator_months"] if c in source_df.columns])
            optional = [metric["low"], metric["high"]]
            keep_cols.extend([c for c in optional if c and c in source_df.columns])
            sub = source_df[[c for c in keep_cols if c in source_df.columns]].copy()
            sub["metric"] = metric["stem"]
            sub["metric_label"] = metric["label"]
            sub["value"] = pd.to_numeric(sub[metric["column"]], errors="coerce")
            low, high = _metric_ci_columns(sub, metric)
            sub["low"] = low if low is not None else np.nan
            sub["high"] = high if high is not None else np.nan
            rows.append(
                sub[
                    [
                        "period_source",
                        "scenario",
                        "scenario_label",
                        "year",
                        "indicator",
                        "heatwave_type",
                        "activity_type",
                        "model_key",
                        *[c for c in ["denominator_id", "denominator_label", "denominator_months"] if c in sub.columns],
                        "metric",
                        "metric_label",
                        "value",
                        "low",
                        "high",
                    ]
                ]
            )

    out = pd.concat(rows, ignore_index=True)
    out = out[out["activity_type"].eq("all")].copy()
    out["heatwave_label"] = out["heatwave_type"].map(HEATWAVE_LABELS)
    out["indicator_label"] = out["indicator"].map(INDICATOR_LABELS)
    out.to_csv(paths.data_plot / "modewise_timeseries_plot_data.csv", index=False)
    return out


def build_city_map_data(future_city_period: pd.DataFrame, paths: PackagePaths) -> pd.DataFrame:
    coords = load_city_coordinates()
    city = future_city_period.copy()
    city = city[city["activity_type"].eq("all")].copy()
    city = city.merge(coords, on="city_standard", how="left")
    city.to_csv(paths.data_plot / "future_city_period_map_data.csv", index=False)
    return city


def load_city_coordinates() -> pd.DataFrame:
    if not GRID10_PATH.exists():
        return pd.DataFrame(columns=["city_standard", "lon", "lat"])
    grid = pd.read_csv(GRID10_PATH, usecols=["city_standard", "grid10_lon", "grid10_lat"])
    grid = grid.replace([np.inf, -np.inf], np.nan).dropna()
    coords = (
        grid.groupby("city_standard", dropna=False)[["grid10_lon", "grid10_lat"]]
        .mean()
        .reset_index()
        .rename(columns={"grid10_lon": "lon", "grid10_lat": "lat"})
    )
    coords.to_csv(PACKAGE_ROOT / "data" / "plot_ready_csv" / "city_centroid_coordinates_from_grid10.csv", index=False)
    return coords


def _split_nan_segments(lons: list[float], lats: list[float]) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for lon, lat in zip(lons, lats):
        if lon is None or lat is None or not np.isfinite(lon) or not np.isfinite(lat):
            if len(current) >= 3:
                segments.append(current)
            current = []
        else:
            current.append((float(lon), float(lat)))
    if len(current) >= 3:
        segments.append(current)
    return segments


def _bokeh_states_to_gdf():
    if gpd is None or Polygon is None:
        return None
    try:
        from bokeh.sampledata.us_states import data as states
    except Exception:
        return None
    rows = []
    for code, state in states.items():
        if code in {"AK", "HI", "PR"}:
            continue
        for seg in _split_nan_segments(state["lons"], state["lats"]):
            try:
                poly = Polygon(seg)
                if not poly.is_empty and poly.area > 0:
                    rows.append({"state": code, "geometry": poly})
            except Exception:
                continue
    if not rows:
        return None
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _read_reference_fishnet_points():
    if gpd is None or not REFERENCE_FISHNET_SHP.exists():
        return None
    ignore_fields = [
        "CMaxExceed",
        "CMeanExcee",
        "CSumExceed",
        "CMaxCEHWI",
        "CMeanCEHWI",
        "CSumDayYr",
        "CSumEvt",
        "CMeanDur",
        "CMaxDur",
    ]
    try:
        pts = gpd.read_file(REFERENCE_FISHNET_SHP, ignore_fields=ignore_fields)
    except Exception:
        pts = gpd.read_file(REFERENCE_FISHNET_SHP)
        pts = pts[["geometry"]].copy()
    if pts.crs is None:
        # The supplied .prj is Web Mercator; set it explicitly if Fiona drops CRS.
        pts = pts.set_crs("EPSG:3857", allow_override=True)
    return pts


def _load_local_fishnet_outline_5070():
    """Build the CONUS outline from the supplied 10 km fishnet-nearest GIS data."""
    if gpd is None or box is None or unary_union is None:
        return None
    MAP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = MAP_CACHE_ROOT / "reference_fishnet_outline_epsg5070.geojson"
    if cache_path.exists():
        try:
            cached = gpd.read_file(cache_path)
            if cached.crs is None:
                cached = cached.set_crs(MAP_TARGET_CRS, allow_override=True)
            return cached.to_crs(MAP_TARGET_CRS)
        except Exception:
            pass

    pts = _read_reference_fishnet_points()
    if pts is None or pts.empty:
        return None

    xs = pts.geometry.x.to_numpy(dtype=float)
    ys = pts.geometry.y.to_numpy(dtype=float)
    # The GIS reference raster is a 10 km grid; points are cell centres.
    half_cell = 5_000.0
    cells = [box(x - half_cell, y - half_cell, x + half_cell, y + half_cell) for x, y in zip(xs, ys)]
    outline_geom = unary_union(cells)
    outline = gpd.GeoDataFrame(geometry=[outline_geom], crs=pts.crs).to_crs(MAP_TARGET_CRS)
    outline["geometry"] = outline.geometry.simplify(2_500, preserve_topology=True)
    try:
        outline.to_file(cache_path, driver="GeoJSON")
    except Exception:
        pass
    return outline


def load_projected_us_map() -> tuple[object | None, object | None]:
    """Load the reference CONUS map style and project it to EPSG:5070."""
    if gpd is None:
        return None, None
    outline = None
    states = None
    if REFERENCE_MAP_GDB.exists() and REFERENCE_STATES_SHP.exists():
        try:
            outline = gpd.read_file(REFERENCE_MAP_GDB, layer=REFERENCE_MAP_GDB_LAYER).to_crs(MAP_TARGET_CRS)
            states = gpd.read_file(REFERENCE_STATES_SHP).to_crs(MAP_TARGET_CRS)
        except Exception:
            outline = None
            states = None
    if outline is None:
        outline = _load_local_fishnet_outline_5070()
    if states is None:
        states_wgs84 = _bokeh_states_to_gdf()
        if states_wgs84 is not None:
            states = states_wgs84.to_crs(MAP_TARGET_CRS)
            if outline is None and unary_union is not None:
                outline = gpd.GeoDataFrame(geometry=[unary_union(states.geometry)], crs=MAP_TARGET_CRS)
    return outline, states


def make_lonlat_line(lon: float | None = None, lat: float | None = None, n: int = 1200):
    if gpd is None or LineString is None:
        return None
    if lon is not None:
        lats = np.linspace(MAP_GRID_EXTEND_LAT_MIN, MAP_GRID_EXTEND_LAT_MAX, n)
        coords = [(lon, y) for y in lats]
    elif lat is not None:
        lons = np.linspace(MAP_GRID_EXTEND_LON_MIN, MAP_GRID_EXTEND_LON_MAX, n)
        coords = [(x, lat) for x in lons]
    else:
        raise ValueError("Either lon or lat must be specified.")
    return gpd.GeoDataFrame(geometry=[LineString(coords)], crs="EPSG:4326")


def _interp_x_at_y(x: np.ndarray, y: np.ndarray, y0: float, xlim: tuple[float, float]) -> float | None:
    xs = []
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        x1, x2 = x[i], x[i + 1]
        if (y1 - y0) * (y2 - y0) <= 0 and y1 != y2:
            t = (y0 - y1) / (y2 - y1)
            xi = x1 + t * (x2 - x1)
            if xlim[0] <= xi <= xlim[1]:
                xs.append(float(xi))
    if not xs:
        return None
    center = (xlim[0] + xlim[1]) / 2
    return min(xs, key=lambda v: abs(v - center))


def _interp_y_at_x(x: np.ndarray, y: np.ndarray, x0: float, ylim: tuple[float, float]) -> float | None:
    ys = []
    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]
        if (x1 - x0) * (x2 - x0) <= 0 and x1 != x2:
            t = (x0 - x1) / (x2 - x1)
            yi = y1 + t * (y2 - y1)
            if ylim[0] <= yi <= ylim[1]:
                ys.append(float(yi))
    if not ys:
        return None
    center = (ylim[0] + ylim[1]) / 2
    return min(ys, key=lambda v: abs(v - center))


def add_projected_graticules(ax: plt.Axes) -> None:
    if gpd is None:
        return
    ax.set_xticks([])
    ax.set_yticks([])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    lon_lines = {}
    lat_lines = {}
    for lon in MAP_GRID_LONS:
        line = make_lonlat_line(lon=lon)
        if line is None:
            continue
        geom = line.to_crs(MAP_TARGET_CRS).geometry.iloc[0]
        x, y = geom.xy
        x = np.asarray(x)
        y = np.asarray(y)
        lon_lines[lon] = (x, y)
        ax.plot(x, y, color=MAP_GRID_COLOR, linewidth=MAP_GRID_WIDTH, alpha=MAP_GRID_ALPHA, zorder=2.6, clip_on=True)
    for lat in MAP_GRID_LATS:
        line = make_lonlat_line(lat=lat)
        if line is None:
            continue
        geom = line.to_crs(MAP_TARGET_CRS).geometry.iloc[0]
        x, y = geom.xy
        x = np.asarray(x)
        y = np.asarray(y)
        lat_lines[lat] = (x, y)
        ax.plot(x, y, color=MAP_GRID_COLOR, linewidth=MAP_GRID_WIDTH, alpha=MAP_GRID_ALPHA, zorder=2.6, clip_on=True)
    y_axis = ylim[0]
    bottom_tick_len = y_range * 0.018
    bottom_label_offset = y_range * 0.025
    ax.plot([xlim[0], xlim[1]], [y_axis, y_axis], color="black", linewidth=0.8, zorder=12, clip_on=False)
    for lon, (x_line, y_line) in lon_lines.items():
        x_label = _interp_x_at_y(x_line, y_line, y_axis, xlim)
        if x_label is None:
            continue
        ax.plot([x_label, x_label], [y_axis, y_axis - bottom_tick_len], color="black", linewidth=0.8, zorder=12, clip_on=False)
        ax.text(x_label, y_axis - bottom_label_offset, f"{abs(lon):.0f}°W", ha="center", va="top", fontsize=MAP_TICK_SIZE, clip_on=False, zorder=12)
    x_axis = xlim[0]
    left_tick_len = x_range * 0.018
    left_label_offset = x_range * 0.030
    ax.plot([x_axis, x_axis], [ylim[0], ylim[1]], color="black", linewidth=0.8, zorder=12, clip_on=False)
    for lat in MAP_LAT_LABELS:
        if lat not in lat_lines:
            continue
        x_line, y_line = lat_lines[lat]
        y_label = _interp_y_at_x(x_line, y_line, x_axis, ylim)
        if y_label is None:
            continue
        ax.plot([x_axis, x_axis - left_tick_len], [y_label, y_label], color="black", linewidth=0.8, zorder=12, clip_on=False)
        ax.text(x_axis - left_label_offset, y_label, f"{lat:.0f}°N", ha="right", va="center", fontsize=MAP_TICK_SIZE, clip_on=False, zorder=12)


def add_projected_scale_bar(ax: plt.Axes) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    x0 = xlim[0] + MAP_SCALE_X_FRAC * x_range
    y0 = ylim[0] + MAP_SCALE_Y_FRAC * y_range
    rect = Rectangle((x0, y0), MAP_SCALE_LENGTH_M, MAP_SCALE_HEIGHT_M, facecolor="black", edgecolor="black", linewidth=0, zorder=13)
    ax.add_patch(rect)
    ax.text(x0 + MAP_SCALE_LENGTH_M / 2, y0 - MAP_SCALE_TEXT_OFFSET_M, "100 km", ha="center", va="top", fontsize=MAP_SCALE_TEXT_SIZE, color="black", zorder=13)


def get_projected_display_limits(outline_5070) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, ymin, xmax, ymax = outline_5070.total_bounds
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    raw_width = xmax - xmin
    raw_height = ymax - ymin
    width = raw_width * (1 + 2 * MAP_PAD_X_RATIO)
    height = raw_height * (1 + 2 * MAP_PAD_Y_RATIO)
    target_width = width
    target_height = max(target_width * MAP_DATA_FRAME_ASPECT_RATIO, height)
    target_width *= MAP_SHRINK_SCALE
    target_height *= MAP_SHRINK_SCALE
    xlim = (x_center - target_width / 2, x_center + target_width / 2)
    ylim = (y_center - target_height / 2, y_center + target_height / 2)
    return xlim, ylim


def draw_projected_us_background(ax: plt.Axes, outline_5070, states_5070) -> None:
    if outline_5070 is None:
        draw_us_background_lonlat_fallback(ax)
        return
    xlim, ylim = get_projected_display_limits(outline_5070)
    ax.set_facecolor("white")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    outline_5070.plot(ax=ax, facecolor=MAP_US_FILL, edgecolor="none", zorder=0.5)
    add_projected_graticules(ax)
    if states_5070 is not None:
        states_5070.boundary.plot(ax=ax, color=MAP_STATE_LINE_COLOR, linewidth=MAP_STATE_LINE_WIDTH, zorder=3)
    outline_5070.boundary.plot(ax=ax, color=MAP_OUTLINE_COLOR, linewidth=MAP_OUTLINE_WIDTH, zorder=4)
    add_projected_scale_bar(ax)
    for spine in ax.spines.values():
        spine.set_linewidth(MAP_SPINE_WIDTH)
        spine.set_color("black")


def draw_us_background_lonlat_fallback(ax: plt.Axes) -> None:
    try:
        from bokeh.sampledata.us_states import data as states
        for code, state in states.items():
            if code in {"AK", "HI", "PR"}:
                continue
            ax.fill(state["lons"], state["lats"], facecolor=MAP_US_FILL, edgecolor="#C9C9C9", linewidth=0.35, zorder=0)
    except Exception:
        outline_lon = [-124.8, -123, -121, -118, -114, -111, -106, -102, -97, -91, -86, -80, -75, -70, -67, -70, -76, -82, -89, -95, -102, -110, -118, -124.8]
        outline_lat = [32, 38, 45, 49, 49, 45, 45, 49, 49, 47, 46, 44, 41, 43, 45, 37, 35, 30, 29, 28, 29, 31, 34, 32]
        ax.fill(outline_lon, outline_lat, facecolor=MAP_US_FILL, edgecolor="#C9C9C9", linewidth=0.6, zorder=0)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_reference_style_colorbar(ax: plt.Axes, mappable, vmin: float, vmax: float) -> None:
    cax = inset_axes(
        ax,
        width=MAP_CBAR_WIDTH,
        height=MAP_CBAR_HEIGHT,
        loc="lower left",
        bbox_to_anchor=(MAP_CBAR_X_FRAC, MAP_CBAR_Y_FRAC, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    ticks = [vmin, (vmin + vmax) / 2, vmax]
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels([f"{x:.2f}" for x in ticks], fontsize=MAP_CBAR_TICK_SIZE)
    cb.outline.set_linewidth(0.6)


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white", format="svg")
    plt.close(fig)


def plot_modewise_timeseries(plot_df: pd.DataFrame, paths: PackagePaths) -> None:
    for heatwave_type in HEATWAVE_ORDER:
        fig, axes = plt.subplots(
            nrows=len(METRICS),
            ncols=len(INDICATOR_ORDER),
            figsize=(13.2, 10.6),
            dpi=220,
            sharex=True,
        )
        for r, metric in enumerate(METRICS):
            for c, indicator in enumerate(INDICATOR_ORDER):
                ax = axes[r, c]
                sub = plot_df[
                    plot_df["heatwave_type"].eq(heatwave_type)
                    & plot_df["indicator"].eq(indicator)
                    & plot_df["metric"].eq(metric["stem"])
                ].copy()
                hist = sub[sub["period_source"].eq("historical")].sort_values("year")
                if not hist.empty:
                    ax.plot(
                        hist["year"].to_numpy(float),
                        hist["value"].to_numpy(float),
                        color="#404040",
                        linewidth=2.15,
                        marker="o",
                        markersize=3.8,
                        markerfacecolor="#404040",
                        markeredgecolor="white",
                        markeredgewidth=0.45,
                        label="Historical" if r == 0 and c == 0 else None,
                        zorder=5,
                    )

                for scenario in SCENARIO_ORDER:
                    fut = sub[sub["scenario"].eq(scenario)].sort_values("year")
                    if fut.empty:
                        continue
                    line_fut = _connected_future_series(hist, fut)
                    colour = SCENARIOS[scenario]["color"]
                    label = SCENARIOS[scenario]["label"] if r == 0 and c == 0 else None
                    if fut["low"].notna().any() and fut["high"].notna().any():
                        ax.fill_between(
                            fut["year"].to_numpy(float),
                            fut["low"].to_numpy(float),
                            fut["high"].to_numpy(float),
                            color=colour,
                            alpha=0.15,
                            linewidth=0,
                            zorder=1,
                        )
                    ax.plot(
                        line_fut["year"].to_numpy(float),
                        line_fut["value"].to_numpy(float),
                        color=colour,
                        linewidth=2.25,
                        marker="o",
                        markersize=3.8,
                        markerfacecolor=colour,
                        markeredgecolor="white",
                        markeredgewidth=0.45,
                        label=label,
                        zorder=4,
                    )

                ax.axvline(2024, color="0.55", linestyle="--", linewidth=1.0, zorder=2)
                ax.grid(True, color="0.82", linestyle="--", linewidth=0.55, alpha=0.65)
                ax.set_axisbelow(True)
                ax.tick_params(axis="both", labelsize=9)
                if r == 0:
                    ax.set_title(INDICATOR_LABELS[indicator], fontsize=12, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(metric["label"], fontsize=10)
                if r == len(METRICS) - 1:
                    ax.set_xlabel("Year", fontsize=10)

                values = sub[["value", "low", "high"]].to_numpy(dtype=float).ravel()
                values = values[np.isfinite(values)]
                if values.size:
                    ymin = float(np.nanmin(values))
                    ymax = float(np.nanmax(values))
                    if math.isclose(ymin, ymax):
                        pad = max(abs(ymax) * 0.2, 0.05)
                    else:
                        pad = (ymax - ymin) * 0.13
                    ax.set_ylim(ymin - pad, ymax + pad)

        title = f"{HEATWAVE_LABELS[heatwave_type]} heatwaves: historical and projected PA-risk trajectories"
        fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=15, fontweight="bold")
        fig.text(
            0.02,
            0.952,
            "Historical line is the observed 2010-2024 trajectory; future lines show SSP ensemble means with available response-curve CI.",
            ha="left",
            fontsize=9.5,
            color="#555555",
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.933),
            ncol=4,
            frameon=True,
            framealpha=0.82,
            fontsize=10,
            handlelength=2.2,
        )
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.065, top=0.88, hspace=0.26, wspace=0.18)
        save_fig(fig, paths.figures_timeseries / f"modewise_projection_{heatwave_type}")


def plot_metric_single_timeseries(plot_df: pd.DataFrame, paths: PackagePaths) -> None:
    """Also export cleaner single-metric figures for manuscript assembly."""
    for metric in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), dpi=220, sharey=False)
        for ax, heatwave_type in zip(axes, HEATWAVE_ORDER):
            sub = plot_df[
                plot_df["heatwave_type"].eq(heatwave_type)
                & plot_df["indicator"].eq("cehwi")
                & plot_df["metric"].eq(metric["stem"])
            ].copy()
            hist = sub[sub["period_source"].eq("historical")].sort_values("year")
            if not hist.empty:
                ax.plot(hist["year"].to_numpy(float), hist["value"].to_numpy(float), color="#404040", linewidth=2.1, marker="o", markersize=3.4, label="Historical")
            for scenario in SCENARIO_ORDER:
                fut = sub[sub["scenario"].eq(scenario)].sort_values("year")
                if fut.empty:
                    continue
                line_fut = _connected_future_series(hist, fut)
                colour = SCENARIOS[scenario]["color"]
                if fut["low"].notna().any() and fut["high"].notna().any():
                    ax.fill_between(fut["year"].to_numpy(float), fut["low"].to_numpy(float), fut["high"].to_numpy(float), color=colour, alpha=0.15, linewidth=0)
                ax.plot(line_fut["year"].to_numpy(float), line_fut["value"].to_numpy(float), color=colour, linewidth=2.15, marker="o", markersize=3.4, label=SCENARIOS[scenario]["label"])
            ax.axvline(2024, color="0.55", linestyle="--", linewidth=1)
            ax.set_title(HEATWAVE_LABELS[heatwave_type], fontsize=11, fontweight="bold")
            ax.grid(True, color="0.82", linestyle="--", linewidth=0.55, alpha=0.65)
            ax.set_xlabel("Year", fontsize=9.5)
            ax.tick_params(labelsize=8.8)
            vals = sub[["value", "low", "high"]].to_numpy(dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ymin, ymax = float(vals.min()), float(vals.max())
                pad = (ymax - ymin) * 0.13 if ymax > ymin else max(abs(ymax) * 0.2, 0.05)
                ax.set_ylim(ymin - pad, ymax + pad)
        axes[0].set_ylabel(metric["label"], fontsize=10)
        fig.suptitle(f"CEHWI-based {metric['short']} by heatwave mode", x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.96), ncol=4, frameon=False, fontsize=9.5)
        fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.16, wspace=0.22)
        save_fig(fig, paths.figures_timeseries / f"single_metric_cehwi_{metric['stem']}")


def plot_city_maps(city_df: pd.DataFrame, paths: PackagePaths) -> None:
    city_df = city_df.dropna(subset=["lon", "lat"]).copy()
    city_df = city_df[city_df["activity_type"].eq("all")].copy()
    outline_5070, states_5070 = load_projected_us_map()
    map_cmap = mcolors.LinearSegmentedColormap.from_list("future_pa_reference_blue_yellow_red", MAP_COLOR_LIST, N=256)
    periods = ["2025-2030", "2031-2040", "2041-2050"]
    for indicator in INDICATOR_ORDER:
        for heatwave_type in HEATWAVE_ORDER:
            for scenario in SCENARIO_ORDER:
                sub = city_df[
                    city_df["indicator"].eq(indicator)
                    & city_df["heatwave_type"].eq(heatwave_type)
                    & city_df["scenario"].eq(scenario)
                    & city_df["period"].isin(periods)
                ].copy()
                if sub.empty:
                    continue
                vals_all = pd.to_numeric(sub["pa_loss_fraction_percent_all_days"], errors="coerce")
                vmin = float(np.nanmin(vals_all))
                vmax = float(np.nanmax(vals_all))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
                    vmin, vmax = float(np.nanmin(vals_all)), float(np.nanmax(vals_all))
                if math.isclose(vmin, vmax):
                    vmax = vmin + 1.0
                norm = Normalize(vmin=vmin, vmax=vmax)
                with plt.rc_context(
                    {
                        "font.family": MAP_FONT_FAMILY,
                        "font.serif": [MAP_FONT_FAMILY],
                        "font.sans-serif": [MAP_FONT_FAMILY],
                        "svg.fonttype": "none",
                        "pdf.fonttype": 42,
                        "ps.fonttype": 42,
                    }
                ):
                    fig, axes = plt.subplots(
                        1,
                        3,
                        figsize=(MAP_FIG_WIDTH * 3.05, MAP_FIG_HEIGHT + 0.62),
                        dpi=MAP_DPI,
                        facecolor="white",
                    )
                    for ax, period in zip(axes, periods):
                        draw_projected_us_background(ax, outline_5070, states_5070)
                        psub = sub[sub["period"].eq(period)].copy()
                        vals = pd.to_numeric(psub["pa_loss_fraction_percent_all_days"], errors="coerce")
                        scaled = np.clip((vals - vmin) / max(vmax - vmin, 1e-9), 0, 1)
                        sizes = MAP_POINT_SIZE_MIN + (MAP_POINT_SIZE_MAX - MAP_POINT_SIZE_MIN) * scaled
                        psub = psub.assign(_value=vals, _size=sizes).sort_values("_value", ascending=True)
                        if gpd is not None and outline_5070 is not None:
                            point_gdf = gpd.GeoDataFrame(
                                psub,
                                geometry=gpd.points_from_xy(psub["lon"], psub["lat"]),
                                crs="EPSG:4326",
                            ).to_crs(MAP_TARGET_CRS)
                            xs = point_gdf.geometry.x
                            ys = point_gdf.geometry.y
                        else:
                            xs = psub["lon"]
                            ys = psub["lat"]
                        sc = ax.scatter(
                            xs,
                            ys,
                            c=psub["_value"],
                            s=psub["_size"],
                            cmap=map_cmap,
                            norm=norm,
                            alpha=MAP_POINT_ALPHA,
                            edgecolors=MAP_POINT_EDGE_COLOR,
                            linewidths=MAP_POINT_EDGE_WIDTH,
                            zorder=8,
                        )
                        add_reference_style_colorbar(ax, sc, vmin, vmax)
                        ax.text(
                            0.5,
                            -0.100,
                            period,
                            transform=ax.transAxes,
                            ha="center",
                            va="top",
                            fontsize=MAP_LABEL_SIZE,
                        )
                    fig.suptitle(
                        f"{SCENARIOS[scenario]['label']} {HEATWAVE_LABELS[heatwave_type]} {INDICATOR_LABELS[indicator]} city projections",
                        x=0.018,
                        y=0.985,
                        ha="left",
                        fontsize=15,
                        fontweight="bold",
                    )
                    fig.text(
                        0.018,
                        0.930,
                        "Point colour and size show period-mean future heatwave-attributable PA loss fraction (%).",
                        ha="left",
                        fontsize=9.5,
                        color="#4A4A4A",
                    )
                    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.18, top=0.875, wspace=0.065)
                    save_fig(fig, paths.figures_maps / f"city_map_{indicator}_{heatwave_type}_{scenario}")


def _r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ok = np.isfinite(observed) & np.isfinite(predicted)
    if ok.sum() < 3:
        return np.nan
    observed = observed[ok]
    predicted = predicted[ok]
    denom = float(np.sum((observed - observed.mean()) ** 2))
    if denom <= 1e-14:
        return np.nan
    return 1.0 - float(np.sum((observed - predicted) ** 2)) / denom


def _bootstrap_r2_ci(group: pd.DataFrame, n_boot: int = 1000, seed: int = 20260618) -> tuple[float, float, float]:
    obs = pd.to_numeric(group["observed"], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(group["predicted"], errors="coerce").to_numpy(float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    point = _r2_score(obs, pred)
    if len(obs) < 8:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n = len(obs)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = _r2_score(obs[idx], pred[idx])
    draws = draws[np.isfinite(draws)]
    if len(draws) < max(25, n_boot // 10):
        return point, np.nan, np.nan
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def _validation_raw_summary(paths: PackagePaths, targets: list[str]) -> pd.DataFrame:
    if not VALIDATION_PRIMARY_RAW.exists():
        return pd.DataFrame()
    raw = pd.read_csv(VALIDATION_PRIMARY_RAW)
    raw = raw[raw["target"].isin(targets)].copy()
    raw.to_csv(paths.data_plot / "validation_primary_national_raw_city_year_predictions.csv", index=False)
    rows: list[dict] = []
    group_cols = ["model_key", "indicator", "heatwave_type", "activity_type", "target"]
    for group_key, group in raw.groupby(group_cols, dropna=False):
        model_key, indicator, heatwave_type, activity_type, target = group_key
        point, lo, hi = _bootstrap_r2_ci(group)
        obs = pd.to_numeric(group["observed"], errors="coerce")
        pred = pd.to_numeric(group["predicted"], errors="coerce")
        obs_sd = float(obs.std(skipna=True))
        status = "display"
        if not np.isfinite(point):
            status = "undefined_low_observed_variance"
        elif point < 0:
            status = "negative_r2_low_skill_or_low_variance"
        elif point > 1:
            status = "out_of_range"
        rows.append(
            {
                "model_key": model_key,
                "indicator": indicator,
                "mode": heatwave_type,
                "activity_type": activity_type,
                "target": target,
                "n": int(len(group)),
                "observed_mean": float(obs.mean(skipna=True)),
                "observed_sd": obs_sd,
                "predicted_mean": float(pred.mean(skipna=True)),
                "r2": point,
                "r2_low": lo,
                "r2_high": hi,
                "display_status": status,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.data_plot / "validation_primary_national_r2_bootstrap_ci.csv", index=False)
    return out


def _load_primary_validation_raw(paths: PackagePaths, targets: list[str]) -> pd.DataFrame:
    """Load the raw city-year hold-out predictions used for scatter and temporal checks."""
    if not VALIDATION_PRIMARY_RAW.exists():
        return pd.DataFrame()
    raw = pd.read_csv(VALIDATION_PRIMARY_RAW)
    raw = raw[raw["target"].isin(targets)].copy()
    raw["indicator"] = raw["indicator"].astype(str)
    raw["heatwave_type"] = raw["heatwave_type"].astype(str)
    raw["target"] = raw["target"].astype(str)
    raw["observed"] = pd.to_numeric(raw["observed"], errors="coerce")
    raw["predicted"] = pd.to_numeric(raw["predicted"], errors="coerce")
    raw = raw[np.isfinite(raw["observed"]) & np.isfinite(raw["predicted"])].copy()
    raw.to_csv(paths.data_plot / "validation_primary_national_raw_city_year_predictions.csv", index=False)
    return raw


def _plot_validation_observed_vs_predicted(raw: pd.DataFrame, paths: PackagePaths, targets: list[str]) -> None:
    """Nature-style 1:1 validation scatter panels from raw hold-out city-year predictions."""
    if raw.empty:
        return
    mode_colours = {"composite": "#a84238", "day": "#da9c15", "night": "#679dbf"}
    for target in targets:
        sub_target = raw[raw["target"].eq(target)].copy()
        if sub_target.empty:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.0), dpi=220)
        for r, indicator in enumerate(INDICATOR_ORDER):
            for c, mode in enumerate(HEATWAVE_ORDER):
                ax = axes[r, c]
                sub = sub_target[sub_target["indicator"].eq(indicator) & sub_target["heatwave_type"].eq(mode)].copy()
                colour = mode_colours.get(mode, "#777777")
                if sub.empty:
                    ax.axis("off")
                    continue
                obs = sub["observed"].to_numpy(float)
                pred = sub["predicted"].to_numpy(float)
                both = np.r_[obs, pred]
                finite = np.isfinite(both)
                if finite.any():
                    lo = float(np.nanpercentile(both[finite], 1))
                    hi = float(np.nanpercentile(both[finite], 99))
                    if math.isclose(lo, hi):
                        pad = max(abs(hi) * 0.2, 0.02)
                    else:
                        pad = (hi - lo) * 0.10
                    lo -= pad
                    hi += pad
                else:
                    lo, hi = -0.05, 1.0
                ax.scatter(obs, pred, s=13, color=colour, alpha=0.36, linewidths=0)
                ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=0.9)
                point_r2 = _r2_score(obs, pred)
                pearson = np.corrcoef(obs, pred)[0, 1] if len(obs) > 2 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan
                ax.text(
                    0.04,
                    0.96,
                    f"$R^2$={point_r2:.2f}\nr={pearson:.2f}\nn={len(sub)}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.4,
                    color="#333333",
                )
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.grid(True, color="0.86", linestyle="--", linewidth=0.45, alpha=0.7)
                ax.tick_params(labelsize=7.3)
                if r == 0:
                    ax.set_title(HEATWAVE_LABELS[mode], fontsize=9.2, fontweight="bold", color=colour)
                if c == 0:
                    ax.set_ylabel(f"{INDICATOR_LABELS[indicator]}\nPredicted", fontsize=8.8)
                if r == 1:
                    ax.set_xlabel("Observed", fontsize=8.8)
        fig.suptitle(
            f"Primary national hold-out: observed versus predicted {target_labels_for_validation(target)}",
            x=0.02,
            y=0.985,
            ha="left",
            fontsize=13,
            fontweight="bold",
        )
        fig.text(
            0.02,
            0.945,
            "Each point is a city-year in 2019-2024; the dashed line is the 1:1 reference.",
            ha="left",
            fontsize=8.2,
            color="#555555",
        )
        fig.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.08, hspace=0.32, wspace=0.23)
        save_fig(fig, paths.figures_validation / f"observed_vs_predicted_{target}")


def _plot_validation_temporal_holdout(raw: pd.DataFrame, paths: PackagePaths, targets: list[str]) -> None:
    """Observed-predicted annual hold-out trajectories, matching the prior fig6 checks."""
    if raw.empty or "year" not in raw.columns:
        return
    mode_colours = {"composite": "#a84238", "day": "#da9c15", "night": "#679dbf"}
    raw = raw.copy()
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw = raw[np.isfinite(raw["year"])].copy()
    annual = (
        raw.groupby(["target", "indicator", "heatwave_type", "year"], dropna=False)
        .agg(observed=("observed", "mean"), predicted=("predicted", "mean"), n=("observed", "size"))
        .reset_index()
        .sort_values(["target", "indicator", "heatwave_type", "year"])
    )
    annual.to_csv(paths.data_plot / "validation_primary_national_temporal_holdout_plot_data.csv", index=False)
    for target in targets:
        fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8), dpi=220, sharex=True)
        sub_target = annual[annual["target"].eq(target)].copy()
        for r, indicator in enumerate(INDICATOR_ORDER):
            for c, mode in enumerate(HEATWAVE_ORDER):
                ax = axes[r, c]
                sub = sub_target[sub_target["indicator"].eq(indicator) & sub_target["heatwave_type"].eq(mode)].copy()
                colour = mode_colours.get(mode, "#777777")
                if sub.empty:
                    ax.axis("off")
                    continue
                x = sub["year"].to_numpy(float)
                ax.plot(x, sub["observed"].to_numpy(float), color="#333333", linewidth=1.8, marker="o", markersize=3.0, label="Observed")
                ax.plot(x, sub["predicted"].to_numpy(float), color=colour, linewidth=1.8, marker="o", markersize=3.0, label="Predicted")
                vals = sub[["observed", "predicted"]].to_numpy(float).ravel()
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ymin, ymax = float(vals.min()), float(vals.max())
                    pad = (ymax - ymin) * 0.16 if ymax > ymin else max(abs(ymax) * 0.2, 0.01)
                    ax.set_ylim(ymin - pad, ymax + pad)
                ax.grid(True, color="0.86", linestyle="--", linewidth=0.45, alpha=0.7)
                ax.tick_params(labelsize=7.3)
                if r == 0:
                    ax.set_title(HEATWAVE_LABELS[mode], fontsize=9.2, fontweight="bold", color=colour)
                if c == 0:
                    ax.set_ylabel(f"{INDICATOR_LABELS[indicator]}\n{target_labels_for_validation(target)}", fontsize=8.6)
                if r == 1:
                    ax.set_xlabel("Hold-out year", fontsize=8.8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.94), ncol=2, frameon=False, fontsize=8.5)
        fig.suptitle(
            f"Primary national temporal hold-out: {target_labels_for_validation(target)}",
            x=0.02,
            y=0.985,
            ha="left",
            fontsize=13,
            fontweight="bold",
        )
        fig.text(
            0.02,
            0.945,
            "Lines are annual means across cities during 2019-2024.",
            ha="left",
            fontsize=8.2,
            color="#555555",
        )
        fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.09, hspace=0.31, wspace=0.23)
        save_fig(fig, paths.figures_validation / f"temporal_holdout_{target}")


def _current_scope_from_validation(val: pd.DataFrame) -> str:
    """Resolve which validation scope belongs to this replot package."""
    available = set(val.get("scope", pd.Series(dtype=str)).dropna().astype(str))
    candidates = [
        os.environ.get("FUTURE_PA_REPLOT_SCOPE"),
        FUTURE_DIR.name,
        PACKAGE_ROOT.name,
    ]
    for candidate in candidates:
        if candidate and candidate in available:
            return str(candidate)
    return "primary_national" if "primary_national" in available else (sorted(available)[0] if available else "")


def _plot_scope_specific_validation(scope_val: pd.DataFrame, paths: PackagePaths, scope_name: str, targets: list[str]) -> None:
    """Plot validation metrics from the current projection scope's source table."""
    if scope_val.empty:
        return
    scope_val = scope_val.copy()
    scope_val["indicator"] = pd.Categorical(scope_val["indicator"], INDICATOR_ORDER, ordered=True)
    scope_val["mode"] = pd.Categorical(scope_val["mode"], HEATWAVE_ORDER, ordered=True)
    scope_val["target"] = pd.Categorical(scope_val["target"], targets, ordered=True)
    scope_val["r2_numeric"] = pd.to_numeric(scope_val["r2"], errors="coerce")
    scope_val = scope_val.sort_values(["indicator", "mode", "target"]).copy()
    scope_val.to_csv(paths.data_plot / "validation_scope_specific_holdout_metrics.csv", index=False)

    x_order = [(indicator, mode) for indicator in INDICATOR_ORDER for mode in HEATWAVE_ORDER]
    x_lookup = {pair: i for i, pair in enumerate(x_order)}
    xlabels = [
        f"{INDICATOR_LABELS[ind].replace('Exceeded cumulative intensity', 'EQ')}\n{HEATWAVE_LABELS[mode].replace('Daytime', 'Day').replace('Nighttime', 'Night').replace('Compound', 'Comp.')}"
        for ind, mode in x_order
    ]
    mode_colours = {"composite": "#a84238", "day": "#da9c15", "night": "#679dbf"}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.25), dpi=220, sharey=True)
    for ax, target in zip(axes, targets):
        sub = scope_val[scope_val["target"].astype(str).eq(target)].copy()
        for _, row in sub.iterrows():
            key = (str(row["indicator"]), str(row["mode"]))
            if key not in x_lookup:
                continue
            x = x_lookup[key]
            color = mode_colours.get(str(row["mode"]), "#777777")
            r2 = float(row["r2_numeric"]) if np.isfinite(row["r2_numeric"]) else np.nan
            if np.isfinite(r2) and 0 <= r2 <= 1:
                ax.bar(x, r2, color=color, alpha=0.86, width=0.34, edgecolor="white", linewidth=0.5)
            else:
                marker = "v" if np.isfinite(r2) and r2 < 0 else "x"
                if marker == "x":
                    ax.scatter(x, -0.075, marker=marker, s=28, color=color, linewidths=0.9, zorder=5)
                else:
                    ax.scatter(
                        x,
                        -0.075,
                        marker=marker,
                        s=28,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=0.9,
                        zorder=5,
                    )
                label = "R2<0" if np.isfinite(r2) and r2 < 0 else "NA"
                ax.text(x, -0.145, label, ha="center", va="top", fontsize=5.8, color=color, rotation=90)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_ylim(-0.18, 1.08)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=6.6)
        ax.set_title(target_labels_for_validation(target), fontsize=9.2, fontweight="bold")
        ax.set_ylabel("Hold-out $R^2$", fontsize=9)
        ax.grid(True, axis="y", color="0.84", linestyle="--", linewidth=0.5)
    clean_scope = scope_name.replace("_", " ")
    fig.suptitle(f"{clean_scope}: scope-specific hold-out validation", x=0.02, y=0.98, ha="left", fontsize=13, fontweight="bold")
    fig.text(
        0.02,
        0.925,
        "Bars use this scope's rows from validation_holdout_all_scopes.csv. Downward triangles mark negative R2; crosses mark undefined R2.",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.81, bottom=0.31, wspace=0.13)
    save_fig(fig, paths.figures_validation / "holdout_validation_scope_specific_r2")

    heat_rows = [
        (indicator, mode)
        for indicator in INDICATOR_ORDER
        for mode in HEATWAVE_ORDER
    ]
    heat_index = pd.MultiIndex.from_tuples(heat_rows, names=["indicator", "mode"])
    heat = (
        scope_val.pivot_table(index=["indicator", "mode"], columns="target", values="r2_numeric", aggfunc="first", observed=False)
        .reindex(index=heat_index, columns=targets)
    )
    if not heat.empty:
        arr = heat.to_numpy(dtype=float)
        finite = np.isfinite(arr)
        fig_h, ax_h = plt.subplots(figsize=(6.9, 4.3), dpi=220)
        vmax = 1.0
        plot_arr = np.clip(arr, -vmax, vmax)
        im = ax_h.imshow(plot_arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax_h.set_xticks(np.arange(len(targets)))
        ax_h.set_xticklabels([target_labels_for_validation(t) for t in targets], fontsize=8.3)
        ylabels = [
            f"{INDICATOR_LABELS[ind].replace('Exceeded cumulative intensity', 'EQ')} | {HEATWAVE_LABELS[mode].replace('Daytime', 'Day').replace('Nighttime', 'Night')}"
            for ind, mode in heat_rows
        ]
        ax_h.set_yticks(np.arange(len(ylabels)))
        ax_h.set_yticklabels(ylabels, fontsize=7.8)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                value = arr[i, j]
                if np.isfinite(value):
                    text = f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
                    ax_h.text(j, i, text, ha="center", va="center", fontsize=7.0, color="#111111")
                else:
                    ax_h.text(j, i, "NA", ha="center", va="center", fontsize=7.0, color="#777777")
        clean_scope = scope_name.replace("_", " ")
        fig_h.text(
            0.08,
            0.97,
            f"{clean_scope}: actual hold-out $R^2$ values",
            ha="left",
            va="top",
            fontsize=11.2,
            fontweight="bold",
        )
        fig_h.text(
            0.08,
            0.915,
            "Colours are capped at [-1, 1] for readability; numbers show the uncapped R2 values.",
            ha="left",
            va="top",
            fontsize=7.4,
            color="#555555",
        )
        ax_h.tick_params(length=0)
        for spine in ax_h.spines.values():
            spine.set_visible(False)
        cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
        cbar.set_label("display-scaled $R^2$", fontsize=8)
        cbar.ax.tick_params(labelsize=7.2)
        fig_h.subplots_adjust(left=0.31, right=0.90, top=0.80, bottom=0.13)
        save_fig(fig_h, paths.figures_validation / "holdout_validation_scope_specific_r2_values_heatmap")


def plot_validation(paths: PackagePaths) -> None:
    if not VALIDATION_ALL_SCOPES.exists():
        return
    val = pd.read_csv(VALIDATION_ALL_SCOPES)
    val.to_csv(paths.data_plot / "validation_holdout_all_scopes.csv", index=False)
    targets = [
        "pa_loss_fraction_percent_all_days",
        "annualized_asri_percent",
        "benefit_adjusted_asri_percent",
    ]
    val = val[val["target"].isin(targets)].copy()
    if val.empty:
        return
    val["r2_numeric"] = pd.to_numeric(val["r2"], errors="coerce")
    extreme = val[(val["r2_numeric"].lt(0)) | (val["r2_numeric"].gt(1)) | (val["r2_numeric"].isna())].copy()
    extreme.to_csv(paths.data_plot / "validation_extreme_diagnostics.csv", index=False)

    current_scope = _current_scope_from_validation(val)
    (paths.data_plot / "validation_current_scope.txt").write_text(current_scope, encoding="utf-8")
    scope_val = val[val["scope"].astype(str).eq(current_scope)].copy() if current_scope else pd.DataFrame()
    _plot_scope_specific_validation(scope_val, paths, current_scope, targets)

    primary_raw = _load_primary_validation_raw(paths, targets)
    _plot_validation_observed_vs_predicted(primary_raw, paths, targets)
    _plot_validation_temporal_holdout(primary_raw, paths, targets)

    primary_boot = _validation_raw_summary(paths, targets)
    if not primary_boot.empty:
        primary_boot["indicator"] = pd.Categorical(primary_boot["indicator"], INDICATOR_ORDER, ordered=True)
        primary_boot["mode"] = pd.Categorical(primary_boot["mode"], HEATWAVE_ORDER, ordered=True)
        primary_boot = primary_boot.sort_values(["indicator", "mode", "target"]).copy()
        primary_boot.to_csv(paths.data_plot / "validation_primary_national_display_data.csv", index=False)

        x_order = [(indicator, mode) for indicator in INDICATOR_ORDER for mode in HEATWAVE_ORDER]
        x_lookup = {pair: i for i, pair in enumerate(x_order)}
        xlabels = [
            f"{INDICATOR_LABELS[ind].replace('Exceeded cumulative intensity', 'EQ')}\n{HEATWAVE_LABELS[mode].replace('Daytime', 'Day').replace('Nighttime', 'Night').replace('Compound', 'Comp.')}"
            for ind, mode in x_order
        ]
        mode_colours = {"composite": "#a84238", "day": "#da9c15", "night": "#679dbf"}
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.25), dpi=220, sharey=True)
        for ax, target in zip(axes, targets):
            sub = primary_boot[primary_boot["target"].eq(target)].copy()
            for _, row in sub.iterrows():
                key = (str(row["indicator"]), str(row["mode"]))
                if key not in x_lookup:
                    continue
                x = x_lookup[key]
                color = mode_colours.get(str(row["mode"]), "#777777")
                r2 = float(row["r2"]) if np.isfinite(row["r2"]) else np.nan
                lo = float(row["r2_low"]) if np.isfinite(row["r2_low"]) else np.nan
                hi = float(row["r2_high"]) if np.isfinite(row["r2_high"]) else np.nan
                if np.isfinite(r2) and 0 <= r2 <= 1:
                    ax.bar(x, r2, color=color, alpha=0.86, width=0.34, edgecolor="white", linewidth=0.5)
                    if np.isfinite(lo) and np.isfinite(hi):
                        yerr = np.array([[max(0.0, r2 - lo)], [max(0.0, hi - r2)]])
                        ax.errorbar(
                            x,
                            r2,
                            yerr=yerr,
                            fmt="none",
                            ecolor="#333333",
                            elinewidth=0.8,
                            capsize=2.5,
                            capthick=0.8,
                            zorder=4,
                        )
                else:
                    marker = "v" if np.isfinite(r2) and r2 < 0 else "x"
                    if marker == "x":
                        ax.scatter(x, -0.075, marker=marker, s=28, color=color, linewidths=0.9, zorder=5)
                    else:
                        ax.scatter(
                            x,
                            -0.075,
                            marker=marker,
                            s=28,
                            facecolors="white",
                            edgecolors=color,
                            linewidths=0.9,
                            zorder=5,
                        )
                    label = "R2<0" if np.isfinite(r2) and r2 < 0 else "NA"
                    ax.text(x, -0.145, label, ha="center", va="top", fontsize=5.8, color=color, rotation=90)
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_ylim(-0.18, 1.08)
            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=6.6)
            ax.set_title(target_labels_for_validation(target), fontsize=9.2, fontweight="bold")
            ax.set_ylabel("Hold-out $R^2$", fontsize=9)
            ax.grid(True, axis="y", color="0.84", linestyle="--", linewidth=0.5)
        fig.suptitle("Primary national hold-out validation with city-year bootstrap intervals", x=0.02, y=0.98, ha="left", fontsize=13, fontweight="bold")
        fig.text(
            0.02,
            0.925,
            "Bars show deterministic hold-out R2; whiskers are 95% city-year bootstrap intervals. CEHWI-night is shown as a diagnostic when observed variance is near zero.",
            fontsize=8.2,
            color="#555555",
        )
        fig.subplots_adjust(left=0.06, right=0.985, top=0.81, bottom=0.31, wspace=0.13)
        save_fig(fig, paths.figures_validation / "holdout_validation_primary_national_r2_bootstrap_ci")

    summary = (
        val.groupby(["scope", "target"], dropna=False)
        .agg(
            n=("r2", "size"),
            median_r2=("r2", "median"),
            median_rmse=("rmse", "median"),
            median_mae=("mae", "median"),
            median_pearson=("pearson", "median"),
        )
        .reset_index()
    )
    summary.to_csv(paths.data_plot / "validation_holdout_summary_by_scope_target.csv", index=False)

    target_labels = {
        "pa_loss_fraction_percent_all_days": "PA loss",
        "annualized_asri_percent": "ASRI",
        "benefit_adjusted_asri_percent": "Net ASRI",
    }
    heat = summary.pivot(index="scope", columns="target", values="median_r2").reindex(columns=targets)
    if not heat.empty:
        heat = heat.sort_index()
        fig_h, ax_h = plt.subplots(figsize=(6.9, max(3.2, 0.50 * len(heat) + 1.7)), dpi=220)
        arr = heat.to_numpy(dtype=float)
        finite = np.isfinite(arr)
        vmax = float(np.nanpercentile(np.abs(arr[finite]), 95)) if finite.any() else 1.0
        vmax = max(vmax, 1.0)
        im = ax_h.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax_h.set_xticks(np.arange(len(heat.columns)))
        ax_h.set_xticklabels([target_labels.get(c, c) for c in heat.columns], fontsize=8.5)
        ax_h.set_yticks(np.arange(len(heat.index)))
        ax_h.set_yticklabels([str(s).replace("_", " ") for s in heat.index], fontsize=8.2)
        fig_h.text(
            0.08,
            0.965,
            "Hold-out validation median $R^2$ by projection scope",
            ha="left",
            va="top",
            fontsize=10.8,
            fontweight="bold",
        )
        fig_h.text(
            0.08,
            0.905,
            "Training period: 2010-2018; evaluation period: 2019-2024. Values are medians across CEHWI/exceeded and compound/day/night targets.",
            ha="left",
            va="top",
            fontsize=7.2,
            color="#555555",
        )
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                value = arr[i, j]
                if np.isfinite(value):
                    ax_h.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.4, color="#222222")
                else:
                    ax_h.text(j, i, "NA", ha="center", va="center", fontsize=7.2, color="#777777")
        ax_h.tick_params(length=0)
        for spine in ax_h.spines.values():
            spine.set_visible(False)
        cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
        cbar.set_label("median hold-out $R^2$", fontsize=8)
        cbar.ax.tick_params(labelsize=7.5)
        fig_h.subplots_adjust(left=0.26, right=0.91, top=0.78, bottom=0.12)
        save_fig(fig_h, paths.figures_validation / "holdout_validation_scope_target_heatmap")


def write_docs(paths: PackagePaths, manifest: pd.DataFrame, plot_df: pd.DataFrame, city_df: pd.DataFrame) -> None:
    denominator_id = "unspecified_in_source"
    denominator_label = "Unspecified in source projection table"
    denominator_months = "unspecified_in_source"
    if "denominator_id" in plot_df.columns:
        denom_rows = plot_df[plot_df["period_source"].eq("future")].copy()
        if not denom_rows.empty:
            denominator_id = str(denom_rows["denominator_id"].dropna().iloc[0]) if denom_rows["denominator_id"].notna().any() else denominator_id
            if "denominator_label" in denom_rows.columns and denom_rows["denominator_label"].notna().any():
                denominator_label = str(denom_rows["denominator_label"].dropna().iloc[0])
            if "denominator_months" in denom_rows.columns and denom_rows["denominator_months"].notna().any():
                denominator_months = str(denom_rows["denominator_months"].dropna().iloc[0])
    counts = {
        "timeseries_rows": int(len(plot_df)),
        "city_period_rows": int(len(city_df)),
        "copied_files": int(manifest["status"].eq("copied").sum()) if not manifest.empty else 0,
        "missing_files": int(manifest["status"].eq("missing").sum()) if not manifest.empty else 0,
        "denominator_id": denominator_id,
    }
    (paths.docs / "run_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    readme = f"""# Future projection modewise replot package

Generated on 2026-06-19 from the updated CMIP-mean input directory.

## What this package contains

- `data/raw_source_csv`: exact source CSVs copied from the final projection run.
- `data/model_relationships`: pooled exposure-response curve inventory and copied pooled RR curve CSVs.
- `data/plot_ready_csv`: cleaned tables used directly by the new figures.
- `code`: copied upstream calculation scripts, the current plotting/repackaging script, the fast all-scope projection script, the PowerShell runner, and `future_pa_heat_risk_projection_with_annualized_asri_ci_new_cmip.R`, the R reference projection script that exports annualized-ASRI lower/upper columns from daily RR bounds.
- `figures/timeseries_modewise`: three modewise historical-future trajectory figures.
- `figures/maps_city_change`: city-level future PA-loss maps by period, SSP and heatwave mode.
- `figures/validation`: hold-out validation summaries, including city-year bootstrap intervals where raw validation predictions are available.

## Final projection source

Future source directory:
`{FUTURE_DIR}`

Historical source directory:
`{HIST_DIR}`

The primary projection uses the national pooled exposure-response function.
This follows the conservative main-analysis choice in the final projection notes.

## Projection denominator

Denominator ID:
`{denominator_id}`

Denominator label:
`{denominator_label}`

Denominator months:
`{denominator_months}`

All days inside the denominator are retained. Non-heatwave days contribute zero
because their exposure is evaluated at the RR reference.

## ASRI uncertainty status

This package is generated after rerunning the daily-to-annual projection with the updated CMIP-mean inputs.
The annual source tables now contain lower/upper columns for PA loss fraction, annualized ASRI and benefit-adjusted ASRI.
The ASRI uncertainty band is drawn from daily RR lower/upper aggregation before annual summarization.

The calculation entry point for the final package is:

`code/future_pa_projection_fast_new_cmip_all_scopes_20260619.py`

The R script is retained as a readable reference implementation:

`code/future_pa_heat_risk_projection_with_annualized_asri_ci_new_cmip.R`

It computes daily `asri_low = max(0, -log(RR_high))` and `asri_high = max(0, -log(RR_low))`.
These values are then aggregated to `annualized_asri_low_percent` and `annualized_asri_high_percent`.

## Main figure logic

The three primary figures are organised by heatwave mode:

1. Compound heatwaves.
2. Daytime-only heatwaves.
3. Nighttime-only heatwaves.

Within each figure, columns separate CEHWI and exceeded cumulative intensity.
Rows show PA loss fraction, ASRI and benefit-adjusted ASRI.
The historical trajectory is drawn as a single grey line.
Future trajectories are drawn for SSP2-4.5, SSP3-7.0 and SSP5-8.5.

## Colour system

The colours follow the archived reference palette documented with the compact release data:

- Historical: `#404040`.
- SSP2-4.5: `#679dbf`.
- SSP3-7.0: `#da9c15`.
- SSP5-8.5: `#a84238`.

## Re-run

To reproduce the current clean package from the updated CMIP-mean daily grid files, run:

`python analysis_code/06_future_projection/code/project_future_pa_loss_all_scopes.py`

The runner rebuilds daily-to-annual projection CSVs, plot-ready CSVs and all figures for each projection scope.

## Output counts

```json
{json.dumps(counts, indent=2)}
```
"""
    (paths.docs / "README.md").write_text(readme, encoding="utf-8")

    method = f"""# Method note

This package has two reproducibility layers. First, `future_pa_projection_fast_new_cmip_all_scopes_20260619.py` reruns the daily-to-annual projection from the updated CMIP-mean grid files and writes national, external-control and sensitivity scopes from one pass through the CMIP grid files. Second, `make_future_projection_modewise_replot_new_cmip_20260619.py` reconstructs publication-ready trajectory, map and validation figures from those newly exported annual and period summaries. The patched R script is retained as a reference implementation of the daily RR-bound aggregation.

The primary metric is the heatwave-attributable physical-activity loss fraction.
It is defined as the grid-day mean of max(0, 1 - RR(x)) across the specified denominator multiplied by 100.
ASRI is the corresponding denominator mean of max(0, -log RR(x)) multiplied by 100.
Benefit-adjusted ASRI is max(0, mean[-log RR(x)]) multiplied by 100.

For this package, the denominator is `{denominator_id}`: {denominator_label}.

The historical segment uses 2010-2024 estimates.
The future segment uses 2025-2050 projections under SSP2-4.5, SSP3-7.0 and SSP5-8.5.
Future SSP lines are connected to the last observed historical point in 2024 for visual continuity.
Shaded uncertainty bands describe the projected period from 2025 onward.
For all three projection indices, available lower and upper bounds are shown as shaded bands.
Annualized ASRI intervals are computed at the daily RR stage before annual aggregation.

Validation is redrawn from raw city-year hold-out predictions where available.
The displayed whiskers are non-parametric 95% bootstrap intervals over city-year rows.
The CEHWI nighttime diagnostic reports its near-zero observed hold-out variance alongside absolute-error measures; under this variance structure, conventional R2 can be negative or undefined even when absolute errors are small.
"""
    (paths.docs / "method_note.md").write_text(method, encoding="utf-8")


def main() -> None:
    set_publication_style()
    paths = ensure_package()
    map_only = os.environ.get("FUTURE_PA_MAP_ONLY", "").strip().upper() in {"1", "TRUE", "YES", "Y"}
    timeseries_only = os.environ.get("FUTURE_PA_TIMESERIES_ONLY", "").strip().upper() in {"1", "TRUE", "YES", "Y"}
    if timeseries_only:
        try:
            shutil.copy2(Path(__file__).resolve(), paths.code / Path(__file__).name)
        except Exception:
            pass
        future_national, historical_national, _, _ = load_projection_tables()
        plot_df = build_plot_ready_timeseries(future_national, historical_national, paths)
        plot_modewise_timeseries(plot_df, paths)
        plot_metric_single_timeseries(plot_df, paths)
        print(f"Timeseries-only package completed: {PACKAGE_ROOT}")
        print(f"Timeseries figures: {paths.figures_timeseries}")
        return

    if map_only:
        try:
            shutil.copy2(Path(__file__).resolve(), paths.code / Path(__file__).name)
        except Exception:
            pass
        _, _, future_city_period, _ = load_projection_tables()
        city_df = build_city_map_data(future_city_period, paths)
        plot_city_maps(city_df, paths)
        print(f"Map-only package completed: {PACKAGE_ROOT}")
        print(f"Map figures: {paths.figures_maps}")
        return

    manifest = copy_source_material(paths)
    future_national, historical_national, future_city_period, future_city_year = load_projection_tables()

    # Preserve an explicit copy of the high-volume city-year table in plot-ready data.
    future_city_year.to_csv(paths.data_plot / "future_city_year_projection_data.csv", index=False)

    plot_df = build_plot_ready_timeseries(future_national, historical_national, paths)
    city_df = build_city_map_data(future_city_period, paths)
    plot_modewise_timeseries(plot_df, paths)
    plot_metric_single_timeseries(plot_df, paths)
    plot_city_maps(city_df, paths)
    plot_validation(paths)
    write_docs(paths, manifest, plot_df, city_df)
    print(f"Package completed: {PACKAGE_ROOT}")
    print(f"Timeseries figures: {paths.figures_timeseries}")
    print(f"Map figures: {paths.figures_maps}")
    print(f"Validation figures: {paths.figures_validation}")


if __name__ == "__main__":
    main()
