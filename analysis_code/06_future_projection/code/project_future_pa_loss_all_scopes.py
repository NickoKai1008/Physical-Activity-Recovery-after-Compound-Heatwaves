"""Fast future PA heat-risk projection using updated CMIP-mean grid files.

This script reproduces the daily-to-annual projection logic of
future_pa_heat_risk_projection_with_annualized_asri_ci_new_cmip.R, but reads the
updated CMIP files once and writes all projection scopes:
primary national, zone, region, city-specific and DTW4lag12 sensitivity.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
REPO_ROOT = MODULE_DIR.parents[1]
EXTERNAL_DATA_ROOT = Path(os.environ.get("HEATPA_DATA_ROOT", REPO_ROOT / "external_data"))
FINAL_ROOT = Path(os.environ.get("FUTURE_PA_FINAL_ROOT", MODULE_DIR))
OUT_ROOT = Path(os.environ.get("FUTURE_PA_OUT_ROOT", MODULE_DIR / "output" / "model"))
CMIP_ROOT = Path(os.environ.get("FUTURE_PA_CMIP_ROOT", EXTERNAL_DATA_ROOT / "cmip6"))
STAGE2_ROOT = Path(os.environ.get("FUTURE_PA_STAGE2_ROOT", EXTERNAL_DATA_ROOT / "dlnm_stage2"))
GRID_MAP_PATH = Path(os.environ.get("FUTURE_PA_GRID_MAP", EXTERNAL_DATA_ROOT / "grid10_environment_predictors.csv"))

FUTURE_START_YEAR = int(os.environ.get("FUTURE_START_YEAR", "2025"))
FUTURE_END_YEAR = int(os.environ.get("FUTURE_END_YEAR", "2050"))
WARM_SEASON_MONTHS = {int(x) for x in os.environ.get("WARM_SEASON_MONTHS", "1,2,3,4,5,6,7,8,9,10,11,12").split(",") if x.strip()}
DENOMINATOR_ID = os.environ.get("DENOMINATOR_ID", "annual_all_days_primary")
DENOMINATOR_LABEL = os.environ.get(
    "DENOMINATOR_LABEL",
    "All calendar-year days; non-heatwave days contribute zero" if WARM_SEASON_MONTHS == set(range(1, 13))
    else f"Selected denominator months {sorted(WARM_SEASON_MONTHS)}; non-heatwave days contribute zero",
)
DENOMINATOR_MONTHS = ",".join(str(x) for x in sorted(WARM_SEASON_MONTHS))
ACTIVITY_FILTER = {x.strip().lower() for x in os.environ.get("PROJECTION_ACTIVITY_TYPES", "all").split(",") if x.strip()}
if not ACTIVITY_FILTER:
    ACTIVITY_FILTER = {"all"}
RUN_FULL_CMIP = os.environ.get("RUN_FULL_CMIP", "TRUE").upper() in {"TRUE", "1", "YES"}
MAX_FILES_PER_SCENARIO = int(os.environ.get("MAX_FILES_PER_SCENARIO", "30"))

SCOPES = {
    "primary_national": {"curve_scope": "national", "family_prefix": None, "family_label": "national"},
    "control_zone": {"curve_scope": "zone", "family_prefix": "ZONE", "family_label": "ZONE"},
    "control_region": {"curve_scope": "region", "family_prefix": "REGION", "family_label": "REGION"},
    "sensitivity_city_specific": {"curve_scope": "city", "family_prefix": None, "family_label": "city_specific"},
    "sensitivity_dtw4lag12": {"curve_scope": "DTW4lag12", "family_prefix": "DTW4lag12", "family_label": "DTW4lag12"},
}

SUM_COLS = [
    "n_days",
    "heatwave_days",
    "asri_sum_days",
    "asri_low_sum_days",
    "asri_high_sum_days",
    "pa_loss_sum_days",
    "pa_loss_low_sum_days",
    "pa_loss_high_sum_days",
    "pa_loss_linear_tail_sum_days",
    "pa_loss_linear_tail_low_sum_days",
    "pa_loss_linear_tail_high_sum_days",
    "net_log_suppression_sum_days",
    "net_log_suppression_low_sum_days",
    "net_log_suppression_high_sum_days",
]

MEAN_COLS = [
    "exposure_mean_all_days",
    "exposure_p95_all_days",
    "rr_mean_all_days",
    "signed_pa_change_percent_all_days",
    "pa_loss_fraction_percent_heatwave_days",
    "asri_mean_all_days",
    "exposure_clipped_share",
]

NATIONAL_MEAN_COLS = [
    "heatwave_days",
    "signed_pa_change_percent_all_days",
    "pa_loss_fraction_percent_all_days",
    "pa_loss_fraction_percent_heatwave_days",
    "asri_mean_all_days",
    "annualized_asri_percent",
    "annualized_asri_low_percent",
    "annualized_asri_high_percent",
    "benefit_adjusted_asri_percent",
    "benefit_adjusted_asri_low_percent",
    "benefit_adjusted_asri_high_percent",
    "exposure_clipped_share",
    "pa_loss_fraction_low_percent_all_days",
    "pa_loss_fraction_high_percent_all_days",
    "pa_loss_fraction_linear_tail_percent_all_days",
    "pa_loss_fraction_linear_tail_low_percent_all_days",
    "pa_loss_fraction_linear_tail_high_percent_all_days",
]

BASE_CITY_YEAR_COLS = [
    "scenario",
    "city_standard",
    "year",
    "period",
    "model_key",
    "indicator",
    "heatwave_type",
    "activity_type",
    "curve_scope",
    "partition_family",
    "partition_name",
]

DERIVED_COLS = [
    "annualized_asri_percent",
    "annualized_asri_low_percent",
    "annualized_asri_high_percent",
    "pa_loss_fraction_percent_all_days",
    "pa_loss_fraction_low_percent_all_days",
    "pa_loss_fraction_high_percent_all_days",
    "pa_loss_fraction_linear_tail_percent_all_days",
    "pa_loss_fraction_linear_tail_low_percent_all_days",
    "pa_loss_fraction_linear_tail_high_percent_all_days",
    "benefit_adjusted_asri_percent",
    "benefit_adjusted_asri_low_percent",
    "benefit_adjusted_asri_high_percent",
]


def msg(text: str) -> None:
    print(pd.Timestamp.now().strftime("%H:%M:%S"), text, flush=True)


def add_denominator_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["denominator_id"] = DENOMINATOR_ID
    df["denominator_label"] = DENOMINATOR_LABEL
    df["denominator_months"] = DENOMINATOR_MONTHS
    return df


def normalize_city(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).lower()


def period_label(year: int) -> str:
    if year <= 2030:
        return f"{FUTURE_START_YEAR}-2030"
    if year <= 2040:
        return "2031-2040"
    return "2041-2050"


def parse_model_key(folder_name: str) -> dict | None:
    key = re.sub(r"^POOLED_META_", "", folder_name, flags=re.I).lower()
    if key.startswith("exceeded_quantity_"):
        indicator = "exceeded_quantity"
        rest = key.removeprefix("exceeded_quantity_")
    elif key.startswith("cehwi_"):
        indicator = "cehwi"
        rest = key.removeprefix("cehwi_")
    else:
        return None
    parts = rest.split("_")
    if len(parts) < 2:
        return None
    return {
        "model_key": key,
        "indicator": indicator,
        "heatwave_type": parts[0],
        "activity_type": "_".join(parts[1:]),
    }


def exposure_col_for(indicator: str, heatwave_type: str) -> str:
    prefix = {"composite": "Composite", "day": "Daytime", "night": "Nighttime"}[heatwave_type]
    if indicator == "cehwi":
        return f"{prefix}_CEHWI"
    return f"{prefix}_exceeded_quantity"


def flag_col_for(heatwave_type: str) -> str:
    return {"composite": "Composite_heatwave", "day": "Daytime_heatwave", "night": "Nighttime_heatwave"}[heatwave_type]


def read_curve_file(path: Path, meta: dict, curve_scope: str, partition_family: str, partition_name: str, city_standard: str | None) -> tuple[dict, dict] | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "rr" not in df.columns:
        return None
    exposure = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    rr = pd.to_numeric(df["rr"], errors="coerce").to_numpy(float)
    rr_low = pd.to_numeric(df["rr_low"], errors="coerce").to_numpy(float) if "rr_low" in df else rr.copy()
    rr_high = pd.to_numeric(df["rr_high"], errors="coerce").to_numpy(float) if "rr_high" in df else rr.copy()
    keep = np.isfinite(exposure) & np.isfinite(rr) & (rr > 0)
    if keep.sum() < 2:
        return None
    tmp = pd.DataFrame(
        {
            "x": exposure[keep],
            "logrr": np.log(np.maximum(rr[keep], 1e-12)),
            "logrr_low": np.log(np.maximum(np.where(np.isfinite(rr_low[keep]) & (rr_low[keep] > 0), rr_low[keep], rr[keep]), 1e-12)),
            "logrr_high": np.log(np.maximum(np.where(np.isfinite(rr_high[keep]) & (rr_high[keep] > 0), rr_high[keep], rr[keep]), 1e-12)),
        }
    ).sort_values("x")
    tmp = tmp.groupby("x", as_index=False).mean(numeric_only=True)
    city_key = normalize_city(city_standard) if city_standard else ""
    curve_id = "::".join([curve_scope, partition_name, city_standard or "", meta["model_key"]])
    curve = {
        "curve_id": curve_id,
        "x": tmp["x"].to_numpy(float),
        "logrr": tmp["logrr"].to_numpy(float),
        "logrr_low": tmp["logrr_low"].to_numpy(float),
        "logrr_high": tmp["logrr_high"].to_numpy(float),
    }
    row = {
        "curve_id": curve_id,
        "model_key": meta["model_key"],
        "indicator": meta["indicator"],
        "heatwave_type": meta["heatwave_type"],
        "activity_type": meta["activity_type"],
        "curve_scope": curve_scope,
        "partition_family": partition_family,
        "partition_name": partition_name,
        "city_standard": city_standard or "",
        "city_key": city_key,
        "curve_file": str(path),
        "curve_x_min": float(tmp["x"].min()),
        "curve_x_max": float(tmp["x"].max()),
    }
    return curve, row


def read_partition_info(partition_dir: Path) -> list[str]:
    candidates = [
        partition_dir / "partition_info.csv",
        partition_dir / "zone_info.csv",
        partition_dir / "region_info.csv",
        *partition_dir.glob("*_info.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "cities" in df.columns and not df.empty:
            cities = [x.strip() for x in str(df["cities"].iloc[0]).split(",") if x.strip()]
            return [re.sub(r"_cehwi$", "", x, flags=re.I).replace("_", " ") for x in cities]
    return []


def should_keep_activity(activity_type: str) -> bool:
    a = activity_type.lower()
    return "*" in ACTIVITY_FILTER or "all_activity_types" in ACTIVITY_FILTER or "all_modes" in ACTIVITY_FILTER or a in ACTIVITY_FILTER


def load_national_curves() -> tuple[dict, pd.DataFrame]:
    curves: dict[str, dict] = {}
    rows: list[dict] = []
    for path in STAGE2_ROOT.rglob("pooled_RR_curve.csv"):
        if "POOLED_META_" not in str(path):
            continue
        meta = parse_model_key(path.parent.name)
        if meta is None or not should_keep_activity(meta["activity_type"]):
            continue
        obj = read_curve_file(path, meta, "national", "national", "National", None)
        if obj is None:
            continue
        curve, row = obj
        curves[curve["curve_id"]] = curve
        rows.append(row)
    return curves, pd.DataFrame(rows)


def load_partition_curves(family_prefix: str, family_label: str) -> tuple[dict, pd.DataFrame]:
    curves: dict[str, dict] = {}
    rows: list[dict] = []
    for pdir in sorted(STAGE2_ROOT.iterdir()):
        if not pdir.is_dir() or not pdir.name.startswith(f"{family_prefix}_"):
            continue
        cities = read_partition_info(pdir)
        if not cities:
            continue
        for cdir in sorted([x for x in pdir.iterdir() if x.is_dir()]):
            meta = parse_model_key(cdir.name)
            if meta is None or not should_keep_activity(meta["activity_type"]):
                continue
            curve_path = cdir / "pooled_RR_data.csv"
            if not curve_path.exists():
                curve_path = cdir / "pooled_RR_curve.csv"
            if not curve_path.exists():
                continue
            for city in cities:
                obj = read_curve_file(curve_path, meta, family_label, family_label, pdir.name, city)
                if obj is None:
                    continue
                curve, row = obj
                curves[curve["curve_id"]] = curve
                rows.append(row)
    return curves, pd.DataFrame(rows)


def load_city_curves() -> tuple[dict, pd.DataFrame]:
    curves: dict[str, dict] = {}
    rows: list[dict] = []
    for pdir in sorted(STAGE2_ROOT.iterdir()):
        if not pdir.is_dir() or not re.search(r"_(cehwi|exceeded_quantity)$", pdir.name, flags=re.I):
            continue
        indicator = "exceeded_quantity" if pdir.name.lower().endswith("_exceeded_quantity") else "cehwi"
        city_stub = re.sub(r"_(cehwi|exceeded_quantity)$", "", pdir.name, flags=re.I)
        city_standard = city_stub.replace("_", " ")
        for path in pdir.glob("*_DLNM_pred.csv"):
            core = re.sub(r"_DLNM_pred\.csv$", "", path.name, flags=re.I)
            prefix = f"{city_stub}_"
            if not core.lower().startswith(prefix.lower()):
                continue
            model_part = core[len(prefix):].lower()
            parts = model_part.split("_")
            if len(parts) < 2 or parts[0] not in {"composite", "day", "night"}:
                continue
            meta = {
                "model_key": "_".join([indicator, parts[0], "_".join(parts[1:])]),
                "indicator": indicator,
                "heatwave_type": parts[0],
                "activity_type": "_".join(parts[1:]),
            }
            if not should_keep_activity(meta["activity_type"]):
                continue
            obj = read_curve_file(path, meta, "city", "city_specific", city_standard, city_standard)
            if obj is None:
                continue
            curve, row = obj
            curves[curve["curve_id"]] = curve
            rows.append(row)
    return curves, pd.DataFrame(rows)


def load_curves_by_scope() -> tuple[dict, dict, pd.DataFrame]:
    out_curves: dict[str, dict] = {}
    out_meta: dict[str, pd.DataFrame] = {}
    for scope, spec in SCOPES.items():
        if spec["curve_scope"] == "national":
            curves, meta = load_national_curves()
        elif spec["curve_scope"] == "city":
            curves, meta = load_city_curves()
        else:
            curves, meta = load_partition_curves(spec["family_prefix"], spec["family_label"])
        if meta.empty:
            raise RuntimeError(f"No curves loaded for {scope}")
        out_curves[scope] = curves
        out_meta[scope] = meta
        msg(f"Loaded {len(curves)} curves for {scope}.")
    all_meta = pd.concat([m.assign(package_scope=s) for s, m in out_meta.items()], ignore_index=True)
    return out_curves, out_meta, all_meta


def predict_rr(curve: dict, exposure: np.ndarray, linear_tail: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(exposure, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    lo = float(np.nanmin(curve["x"]))
    hi = float(np.nanmax(curve["x"]))
    x_clip = np.minimum(np.maximum(x, lo), hi)
    clipped = x != x_clip

    def interp(log_values: np.ndarray) -> np.ndarray:
        y = np.interp(x_clip, curve["x"], log_values, left=log_values[0], right=log_values[-1])
        if linear_tail and len(curve["x"]) >= 2:
            below = x < curve["x"][0]
            above = x > curve["x"][-1]
            if below.any():
                slope = (log_values[1] - log_values[0]) / (curve["x"][1] - curve["x"][0])
                y[below] = log_values[0] + slope * (x[below] - curve["x"][0])
            if above.any():
                slope = (log_values[-1] - log_values[-2]) / (curve["x"][-1] - curve["x"][-2])
                y[above] = log_values[-1] + slope * (x[above] - curve["x"][-1])
        return np.exp(np.clip(y, math.log(1e-8), math.log(1e8)))

    return interp(curve["logrr"]), interp(curve["logrr_low"]), interp(curve["logrr_high"]), clipped


def empty_accumulator() -> dict:
    d = {c: 0.0 for c in SUM_COLS}
    d["_mean_sum"] = {c: 0.0 for c in MEAN_COLS}
    d["_mean_n"] = {c: 0 for c in MEAN_COLS}
    return d


def add_mean(acc: dict, name: str, value: float) -> None:
    if np.isfinite(value):
        acc["_mean_sum"][name] += float(value)
        acc["_mean_n"][name] += 1


def build_time_template(first_file: Path) -> dict:
    t = pd.to_datetime(pd.read_csv(first_file, usecols=["Time"])["Time"], errors="coerce")
    years_all = t.dt.year.to_numpy()
    months_all = t.dt.month.to_numpy()
    mask = (years_all >= FUTURE_START_YEAR) & (years_all <= FUTURE_END_YEAR) & np.isin(months_all, list(WARM_SEASON_MONTHS))
    years = years_all[mask]
    year_to_idx = {int(y): np.where(years == y)[0] for y in np.unique(years)}
    return {"mask": mask, "years": years, "year_to_idx": year_to_idx, "template_len": len(t)}


def aggregate_grid_year(
    acc: dict,
    key: tuple,
    years_idx: dict[int, np.ndarray],
    exposure: np.ndarray,
    hw_flag: np.ndarray,
    rr: np.ndarray,
    rr_low: np.ndarray,
    rr_high: np.ndarray,
    rr_tail: np.ndarray,
    rr_tail_low: np.ndarray,
    rr_tail_high: np.ndarray,
    clipped: np.ndarray,
) -> None:
    signed_change = rr - 1.0
    loss = np.maximum(0.0, 1.0 - rr)
    loss_low = np.maximum(0.0, 1.0 - rr_high)
    loss_high = np.maximum(0.0, 1.0 - rr_low)
    loss_tail = np.maximum(0.0, 1.0 - rr_tail)
    loss_tail_low = np.maximum(0.0, 1.0 - rr_tail_high)
    loss_tail_high = np.maximum(0.0, 1.0 - rr_tail_low)
    asri = np.maximum(0.0, -np.log(np.maximum(rr, 1e-12)))
    asri_low = np.maximum(0.0, -np.log(np.maximum(rr_high, 1e-12)))
    asri_high = np.maximum(0.0, -np.log(np.maximum(rr_low, 1e-12)))
    net = -np.log(np.maximum(rr, 1e-12))
    net_low = -np.log(np.maximum(rr_high, 1e-12))
    net_high = -np.log(np.maximum(rr_low, 1e-12))

    for year, idx in years_idx.items():
        year_key = (*key[:2], year, period_label(year), *key[2:])
        item = acc[year_key]
        hw = hw_flag[idx] | (exposure[idx] > 0)
        n = int(idx.size)
        item["n_days"] += n
        item["heatwave_days"] += int(np.nansum(hw))
        item["asri_sum_days"] += float(np.nansum(asri[idx]))
        item["asri_low_sum_days"] += float(np.nansum(asri_low[idx]))
        item["asri_high_sum_days"] += float(np.nansum(asri_high[idx]))
        item["pa_loss_sum_days"] += float(np.nansum(loss[idx]))
        item["pa_loss_low_sum_days"] += float(np.nansum(loss_low[idx]))
        item["pa_loss_high_sum_days"] += float(np.nansum(loss_high[idx]))
        item["pa_loss_linear_tail_sum_days"] += float(np.nansum(loss_tail[idx]))
        item["pa_loss_linear_tail_low_sum_days"] += float(np.nansum(loss_tail_low[idx]))
        item["pa_loss_linear_tail_high_sum_days"] += float(np.nansum(loss_tail_high[idx]))
        item["net_log_suppression_sum_days"] += float(np.nansum(net[idx]))
        item["net_log_suppression_low_sum_days"] += float(np.nansum(net_low[idx]))
        item["net_log_suppression_high_sum_days"] += float(np.nansum(net_high[idx]))

        add_mean(item, "exposure_mean_all_days", float(np.nanmean(exposure[idx])))
        add_mean(item, "exposure_p95_all_days", float(np.nanquantile(exposure[idx], 0.95)))
        add_mean(item, "rr_mean_all_days", float(np.nanmean(rr[idx])))
        add_mean(item, "signed_pa_change_percent_all_days", float(np.nanmean(signed_change[idx]) * 100.0))
        add_mean(item, "pa_loss_fraction_percent_heatwave_days", float(np.nanmean(loss[idx][hw]) * 100.0) if hw.any() else np.nan)
        add_mean(item, "asri_mean_all_days", float(np.nanmean(asri[idx])))
        add_mean(item, "exposure_clipped_share", float(np.nanmean(clipped[idx])))


def accumulator_to_city_year(scope: str, acc: dict) -> pd.DataFrame:
    rows = []
    for key, item in acc.items():
        (
            scenario,
            city_standard,
            year,
            period,
            model_key,
            indicator,
            heatwave_type,
            activity_type,
            curve_scope,
            partition_family,
            partition_name,
        ) = key
        row = {
            "scenario": scenario,
            "city_standard": city_standard,
            "year": year,
            "period": period,
            "model_key": model_key,
            "indicator": indicator,
            "heatwave_type": heatwave_type,
            "activity_type": activity_type,
            "curve_scope": curve_scope,
            "partition_family": partition_family,
            "partition_name": partition_name,
        }
        for col in SUM_COLS:
            row[col] = item[col]
        for col in MEAN_COLS:
            n = item["_mean_n"][col]
            row[col] = item["_mean_sum"][col] / n if n else np.nan
        n_days = row["n_days"] if row["n_days"] else np.nan
        row["annualized_asri_percent"] = row["asri_mean_all_days"] * 100.0
        row["annualized_asri_low_percent"] = row["asri_low_sum_days"] / n_days * 100.0
        row["annualized_asri_high_percent"] = row["asri_high_sum_days"] / n_days * 100.0
        row["pa_loss_fraction_percent_all_days"] = row["pa_loss_sum_days"] / n_days * 100.0
        row["pa_loss_fraction_low_percent_all_days"] = row["pa_loss_low_sum_days"] / n_days * 100.0
        row["pa_loss_fraction_high_percent_all_days"] = row["pa_loss_high_sum_days"] / n_days * 100.0
        row["pa_loss_fraction_linear_tail_percent_all_days"] = row["pa_loss_linear_tail_sum_days"] / n_days * 100.0
        row["pa_loss_fraction_linear_tail_low_percent_all_days"] = row["pa_loss_linear_tail_low_sum_days"] / n_days * 100.0
        row["pa_loss_fraction_linear_tail_high_percent_all_days"] = row["pa_loss_linear_tail_high_sum_days"] / n_days * 100.0
        row["benefit_adjusted_asri_percent"] = max(0.0, row["net_log_suppression_sum_days"] / n_days) * 100.0
        row["benefit_adjusted_asri_low_percent"] = max(0.0, row["net_log_suppression_low_sum_days"] / n_days) * 100.0
        row["benefit_adjusted_asri_high_percent"] = max(0.0, row["net_log_suppression_high_sum_days"] / n_days) * 100.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return add_denominator_columns(pd.DataFrame(columns=BASE_CITY_YEAR_COLS + SUM_COLS + MEAN_COLS + DERIVED_COLS))
    out = add_denominator_columns(out)
    return out.sort_values(["scenario", "city_standard", "year", "model_key"]).reset_index(drop=True)


def derive_outputs(scope: str, city_year: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    city_year = add_denominator_columns(city_year)
    city_year.to_csv(out_dir / "future_heat_pa_risk_city_year_trial.csv", index=False)

    national_year = (
        city_year.groupby(["scenario", "year", "period", "model_key", "indicator", "heatwave_type", "activity_type"], as_index=False)[NATIONAL_MEAN_COLS]
        .mean(numeric_only=True)
    )
    national_year["curve_scope"] = SCOPES[scope]["curve_scope"]
    fam = SCOPES[scope]["family_label"]
    if scope == "primary_national":
        national_year["partition_family"] = "national"
    elif scope == "sensitivity_city_specific":
        national_year["partition_family"] = "city_specific_mixed"
    else:
        national_year["partition_family"] = f"{fam}_mixed"
    national_year["partition_name"] = "All_cities"
    national_year = add_denominator_columns(national_year)
    national_year.to_csv(out_dir / "future_heat_pa_risk_national_year_trial.csv", index=False)

    group_cols = ["scenario", "city_standard", "period", "model_key", "indicator", "heatwave_type", "activity_type", "curve_scope", "partition_family", "partition_name"]
    city_period = city_year.groupby(group_cols, as_index=False)[SUM_COLS].sum(numeric_only=True)
    mean_part = city_year.groupby(group_cols, as_index=False)[MEAN_COLS].mean(numeric_only=True)
    city_period = city_period.merge(mean_part, on=group_cols, how="left")
    n = city_period["n_days"].replace(0, np.nan)
    city_period["annualized_asri_percent"] = city_period["asri_mean_all_days"] * 100.0
    city_period["annualized_asri_low_percent"] = city_period["asri_low_sum_days"] / n * 100.0
    city_period["annualized_asri_high_percent"] = city_period["asri_high_sum_days"] / n * 100.0
    city_period["pa_loss_fraction_percent_all_days"] = city_period["pa_loss_sum_days"] / n * 100.0
    city_period["pa_loss_fraction_low_percent_all_days"] = city_period["pa_loss_low_sum_days"] / n * 100.0
    city_period["pa_loss_fraction_high_percent_all_days"] = city_period["pa_loss_high_sum_days"] / n * 100.0
    city_period["pa_loss_fraction_linear_tail_percent_all_days"] = city_period["pa_loss_linear_tail_sum_days"] / n * 100.0
    city_period["pa_loss_fraction_linear_tail_low_percent_all_days"] = city_period["pa_loss_linear_tail_low_sum_days"] / n * 100.0
    city_period["pa_loss_fraction_linear_tail_high_percent_all_days"] = city_period["pa_loss_linear_tail_high_sum_days"] / n * 100.0
    city_period["benefit_adjusted_asri_percent"] = np.maximum(0.0, city_period["net_log_suppression_sum_days"] / n) * 100.0
    city_period["benefit_adjusted_asri_low_percent"] = np.maximum(0.0, city_period["net_log_suppression_low_sum_days"] / n) * 100.0
    city_period["benefit_adjusted_asri_high_percent"] = np.maximum(0.0, city_period["net_log_suppression_high_sum_days"] / n) * 100.0
    city_period = add_denominator_columns(city_period)
    city_period.to_csv(out_dir / "future_heat_pa_risk_city_period_trial.csv", index=False)

    national_period = (
        city_period.groupby(["scenario", "period", "model_key", "indicator", "heatwave_type", "activity_type"], as_index=False)[NATIONAL_MEAN_COLS]
        .mean(numeric_only=True)
    )
    national_period["curve_scope"] = national_year["curve_scope"].iloc[0]
    national_period["partition_family"] = national_year["partition_family"].iloc[0]
    national_period["partition_name"] = "All_cities"
    national_period = add_denominator_columns(national_period)
    national_period.to_csv(out_dir / "future_heat_pa_risk_national_period_trial.csv", index=False)

    clipping = city_period[
        [
            "scenario",
            "city_standard",
            "period",
            "model_key",
            "indicator",
            "heatwave_type",
            "activity_type",
            "curve_scope",
            "partition_family",
            "partition_name",
            "exposure_clipped_share",
            "pa_loss_fraction_percent_all_days",
            "pa_loss_fraction_linear_tail_percent_all_days",
            "denominator_id",
            "denominator_label",
            "denominator_months",
        ]
    ]
    clipping.to_csv(out_dir / "future_exposure_clipping_diagnostics_city_period.csv", index=False)

    primary = national_period[
        [
            "scenario",
            "period",
            "model_key",
            "indicator",
            "heatwave_type",
            "activity_type",
            "curve_scope",
            "partition_family",
            "partition_name",
            "pa_loss_fraction_percent_all_days",
            "pa_loss_fraction_low_percent_all_days",
            "pa_loss_fraction_high_percent_all_days",
            "annualized_asri_percent",
            "annualized_asri_low_percent",
            "annualized_asri_high_percent",
            "benefit_adjusted_asri_percent",
            "benefit_adjusted_asri_low_percent",
            "benefit_adjusted_asri_high_percent",
            "exposure_clipped_share",
            "denominator_id",
            "denominator_label",
            "denominator_months",
        ]
    ]
    primary.to_csv(out_dir / "future_primary_pa_loss_fraction_summary.csv", index=False)


def write_notes(scope: str, out_dir: Path, cmip_files: int, curve_meta: pd.DataFrame) -> None:
    notes = [
        "Future PA heat-risk projection, fast Python implementation.",
        f"CMIP root: {CMIP_ROOT}",
        f"Stage-2 root: {STAGE2_ROOT}",
        f"Grid map: {GRID_MAP_PATH}",
        f"Future years: {FUTURE_START_YEAR}-{FUTURE_END_YEAR}",
        f"Denominator ID: {DENOMINATOR_ID}",
        f"Denominator label: {DENOMINATOR_LABEL}",
        f"Denominator months: {sorted(WARM_SEASON_MONTHS)}",
        "All denominator days are included; non-heatwave days contribute zero through zero exposure / RR reference.",
        f"Package scope: {scope}",
        f"Projection curve scope: {SCOPES[scope]['curve_scope']}",
        f"Projection activity types: {sorted(ACTIVITY_FILTER)}",
        f"Processed CMIP files: {cmip_files}",
        f"Historical curves used: {len(curve_meta)}",
        "Main projection: national pooled curve.",
        "External checks: zone and region pooled curves.",
        "Sensitivity checks: city-specific and DTW4lag12 phenotype curves.",
    ]
    (out_dir / "future_projection_run_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(GRID_MAP_PATH, dtype={"grid10_id": str})
    grid = grid.drop_duplicates("grid10_id")
    grid["city_key"] = grid["city_standard"].map(normalize_city)
    grid_lookup = grid.set_index("grid10_id")[["city", "city_standard", "grid10_lon", "grid10_lat", "city_key"]].to_dict("index")

    all_curves, meta_by_scope, all_meta = load_curves_by_scope()
    for scope, meta in meta_by_scope.items():
        out_dir = OUT_ROOT / scope
        out_dir.mkdir(parents=True, exist_ok=True)
        meta.to_csv(out_dir / "historical_pooled_curve_inventory.csv", index=False)
    all_meta.to_csv(OUT_ROOT / "historical_pooled_curve_inventory_all_scopes.csv", index=False)

    first_file = next((CMIP_ROOT / "ssp245").glob("*.csv"))
    time_info = build_time_template(first_file)
    required_cols = [
        "Composite_CEHWI",
        "Daytime_CEHWI",
        "Nighttime_CEHWI",
        "Composite_exceeded_quantity",
        "Daytime_exceeded_quantity",
        "Nighttime_exceeded_quantity",
        "Composite_heatwave",
        "Daytime_heatwave",
        "Nighttime_heatwave",
    ]
    mask = time_info["mask"]
    years_idx = time_info["year_to_idx"]

    acc_by_scope: dict[str, defaultdict] = {s: defaultdict(empty_accumulator) for s in SCOPES}
    file_count = 0

    for sdir in sorted([p for p in CMIP_ROOT.iterdir() if p.is_dir() and p.name in {"ssp245", "ssp370", "ssp585"}]):
        scenario = sdir.name
        files = sorted(sdir.glob("*.csv"))
        if not RUN_FULL_CMIP and len(files) > MAX_FILES_PER_SCENARIO:
            matched_files = [f for f in files if re.sub(r"_temperature.*$", "", f.name) in grid_lookup]
            if matched_files:
                files = matched_files
            rng = np.random.default_rng(20260506)
            sample_size = min(MAX_FILES_PER_SCENARIO, len(files))
            files = sorted(rng.choice(np.array(files, dtype=object), size=sample_size, replace=False).tolist())
        msg(f"Scenario {scenario}: processing {len(files)} grid files once for all scopes.")
        for idx, path in enumerate(files, start=1):
            grid_id = re.sub(r"_temperature.*$", "", path.name)
            gm = grid_lookup.get(grid_id)
            if gm is None:
                continue
            try:
                dat = pd.read_csv(path, usecols=required_cols)
            except ValueError:
                dat = pd.read_csv(path)
            if len(dat) != time_info["template_len"]:
                d = pd.to_datetime(pd.read_csv(path, usecols=["Time"])["Time"], errors="coerce")
                local_mask = (d.dt.year >= FUTURE_START_YEAR) & (d.dt.year <= FUTURE_END_YEAR) & d.dt.month.isin(WARM_SEASON_MONTHS)
                years = d.dt.year[local_mask].to_numpy()
                local_years_idx = {int(y): np.where(years == y)[0] for y in np.unique(years)}
                sub = dat.loc[local_mask.to_numpy(), required_cols]
            else:
                local_years_idx = years_idx
                sub = dat.loc[mask, required_cols]
            cache = {col: pd.to_numeric(sub[col], errors="coerce").to_numpy(float) for col in required_cols}

            for scope, meta in meta_by_scope.items():
                city_key = gm["city_key"]
                applicable = meta[(meta["city_key"].eq("")) | (meta["city_key"].eq(city_key))]
                if applicable.empty:
                    continue
                curves = all_curves[scope]
                for _, m in applicable.iterrows():
                    xcol = exposure_col_for(m["indicator"], m["heatwave_type"])
                    hcol = flag_col_for(m["heatwave_type"])
                    exposure = cache[xcol]
                    hw_flag = cache[hcol] > 0
                    rr, rr_low, rr_high, clipped = predict_rr(curves[m["curve_id"]], exposure, linear_tail=False)
                    rr_tail, rr_tail_low, rr_tail_high, _ = predict_rr(curves[m["curve_id"]], exposure, linear_tail=True)
                    key = (
                        scenario,
                        gm["city_standard"],
                        m["model_key"],
                        m["indicator"],
                        m["heatwave_type"],
                        m["activity_type"],
                        m["curve_scope"],
                        m["partition_family"],
                        m["partition_name"],
                    )
                    aggregate_grid_year(
                        acc_by_scope[scope],
                        key,
                        local_years_idx,
                        exposure,
                        hw_flag,
                        rr,
                        rr_low,
                        rr_high,
                        rr_tail,
                        rr_tail_low,
                        rr_tail_high,
                        clipped,
                    )
            file_count += 1
            if idx % 500 == 0:
                msg(f"  {scenario}: {idx}/{len(files)} files processed.")

    manifest = []
    for scope, acc in acc_by_scope.items():
        out_dir = OUT_ROOT / scope
        msg(f"Writing outputs for {scope} with {len(acc)} city-year keys.")
        city_year = accumulator_to_city_year(scope, acc)
        derive_outputs(scope, city_year, out_dir)
        write_notes(scope, out_dir, file_count, meta_by_scope[scope])
        manifest.append(
            {
                "scope": scope,
                "output_dir": str(out_dir),
                "city_year_rows": int(len(city_year)),
                "city_period_rows": int(len(pd.read_csv(out_dir / "future_heat_pa_risk_city_period_trial.csv"))),
                "national_year_rows": int(len(pd.read_csv(out_dir / "future_heat_pa_risk_national_year_trial.csv"))),
            }
        )
    (OUT_ROOT / "fast_projection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    msg("All fast projection scopes completed.")


if __name__ == "__main__":
    main()
