"""
This script implements the heatwave definitions used in the manuscript:

* A grid- and year-specific threshold is the 95th percentile of all valid observations from 1985 through the year preceding the target year.
* The target year is excluded from its own threshold calculation.
* Tmax and Tmin thresholds are calculated separately and bounded below at 24 degrees Celsius.
* A daytime heatwave (DHW), nighttime heatwave (NHW), or compound heatwave (CHW) requires at least three consecutive days of the corresponding mutually exclusive daily condition.
* Daily exceeded quantity (EQ) and cumulative excess heatwave index (CEHWI) follow the equations reported in the manuscript.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NONE = "NONE"
DHW = "DHW"
NHW = "NHW"
CHW = "CHW"
EVENT_TYPES = (DHW, NHW, CHW)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Identify independently qualified daytime and nighttime heatwaves, "
            "convert their overlapping days to compound heatwaves, and calculate "
            "EQ and CEHWI for each grid cell."
        )
    )
    parser.add_argument(
        "--grid-list",
        type=Path,
        help="CSV containing the grid-cell identifiers.",
    )
    parser.add_argument(
        "--id-column",
        default="FID_USA_fi",
        help="Identifier column in --grid-list (default: FID_USA_fi).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing the daily Tmax and Tmin CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for one standardized heatwave CSV per grid cell.",
    )
    parser.add_argument(
        "--tmax-template",
        default="{id}_temperature_max.csv",
        help="Tmax filename template (default: {id}_temperature_max.csv).",
    )
    parser.add_argument(
        "--tmin-template",
        default="{id}_temperature_min.csv",
        help="Tmin filename template (default: {id}_temperature_min.csv).",
    )
    parser.add_argument(
        "--temperature-column",
        default="temp_temperature__c",
        help="Temperature column in both input files.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="First target year (default: 2010).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Last target year, inclusive (default: 2024).",
    )
    parser.add_argument(
        "--history-start-year",
        type=int,
        default=1985,
        help="First year in the expanding threshold record (default: 1985).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.95,
        help="Relative threshold quantile (default: 0.95).",
    )
    parser.add_argument(
        "--absolute-floor-c",
        type=float,
        default=24.0,
        help="Lower bound applied to both thresholds in degrees C (default: 24).",
    )
    parser.add_argument(
        "--minimum-duration",
        type=int,
        default=3,
        help=(
            "Minimum consecutive duration applied independently to daytime and "
            "nighttime heatwaves; no additional duration filter is applied to "
            "compound overlap days (default: 3)."
        ),
    )
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help=(
            "Fail when the daily record does not span the complete period from "
            "1 January of --history-start-year through 31 December of --end-year."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in synthetic validation tests and exit.",
    )
    return parser.parse_args()


def load_daily_temperature(
    path: Path,
    temperature_column: str,
    output_name: str,
) -> pd.Series:
    """Load one daily temperature series from a CSV with dates in column 1."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_csv(path, index_col=0)
    if temperature_column not in data.columns:
        raise KeyError(
            f"Column '{temperature_column}' is absent from {path}. "
            f"Available columns: {list(data.columns)}"
        )

    dates = pd.to_datetime(data.index, errors="coerce").normalize()
    if dates.isna().any():
        bad_count = int(dates.isna().sum())
        raise ValueError(f"{path} contains {bad_count} invalid date value(s).")

    values = pd.to_numeric(data[temperature_column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name=output_name).sort_index()

    if series.index.duplicated().any():
        duplicates = series.index[series.index.duplicated()].unique()
        raise ValueError(
            f"{path} contains duplicate dates; first duplicate: {duplicates[0].date()}"
        )
    return series


def validate_coverage(
    data: pd.DataFrame,
    history_start_year: int,
    end_year: int,
    strict: bool,
) -> None:
    """Validate the date extent and optionally require complete calendar coverage."""
    required_start = pd.Timestamp(history_start_year, 1, 1)
    required_end = pd.Timestamp(end_year, 12, 31)

    if data.index.min() > required_start or data.index.max() < required_end:
        raise ValueError(
            "The input record does not span the required threshold and target "
            f"period ({required_start.date()} to {required_end.date()})."
        )

    if strict:
        expected = pd.date_range(required_start, required_end, freq="D")
        observed = data.loc[required_start:required_end].index
        missing_dates = expected.difference(observed)
        if len(missing_dates):
            raise ValueError(
                f"Strict coverage check failed: {len(missing_dates)} calendar "
                f"date(s) are missing; first missing date: {missing_dates[0].date()}."
            )


def add_expanding_thresholds(
    data: pd.DataFrame,
    start_year: int = 2010,
    end_year: int = 2024,
    history_start_year: int = 1985,
    percentile: float = 0.95,
    absolute_floor_c: float = 24.0,
) -> pd.DataFrame:
    """Add fixed annual Tmax and Tmin thresholds based on preceding years."""
    if not 0.0 < percentile < 1.0:
        raise ValueError("percentile must be between 0 and 1.")
    if start_year <= history_start_year:
        raise ValueError("start_year must be later than history_start_year.")
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year.")

    result = data.copy()
    result["Tmax_threshold_95_C"] = np.nan
    result["Tmin_threshold_95_C"] = np.nan

    for year in range(start_year, end_year + 1):
        history_mask = (
            (result.index.year >= history_start_year)
            & (result.index.year < year)
        )
        target_mask = result.index.year == year

        historical_tmax = result.loc[history_mask, "Tmax_C"].dropna()
        historical_tmin = result.loc[history_mask, "Tmin_C"].dropna()
        if historical_tmax.empty or historical_tmin.empty:
            raise ValueError(
                f"No valid historical observations are available before {year}."
            )

        tmax_threshold = max(
            float(historical_tmax.quantile(percentile)), absolute_floor_c
        )
        tmin_threshold = max(
            float(historical_tmin.quantile(percentile)), absolute_floor_c
        )
        result.loc[target_mask, "Tmax_threshold_95_C"] = tmax_threshold
        result.loc[target_mask, "Tmin_threshold_95_C"] = tmin_threshold

    return result


def _consecutive_run_metadata(labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return a run identifier and run length, breaking runs at date gaps."""
    date_gap = labels.index.to_series().diff().ne(pd.Timedelta(days=1))
    label_change = labels.ne(labels.shift())
    run_id = (date_gap | label_change).cumsum()
    run_length = labels.groupby(run_id).transform("size")
    return run_id, run_length


def identify_heatwaves(
    data: pd.DataFrame,
    minimum_duration: int = 3,
) -> pd.DataFrame:
    """Classify mutually exclusive events and calculate EQ and CEHWI.

    Tmax and Tmin exceedance runs are qualified independently. 
    Overlap between the two qualified series is then converted to CHW.
    The final DHW, NHW and CHW indicators are mutually exclusive.
    """
    if minimum_duration < 1:
        raise ValueError("minimum_duration")

    required = {
        "Tmax_C",
        "Tmin_C",
        "Tmax_threshold_95_C",
        "Tmin_threshold_95_C",
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {sorted(missing)}")

    result = data.copy()
    valid = result[list(required)].notna().all(axis=1)
    result["hot_daytime"] = valid & (
        result["Tmax_C"] >= result["Tmax_threshold_95_C"]
    )
    result["hot_nighttime"] = valid & (
        result["Tmin_C"] >= result["Tmin_threshold_95_C"]
    )

    # Record the unfiltered daily threshold-exceedance state for quality control.
    raw_type = pd.Series(NONE, index=result.index, dtype="object")
    raw_type.loc[result["hot_daytime"] & ~result["hot_nighttime"]] = DHW
    raw_type.loc[~result["hot_daytime"] & result["hot_nighttime"]] = NHW
    raw_type.loc[result["hot_daytime"] & result["hot_nighttime"]] = CHW
    result["daily_heat_state"] = raw_type

    # Apply the duration rule independently to Tmax and Tmin exceedance runs.
    # Date gaps interrupt a run and therefore cannot be counted as consecutive.
    _, daytime_run_length = _consecutive_run_metadata(result["hot_daytime"])
    _, nighttime_run_length = _consecutive_run_metadata(result["hot_nighttime"])
    daytime_heatwave = result["hot_daytime"] & daytime_run_length.ge(
        minimum_duration
    )
    nighttime_heatwave = result["hot_nighttime"] & nighttime_run_length.ge(
        minimum_duration
    )
    result["daytime_heatwave_base"] = daytime_heatwave.astype("int8")
    result["nighttime_heatwave_base"] = nighttime_heatwave.astype("int8")

    # Convert overlapping qualified days to CHW. No independent duration rule
    # is imposed on CHW, so even a one-day overlap is retained as compound heat.
    compound_heatwave = daytime_heatwave & nighttime_heatwave
    heatwave_type = pd.Series(NONE, index=result.index, dtype="object")
    heatwave_type.loc[daytime_heatwave & ~compound_heatwave] = DHW
    heatwave_type.loc[nighttime_heatwave & ~compound_heatwave] = NHW
    heatwave_type.loc[compound_heatwave] = CHW
    result["heatwave_type"] = heatwave_type

    for event_type in EVENT_TYPES:
        is_event = result["heatwave_type"].eq(event_type)
        event_start = is_event & (
            ~is_event.shift(fill_value=False)
            | result.index.to_series().diff().ne(pd.Timedelta(days=1))
        )
        event_id = event_start.cumsum().where(is_event).astype("Int64")
        result[event_type] = is_event.astype("int8")
        result[f"{event_type}_event_id"] = event_id
        result[f"{event_type}_event_day"] = (
            result.loc[is_event].groupby(event_id[is_event]).cumcount().add(1)
        ).reindex(result.index).astype("Int64")

    # Equation (1): daily exceeded quantity for each mutually exclusive event type.
    result["EQ_DHW_C"] = np.where(
        result[DHW].eq(1),
        result["Tmax_C"] - result["Tmax_threshold_95_C"],
        0.0,
    )
    result["EQ_NHW_C"] = np.where(
        result[NHW].eq(1),
        result["Tmin_C"] - result["Tmin_threshold_95_C"],
        0.0,
    )
    result["EQ_CHW_C"] = np.where(
        result[CHW].eq(1),
        (
            (result["Tmax_C"] - result["Tmax_threshold_95_C"])
            + (result["Tmin_C"] - result["Tmin_threshold_95_C"])
        )
        / 2.0,
        0.0,
    )

    # CEHWI accumulates EQ within each continuous event and resets afterwards.
    for event_type in EVENT_TYPES:
        event_id_column = f"{event_type}_event_id"
        eq_column = f"EQ_{event_type}_C"
        cehwi_column = f"CEHWI_{event_type}_C_days"
        result[cehwi_column] = 0.0
        event_mask = result[event_type].eq(1)
        result.loc[event_mask, cehwi_column] = (
            result.loc[event_mask]
            .groupby(event_id_column, sort=False)[eq_column]
            .cumsum()
        )

    return result


def process_grid(
    grid_id: str,
    input_dir: Path,
    output_dir: Path,
    tmax_template: str,
    tmin_template: str,
    temperature_column: str,
    start_year: int,
    end_year: int,
    history_start_year: int,
    percentile: float,
    absolute_floor_c: float,
    minimum_duration: int,
    strict_coverage: bool,
) -> Path:
    """Process one grid cell and write its standardized daily output."""
    tmax_path = input_dir / tmax_template.format(id=grid_id)
    tmin_path = input_dir / tmin_template.format(id=grid_id)

    tmax = load_daily_temperature(tmax_path, temperature_column, "Tmax_C")
    tmin = load_daily_temperature(tmin_path, temperature_column, "Tmin_C")
    daily = pd.concat([tmax, tmin], axis=1, join="outer").sort_index()
    validate_coverage(daily, history_start_year, end_year, strict_coverage)

    daily = add_expanding_thresholds(
        daily,
        start_year=start_year,
        end_year=end_year,
        history_start_year=history_start_year,
        percentile=percentile,
        absolute_floor_c=absolute_floor_c,
    )
    target = daily.loc[
        (daily.index.year >= start_year) & (daily.index.year <= end_year)
    ].copy()
    result = identify_heatwaves(target, minimum_duration=minimum_duration)
    result.index.name = "date"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{grid_id}_temperature_heatwaves.csv"
    result.to_csv(output_path, index=True, date_format="%Y-%m-%d")
    return output_path


def _normalise_grid_ids(values: Iterable[object]) -> list[str]:
    """Convert numeric-looking identifiers to stable filename strings."""
    identifiers: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, (int, np.integer)):
            identifiers.append(str(int(value)))
        elif isinstance(value, (float, np.floating)) and float(value).is_integer():
            identifiers.append(str(int(value)))
        else:
            identifiers.append(str(value).strip())
    return identifiers


def run_self_test() -> None:
    """Validate thresholds, duration rules, overlap conversion and accumulation."""
    dates = pd.date_range("1985-01-01", "2010-01-15", freq="D")
    synthetic = pd.DataFrame({"Tmax_C": 20.0, "Tmin_C": 10.0}, index=dates)

    # The historical 95th percentiles are below 24 C, so the absolute floor applies.
    # In 2010: three DHW days, three CHW days and a two-day NHW spell are inserted.
    synthetic.loc["2010-01-01":"2010-01-03", ["Tmax_C", "Tmin_C"]] = [26.0, 20.0]
    synthetic.loc["2010-01-05":"2010-01-07", ["Tmax_C", "Tmin_C"]] = [28.0, 26.0]
    synthetic.loc["2010-01-09":"2010-01-10", ["Tmax_C", "Tmin_C"]] = [20.0, 27.0]

    thresholded = add_expanding_thresholds(synthetic, end_year=2010)
    result = identify_heatwaves(thresholded.loc["2010"], minimum_duration=3)

    assert (result["Tmax_threshold_95_C"] == 24.0).all()
    assert (result["Tmin_threshold_95_C"] == 24.0).all()
    assert result[DHW].sum() == 3
    assert result[CHW].sum() == 3
    assert result[NHW].sum() == 0
    assert np.isclose(result.loc["2010-01-03", "CEHWI_DHW_C_days"], 6.0)
    assert np.isclose(result.loc["2010-01-07", "CEHWI_CHW_C_days"], 9.0)

    # Tmax is hot on days 1-3 and Tmin is hot on days 3-5. Both source events
    # satisfy the three-day rule independently, but their one-day overlap on
    # day 3 must still be classified as CHW.
    one_day_overlap = pd.DataFrame(
        {
            "Tmax_C": [26.0, 26.0, 26.0, 20.0, 20.0],
            "Tmin_C": [20.0, 20.0, 26.0, 26.0, 26.0],
            "Tmax_threshold_95_C": [24.0] * 5,
            "Tmin_threshold_95_C": [24.0] * 5,
        },
        index=pd.date_range("2010-07-01", periods=5, freq="D"),
    )
    overlap_result = identify_heatwaves(one_day_overlap, minimum_duration=3)
    assert overlap_result["daytime_heatwave_base"].sum() == 3
    assert overlap_result["nighttime_heatwave_base"].sum() == 3
    assert overlap_result[DHW].sum() == 2
    assert overlap_result[CHW].sum() == 1
    assert overlap_result[NHW].sum() == 2
    assert overlap_result.loc["2010-07-03", "heatwave_type"] == CHW
    assert overlap_result.loc["2010-07-03", "CHW_event_day"] == 1
    assert np.isclose(overlap_result.loc["2010-07-03", "EQ_CHW_C"], 2.0)
    assert np.isclose(
        overlap_result.loc["2010-07-03", "CEHWI_CHW_C_days"], 2.0
    )

    # A simultaneous hot day that is not part of qualified Tmax and Tmin runs
    # must not be classified as CHW merely because both thresholds are exceeded.
    isolated = one_day_overlap.copy()
    isolated[["Tmax_C", "Tmin_C"]] = 20.0
    isolated.loc["2010-07-03", ["Tmax_C", "Tmin_C"]] = 26.0
    isolated_result = identify_heatwaves(isolated, minimum_duration=3)
    assert isolated_result[list(EVENT_TYPES)].to_numpy().sum() == 0

    print("All synthetic heatwave-identification tests passed.")


def main() -> None:
    """Run the batch workflow."""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    missing_paths = [
        option
        for option, value in (
            ("--grid-list", args.grid_list),
            ("--input-dir", args.input_dir),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing_paths:
        raise SystemExit(
            "The following arguments are required for batch processing: "
            + ", ".join(missing_paths)
        )

    grid_table = pd.read_csv(args.grid_list)
    if args.id_column not in grid_table.columns:
        raise KeyError(
            f"Column '{args.id_column}' is absent from {args.grid_list}. "
            f"Available columns: {list(grid_table.columns)}"
        )
    grid_ids = _normalise_grid_ids(grid_table[args.id_column])
    if not grid_ids:
        raise ValueError("No valid grid-cell identifiers were found.")

    failures: list[tuple[str, str]] = []
    for position, grid_id in enumerate(grid_ids, start=1):
        try:
            output_path = process_grid(
                grid_id=grid_id,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                tmax_template=args.tmax_template,
                tmin_template=args.tmin_template,
                temperature_column=args.temperature_column,
                start_year=args.start_year,
                end_year=args.end_year,
                history_start_year=args.history_start_year,
                percentile=args.percentile,
                absolute_floor_c=args.absolute_floor_c,
                minimum_duration=args.minimum_duration,
                strict_coverage=args.strict_coverage,
            )
            print(f"[{position}/{len(grid_ids)}] Wrote {output_path}")
        except Exception as exc:  # Continue processing other grid cells.
            failures.append((grid_id, str(exc)))
            print(f"[{position}/{len(grid_ids)}] FAILED grid {grid_id}: {exc}")

    if failures:
        failure_path = args.output_dir / "failed_grid_cells.csv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures, columns=[args.id_column, "error"]).to_csv(
            failure_path, index=False
        )
        raise RuntimeError(
            f"{len(failures)} grid cell(s) failed. See {failure_path} for details."
        )


if __name__ == "__main__":
    main()
