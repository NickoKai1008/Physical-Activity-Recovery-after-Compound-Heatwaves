#!/usr/bin/env python
"""Re-estimate phenotype mean trajectories after excluding three cities."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "input"
OUTPUT = ROOT / "output" / "csv"

EXCLUDED_CITIES = {"Clearwater", "Gilbert", "Palm Bay"}
EXPECTED_PRIMARY_COUNTS = {1: 18, 2: 14, 3: 11, 4: 20}
EXPECTED_EXCLUSION_COUNTS = {1: 18, 2: 14, 3: 11, 4: 17}


def standardize_rows(values: np.ndarray) -> np.ndarray:
    standardized = np.zeros_like(values, dtype=float)
    for index, row in enumerate(values):
        standard_deviation = float(np.std(row))
        if standard_deviation > 1e-6:
            standardized[index] = (row - np.mean(row)) / standard_deviation
    return np.nan_to_num(standardized)


def summarize_profiles(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for cluster in range(1, 5):
        members = labels.index[labels == cluster]
        values = features.loc[members].to_numpy(float)
        mean = values.mean(axis=0)
        standard_error = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        for position, lag in enumerate(features.columns):
            rows.append(
                {
                    "cluster": cluster,
                    "lag": int(lag),
                    "n_cities": values.shape[0],
                    "mean_standardized_response": mean[position],
                    "ci_low": mean[position] - 1.96 * standard_error[position],
                    "ci_high": mean[position] + 1.96 * standard_error[position],
                }
            )
    return pd.DataFrame(rows)


def profile_metrics(primary: pd.DataFrame, exclusion: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for cluster in range(1, 5):
        primary_values = primary.loc[
            primary["cluster"] == cluster, "mean_standardized_response"
        ].to_numpy(float)
        exclusion_values = exclusion.loc[
            exclusion["cluster"] == cluster, "mean_standardized_response"
        ].to_numpy(float)
        difference = exclusion_values - primary_values
        correlation = (
            1.0
            if np.allclose(primary_values, exclusion_values)
            else float(np.corrcoef(primary_values, exclusion_values)[0, 1])
        )
        rows.append(
            {
                "cluster": cluster,
                "n_primary": int(
                    primary.loc[primary["cluster"] == cluster, "n_cities"].iloc[0]
                ),
                "n_exclusion": int(
                    exclusion.loc[exclusion["cluster"] == cluster, "n_cities"].iloc[0]
                ),
                "pearson_r": correlation,
                "max_abs_standardized_difference": float(np.max(np.abs(difference))),
                "rmse_standardized_difference": float(np.sqrt(np.mean(difference**2))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    estimates = pd.read_csv(INPUT / "all_cities_eventlag_resultspost12.csv")
    canonical = pd.read_csv(
        INPUT / "canonical_city_cluster_optimized_12d_ward_k4.csv"
    )
    estimates = estimates.dropna(subset=["estimate"])

    beta = estimates.pivot(index="city", columns="lag", values="estimate").sort_index()
    p_value = estimates.pivot(index="city", columns="lag", values="p.value").reindex(
        index=beta.index, columns=beta.columns
    )
    if beta.shape != (63, 12) or p_value.isna().any().any():
        raise RuntimeError("Expected a complete 63-city by 12-lag PPML table")

    filtered = np.where(
        p_value.to_numpy(float) < 0.05,
        beta.to_numpy(float),
        0.0,
    )
    features = pd.DataFrame(
        standardize_rows(filtered),
        index=beta.index,
        columns=beta.columns,
    )
    labels = canonical.set_index("city")["cluster"].reindex(features.index)
    if labels.isna().any():
        raise RuntimeError("Canonical phenotype labels are missing for one or more cities")
    labels = labels.astype(int)

    primary_counts = labels.value_counts().sort_index().to_dict()
    if primary_counts != EXPECTED_PRIMARY_COUNTS:
        raise RuntimeError(f"Unexpected primary cluster sizes: {primary_counts}")
    if not EXCLUDED_CITIES.issubset(features.index):
        raise RuntimeError("One or more single-grid cities are absent from the PPML table")
    if not (labels.loc[list(EXCLUDED_CITIES)] == 4).all():
        raise RuntimeError("All three excluded cities must belong to C4")

    retained_index = features.index[~features.index.isin(EXCLUDED_CITIES)]
    retained_features = features.loc[retained_index]
    retained_labels = labels.loc[retained_index]
    exclusion_counts = retained_labels.value_counts().sort_index().to_dict()
    if exclusion_counts != EXPECTED_EXCLUSION_COUNTS:
        raise RuntimeError(f"Unexpected exclusion cluster sizes: {exclusion_counts}")

    primary_profiles = summarize_profiles(features, labels)
    exclusion_profiles = summarize_profiles(retained_features, retained_labels)
    metrics = profile_metrics(primary_profiles, exclusion_profiles)

    city_profiles = features.copy()
    city_profiles.insert(0, "cluster", labels)
    city_profiles.index.name = "city"
    city_profiles.to_csv(OUTPUT / "city_standardized_post_event_profiles.csv")
    primary_profiles.to_csv(
        OUTPUT / "phenotype_profiles_primary_63.csv", index=False
    )
    exclusion_profiles.to_csv(
        OUTPUT / "phenotype_profiles_exclusion_60.csv", index=False
    )
    metrics.to_csv(OUTPUT / "phenotype_profile_influence_metrics.csv", index=False)

    c4 = metrics.loc[metrics["cluster"] == 4].iloc[0]
    print(
        "C4 profile: "
        f"n={int(c4['n_primary'])} to {int(c4['n_exclusion'])}; "
        f"r={c4['pearson_r']:.3f}; "
        f"max |difference|={c4['max_abs_standardized_difference']:.3f}; "
        f"RMSE={c4['rmse_standardized_difference']:.3f}"
    )


if __name__ == "__main__":
    main()
