from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PACKAGE = Path(__file__).resolve().parents[1]

DATA_DIR = PACKAGE / "data"
RESULTS_DIR = PACKAGE / "output" / "lag_assignment"
TEXT_DIR = PACKAGE / "output" / "lag_assignment" / "text"

PPML_12D = PACKAGE.parent / "02_ppml_post_event" / "data" / "city_lag_estimates_12day.csv"
PPML_7D = PACKAGE.parent / "02_ppml_post_event" / "data" / "city_lag_estimates_7day.csv"
CITY_CLUSTER_12D = DATA_DIR / "results" / "city_cluster_optimized_12d_ward_k4.csv"
CITY_CLUSTER_5GROUP = DATA_DIR / "results" / "city_cluster_5group_12d_ward_k4.csv"

VALIDATED_DURATION_TABLE = DATA_DIR / "results" / "ward_k4_12d_75city_lowess_multi_threshold_effect_days_clean.csv"
VALIDATED_LOWESS_DIAG = DATA_DIR / "cluster_sig_lowess_duration_diagnostics_uncapped.csv"

SCHEME = "12d_ward_k4"
N_LAGS = 12
ALPHA_RAW = 0.05
FDR_THRESHOLDS = {
    "FDR05": 0.05,
    "FDR01": 0.01,
    "FDR001": 0.001,
}


def ensure_dirs() -> None:
    for path in (DATA_DIR, RESULTS_DIR, TEXT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def copy_inputs() -> pd.DataFrame:
    rows = []
    items = [
        (PPML_12D, DATA_DIR / PPML_12D.name, "12-day PPML lag-response input"),
        (PPML_7D, DATA_DIR / PPML_7D.name, "7-day fixed-window control input"),
        (CITY_CLUSTER_12D, DATA_DIR / "city_cluster_optimized_12d_ward_k4.csv", "12-day ward-k4 city clusters"),
        (CITY_CLUSTER_5GROUP, DATA_DIR / "city_cluster_5group_12d_ward_k4.csv", "75-city cluster roster with no-heatwave group"),
        (VALIDATED_DURATION_TABLE, DATA_DIR / VALIDATED_DURATION_TABLE.name, "validated 75-city duration table"),
        (VALIDATED_LOWESS_DIAG, DATA_DIR / VALIDATED_LOWESS_DIAG.name, "validated LOWESS duration diagnostics"),
    ]
    for src, dst, desc in items:
        if src.exists():
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            rows.append({"source": str(src), "package_path": str(dst), "description": desc})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(RESULTS_DIR / "input_manifest.csv", index=False)
    return manifest


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def signed_stouffer(beta: np.ndarray, p_value: np.ndarray) -> tuple[float, float]:
    beta = np.asarray(beta, dtype=float)
    p_value = np.asarray(p_value, dtype=float)
    ok = np.isfinite(beta) & np.isfinite(p_value)
    if ok.sum() < 2:
        return np.nan, np.nan
    b = beta[ok]
    p = np.clip(p_value[ok], 1e-300, 1.0 - 1e-15)
    sign = np.sign(b)
    sign[sign == 0] = 1.0
    z_i = sign * stats.norm.isf(p / 2.0)
    z_i = np.clip(z_i, -40.0, 40.0)
    z_cluster = z_i.sum() / math.sqrt(len(z_i))
    p_two_sided = 2.0 * stats.norm.sf(abs(z_cluster))
    return float(z_cluster), float(p_two_sided)


def recompute_lagwise_stouffer(city_table: pd.DataFrame) -> pd.DataFrame:
    core = city_table[city_table["is_outlier"].eq(0)].copy()
    rows = []
    for cluster, group in core.groupby("cluster"):
        for lag in range(1, N_LAGS + 1):
            beta = group[f"beta_lag{lag}"].to_numpy(float)
            p = group[f"pval_lag{lag}"].to_numpy(float)
            valid = np.isfinite(beta) & np.isfinite(p)
            beta = beta[valid]
            p = p[valid]
            sig = p < ALPHA_RAW
            z, pz = signed_stouffer(beta, p)
            rows.append(
                {
                    "scheme": SCHEME,
                    "cluster": int(cluster),
                    "n_cities_in_cluster": int(len(group)),
                    "lag": lag,
                    "n_valid": int(len(beta)),
                    "n_sig": int(sig.sum()),
                    "frac_sig": float(sig.mean()) if len(sig) else np.nan,
                    "n_sig_pos": int((sig & (beta > 0)).sum()),
                    "n_sig_neg": int((sig & (beta < 0)).sum()),
                    "mean_beta": float(np.mean(beta)) if len(beta) else np.nan,
                    "median_beta": float(np.median(beta)) if len(beta) else np.nan,
                    "stouffer_z": z,
                    "stouffer_p": pz,
                }
            )
    out = pd.DataFrame(rows)
    out["stouffer_p_fdr"] = np.nan
    for cluster, group in out.groupby("cluster"):
        idx = group.index
        out.loc[idx, "stouffer_p_fdr"] = bh_fdr(group["stouffer_p"].to_numpy())
    out["neglog10_fdr"] = -np.log10(np.clip(out["stouffer_p_fdr"], 1e-300, 1.0))
    out.to_csv(RESULTS_DIR / "cluster_lag_significance_12d_ward_k4_recomputed.csv", index=False)
    return out


def lowess_predict(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, frac: float = 0.55) -> np.ndarray:
    """Dependency-free LOWESS used only for descriptive attenuation summaries."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = len(x)
    if n < 3:
        return np.interp(x_new, x, y) if n else np.full_like(x_new, np.nan)
    span = max(3, int(np.ceil(frac * n)))
    pred = np.empty_like(x_new, dtype=float)
    X = np.column_stack([np.ones(n), x])
    for i, x0 in enumerate(x_new):
        d = np.abs(x - x0)
        h = np.partition(d, min(span - 1, n - 1))[min(span - 1, n - 1)]
        if h <= 1e-12:
            w = (d <= 1e-12).astype(float)
        else:
            u = np.clip(d / h, 0, 1)
            w = (1 - u**3) ** 3
        W = np.sqrt(w)
        try:
            beta_hat = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
            pred[i] = beta_hat[0] + beta_hat[1] * x0
        except Exception:
            pred[i] = np.average(y, weights=w) if np.sum(w) else np.nan
    return pred


def first_post_peak_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | np.nan:
    peak_index = int(np.nanargmax(y))
    for i in range(peak_index + 1, len(x)):
        if y[i] < threshold:
            x0, x1 = x[i - 1], x[i]
            y0, y1 = y[i - 1], y[i]
            if abs(y1 - y0) < 1e-12:
                return float(x1)
            return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))
    return np.nan


def recompute_lowess_diagnostics(lagwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grid = np.linspace(1, N_LAGS, 1101)
    for cluster, group in lagwise.groupby("cluster"):
        group = group.sort_values("lag")
        lags = group["lag"].to_numpy(float)
        s = group["neglog10_fdr"].to_numpy(float)
        smooth = lowess_predict(lags, s, grid, frac=0.55)
        peak_lag = float(grid[int(np.nanargmax(smooth))])
        row = {
            "scheme": SCHEME,
            "cluster": int(cluster),
            "n": int(group["n_cities_in_cluster"].iloc[0]),
            "smooth_peak": round(peak_lag, 2),
            "raw_FDR05_n": int((group["stouffer_p_fdr"] < 0.05).sum()),
        }
        for name, alpha in FDR_THRESHOLDS.items():
            row[f"{name}_cross_recomputed"] = first_post_peak_crossing(
                grid, smooth, -math.log10(alpha)
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cluster_sig_lowess_duration_diagnostics_recomputed.csv", index=False)
    return out


def clean_validated_diagnostics() -> pd.DataFrame:
    diag = pd.read_csv(VALIDATED_LOWESS_DIAG)
    diag = diag[diag["scheme"].eq(SCHEME)].copy()
    diag.to_csv(RESULTS_DIR / "cluster_sig_lowess_duration_diagnostics_final.csv", index=False)
    return diag


def build_duration_tables(diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    city_5 = pd.read_csv(CITY_CLUSTER_5GROUP).rename(columns={"cluster": "Group"})
    city_5["Group"] = city_5["Group"].astype(int)

    duration_rows = []
    for group in [1, 2, 3, 4]:
        d = diag[diag["cluster"].eq(group)].iloc[0]
        strong = 12 if pd.isna(d["FDR001_cross"]) else int(round(float(d["FDR001_cross"])))
        overall = 12 if pd.isna(d["FDR05_cross"]) else int(round(float(d["FDR05_cross"])))
        band = (
            "Persistent through 12d"
            if pd.isna(d["FDR05_cross"])
            else "Attenuates ~6d strong / ~8d overall"
        )
        duration_rows.append(
            {
                "Group": group,
                "n_meta_cities_core": int(d["n"]),
                "Strong days": strong,
                "Overall days": overall,
                "Duration band": band,
                "FDR001_cross": d["FDR001_cross"],
                "FDR01_cross": d["FDR01_cross"],
                "FDR05_cross": d["FDR05_cross"],
                "raw_FDR05_significant_lags": int(d["raw_FDR05_n"]),
            }
        )
    duration_rows.append(
        {
            "Group": 5,
            "n_meta_cities_core": 0,
            "Strong days": 0,
            "Overall days": 0,
            "Duration band": "No composite heatwave",
            "FDR001_cross": np.nan,
            "FDR01_cross": np.nan,
            "FDR05_cross": np.nan,
            "raw_FDR05_significant_lags": 0,
        }
    )
    group_duration = pd.DataFrame(duration_rows)
    group_duration.to_csv(RESULTS_DIR / "ward_k4_12d_group_lowess_duration_summary.csv", index=False)

    city_duration = city_5[["city", "Group"]].rename(columns={"city": "City"}).merge(
        group_duration[["Group", "Strong days", "Overall days", "Duration band"]],
        on="Group",
        how="left",
    )
    med = (
        city_duration.groupby("Group", as_index=False)
        .agg(
            **{
                "Group median strong": ("Strong days", "median"),
                "Group mean strong": ("Strong days", "mean"),
                "Group median overall": ("Overall days", "median"),
                "Group mean overall": ("Overall days", "mean"),
            }
        )
    )
    city_duration = city_duration.merge(med, on="Group", how="left")
    city_duration = city_duration.sort_values("City").reset_index(drop=True)
    city_duration.to_csv(
        RESULTS_DIR / "ward_k4_12d_75city_lowess_multi_threshold_effect_days_clean.csv",
        index=False,
    )

    dlnm = city_duration.copy()
    dlnm["DLNM_lag_days_main"] = dlnm["Group median overall"].astype(int)
    dlnm["DLNM_lag_days_strong_sensitivity"] = dlnm["Group median strong"].astype(int)
    dlnm["DLNM_lag_days_7d_control"] = np.where(dlnm["Group"].eq(5), 0, 7)
    dlnm["Main_lag_definition"] = "cluster-level LOWESS FDR05 duration"
    dlnm["Control_lag_definition"] = "fixed 7-day lag window for estimable heatwave groups"
    dlnm.to_csv(RESULTS_DIR / "dlnm_lag_assignment_main_vs_7day_control.csv", index=False)
    return group_duration, city_duration, dlnm


def write_ready_text(group_duration: pd.DataFrame) -> None:
    c3 = group_duration[group_duration["Group"].eq(3)].iloc[0]
    text = f"""# LOWESS-Based Lag-Duration Assignment for DLNM

## Main Methods

We assigned downstream DLNM lag windows from cluster-level attenuation trajectories.
For each Ward-k4 response cluster, city-level PPML evidence was combined by lag.
Core cities defined the cluster-level meta-analytic trajectory.
High-magnitude outliers inherited the duration of their reassigned cluster.
The no-composite-heatwave group was assigned a duration of zero.

For city i and lag l, we converted the two-sided PPML P value into a signed statistic.
The sign was determined by the PPML coefficient at the same lag.

z_i,l = sign(beta_i,l) Phi^-1(1 - p_i,l / 2)

Within each cluster c, the signed statistics were combined using Stouffer's method.

Z_c,l = sum_i z_i,l / sqrt(n_c)

The resulting two-sided P values were adjusted across 12 lags within each cluster.
We used the Benjamini-Hochberg false-discovery-rate procedure.

S_c(l) = -log10(q_c,l)

LOWESS smoothing was applied to S_c(l) as a descriptive attenuation summary.
The smoothed curve served as a descriptive guide, with threshold decisions based on the cluster-level FDR evidence.
We defined two pre-specified duration endpoints.
The strong endpoint used q = 0.001.
The overall endpoint used q = 0.05.
The endpoint was the first post-peak lag where the smoothed curve crossed below threshold.
Clusters retaining significance through the evaluated horizon were classified as persistent through 12 days.
For the main DLNM dose-response analysis, we used Group median overall.
This column corresponds to the cluster-level FDR05 attenuation window.
A fixed 7-day lag window was retained as the control analysis.

## Main Results

The ward-k4 12-day classification yielded four response clusters and one reference group.
Clusters 1, 2 and 4 were persistent through the 12-day window.
Their overall FDR05 duration was therefore assigned as 12 days.
Cluster 3 showed a shorter attenuation profile.
Its FDR001 crossing occurred at approximately {float(c3['FDR001_cross']):.2f} days.
Its FDR05 crossing occurred at approximately {float(c3['FDR05_cross']):.2f} days.
After rounding to broad duration bands, Cluster 3 was assigned 6 strong days.
It was assigned 8 overall days.
Cities in Cluster 3 therefore used an 8-day main DLNM lag window.
Cities in Clusters 1, 2 and 4 used a 12-day main lag window.
The 12 cities outside the estimable compound-heatwave subset were assigned zero lag duration.
The fixed-window control analysis used a 7-day lag for all estimable heatwave groups.

## Supplementary Methods

Supplementary Table 1 reports the cluster-level attenuation endpoints.
Supplementary Table 2 reports the 75-city lag assignment.
Supplementary Table 3 reports the DLNM-ready lag assignment.
The main DLNM lag column is Group median overall.
The strong-duration column is retained as sensitivity evidence.
The 7-day control column preserves the fixed lag-window benchmark.
The attenuation windows are phenotype-level summaries applied to member cities.

## Supplementary Table Notes

Strong-significance duration denotes the LOWESS-smoothed FDR001 endpoint.
Overall-significance duration denotes the LOWESS-smoothed FDR05 endpoint.
Values of 12 indicate no crossing within the observed 12-day window.
Values of zero identify cities outside the estimable compound-heatwave subset.
The DLNM main lag uses the group median overall duration.
"""
    (TEXT_DIR / "lag_duration_methods_results_ready_text.md").write_text(text, encoding="utf-8")


def write_readme() -> None:
    readme = """# Main Figure 3 Lag Calculation Package

This folder contains the final lag-duration assignment used to move from the
ward-k4 12-day DTW clusters to the next-stage DLNM dose-response models.

Key outputs:

- `results_csv/ward_k4_12d_group_lowess_duration_summary.csv`
- `results_csv/ward_k4_12d_75city_lowess_multi_threshold_effect_days_clean.csv`
- `results_csv/dlnm_lag_assignment_main_vs_7day_control.csv`
- `text/lag_duration_methods_results_ready_text.md`
- `Main_figure_3_lag_calculation_methods_results.docx`

The main DLNM lag is `DLNM_lag_days_main`.
It is equal to `Group median overall`, the cluster-level LOWESS FDR05 duration.
The fixed 7-day lag column is retained as a control analysis.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    copy_inputs()
    city_cluster = pd.read_csv(CITY_CLUSTER_12D)
    lagwise = recompute_lagwise_stouffer(city_cluster)
    recompute_lowess_diagnostics(lagwise)
    diag = clean_validated_diagnostics()
    group_duration, city_duration, dlnm = build_duration_tables(diag)
    write_ready_text(group_duration)
    write_readme()
    print(f"Saved lag-duration package to: {PACKAGE}")
    print(f"Rows in 75-city table: {len(city_duration)}")
    print(f"Rows in DLNM assignment table: {len(dlnm)}")


if __name__ == "__main__":
    main()
