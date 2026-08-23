# Leave-three-out influence analysis for single-grid cities

This package reproduces the sensitivity analysis that excludes Clearwater, Gilbert and Palm Bay from the 63-city post-heatwave phenotype analysis. Each city contributed one analytical grid. Their complete 12-day point-estimate trajectories are retained in the primary analysis, while lag-level cluster-robust significance is not interpreted because uncertainty cannot be reliably estimated from a single cluster unit.

The sensitivity analysis holds the primary C1-C4 assignments of the remaining 60 cities fixed and recalculates:

1. the four phenotype mean trajectories;
2. the national pooled CEHWI-physical-activity response; and
3. the C4 pooled response, which changes from 20 to 17 cities.

This is an influence analysis of the reported phenotype profiles and pooled response estimates. It does not alter the primary 63-city partition or any remaining city model.

## Directory structure

```text
code/
  01_leave_three_out_influence_analysis.py
  02_pooled_curve_exclusion_sensitivity.R
  03_make_submission_figure.R
data/input/
  all_cities_eventlag_resultspost12.csv
  canonical_city_cluster_optimized_12d_ward_k4.csv
  c4_archived_pooled_RR_data.csv
  national_archived_pooled_RR_curve.csv
data/model_objects/
  c4_cehwi_composite_all/
  national_cehwi_composite_all/
environment/
  requirements-python.txt
  requirements-r.txt
output/csv/
  city_standardized_post_event_profiles.csv
  phenotype_profiles_primary_63.csv
  phenotype_profiles_exclusion_60.csv
  phenotype_profile_influence_metrics.csv
  national_pooled_curve_*.csv
  c4_pooled_curve_*.csv
  figure_panel_*.csv
output/figures/
  single_grid_exclusion_robustness.png
  single_grid_exclusion_robustness.svg
  single_grid_exclusion_robustness.pdf
```

## Inputs and estimands

`all_cities_eventlag_resultspost12.csv` contains the submitted 63-city PPML lag estimates. Non-significant city-lag coefficients are set to zero and each 12-day city trajectory is centred and standardized, matching the primary phenotype workflow. `canonical_city_cluster_optimized_12d_ward_k4.csv` supplies the primary C1-C4 assignments.

The pooled response analysis reads the archived city-level DLNM coefficient vectors and covariance matrices. The same REML estimator is applied before and after exclusion. Comparisons are evaluated within the 95th percentile of each positive CEHWI exposure distribution.

## Main results

- C1-C3 phenotype means are unchanged because all three excluded cities belong to C4.
- The C4 mean trajectory remains highly concordant after exclusion: Pearson `r=0.962`, maximum absolute standardized difference `0.245`, and RMSE `0.107`.
- The national pooled curves remain highly concordant: Pearson `r=0.990` and maximum absolute change in log-RR `0.011` within the 95th-percentile exposure support.
- The C4 pooled response changes modestly: maximum absolute change in log-RR `0.101` and RMSE `0.065` within the same support criterion.

## Reproduction

From the package root, run:

The three analysis steps separately:

```powershell
python code\01_leave_three_out_influence_analysis.py
Rscript code\02_pooled_curve_exclusion_sensitivity.R
Rscript code\03_make_submission_figure.R
```
