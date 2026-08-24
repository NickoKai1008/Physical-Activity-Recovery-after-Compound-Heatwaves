# DLNM exposure-response functions

## Purpose

Estimates city-specific nonlinear exposure-response functions using positive-exposure p50/p90 knots and phenotype-specific 8/12-day lag windows. Each cumulative curve is reconstructed in its city-specific basis, then log-relative risks are synthesized pointwise at common exposure values. A common-support shared-basis analysis provides the supplementary coordinate-system sensitivity.

## Primary workflow

The primary model and synthesis are:

```bash
Rscript analysis_code/04_dlnm_response/code/fit_city_dlnm_primary_p50_p90.R
Rscript analysis_code/04_dlnm_response/code/pool_city_specific_p50_p90_curves_pointwise.R
```

The primary model entry uses the 2010-2024 period, All + Cycling + Running + Walking outcomes and the phenotype-specific group-median lag assignment (C3: lag 0-7; C1/C2/C4: lag 0-11). The synthesis entry audits the 189 city/type fits, reconstructs each curve in its native p50/p90 basis and applies pointwise random-effects REML pooling.

Primary pointwise 95% confidence intervals are the normal-theory REML intervals; AF limits are the monotonic transformation of the log-RR limits. The common-basis sensitivity uses 500 multivariate-normal coefficient draws.

## Retained core calculations

| Script | Role | Principal output |
|---|---|---|
| `code/fit_city_dlnm_primary_p50_p90.R` | Fits the city-specific primary CEHWI DLNMs with native positive-exposure p50/p90 knots and phenotype-specific 8/12-day lag windows. | City model objects, reduced coefficients and city exposure-response tables |
| `code/pool_city_specific_p50_p90_curves_pointwise.R` | Reconstructs each city curve in its native basis and performs pointwise random-effects REML synthesis on common exposure values. | Primary pooled curves, dose-specific RR/AF nodes and model audit |

## Retained formal figure code

| Script | Formal use | Principal input | Principal output |
|---|---|---|---|
| `code/make_figure4_panels.py` | Main Figure 4 and its support-range companion | `data/main_figure_input/` | `output/main_panels/` PNG and SVG |
| `code/make_city_dlnm_lag_atlas.R` | Supplementary Figures 15-18 | `data/lag_window/` | Phenotype-arranged city DLNM atlases |
| `code/make_exact_common_basis_fig4.py` | Supplementary Figure 20 | `data/common_basis/cehwi/` | Common-basis phenotype RR/AF and tail-sensitivity panels |
| `code/make_integrated_common_basis_figure.py` | Extended Data Figure 6 | `data/common_basis/` | Integrated phenotype, AF and clustering-robustness figure |
| `code/make_partition_robustness_figure.py` | Extended Data Figure 6 and Supplementary Figure 21 | `data/common_basis/` | Partition and clustering-method robustness panels |
| `code/make_national_lag_window_comparison.py` | Extended Data Figure 7 | `data/lag_window/` | National fixed-7, fixed-12 and dynamic-8/12 comparison |
| `code/make_exceeded_quantity_figures.py` | Supplementary Figure 19 | `data/main_figure_input/`; `data/common_basis/exceeded_quantity/` | Exceeded-quantity RR/AF panels in PNG and SVG |

## Inputs

Primary pointwise-pooled curves and AF nodes are under `data/main_figure_input/`. The weighted exposure-support table used by the model-level synthesis is under `data/model_reporting_input/`. Common-basis and fixed-lag sensitivity inputs are organized under `data/common_basis/` and `data/lag_window/`.

## Outputs

Main, lag-window, common-basis and exceeded-quantity panels are under `output/`.

## Reproduction mode

Direct redraw from the compact primary and sensitivity tables. The two retained R scripts provide the canonical primary model and synthesis workflow; the seven figure scripts reproduce the submitted Figure 4, Extended Data and Supplementary panels. Canonical file identities are recorded in `../../docs/data_manifest.csv`.
