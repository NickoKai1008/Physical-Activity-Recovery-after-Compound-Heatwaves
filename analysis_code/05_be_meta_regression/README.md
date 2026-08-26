# Built-environment meta-regression

## Purpose

Relates between-city variation in reduced exposure-response coefficients to Mean and Gini summaries of eight prespecified contextual predictors: NDVI, population aged 20-55 years, GDP, unemployment, building density, urbanization rate, intersection density and walkability.

## Retained code

| Script | Role | Principal input | Principal output |
|---|---|---|---|
| `code/fit_be_meta_regression.R` | Core model: fits the Mean and Gini multivariate meta-regressions for the eight prespecified contextual predictors. | City reduced coefficients, covariance matrices and Mean/Gini predictor tables | Predictor tests, model summaries and conditional response estimates |
| `code/make_figure5_ab.ipynb` | Formal panel redraw: renders the Mean and Gini predictor forests and their reporting tables. | `data/figure5_ab/` | Figure 5a-b panels, editable SVG and source CSV |
| `code/make_figure5_c.ipynb` | Formal panel redraw: visualizes the reduction in between-city heterogeneity. | `data/figure5_c/fig5c_plot_data.csv` | Figure 5c PNG and editable SVG |
| `code/make_figure5_d.ipynb` | Formal panel redraw: reconstructs the percentile-by-lag response surface with contours and significance markers. | `data/figure5_d/pooled_lag_response_p25-p95.csv` | Figure 5d PNG, editable SVG and interpolated surface CSV |

The confidential Figshare record documents the spatial predictor construction and diagnostics.

## Inputs

Compact model summaries, contextual predictor tables and plotted lag-response profiles are organized under `data/`.

## Outputs

Mean/Gini forests, heterogeneity reduction and the lag-response surface are under `output/`. `figure5d_lag_response_surface.png` and `.svg` include the white dashed response contours, black zero-effect contour and 90% CI-excludes-null markers used in the submitted panel.

## Reproduction mode

Direct panel redraw from the deposited tables. `sensitivity_common_coordinate_reprojection/` expresses each city response at CEHWI 2, 4 and 6 relative to 0 and repeats the Mean/Gini models as a coordinate-system sensitivity. Every retained file is a core model or a submitted panel redraw. Canonical file identities are recorded in `../../docs/data_manifest.csv`.
