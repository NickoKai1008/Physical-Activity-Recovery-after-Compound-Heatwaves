# Figure 5 sensitivity: common exposure-coordinate reprojection

## Role in the analysis

The parent directory contains the submitted Figure 5 analysis of three reduced natural-spline coefficients from each city model. This sensitivity re-expresses each city-specific response on the same three CEHWI contrasts, then repeats the Mean and Gini meta-regressions with the same city set, predictors and grid-row design.

## Reprojection

For city `c`, let `beta_c` and `V_beta,c` denote the native reduced spline coefficient vector and covariance matrix. Evaluating the city-specific natural-spline basis at CEHWI 0, 2, 4 and 6 and subtracting the CEHWI 0 row gives transformation matrix `C_c`:

```text
theta_c = C_c beta_c
V_theta,c = C_c V_beta,c C_c^T.
```

The three elements of `theta_c` represent log relative risk at CEHWI 2, 4 and 6 relative to CEHWI 0 in every city. Joint predictor evidence uses a three-degree-of-freedom Wald test across these coordinates. Forest points summarize the equally weighted mean of the three coordinate-specific predictor effects; intervals use the transformed covariance matrix.

## Contents

- `code/fit_be_meta_regression_common_coordinate_sensitivity.R`: reprojection, covariance propagation and Mean/Gini models for all activity, cycling, running and walking.
- `code/make_figure5_ab_common_coordinate_sensitivity.py`: sensitivity forest panels.
- `data/`: native city coefficients, covariance matrices, basis definitions and grid-row contextual predictors.
- `output/`: transformed parameters, model tests, comparison tables and PNG/SVG figures.
- `data/data_manifest.csv` and `output/output_manifest.csv`: row counts, file sizes and SHA-256 checksums.

## Run

```powershell
Rscript code/fit_be_meta_regression_common_coordinate_sensitivity.R
python code/make_figure5_ab_common_coordinate_sensitivity.py
```

Required R packages are `dlnm` and `mvmeta`; Python requires `pandas`, `numpy` and `matplotlib`.

## Verified reproduction

The deposited workflow produces 252 city-activity models, 756 common-coordinate log-RR values, 128 representation-specific joint-test rows and 32 plotted rows in each Mean/Gini forest panel. Six of the 16 all-activity predictor tests changed their nominal 0.05 significance classification between the native-coordinate and common-coordinate representations.

## Interpretation

This sensitivity isolates the coordinate representation of city response functions. City models, membership, activity definitions, contextual predictors and the grid-row design follow the primary analysis.
