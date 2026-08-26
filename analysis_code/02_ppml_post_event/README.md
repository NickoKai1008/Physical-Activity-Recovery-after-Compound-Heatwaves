# PPML post-event response

## Purpose

Estimates city-specific changes in physical activity during calendar days 1-12 after a compound heatwave event. A fixed 7-day window provides the lag-window sensitivity analysis.

## Retained code

| Script | Role | Principal input | Principal output |
|---|---|---|---|
| `code/fit_city_ppml.R` | Core model: fits each city's grid fixed-effects PPML model for post-event days 1-12. | Model-ready city-grid-day panel | City-lag coefficients, standard errors and confidence intervals |
| `code/make_figure2_panels.py` | Formal panel redraw: creates the pooled lag-response, lag-duration tally and city-by-lag heatmap, plus the fixed 7-day sensitivity panels. | `data/city_lag_estimates_12day.csv`; `data/city_lag_estimates_7day.csv` | `output/lag12_panels/`; `output/lag7_panels/` |
| `code/make_main_figure2.py` | Main-figure assembly: composes the submitted Figure 2 structure from the formal 12-day panels. | `output/lag12_panels/` | `output/main_figure/main_figure_02_reproduced.png`; editable SVG |
| `code/make_city_lag_atlas.py` | Formal supplementary redraw: renders all 75 fixed city positions, with 63 estimable trajectories and 12 empty positions. | `data/city_atlas/` | `output/city_atlas/fig2_ppml_city_lag_atlas_5x15.png`; editable SVG |

## Inputs

`data/city_lag_estimates_12day.csv` contains the 756 primary city-lag estimates. `data/city_lag_estimates_7day.csv` contains the fixed-window sensitivity estimates.

## Outputs

The complete Main Figure 2 redraw is written to `output/main_figure/`. `output/lag12_panels/` contains Fig. 2a (`fig2_pooled_lag_response`), Fig. 2b-c (`fig5_lag_tally_distribution`) and Fig. 2d (`fig3_heatmap_city_lag`) in PNG and editable SVG. The 7-day sensitivity set is under `output/lag7_panels/`. The 75-city atlas is supplied in PNG and editable SVG.

## Reproduction mode

Direct redraw from the deposited result tables. The model archive defines the PPML estimator and input schema used to produce those tables. Canonical file identities are recorded in `../../docs/data_manifest.csv`.
