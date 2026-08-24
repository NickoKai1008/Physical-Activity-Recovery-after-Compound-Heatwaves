# Future heatwave-attributable PA-loss projection

## Purpose

Combines historical response functions with 2025-2050 CMIP heatwave exposure under SSP2-4.5, SSP3-7.0 and SSP5-8.5. Outputs include maps, annual trajectories, city rankings, response-scope comparisons and holdout validation for the 63 cities with estimable compound-heatwave response phenotypes.

## Retained code

| Script | Role | Principal input | Principal output |
|---|---|---|---|
| `code/project_future_pa_loss_all_scopes.py` | Core projection: combines historical response functions with provider-processed CMIP grid-day exposure under national, DTW, climate-zone, regional and city-specific scopes. | Historical response tables, CMIP grid-day tables and city-grid mapping | Annual and period-specific PA-loss summaries for 2025-2050 |
| `code/make_figure6_maps.py` | Formal panel redraw: renders the three SSP city maps with a common colour and point-size classification. | `data/maps/city_map_2025_2050_plot_data.csv`; phenotype map | Figure 6a-c PNG and editable SVG |
| `code/make_figure6_ad.py` | Formal trajectory and validation redraw: creates Figure 6d and the deposited validation diagnostics; it also supplies shared mapping utilities used by `make_figure6_maps.py`. | Annual time series and validation tables under `data/` | Figure 6d trajectory and validation PNG/SVG files |
| `code/make_figure6e.py` | Formal panel redraw: ranks city-level 2025-2050 projected loss under SSP5-8.5. | `data/figure6e/` | Figure 6e PNG and editable SVG |
| `code/make_figure6f.py` | Formal panel redraw: compares response-function scopes for compound, daytime and nighttime heatwaves. | `data/figure6f/` | Figure 6f PNG and editable SVG |

## Inputs

Compact annual, map, city-forest, response-scope and validation tables are organized under `data/`. The confidential Figshare record supplies the 75-city spatial reference used for map auditing.

## Outputs

Figure 6 panels and validation diagnostics are under `output/`.

## Reproduction mode

Direct or panel-level redraw from the compact projection tables. The archived calculation entry defines the expected scenario-level climate and historical-response inputs. The map workflow selects the 63 DTW-phenotyped cities used by the manuscript projection. Every retained file is a core projection or a formal panel redraw. Canonical file identities are recorded in `../../docs/data_manifest.csv`.
