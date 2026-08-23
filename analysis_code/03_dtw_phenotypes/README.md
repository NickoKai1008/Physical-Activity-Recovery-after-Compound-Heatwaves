# DTW response phenotypes

## Purpose

Clusters standardized PPML lag-response trajectories using DTW with the archived Ward k=4 specification and derives phenotype-specific lag-duration assignments.

## Retained code

| Script | Role | Principal input | Principal output |
|---|---|---|---|
| `code/cluster_dtw_ward_k4.py` | Core model: standardizes city PPML trajectories and applies the archived DTW, Ward-linkage and k=4 phenotype specification. | Module 02 12-day city-lag estimates | City phenotype assignments, cluster diagnostics and standardized trajectories |
| `code/assign_cluster_lag_duration.py` | Core decision calculation: converts phenotype-level lag evidence into the C3 8-day and C1/C2/C4 12-day DLNM windows. | Cluster assignments and lag-significance summaries | City and phenotype lag-window tables |
| `code/make_lowess_lag_decision_figure.py` | Formal methods redraw: visualizes LOWESS/FDR lag-duration evidence and the final phenotype lag assignments. | Lag-significance and assignment tables | Main/Extended Data lag-decision panels in PNG and SVG |
| `code/make_phenotype_map.ipynb` | Formal panel redraw: maps the four response phenotypes across the 75-city frame. | City coordinates and phenotype assignments | Phenotype map panel in PNG and editable SVG |
| `code/make_phenotype_scatter.ipynb` | Formal panel redraw: plots temporal-response shape features and phenotype membership. | Standardized shape-feature table | Feature-space scatter panel and plotted CSV |
| `code/make_phenotype_trajectories.ipynb` | Formal panel redraw: renders cluster trajectories, dendrogram-related summaries and phenotype profiles. | PPML trajectories and final assignments | Trajectory panels, cluster tables and editable SVG |

## Inputs

The PPML lag-response table is referenced from Module 02. Canonical assignments and lag decisions are stored under `data/results/`.

## Outputs

Phenotype assignments, lag-window decisions, trajectories and figure panels are written to `output/`.

## Reproduction mode

Direct reproduction of clustering and lag-decision tables, with panel-level reproduction of the submitted Main Figure 3 composition. Every retained file is either a core calculation or a formal panel redraw. Canonical file identities are recorded in `../../docs/data_manifest.csv`.
