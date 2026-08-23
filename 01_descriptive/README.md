# Descriptive coverage and heatwave inputs

## Purpose

Constructs heatwave indicators and descriptive summaries for the 75-city analytical frame.

## Retained code

| Script | Role | Principal input | Principal output |
|---|---|---|---|
| `code/identify_heatwaves.py` | Core exposure calculation: identifies daytime, nighttime and compound heatwave events from daily city-grid temperature series. | Daily city-grid temperature records | Heatwave-day and event classifications |
| `code/summarize_heatwave_factors.py` | Core descriptive calculation: derives exceeded quantity, CEHWI and event-level summaries used by Figure 1 and its supporting displays. | Heatwave classifications and activity summaries | Descriptive heatwave and activity tables |

## Inputs

The scripts define the required schemas for daily city-grid temperature and physical-activity records. The confidential Figshare record supplies the 75-city spatial reference and an Austin linkage example.

## Outputs

Submitted Figure 1, Extended Data Figures 1-2 and Supplementary Figures 1-3 are stored under `../../figures/` as reference exhibits.

## Reproduction mode

Method archive and submitted-reference mapping. Both retained scripts are core calculations. Canonical file identities are recorded in `../../docs/data_manifest.csv`.
