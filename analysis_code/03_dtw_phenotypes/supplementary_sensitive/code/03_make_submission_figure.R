#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else getwd()
root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
csv_dir <- file.path(root, "output", "csv")
figure_dir <- file.path(root, "output", "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

read_metric <- function(path, name) {
  table <- read.csv(path, stringsAsFactors = FALSE)
  as.numeric(table$value[table$metric == name])
}

prepare_curves <- function(scope, primary_path, exclusion_path, metrics_path) {
  primary <- read.csv(primary_path, stringsAsFactors = FALSE)
  exclusion <- read.csv(exclusion_path, stringsAsFactors = FALSE)
  p95 <- read_metric(metrics_path, "positive_exposure_p95")
  p99 <- read_metric(metrics_path, "positive_exposure_p99")
  primary$scope <- scope
  exclusion$scope <- scope
  list(
    curves = rbind(primary, exclusion),
    p95 = p95,
    p99 = p99,
    metrics_path = metrics_path
  )
}

national <- prepare_curves(
  "National",
  file.path(csv_dir, "national_pooled_curve_primary_n63.csv"),
  file.path(csv_dir, "national_pooled_curve_excluding_single_grid_n60.csv"),
  file.path(csv_dir, "national_pooled_curve_sensitivity_metrics.csv")
)
c4 <- prepare_curves(
  "C4",
  file.path(csv_dir, "c4_pooled_curve_primary_n20.csv"),
  file.path(csv_dir, "c4_pooled_curve_excluding_single_grid_n17.csv"),
  file.path(csv_dir, "c4_pooled_curve_sensitivity_metrics.csv")
)

national_r <- read_metric(national$metrics_path, "pearson_log_rr_within_p95")
national_max <- read_metric(national$metrics_path, "max_abs_log_rr_difference_within_p95")
c4_max <- read_metric(c4$metrics_path, "max_abs_log_rr_difference_within_p95")
c4_rmse <- read_metric(c4$metrics_path, "rmse_log_rr_difference_within_p95")

theme_submission <- function() {
  theme_classic(base_size = 12.5, base_family = "Times New Roman") +
    theme(
      legend.position = "top",
      legend.justification = "left",
      legend.margin = margin(0, 0, 2, 0),
      legend.key.width = grid::unit(1.15, "cm"),
      legend.text = element_text(size = 11.5),
      axis.title = element_text(size = 13),
      axis.text = element_text(size = 11.5),
      axis.line = element_line(linewidth = 0.45, colour = "#24282C"),
      axis.ticks = element_line(linewidth = 0.4, colour = "#24282C"),
      panel.grid.major.y = element_line(linewidth = 0.3, colour = "#E4E6E8"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(6, 8, 6, 8)
    )
}

plot_scope <- function(data, p95, p99, colours, line_types) {
  data <- data[data$cehwi <= p99, ]
  ggplot(data, aes(cehwi, log_rr, colour = specification, fill = specification)) +
    annotate("rect", xmin = p95, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "#F2F3F4", alpha = 0.75) +
    geom_ribbon(aes(ymin = log_rr_low, ymax = log_rr_high), alpha = 0.11, colour = NA) +
    geom_hline(yintercept = 0, linewidth = 0.35, linetype = "dashed", colour = "#73777B") +
    geom_vline(xintercept = p95, linewidth = 0.45, linetype = "dotted", colour = "#6D7277") +
    geom_line(aes(linetype = specification), linewidth = 0.9) +
    annotate(
      "text", x = p95, y = Inf, label = "95th percentile support",
      vjust = 1.4, hjust = 1.03, size = 3.7, colour = "#5B6065",
      family = "Times New Roman"
    ) +
    scale_colour_manual(values = colours) +
    scale_fill_manual(values = colours) +
    scale_linetype_manual(values = line_types) +
    coord_cartesian(xlim = c(0, p99)) +
    labs(
      x = "CEHWI",
      y = "log-RR",
      colour = NULL,
      fill = NULL,
      linetype = NULL
    ) +
    theme_submission()
}

national_colours <- c(
  "Primary national pool (n=63)" = "#30343B",
  "Excluding single-grid cities (n=60)" = "#4C8EBA"
)
national_lines <- c(
  "Primary national pool (n=63)" = "solid",
  "Excluding single-grid cities (n=60)" = "longdash"
)
c4_colours <- c(
  "Primary C4 (n=20)" = "#234F8C",
  "Excluding single-grid cities (n=17)" = "#86B9CF"
)
c4_lines <- c(
  "Primary C4 (n=20)" = "solid",
  "Excluding single-grid cities (n=17)" = "longdash"
)

p_a <- plot_scope(
  national$curves,
  national$p95,
  national$p99,
  national_colours,
  national_lines
)
p_b <- plot_scope(
  c4$curves,
  c4$p95,
  c4$p99,
  c4_colours,
  c4_lines
)

make_delta <- function(scope, comparison_path, p95, primary_col, exclusion_col) {
  data <- read.csv(comparison_path, stringsAsFactors = FALSE)
  data <- data[data$cehwi <= p95, ]
  data.frame(
    scope = scope,
    support_percent = 100 * data$cehwi / p95,
    delta_log_rr = data[[exclusion_col]] - data[[primary_col]],
    stringsAsFactors = FALSE
  )
}

delta_data <- rbind(
  make_delta(
    "National pool",
    file.path(csv_dir, "national_pooled_curve_comparison.csv"),
    national$p95,
    "log_rr_n63",
    "log_rr_n60"
  ),
  make_delta(
    "C4 pool",
    file.path(csv_dir, "c4_pooled_curve_comparison.csv"),
    c4$p95,
    "log_rr_n20",
    "log_rr_n17"
  )
)
write.csv(national$curves, file.path(csv_dir, "figure_panel_a_national_curves.csv"), row.names = FALSE)
write.csv(c4$curves, file.path(csv_dir, "figure_panel_b_c4_curves.csv"), row.names = FALSE)
write.csv(delta_data, file.path(csv_dir, "figure_panel_c_curve_differences.csv"), row.names = FALSE)

p_c <- ggplot(delta_data, aes(support_percent, delta_log_rr, colour = scope)) +
  geom_hline(yintercept = 0, linewidth = 0.4, colour = "#777B7F") +
  geom_line(linewidth = 1.0) +
  scale_colour_manual(values = c("National pool" = "#A04D51", "C4 pool" = "#234F8C")) +
  coord_cartesian(xlim = c(0, 100), ylim = c(-0.50, 0.50)) +
  scale_x_continuous(breaks = c(0, 25, 50, 75, 100)) +
  labs(
    x = "Position within p95 CEHWI support (%)",
    y = expression(Delta * " log-RR (exclusion minus primary)"),
    colour = NULL
  ) +
  theme_submission() +
  theme(legend.position = "top")

phenotype_primary <- read.csv(
  file.path(csv_dir, "phenotype_profiles_primary_63.csv"),
  stringsAsFactors = FALSE
)
phenotype_exclusion <- read.csv(
  file.path(csv_dir, "phenotype_profiles_exclusion_60.csv"),
  stringsAsFactors = FALSE
)
phenotype_primary$series <- "Primary"
phenotype_exclusion$series <- "Exclusion"
phenotype_data <- rbind(phenotype_primary, phenotype_exclusion)
phenotype_data$cluster_label <- factor(
  paste0("C", phenotype_data$cluster),
  levels = c("C1", "C2", "C3", "C4")
)
phenotype_data$series <- factor(
  phenotype_data$series,
  levels = c("Primary", "Exclusion")
)
write.csv(phenotype_data, file.path(csv_dir, "figure_panel_d_phenotype_profiles.csv"), row.names = FALSE)
phenotype_colours <- c(
  "C1" = "#A04D51",
  "C2" = "#E39B05",
  "C3" = "#4C8EBA",
  "C4" = "#234F8C"
)

p_d <- ggplot(
  phenotype_data,
  aes(
    lag,
    mean_standardized_response,
    colour = cluster_label,
    fill = cluster_label,
    linetype = series,
    group = interaction(cluster_label, series)
  )
) +
  geom_hline(yintercept = 0, linewidth = 0.35, linetype = "dashed", colour = "#777B7F") +
  geom_ribbon(
    data = phenotype_primary,
    aes(
      lag,
      ymin = ci_low,
      ymax = ci_high,
      fill = factor(paste0("C", cluster), levels = c("C1", "C2", "C3", "C4")),
      group = cluster
    ),
    inherit.aes = FALSE,
    alpha = 0.08,
    colour = NA
  ) +
  geom_line(linewidth = 1.0) +
  scale_colour_manual(values = phenotype_colours) +
  scale_fill_manual(values = phenotype_colours) +
  scale_linetype_manual(values = c("Primary" = "solid", "Exclusion" = "22")) +
  scale_x_continuous(breaks = c(1, 4, 8, 12), limits = c(1, 12)) +
  scale_y_continuous(breaks = seq(-3, 3, by = 1), limits = c(-3, 3)) +
  labs(
    x = "Post-event lag (days)",
    y = "Mean standardized response",
    colour = NULL,
    fill = NULL,
    linetype = NULL
  ) +
  theme_submission() +
  guides(
    fill = "none",
    colour = guide_legend(order = 1, nrow = 1),
    linetype = guide_legend(order = 2, nrow = 1)
  ) +
  theme(
    legend.position = "top",
    legend.box = "vertical",
    legend.box.just = "left"
  )

combined <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(heights = c(1.45, 1)) +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.tag = element_text(
      family = "Times New Roman", face = "bold", size = 19, colour = "black"
    ),
    plot.tag.position = c(0, 1),
    plot.margin = margin(8, 8, 8, 8)
  )

ggsave(
  file.path(figure_dir, "single_grid_exclusion_robustness.png"),
  combined,
  width = 12.0,
  height = 8.2,
  dpi = 500,
  bg = "white"
)
ggsave(
  file.path(figure_dir, "single_grid_exclusion_robustness.svg"),
  combined,
  width = 12.0,
  height = 8.2,
  bg = "white"
)
ggsave(
  file.path(figure_dir, "single_grid_exclusion_robustness.pdf"),
  combined,
  width = 12.0,
  height = 8.2,
  device = grDevices::cairo_pdf,
  bg = "white"
)

cat("Submission combined figure written to: ", figure_dir, "\n", sep = "")
