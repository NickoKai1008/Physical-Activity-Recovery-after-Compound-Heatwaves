suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(readr)
  library(ggplot2)
  library(scales)
  library(dlnm)
  library(MASS)
})

args <- commandArgs(trailingOnly = TRUE)
package_root <- if (length(args) >= 1) normalizePath(args[[1]], winslash = "/", mustWork = TRUE) else {
  normalizePath(file.path(getwd(), "CITY_DLNM_LAG_COMPARISON_20260801"), winslash = "/", mustWork = TRUE)
}

model_root <- file.path(package_root, "model_rds")
input_dir <- file.path(package_root, "data", "input")
derived_dir <- file.path(package_root, "data", "derived")
figure_dir <- file.path(package_root, "figures")
table_dir <- file.path(package_root, "tables")
report_dir <- file.path(package_root, "report")
dir.create(derived_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

required_packages <- c("ggplot2", "dplyr", "tidyr", "purrr", "readr", "dlnm", "MASS", "svglite", "ragg")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Missing R packages: ", paste(missing_packages, collapse = ", "))
}
if (!dir.exists(model_root)) stop("Model directory does not exist: ", model_root)

normalize_city <- function(x) {
  x <- trimws(as.character(x))
  x <- gsub("[^A-Za-z0-9]+", "_", x)
  gsub("^_+|_+$", "", x)
}

clean_names_base <- function(x) {
  out <- tolower(gsub("[^A-Za-z0-9]+", "_", x))
  gsub("^_+|_+$", "", out)
}

lag_file <- file.path(input_dir, "ward_k4_12d_75city_lowess_multi_threshold_effect_days.csv")
lag_table <- read_csv(lag_file, show_col_types = FALSE)
names(lag_table) <- clean_names_base(names(lag_table))
lag_table <- lag_table %>%
  transmute(
    city = as.character(city),
    city_code = normalize_city(city),
    dtw_group = as.integer(group),
    dtw_cluster = if_else(dtw_group %in% 1:4, paste0("C", dtw_group), "No phenotype"),
    dynamic_lag_days = if_else(dtw_group == 3L, 8L, 12L)
  ) %>%
  arrange(dtw_group, city)
if (nrow(lag_table) != 75 || n_distinct(lag_table$city_code) != 75) {
  stop("Lag table must contain exactly 75 unique cities.")
}

cluster_colors <- c(
  "C1" = "#A93436",
  "C2" = "#E39A08",
  "C3" = "#5A9FC8",
  "C4" = "#234F8C",
  "No phenotype" = "#858B91",
  "Sparse tail" = "#D7D9DC"
)
scenario_levels <- c("lag7", "lag12", "dynamic")
scenario_labels <- c(
  "lag7" = "Fixed 7-day",
  "lag12" = "Fixed 12-day",
  "dynamic" = "DTW 8/12-day"
)
scenario_linetypes <- c("lag7" = "dotted", "lag12" = "22", "dynamic" = "solid")
scenario_linewidths <- c("lag7" = 0.32, "lag12" = 0.38, "dynamic" = 0.62)
scenario_alphas <- c("lag7" = 0.58, "lag12" = 0.74, "dynamic" = 1.00)

scenario_dirs <- c(
  lag7 = "lag7",
  lag12 = "lag12",
  dynamic = "lag_group_median_overall"
)

parse_model_file <- function(path, scenario) {
  file_name <- basename(path)
  model_name <- sub("_DLNM_result[.]rds$", "", file_name)
  model_type <- sub("^.*_(composite_all|day_all|night_all)$", "\\1", model_name)
  city_code <- sub("_(composite_all|day_all|night_all)$", "", model_name)
  indicator <- if (grepl("_exceeded_quantity$", basename(dirname(path)))) "exceeded_quantity" else "cehwi"
  tibble(
    scenario = scenario,
    scenario_label = unname(scenario_labels[scenario]),
    indicator = indicator,
    model_type = model_type,
    city_code = city_code,
    rds_path = normalizePath(path, winslash = "/", mustWork = TRUE)
  )
}

model_manifest <- imap_dfr(scenario_dirs, function(folder, scenario) {
  scenario_root <- file.path(model_root, folder)
  files <- list.files(
    scenario_root,
    pattern = "_(composite_all|day_all|night_all)_DLNM_result[.]rds$",
    recursive = TRUE,
    full.names = TRUE
  )
  map_dfr(files, parse_model_file, scenario = scenario)
}) %>%
  left_join(lag_table, by = "city_code") %>%
  mutate(
    scenario = factor(scenario, levels = scenario_levels),
    heat_type = recode(model_type,
      composite_all = "composite",
      day_all = "daytime",
      night_all = "nighttime"
    )
  )

if (nrow(model_manifest) == 0) stop("No city DLNM RDS files were found under ", model_root)

nearest_psd <- function(sigma, floor_value = 1e-10) {
  sigma <- (sigma + t(sigma)) / 2
  eig <- eigen(sigma, symmetric = TRUE)
  eig$values <- pmax(eig$values, floor_value)
  eig$vectors %*% diag(eig$values, nrow = length(eig$values)) %*% t(eig$vectors)
}

make_basis <- function(model, dose) {
  template <- model$cb
  dose <- as.numeric(dose)
  helper_dose <- unique(c(
    dose,
    as.numeric(attr(template, "knots")),
    as.numeric(attr(template, "Boundary.knots"))
  ))
  helper_basis <- dlnm::onebasis(
    helper_dose,
    fun = attr(template, "fun"),
    knots = attr(template, "knots"),
    Boundary.knots = attr(template, "Boundary.knots"),
    intercept = isTRUE(attr(template, "intercept"))
  )
  helper_basis[match(dose, helper_dose), , drop = FALSE]
}

evaluate_rr_mc <- function(model, doses, weights = NULL, n_draws = 500L, seed = 20260801L) {
  doses <- as.numeric(doses)
  basis <- make_basis(model, doses)
  beta <- as.numeric(model$coef)
  sigma <- nearest_psd(as.matrix(model$vcov))
  point_values <- exp(pmin(700, as.vector(basis %*% beta)))
  set.seed(seed)
  draws <- MASS::mvrnorm(n = n_draws, mu = beta, Sigma = sigma)
  draw_eta <- basis %*% t(draws)
  draw_eta[draw_eta > 700] <- 700
  draw_values <- exp(draw_eta)
  if (is.null(weights)) {
    point <- point_values
    draw_summary <- apply(draw_values, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE)
    return(tibble(
      dose = doses,
      rr = point,
      rr_low = draw_summary[1, ],
      rr_high = draw_summary[2, ]
    ))
  }
  weights <- as.numeric(weights)
  weights <- weights / sum(weights)
  point <- sum(weights * point_values)
  weighted_draws <- as.vector(t(weights) %*% draw_values)
  ci <- quantile(weighted_draws, probs = c(0.025, 0.975), na.rm = TRUE)
  tibble(rr = point, rr_low = ci[[1]], rr_high = ci[[2]])
}

interpolate_curve_row <- function(curve, x_value) {
  if (!is.finite(x_value) || nrow(curve) == 0) return(tibble())
  cols <- c("rr", "rr_low", "rr_high")
  vals <- lapply(cols, function(col) approx(curve$x, curve[[col]], xout = x_value, rule = 2)$y)
  names(vals) <- cols
  tibble(x = x_value, !!!vals)
}

extract_model <- function(path, scenario, indicator, model_type, city_code, cluster, dynamic_lag_days) {
  model <- readRDS(path)
  pred <- as_tibble(model$pred_df)
  names(pred)[1] <- "x"
  pred <- pred %>%
    transmute(
      x = as.numeric(x),
      rr = as.numeric(rr),
      rr_low = as.numeric(rr_low),
      rr_high = as.numeric(rr_high)
    ) %>%
    filter(is.finite(x), is.finite(rr), rr > 0) %>%
    arrange(x)
  exposure <- as.numeric(model$cehwi_data)
  exposure <- exposure[is.finite(exposure) & exposure > 0]
  if (length(exposure) == 0) stop("No positive exposure values in ", path)

  q <- quantile(exposure, probs = c(0.25, 0.95, 0.99), na.rm = TRUE, names = FALSE, type = 7)
  names(q) <- c("p25", "p95", "p99")
  p99 <- min(q[["p99"]], max(pred$x, na.rm = TRUE))
  p95 <- min(q[["p95"]], p99)
  p25 <- min(q[["p25"]], p95)

  pred_plot <- pred %>% filter(x <= p99)
  if (!any(abs(pred_plot$x - p99) < 1e-10)) pred_plot <- bind_rows(pred_plot, interpolate_curve_row(pred, p99))
  if (!any(abs(pred_plot$x - p95) < 1e-10)) pred_plot <- bind_rows(pred_plot, interpolate_curve_row(pred, p95))
  pred_plot <- pred_plot %>% distinct(x, .keep_all = TRUE) %>% arrange(x)

  support_curve <- pred_plot %>% filter(x <= p95) %>% mutate(segment = "Supported range")
  tail_curve <- pred_plot %>% filter(x >= p95) %>% mutate(segment = "Sparse tail")
  curve <- bind_rows(support_curve, tail_curve) %>%
    mutate(
      log_rr = log(rr),
      log_low = if_else(rr_low > 0, log(rr_low), NA_real_),
      log_high = if_else(rr_high > 0, log(rr_high), NA_real_),
      scenario = scenario,
      indicator = indicator,
      model_type = model_type,
      city_code = city_code,
      dtw_cluster = cluster,
      support_boundary = p95,
      plot_max = p99
    )

  hist_values <- exposure[exposure <= p99]
  breaks <- seq(0, p99, length.out = 13)
  if (length(unique(breaks)) < 3) breaks <- pretty(c(0, p99), n = 12)
  h <- hist(hist_values, breaks = breaks, plot = FALSE, include.lowest = TRUE, right = FALSE)
  histogram <- tibble(
    xmin = head(h$breaks, -1),
    xmax = tail(h$breaks, -1),
    count = h$counts,
    density_scaled = if (max(h$counts) > 0) h$counts / max(h$counts) else 0,
    tail = (head(h$breaks, -1) + tail(h$breaks, -1)) / 2 > p95,
    indicator = indicator,
    model_type = model_type,
    city_code = city_code,
    dtw_cluster = cluster,
    support_boundary = p95,
    plot_max = p99
  )

  metric_bins <- hist(exposure[exposure <= p99], breaks = seq(0, p99, length.out = 31), plot = FALSE, include.lowest = TRUE)
  metric_mid <- metric_bins$mids[metric_bins$counts > 0]
  metric_weights <- metric_bins$counts[metric_bins$counts > 0]
  seed_base <- sum(utf8ToInt(paste(city_code, indicator, model_type, scenario, sep = "|"))) + 20260801L
  rr_selected <- evaluate_rr_mc(model, c(p25, p95), n_draws = 500L, seed = seed_base)
  rr_weighted <- evaluate_rr_mc(model, metric_mid, weights = metric_weights, n_draws = 500L, seed = seed_base + 17L)

  expected_lag <- case_when(
    scenario == "lag7" ~ 7L,
    scenario == "lag12" ~ 12L,
    TRUE ~ dynamic_lag_days
  )
  actual_lag <- if (!is.null(model$lag_days)) as.integer(model$lag_days) else as.integer(model$max_lag) + 1L
  metrics <- tibble(
    city_code = city_code,
    indicator = indicator,
    model_type = model_type,
    scenario = scenario,
    dtw_cluster = cluster,
    dynamic_lag_days = dynamic_lag_days,
    actual_lag_days = actual_lag,
    expected_lag_days = expected_lag,
    lag_match = isTRUE(actual_lag == expected_lag),
    includes_temperature_control = isTRUE(model$includes_temperature_control),
    temperature_control_col = if (is.null(model$temperature_control_col)) NA_character_ else as.character(model$temperature_control_col),
    exposure_knots = paste(signif(as.numeric(attr(model$cb, "knots")), 12), collapse = ";"),
    exposure_boundaries = paste(signif(as.numeric(attr(model$cb, "Boundary.knots")), 12), collapse = ";"),
    n_obs = as.integer(model$n_obs),
    n_grids = as.integer(model$n_grids),
    positive_grid_days = length(exposure),
    exposure_p25 = p25,
    support_boundary_p95 = p95,
    plot_boundary_p99 = p99,
    rr_p25 = rr_selected$rr[[1]],
    rr_p25_low = rr_selected$rr_low[[1]],
    rr_p25_high = rr_selected$rr_high[[1]],
    rr_p95 = rr_selected$rr[[2]],
    rr_p95_low = rr_selected$rr_low[[2]],
    rr_p95_high = rr_selected$rr_high[[2]],
    rr_weighted_p99 = rr_weighted$rr[[1]],
    rr_weighted_p99_low = rr_weighted$rr_low[[1]],
    rr_weighted_p99_high = rr_weighted$rr_high[[1]],
    overall_wald_p = if (is.null(model$overall_wald_p)) NA_real_ else as.numeric(model$overall_wald_p),
    deviance_explained = if (is.null(model$dev_explained)) NA_real_ else as.numeric(model$dev_explained),
    rds_path = path
  )

  list(curve = curve, histogram = histogram, metrics = metrics)
}

message("Reading and auditing ", nrow(model_manifest), " city models...")
extracted <- pmap(
  model_manifest %>% dplyr::select(rds_path, scenario, indicator, model_type, city_code, dtw_cluster, dynamic_lag_days),
  function(rds_path, scenario, indicator, model_type, city_code, dtw_cluster, dynamic_lag_days) {
    extract_model(
      path = rds_path,
      scenario = as.character(scenario),
      indicator = indicator,
      model_type = model_type,
      city_code = city_code,
      cluster = dtw_cluster,
      dynamic_lag_days = dynamic_lag_days
    )
  }
)

curve_data <- map_dfr(extracted, "curve")
histogram_data <- map_dfr(extracted, "histogram") %>%
  distinct(indicator, model_type, city_code, xmin, xmax, .keep_all = TRUE)
metric_data <- map_dfr(extracted, "metrics")

qa_fail <- metric_data %>% filter(!lag_match | !includes_temperature_control)
basis_qa <- metric_data %>%
  group_by(city_code, indicator, model_type) %>%
  summarise(
    n_scenarios = n_distinct(scenario),
    n_exposure_knot_sets = n_distinct(exposure_knots),
    n_exposure_boundary_sets = n_distinct(exposure_boundaries),
    n_temperature_columns = n_distinct(temperature_control_col),
    .groups = "drop"
  ) %>%
  mutate(
    basis_consistent = n_exposure_knot_sets == 1L & n_exposure_boundary_sets == 1L,
    temperature_source_consistent = n_temperature_columns == 1L
  )
basis_fail <- basis_qa %>%
  filter(n_scenarios == 3L, !basis_consistent | !temperature_source_consistent)
write_csv(metric_data, file.path(derived_dir, "city_model_metrics_long.csv"))
write_csv(model_manifest %>% dplyr::select(-scenario_label), file.path(derived_dir, "city_model_input_manifest.csv"))
write_csv(qa_fail, file.path(derived_dir, "city_model_qa_failures.csv"))
write_csv(basis_qa, file.path(derived_dir, "city_model_basis_consistency.csv"))
if (nrow(qa_fail) > 0) {
  stop(
    "Model QA failed for ", nrow(qa_fail),
    " city-scenario models. See data/derived/city_model_qa_failures.csv."
  )
}
if (nrow(basis_fail) > 0) {
  stop(
    "Exposure-basis QA failed for ", nrow(basis_fail),
    " city-indicator-model combinations. See data/derived/city_model_basis_consistency.csv."
  )
}

write_csv(curve_data, file.path(derived_dir, "city_curve_source_data.csv"))
write_csv(histogram_data, file.path(derived_dir, "city_histogram_source_data.csv"))

model_types <- c("composite_all", "day_all", "night_all")
indicators <- c("cehwi", "exceeded_quantity")

full_grid <- crossing(
  city_code = lag_table$city_code,
  indicator = indicators,
  model_type = model_types,
  scenario = scenario_levels
) %>%
  left_join(lag_table, by = "city_code") %>%
  left_join(metric_data, by = c(
    "city_code", "indicator", "model_type", "scenario", "dtw_cluster", "dynamic_lag_days"
  )) %>%
  mutate(model_available = !is.na(actual_lag_days))
write_csv(full_grid, file.path(derived_dir, "city_model_coverage_complete_grid.csv"))

format_rr_ci <- function(rr, low, high) {
  format_rr_value <- function(x) {
    ifelse(
      is.finite(x),
      ifelse(x < 0.001, "<0.001", ifelse(x >= 1000, sprintf("%.2e", x), sprintf("%.3f", x))),
      "NA"
    )
  }
  valid <- is.finite(rr) & is.finite(low) & is.finite(high)
  ifelse(valid, paste0(format_rr_value(rr), " (", format_rr_value(low), "-", format_rr_value(high), ")"), "NA")
}

submission_base <- full_grid %>%
  mutate(
    heat_type = recode(model_type,
      composite_all = "composite",
      day_all = "daytime",
      night_all = "nighttime"
    ),
    rr_p25_95ci = format_rr_ci(rr_p25, rr_p25_low, rr_p25_high),
    rr_p95_95ci = format_rr_ci(rr_p95, rr_p95_low, rr_p95_high),
    rr_weighted_p99_95ci = format_rr_ci(rr_weighted_p99, rr_weighted_p99_low, rr_weighted_p99_high),
    scenario = factor(scenario, levels = scenario_levels)
  )

for (indicator_i in indicators) {
  for (model_i in model_types) {
    heat_i <- recode(model_i, composite_all = "composite", day_all = "daytime", night_all = "nighttime")
    table_i <- submission_base %>%
      filter(indicator == indicator_i, model_type == model_i) %>%
      dplyr::select(
        city, city_code, dtw_cluster, dynamic_lag_days, scenario,
        model_available, n_obs, n_grids, positive_grid_days,
        rr_p25_95ci, rr_p95_95ci, rr_weighted_p99_95ci,
        overall_wald_p, deviance_explained
      ) %>%
      pivot_wider(
        names_from = scenario,
        values_from = c(
          model_available, n_obs, n_grids, positive_grid_days,
          rr_p25_95ci, rr_p95_95ci, rr_weighted_p99_95ci,
          overall_wald_p, deviance_explained
        ),
        names_glue = "{scenario}_{.value}"
      ) %>%
      arrange(match(city_code, lag_table$city_code))
    write_csv(
      table_i,
      file.path(table_dir, paste0("nature_submission_75city_", indicator_i, "_", heat_i, ".csv"))
    )
  }
}

theme_set(theme_classic(base_size = 5.2, base_family = "Arial"))

make_city_atlas <- function(indicator_i, model_i) {
  heat_i <- recode(model_i, composite_all = "composite", day_all = "daytime", night_all = "nighttime")
  heat_label <- recode(model_i,
    composite_all = "Compound heatwaves",
    day_all = "Daytime heatwaves",
    night_all = "Nighttime heatwaves"
  )
  indicator_label <- if (indicator_i == "cehwi") "CEHWI" else "Exceeded quantity"

  curves <- curve_data %>%
    filter(indicator == indicator_i, model_type == model_i) %>%
    left_join(lag_table %>% dplyr::select(city_code, city, dtw_cluster, dynamic_lag_days), by = c("city_code", "dtw_cluster")) %>%
    mutate(
      scenario = factor(scenario, levels = scenario_levels),
      panel_id = factor(city_code, levels = lag_table$city_code)
    )
  histograms <- histogram_data %>%
    filter(indicator == indicator_i, model_type == model_i) %>%
    left_join(lag_table %>% dplyr::select(city_code, city, dtw_cluster, dynamic_lag_days), by = c("city_code", "dtw_cluster")) %>%
    mutate(panel_id = factor(city_code, levels = lag_table$city_code))

  available <- curves %>% distinct(city_code) %>% pull(city_code)
  panel_info <- lag_table %>%
    mutate(
      panel_id = factor(city_code, levels = lag_table$city_code),
      panel_label = paste0(city, " | ", dtw_cluster, " | dyn", dynamic_lag_days),
      model_available = city_code %in% available
    )

  panel_ranges <- curves %>%
    group_by(city_code) %>%
    summarise(
      x_max = max(plot_max, na.rm = TRUE),
      curve_min = min(c(log_rr, log_low[scenario == "dynamic" & segment == "Supported range"]), na.rm = TRUE),
      curve_max = max(c(log_rr, log_high[scenario == "dynamic" & segment == "Supported range"]), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      curve_min = pmin(curve_min, 0),
      curve_max = pmax(curve_max, 0),
      curve_span = pmax(curve_max - curve_min, 0.20),
      separator_y = curve_min - 0.08 * curve_span,
      hist_bottom = curve_min - 0.42 * curve_span,
      hist_top = curve_min - 0.12 * curve_span,
      plot_y_min = hist_bottom,
      plot_y_max = curve_max + 0.08 * curve_span
    )

  panel_info <- panel_info %>%
    left_join(panel_ranges, by = "city_code") %>%
    mutate(
      x_max = if_else(is.finite(x_max), x_max, 1),
      plot_y_min = if_else(is.finite(plot_y_min), plot_y_min, -0.5),
      plot_y_max = if_else(is.finite(plot_y_max), plot_y_max, 0.5),
      separator_y = if_else(is.finite(separator_y), separator_y, -0.25),
      hist_bottom = if_else(is.finite(hist_bottom), hist_bottom, -0.5),
      hist_top = if_else(is.finite(hist_top), hist_top, -0.30)
    )

  histograms <- histograms %>%
    left_join(panel_info %>% dplyr::select(city_code, hist_bottom, hist_top), by = "city_code") %>%
    mutate(
      ymin = hist_bottom,
      ymax = hist_bottom + density_scaled * (hist_top - hist_bottom),
      hist_fill = if_else(tail, "Sparse tail", dtw_cluster)
    )

  blank_ranges <- panel_info %>%
    transmute(panel_id, x = 0, y = plot_y_min) %>%
    bind_rows(panel_info %>% transmute(panel_id, x = x_max, y = plot_y_max))
  empty_labels <- panel_info %>%
    filter(!model_available) %>%
    transmute(
      panel_id,
      x = 0.5,
      y = 0,
      label = if_else(model_i == "composite_all", "No compound-heatwave model", "No estimable model")
    )
  facet_labels <- setNames(panel_info$panel_label, panel_info$city_code)

  p <- ggplot() +
    geom_blank(data = blank_ranges, aes(x = x, y = y)) +
    geom_rect(
      data = histograms,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = hist_fill),
      colour = NA,
      alpha = 0.62
    ) +
    geom_hline(
      data = panel_info %>% filter(model_available),
      aes(yintercept = separator_y),
      linewidth = 0.22,
      colour = "#9AA0A6"
    ) +
    geom_hline(yintercept = 0, linewidth = 0.22, colour = "#8B8F94", linetype = "dashed") +
    geom_rect(
      data = curves %>% distinct(city_code, panel_id, support_boundary, plot_max),
      aes(xmin = support_boundary, xmax = plot_max, ymin = -Inf, ymax = Inf),
      inherit.aes = FALSE,
      fill = "#BFC3C7",
      alpha = 0.07
    ) +
    geom_vline(
      data = curves %>% distinct(city_code, panel_id, support_boundary),
      aes(xintercept = support_boundary),
      linewidth = 0.24,
      colour = "#70757A",
      linetype = "33"
    ) +
    geom_ribbon(
      data = curves %>% filter(scenario == "dynamic", segment == "Supported range", is.finite(log_low), is.finite(log_high)),
      aes(x = x, ymin = log_low, ymax = log_high, fill = dtw_cluster, group = city_code),
      alpha = 0.13,
      colour = NA
    ) +
    geom_line(
      data = curves,
      aes(
        x = x,
        y = log_rr,
        colour = dtw_cluster,
        linetype = scenario,
        linewidth = scenario,
        alpha = scenario,
        group = interaction(city_code, scenario, segment)
      ),
      lineend = "round"
    ) +
    geom_text(
      data = empty_labels,
      aes(x = x, y = y, label = label),
      size = 1.55,
      colour = "#777777",
      lineheight = 0.9
    ) +
    facet_wrap(
      vars(panel_id),
      ncol = 5,
      scales = "free",
      axes = "all",
      axis.labels = "all",
      labeller = as_labeller(facet_labels)
    ) +
    scale_colour_manual(values = cluster_colors, breaks = c("C1", "C2", "C3", "C4", "No phenotype"), drop = FALSE) +
    scale_fill_manual(values = cluster_colors, breaks = c("C1", "C2", "C3", "C4", "No phenotype", "Sparse tail"), drop = FALSE) +
    scale_linetype_manual(values = scenario_linetypes, labels = scenario_labels, drop = FALSE) +
    scale_linewidth_manual(values = scenario_linewidths, labels = scenario_labels, drop = FALSE) +
    scale_alpha_manual(values = scenario_alphas, labels = scenario_labels, drop = FALSE) +
    scale_x_continuous(breaks = breaks_pretty(n = 3), expand = expansion(mult = c(0.01, 0.02))) +
    scale_y_continuous(breaks = breaks_pretty(n = 3), expand = expansion(mult = c(0.01, 0.01))) +
    labs(
      title = paste0("City-specific ", indicator_label, " response across lag specifications | ", heat_label),
      subtitle = paste0(
        "Cumulative log-RR: solid, DTW 8/12-day; dashed, fixed 12-day; dotted, fixed 7-day. ",
        "Ribbon is the dynamic-window 95% CI within the positive-exposure p95 boundary; gray marks the p95-p99 sparse tail."
      ),
      x = indicator_label,
      y = "Cumulative log-RR",
      colour = "DTW phenotype",
      fill = "Exposure distribution",
      linetype = "Lag specification",
      linewidth = "Lag specification",
      alpha = "Lag specification"
    ) +
    guides(
      fill = "none",
      alpha = "none",
      linewidth = "none",
      colour = guide_legend(order = 1, override.aes = list(linewidth = 1.3, alpha = 1)),
      linetype = guide_legend(order = 2, override.aes = list(colour = "#3F4448", linewidth = 0.8, alpha = 1))
    ) +
    theme_classic(base_size = 5.2, base_family = "Arial") +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.border = element_rect(fill = NA, colour = "#B9BDC1", linewidth = 0.22),
      axis.line = element_blank(),
      axis.ticks = element_line(linewidth = 0.20, colour = "#55585C"),
      axis.ticks.length = unit(0.8, "mm"),
      axis.text = element_text(size = 3.9, colour = "#2D3033"),
      axis.title = element_text(size = 6.4, face = "bold", colour = "#24272A"),
      strip.background = element_rect(fill = "#F5F6F7", colour = "#B9BDC1", linewidth = 0.22),
      strip.text = element_text(size = 4.7, face = "bold", colour = "#25282B", margin = margin(0.7, 0.4, 0.7, 0.4, unit = "mm")),
      panel.spacing = unit(1.1, "mm"),
      plot.title = element_text(size = 9.2, face = "bold", margin = margin(b = 1.0, unit = "mm")),
      plot.subtitle = element_text(size = 5.5, colour = "#555A60", margin = margin(b = 2.0, unit = "mm")),
      legend.position = "top",
      legend.box = "horizontal",
      legend.justification = "left",
      legend.title = element_text(size = 5.3, face = "bold"),
      legend.text = element_text(size = 5.0),
      legend.key.width = unit(7, "mm"),
      legend.key.height = unit(2.4, "mm"),
      plot.margin = margin(3, 3, 3, 3, unit = "mm")
    ) +
    coord_cartesian(clip = "off")

  out_subdir <- file.path(figure_dir, indicator_i)
  dir.create(out_subdir, recursive = TRUE, showWarnings = FALSE)
  stem <- paste0("city_dlnm_lag_comparison_", indicator_i, "_", heat_i, "_5x15")
  width_in <- 11.7
  height_in <- 18.2
  svglite::svglite(file.path(out_subdir, paste0(stem, ".svg")), width = width_in, height = height_in, bg = "white")
  print(p)
  dev.off()
  ragg::agg_png(
    file.path(out_subdir, paste0(stem, ".png")),
    width = width_in,
    height = height_in,
    units = "in",
    res = 420,
    background = "white"
  )
  print(p)
  dev.off()
  invisible(stem)
}

message("Rendering six 5 x 15 city atlases...")
figure_stems <- character()
for (indicator_i in indicators) {
  for (model_i in model_types) {
    figure_stems <- c(figure_stems, make_city_atlas(indicator_i, model_i))
  }
}

coverage_summary <- full_grid %>%
  count(indicator, model_type, scenario, model_available, name = "n_cities") %>%
  arrange(indicator, model_type, scenario, desc(model_available))
write_csv(coverage_summary, file.path(report_dir, "model_coverage_summary.csv"))

cat(
  paste0(
    "# City DLNM lag-comparison atlas\n\n",
    "This package compares fixed 7-day, fixed 12-day and DTW-assigned 8/12-day cumulative DLNM responses for 75 US cities. ",
    "All three specifications use the same city-specific exposure basis, seasonal adjustment and mean-temperature cross-basis; the columns compare the prespecified lag windows. Fixed 7-day models and the C3 fixed 12-day models were refitted. For DTW C1, C2 and C4, the dynamic specification equals fixed 12 days, and the identical fitted objects were reused for the corresponding fixed 12-day specification.\n\n",
    "## Figure logic\n\n",
    "- Each atlas contains 75 fixed city positions arranged in five columns and 15 rows.\n",
    "- C1-C4 colours identify the archived DTW4 response phenotypes; cities outside the estimable phenotype subset are grey.\n",
    "- Solid, dashed and dotted curves show the dynamic 8/12-day, fixed 12-day and fixed 7-day cumulative log-RR, respectively.\n",
    "- The ribbon is the 95% coefficient-covariance interval for the dynamic specification within the observed positive-exposure p95 boundary.\n",
    "- The histogram uses the true positive grid-day exposure distribution. Grey tail bars retain their original positions between p95 and p99.\n",
    "- Empty composite panels identify the 12 cities outside the estimable compound-heatwave subset.\n\n",
    "## Key-result tables\n\n",
    "The six Nature submission CSV files each contain exactly 75 city rows. They report sample size, positive-exposure grid-days, cumulative RR at positive-exposure p25, cumulative RR at the p95 empirical-support boundary, p99-supported exposure-distribution-weighted RR, the overall Wald p value and deviance explained. Cross-city I2 is reported in pooled analyses.\n\n",
    "`report/lag_sensitivity_summary.csv` provides a compact cross-city comparison of the dynamic specification against each fixed lag window. The six city atlases remain the primary curve-shape diagnostic.\n\n",
    "Sparse-tail estimates are retained and displayed in scientific notation. Empirical-support summaries and cross-city pooled curves provide the primary inference.\n\n",
    "## Reproduction\n\n",
    "Run `code/run_all.ps1`. Outputs are written under this package. Use `-SkipModelFit` to rebuild figures, tables and sensitivity summaries from the archived model objects.\n"
  ),
  file = file.path(report_dir, "README.md")
)

message("Completed city atlas package: ", package_root)
