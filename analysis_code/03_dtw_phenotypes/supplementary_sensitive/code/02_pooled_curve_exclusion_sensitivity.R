#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dlnm)
  library(mvmeta)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else getwd()
root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)

model_dir <- file.path(root, "data", "model_objects", "c4_cehwi_composite_all")
archive_path <- file.path(root, "data", "input", "c4_archived_pooled_RR_data.csv")
national_model_dir <- file.path(root, "data", "model_objects", "national_cehwi_composite_all")
national_archive_path <- file.path(root, "data", "input", "national_archived_pooled_RR_curve.csv")
csv_dir <- file.path(root, "output", "csv")
dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)

excluded_cities <- c("Clearwater", "Gilbert", "Palm_Bay")
model_files <- sort(list.files(model_dir, pattern = "_composite_all_DLNM_result\\.rds$", full.names = TRUE))
if (length(model_files) != 20L) stop("Expected 20 C4 model objects, found ", length(model_files))

models <- lapply(model_files, readRDS)
names(models) <- sub("_composite_all_DLNM_result\\.rds$", "", basename(model_files))
if (!all(excluded_cities %in% names(models))) stop("One or more exclusion cities are absent")

fit_pool <- function(city_models, label) {
  coef_list <- lapply(city_models, function(x) x$coef)
  vcov_list <- lapply(city_models, function(x) x$vcov)
  coef_matrix <- do.call(rbind, coef_list)
  fit <- mvmeta(coef_matrix, S = vcov_list, method = "reml")

  ranges <- lapply(city_models, function(x) x$cehwi_range)
  xmax <- max(unlist(ranges), na.rm = TRUE)
  x <- seq(0, xmax, length.out = 500)
  template <- city_models[[1]]$cb
  cp <- crosspred(
    template,
    coef = coef(fit),
    vcov = vcov(fit),
    model.link = "log",
    at = x,
    cen = 0,
    cumul = TRUE
  )

  data.frame(
    cehwi = x,
    rr = as.numeric(cp$allRRfit),
    rr_low = as.numeric(cp$allRRlow),
    rr_high = as.numeric(cp$allRRhigh),
    log_rr = log(as.numeric(cp$allRRfit)),
    log_rr_low = log(as.numeric(cp$allRRlow)),
    log_rr_high = log(as.numeric(cp$allRRhigh)),
    specification = label,
    n_cities = length(city_models),
    stringsAsFactors = FALSE
  )
}

curve_n20 <- fit_pool(models, "Primary C4 (n=20)")
curve_n17 <- fit_pool(models[!names(models) %in% excluded_cities], "Excluding single-grid cities (n=17)")

archive <- read.csv(archive_path, stringsAsFactors = FALSE)
archive_interp <- approx(archive$cehwi, archive$rr, xout = curve_n20$cehwi, rule = 2)$y
archive_max_abs_rr_diff <- max(abs(archive_interp - curve_n20$rr), na.rm = TRUE)
if (!is.finite(archive_max_abs_rr_diff) || archive_max_abs_rr_diff > 1e-6) {
  stop(sprintf("The n=20 rerun did not reproduce the archived curve (max |RR difference| = %.6g)", archive_max_abs_rr_diff))
}

positive_exposure <- unlist(lapply(models, function(x) {
  z <- as.numeric(x$cehwi_data)
  z[is.finite(z) & z > 0]
}))
percentiles <- quantile(positive_exposure, probs = c(0.25, 0.50, 0.75, 0.90, 0.95, 0.99), na.rm = TRUE)
support_p95 <- unname(percentiles[["95%"]])
support_p99 <- unname(percentiles[["99%"]])

n17_interp <- approx(curve_n17$cehwi, curve_n17$log_rr, xout = curve_n20$cehwi, rule = 2)$y
delta <- n17_interp - curve_n20$log_rr
within_p95 <- curve_n20$cehwi <= support_p95
within_p99 <- curve_n20$cehwi <= support_p99

metrics <- data.frame(
  metric = c(
    "n_primary", "n_exclusion", "archive_max_abs_rr_difference",
    "positive_exposure_p95", "positive_exposure_p99",
    "max_abs_log_rr_difference_within_p95",
    "rmse_log_rr_difference_within_p95",
    "pearson_log_rr_within_p95",
    "max_abs_log_rr_difference_within_p99",
    "rmse_log_rr_difference_within_p99"
  ),
  value = c(
    20, 17, archive_max_abs_rr_diff, support_p95, support_p99,
    max(abs(delta[within_p95]), na.rm = TRUE),
    sqrt(mean(delta[within_p95]^2, na.rm = TRUE)),
    cor(curve_n20$log_rr[within_p95], n17_interp[within_p95]),
    max(abs(delta[within_p99]), na.rm = TRUE),
    sqrt(mean(delta[within_p99]^2, na.rm = TRUE))
  )
)

key_points <- do.call(rbind, lapply(names(percentiles), function(qname) {
  xq <- unname(percentiles[[qname]])
  interpolate <- function(curve, column) {
    approx(curve$cehwi, curve[[column]], xout = xq, rule = 2)$y
  }
  data.frame(
    percentile = qname,
    cehwi = xq,
    primary_rr = interpolate(curve_n20, "rr"),
    primary_rr_low = interpolate(curve_n20, "rr_low"),
    primary_rr_high = interpolate(curve_n20, "rr_high"),
    exclusion_rr = interpolate(curve_n17, "rr"),
    exclusion_rr_low = interpolate(curve_n17, "rr_low"),
    exclusion_rr_high = interpolate(curve_n17, "rr_high"),
    delta_log_rr = interpolate(curve_n17, "log_rr") - interpolate(curve_n20, "log_rr"),
    stringsAsFactors = FALSE
  )
}))

combined <- merge(
  curve_n20[, c("cehwi", "rr", "rr_low", "rr_high", "log_rr", "log_rr_low", "log_rr_high")],
  curve_n17[, c("cehwi", "rr", "rr_low", "rr_high", "log_rr", "log_rr_low", "log_rr_high")],
  by = "cehwi",
  suffixes = c("_n20", "_n17")
)
combined$delta_log_rr <- combined$log_rr_n17 - combined$log_rr_n20

write.csv(curve_n20, file.path(csv_dir, "c4_pooled_curve_primary_n20.csv"), row.names = FALSE)
write.csv(curve_n17, file.path(csv_dir, "c4_pooled_curve_excluding_single_grid_n17.csv"), row.names = FALSE)
write.csv(combined, file.path(csv_dir, "c4_pooled_curve_comparison.csv"), row.names = FALSE)
write.csv(metrics, file.path(csv_dir, "c4_pooled_curve_sensitivity_metrics.csv"), row.names = FALSE)
write.csv(key_points, file.path(csv_dir, "c4_pooled_curve_key_percentiles.csv"), row.names = FALSE)

# National comparison: all 63 estimable cities versus the same pool excluding
# Clearwater, Gilbert and Palm Bay. This leaves every remaining city model and
# the pooling estimator unchanged.
national_files <- sort(list.files(
  national_model_dir,
  pattern = "_composite_all_DLNM_result\\.rds$",
  full.names = TRUE
))
if (length(national_files) != 63L) stop("Expected 63 national model objects, found ", length(national_files))
national_models <- lapply(national_files, readRDS)
names(national_models) <- sub("_composite_all_DLNM_result\\.rds$", "", basename(national_files))
if (!all(excluded_cities %in% names(national_models))) stop("National exclusion cities are absent")

national_n63 <- fit_pool(national_models, "Primary national pool (n=63)")
national_n60 <- fit_pool(
  national_models[!names(national_models) %in% excluded_cities],
  "Excluding single-grid cities (n=60)"
)
national_archive <- read.csv(national_archive_path, stringsAsFactors = FALSE)
national_archive_interp <- approx(
  national_archive$cehwi,
  national_archive$rr,
  xout = national_n63$cehwi,
  rule = 2
)$y
national_archive_max_abs_rr_diff <- max(abs(national_archive_interp - national_n63$rr), na.rm = TRUE)
if (!is.finite(national_archive_max_abs_rr_diff) || national_archive_max_abs_rr_diff > 1e-4) {
  stop(sprintf(
    "The n=63 rerun did not reproduce the archived national curve (max |RR difference| = %.6g)",
    national_archive_max_abs_rr_diff
  ))
}

national_positive <- unlist(lapply(national_models, function(x) {
  z <- as.numeric(x$cehwi_data)
  z[is.finite(z) & z > 0]
}))
national_percentiles <- quantile(national_positive, probs = c(0.95, 0.99), na.rm = TRUE)
national_p95 <- unname(national_percentiles[["95%"]])
national_p99 <- unname(national_percentiles[["99%"]])
national_n60_log <- approx(national_n60$cehwi, national_n60$log_rr, xout = national_n63$cehwi, rule = 2)$y
national_delta <- national_n60_log - national_n63$log_rr
national_within_p95 <- national_n63$cehwi <= national_p95
national_within_p99 <- national_n63$cehwi <= national_p99
national_metrics <- data.frame(
  metric = c(
    "n_primary", "n_exclusion", "archive_max_abs_rr_difference",
    "positive_exposure_p95", "positive_exposure_p99",
    "pearson_log_rr_within_p95", "max_abs_log_rr_difference_within_p95",
    "rmse_log_rr_difference_within_p95", "pearson_log_rr_within_p99",
    "max_abs_log_rr_difference_within_p99", "rmse_log_rr_difference_within_p99"
  ),
  value = c(
    63, 60, national_archive_max_abs_rr_diff, national_p95, national_p99,
    cor(national_n63$log_rr[national_within_p95], national_n60_log[national_within_p95]),
    max(abs(national_delta[national_within_p95]), na.rm = TRUE),
    sqrt(mean(national_delta[national_within_p95]^2, na.rm = TRUE)),
    cor(national_n63$log_rr[national_within_p99], national_n60_log[national_within_p99]),
    max(abs(national_delta[national_within_p99]), na.rm = TRUE),
    sqrt(mean(national_delta[national_within_p99]^2, na.rm = TRUE))
  )
)

national_combined <- merge(
  national_n63[, c("cehwi", "rr", "rr_low", "rr_high", "log_rr", "log_rr_low", "log_rr_high")],
  national_n60[, c("cehwi", "rr", "rr_low", "rr_high", "log_rr", "log_rr_low", "log_rr_high")],
  by = "cehwi",
  suffixes = c("_n63", "_n60")
)
national_combined$delta_log_rr <- national_combined$log_rr_n60 - national_combined$log_rr_n63
write.csv(national_n63, file.path(csv_dir, "national_pooled_curve_primary_n63.csv"), row.names = FALSE)
write.csv(national_n60, file.path(csv_dir, "national_pooled_curve_excluding_single_grid_n60.csv"), row.names = FALSE)
write.csv(national_combined, file.path(csv_dir, "national_pooled_curve_comparison.csv"), row.names = FALSE)
write.csv(national_metrics, file.path(csv_dir, "national_pooled_curve_sensitivity_metrics.csv"), row.names = FALSE)

national_r <- national_metrics$value[national_metrics$metric == "pearson_log_rr_within_p95"]
national_max_delta <- national_metrics$value[national_metrics$metric == "max_abs_log_rr_difference_within_p95"]

cat(sprintf("Archived n=20 curve reproduced: max |RR difference| = %.3g\n", archive_max_abs_rr_diff))
cat(sprintf("Within p95 support: max |delta log-RR| = %.4f; RMSE = %.4f\n",
            metrics$value[metrics$metric == "max_abs_log_rr_difference_within_p95"],
            metrics$value[metrics$metric == "rmse_log_rr_difference_within_p95"]))
cat(sprintf("National within p95 support: Pearson r = %.4f; max |delta log-RR| = %.4f\n",
            national_r, national_max_delta))
