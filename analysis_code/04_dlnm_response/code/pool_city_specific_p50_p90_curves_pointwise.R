suppressPackageStartupMessages({
  library(data.table)
  library(dlnm)
  library(metafor)
  library(readr)
})

# Final Figure 4 cross-city synthesis.
#
# The primary first-stage models use city-specific positive-exposure p50/p90
# spline knots. Their reduced coefficients therefore do not share a common
# coordinate system. This script reconstructs each city curve in its own basis
# and pools log-relative risks at common exposure values using pointwise REML.

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) stop("Run with Rscript")
script_path <- normalizePath(sub("^--file=", "", script_argument), winslash = "/")
module_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/")
repo_root <- normalizePath(file.path(module_root, "..", ".."), winslash = "/")

model_root <- Sys.getenv(
  "FIG4_MODEL_ROOT",
  unset = file.path(
    repo_root, "external_data", "dlnm", "city_specific_p50_p90",
    "dynamic", "lag_group_median_overall"
  )
)
assignment_file <- Sys.getenv(
  "FIG4_ASSIGNMENT_FILE",
  unset = file.path(
    repo_root, "analysis_code", "03_dtw_phenotypes", "data", "results",
    "city_cluster_optimized_12d_ward_k4.csv"
  )
)
support_file <- Sys.getenv(
  "FIG4_SUPPORT_FILE",
  unset = file.path(module_root, "data", "model_reporting_input", "weighted_exposure_support.csv")
)
out_dir <- Sys.getenv(
  "FIG4_POINTWISE_OUTPUT_DIR",
  unset = file.path(module_root, "output", "pointwise_p50_p90_reporting")
)

if (!dir.exists(model_root)) {
  stop(
    "First-stage RDS directory not found: ", model_root,
    "\nSet FIG4_MODEL_ROOT to an authorized local city_specific_p50_p90 model directory."
  )
}
if (!file.exists(assignment_file)) stop("Assignment file not found: ", assignment_file)
if (!file.exists(support_file)) stop("Support file not found: ", support_file)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

heatwave_types <- c("composite", "day", "night")
percentile_names <- c("p25", "p50", "p75", "p90", "p95")

normalize_city <- function(value) {
  gsub("(^_+|_+$)", "", gsub("[^A-Za-z0-9]+", "_", value))
}

find_models <- function(heatwave_type) {
  files <- list.files(
    model_root,
    pattern = paste0("_", heatwave_type, "_all_DLNM_result[.]rds$"),
    recursive = TRUE,
    full.names = TRUE
  )
  cities <- normalize_city(sub("_cehwi$", "", basename(dirname(files))))
  setNames(files, cities)
}

centered_city_basis <- function(model, x) {
  basis <- dlnm::onebasis(
    x,
    fun = "ns",
    knots = as.numeric(model$cehwi_knots),
    Boundary.knots = as.numeric(model$cehwi_boundary_knots),
    intercept = FALSE
  )
  reference <- dlnm::onebasis(
    0,
    fun = "ns",
    knots = as.numeric(model$cehwi_knots),
    Boundary.knots = as.numeric(model$cehwi_boundary_knots),
    intercept = FALSE
  )
  sweep(basis, 2L, as.numeric(reference), "-")
}

city_effect_at <- function(model, x) {
  design <- centered_city_basis(model, x)
  estimate <- as.numeric(design %*% as.numeric(model$coef))
  variance <- as.numeric(design %*% as.matrix(model$vcov) %*% t(design))
  c(estimate = estimate, variance = variance)
}

pool_point <- function(models, x) {
  if (abs(x) < 1e-12) {
    return(list(
      estimate = 0, se = 0, ci_low = 0, ci_high = 0, tau2 = 0,
      qe = 0, i2 = 0, n_used = length(models), method = "reference"
    ))
  }

  effects <- vapply(models, city_effect_at, numeric(2), x = x)
  estimates <- effects["estimate", ]
  variances <- effects["variance", ]
  keep <- is.finite(estimates) & is.finite(variances) & variances > 1e-12
  if (sum(keep) < 3L) stop("Fewer than three finite city effects at x=", x)

  fit <- tryCatch(
    metafor::rma.uni(yi = estimates[keep], vi = variances[keep], method = "REML"),
    error = function(error) NULL
  )
  method <- "pointwise_REML"
  if (is.null(fit)) {
    fit <- tryCatch(
      metafor::rma.uni(yi = estimates[keep], vi = variances[keep], method = "DL"),
      error = function(error) NULL
    )
    method <- "pointwise_DL_fallback"
  }
  if (is.null(fit)) {
    fit <- metafor::rma.uni(yi = estimates[keep], vi = variances[keep], method = "FE")
    method <- "pointwise_FE_fallback"
  }

  list(
    estimate = as.numeric(fit$b),
    se = as.numeric(fit$se),
    ci_low = as.numeric(fit$ci.lb),
    ci_high = as.numeric(fit$ci.ub),
    tau2 = as.numeric(fit$tau2),
    qe = as.numeric(fit$QE),
    i2 = as.numeric(fit$I2),
    n_used = sum(keep),
    method = method
  )
}

format_point <- function(scope, cluster, heatwave_type, percentile, exposure, models,
                         common_support_upper = NA_real_) {
  pooled <- pool_point(models, exposure)
  direct_support <- sum(vapply(
    models,
    function(model) as.numeric(model$cehwi_boundary_knots)[[2]] >= exposure - 1e-12,
    logical(1)
  ))
  data.frame(
    scope = scope,
    cluster = cluster,
    heatwave_type = heatwave_type,
    percentile = percentile,
    exposure = exposure,
    log_rr = pooled$estimate,
    log_rr_low = pooled$ci_low,
    log_rr_high = pooled$ci_high,
    rr = exp(pooled$estimate),
    rr_low = exp(pooled$ci_low),
    rr_high = exp(pooled$ci_high),
    af_percent = 100 * (1 - exp(-pooled$estimate)),
    af_low = 100 * (1 - exp(-pooled$ci_low)),
    af_high = 100 * (1 - exp(-pooled$ci_high)),
    standard_error = pooled$se,
    tau2 = pooled$tau2,
    qe = pooled$qe,
    i2 = pooled$i2,
    n_cities = length(models),
    n_cities_used = pooled$n_used,
    n_cities_with_direct_support = direct_support,
    pooling_method = pooled$method,
    common_support_upper = common_support_upper,
    support_segment = ifelse(
      is.finite(common_support_upper) & exposure <= common_support_upper + 1e-12,
      "common_empirical_support",
      ifelse(is.finite(common_support_upper), "linear_tail", "not_applicable")
    ),
    stringsAsFactors = FALSE
  )
}

assignments <- readr::read_csv(assignment_file, show_col_types = FALSE)
names(assignments) <- tolower(names(assignments))
if (!all(c("city", "cluster") %in% names(assignments))) {
  stop("Assignment file must contain city and cluster columns")
}
assignments <- assignments[, c("city", "cluster")]
assignments$city <- normalize_city(assignments$city)
assignments$cluster <- as.integer(assignments$cluster)
assignments <- assignments[assignments$cluster %in% 1:4, ]
if (nrow(assignments) != 63L || anyDuplicated(assignments$city)) {
  stop("Expected 63 unique cities in the locked DTW assignment")
}

support <- readr::read_csv(support_file, show_col_types = FALSE)
if (nrow(support) != 12L) stop("Expected 12 weighted support rows")

model_sets <- list()
audit_rows <- list()
for (heatwave_type in heatwave_types) {
  files <- find_models(heatwave_type)
  if (length(files) != 63L || !setequal(names(files), assignments$city)) {
    stop("Incomplete model set for ", heatwave_type, ": found ", length(files), " of 63 cities")
  }
  models <- lapply(files[assignments$city], readRDS)
  names(models) <- assignments$city
  model_sets[[heatwave_type]] <- models

  for (city in names(models)) {
    model <- models[[city]]
    positive <- as.numeric(model$cehwi_data)
    positive <- positive[is.finite(positive) & positive > 0]
    expected <- as.numeric(quantile(positive, c(0.50, 0.90), type = 7, na.rm = TRUE))
    knots <- as.numeric(model$cehwi_knots)
    audit_rows[[paste(heatwave_type, city, sep = "_")]] <- data.frame(
      city = city,
      cluster = assignments$cluster[match(city, assignments$city)],
      heatwave_type = heatwave_type,
      n_observations = model$n_obs,
      coefficient_dimension = length(model$coef),
      knot_1 = knots[[1]],
      knot_2 = knots[[2]],
      expected_positive_p50 = expected[[1]],
      expected_positive_p90 = expected[[2]],
      knot_match = length(knots) == 2L && max(abs(knots - expected)) < 1e-8,
      boundary_low = as.numeric(model$cehwi_boundary_knots)[[1]],
      boundary_high = as.numeric(model$cehwi_boundary_knots)[[2]],
      time_control_spec = model$time_control_spec,
      time_df_per_year = model$time_df_per_year,
      lag_days = model$lag_days,
      stringsAsFactors = FALSE
    )
  }
}

audit <- rbindlist(audit_rows)
if (nrow(audit) != 189L || !all(audit$knot_match) ||
    !all(audit$coefficient_dimension == 3L) ||
    !all(audit$time_control_spec == "continuous_time_ns") ||
    !all(audit$time_df_per_year == 7L)) {
  stop("Primary model audit failed")
}

curve_rows <- list()
dose_rows <- list()
for (cluster_id in 1:4) {
  cities <- assignments$city[assignments$cluster == cluster_id]
  for (heatwave_type in heatwave_types) {
    models <- model_sets[[heatwave_type]][cities]
    support_row <- support[
      support$cluster == cluster_id & support$heatwave_type == heatwave_type,
    ]
    if (nrow(support_row) != 1L) stop("Missing support row")

    grid <- sort(unique(c(
      seq(0, support_row$p99[[1]], length.out = 241L),
      support_row$common_support_upper[[1]],
      as.numeric(unlist(
        support_row[1, c("p25", "p50", "p75", "p90", "p95", "p98", "p99")],
        use.names = FALSE
      ))
    )))
    pooled_points <- lapply(grid, function(x) pool_point(models, x))
    boundary_highs <- vapply(
      models,
      function(model) as.numeric(model$cehwi_boundary_knots)[[2]],
      numeric(1)
    )
    key <- paste(cluster_id, heatwave_type, sep = "_")
    curve_rows[[key]] <- data.frame(
      cluster = cluster_id,
      heatwave_type = heatwave_type,
      exposure = grid,
      log_rr = vapply(pooled_points, `[[`, numeric(1), "estimate"),
      log_rr_low = vapply(pooled_points, `[[`, numeric(1), "ci_low"),
      log_rr_high = vapply(pooled_points, `[[`, numeric(1), "ci_high"),
      standard_error = vapply(pooled_points, `[[`, numeric(1), "se"),
      tau2 = vapply(pooled_points, `[[`, numeric(1), "tau2"),
      qe = vapply(pooled_points, `[[`, numeric(1), "qe"),
      i2 = vapply(pooled_points, `[[`, numeric(1), "i2"),
      n_cities = length(models),
      n_cities_used = vapply(pooled_points, `[[`, numeric(1), "n_used"),
      n_cities_with_direct_support = vapply(
        grid, function(x) sum(boundary_highs >= x - 1e-12), integer(1)
      ),
      common_support_upper = support_row$common_support_upper[[1]],
      support_segment = ifelse(
        grid <= support_row$common_support_upper[[1]] + 1e-12,
        "common_empirical_support", "linear_tail_sensitivity"
      ),
      pooling_method = vapply(pooled_points, `[[`, character(1), "method"),
      first_stage_basis = "city-specific positive-exposure p50/p90 natural spline",
      stringsAsFactors = FALSE
    )
    curve_rows[[key]]$rr <- exp(curve_rows[[key]]$log_rr)
    curve_rows[[key]]$rr_low <- exp(curve_rows[[key]]$log_rr_low)
    curve_rows[[key]]$rr_high <- exp(curve_rows[[key]]$log_rr_high)

    for (percentile_name in percentile_names) {
      dose_key <- paste(key, percentile_name, sep = "_")
      dose_rows[[dose_key]] <- format_point(
        scope = "cluster",
        cluster = cluster_id,
        heatwave_type = heatwave_type,
        percentile = percentile_name,
        exposure = as.numeric(support_row[[percentile_name]]),
        models = models,
        common_support_upper = as.numeric(support_row$common_support_upper)
      )
    }
  }
}

reference_specs <- data.frame(
  heatwave_type = c("composite", "day"),
  exposure = c(12.3, 38.7),
  label = c("manuscript_reference_CEHWI_12.3", "manuscript_reference_CEHWI_38.7")
)
national_reference <- rbindlist(lapply(seq_len(nrow(reference_specs)), function(index) {
  row <- reference_specs[index, ]
  format_point(
    scope = "national",
    cluster = NA_integer_,
    heatwave_type = row$heatwave_type,
    percentile = row$label,
    exposure = row$exposure,
    models = model_sets[[row$heatwave_type]]
  )
}))

fwrite(audit, file.path(out_dir, "primary_p50_p90_model_audit.csv"))
fwrite(rbindlist(curve_rows), file.path(out_dir, "p50_p90_pointwise_reml_curves.csv"))
fwrite(rbindlist(dose_rows), file.path(out_dir, "cluster_dose_specific_RR_AF.csv"))
fwrite(national_reference, file.path(out_dir, "national_reference_point_RR_AF.csv"))
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
cat("Completed final city-specific p50/p90 pointwise REML synthesis.\n")
