suppressPackageStartupMessages({
  library(dlnm)
  library(mvmeta)
})

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/", mustWork = TRUE)
package_root <- dirname(dirname(script_path))
input_dir <- file.path(package_root, "data")
result_dir <- file.path(package_root, "output")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

eval_points <- c(2, 4, 6)
predictor_bases <- c(
  "NDVI_2023", "total_20_55", "GDP", "Unemployment", "BD",
  "Urbanization_Rate", "Street_Intersection_Density", "Walkability_Index"
)
predictor_labels <- c(
  NDVI_2023 = "NDVI",
  total_20_55 = "Population (20-55)",
  GDP = "GDP",
  Unemployment = "Unemployment",
  BD = "Building Density",
  Urbanization_Rate = "Urbanization Rate",
  Street_Intersection_Density = "Street Intersection Density",
  Walkability_Index = "Walkability Index"
)
activities <- c("all", "ride", "run", "walk")

parse_numeric <- function(x) as.numeric(strsplit(as.character(x), ";", fixed = TRUE)[[1]])

read_input_csv <- function(path, required) {
  out <- read.csv(
    path,
    check.names = FALSE,
    stringsAsFactors = FALSE,
    fileEncoding = "UTF-8-BOM"
  )
  missing <- setdiff(required, names(out))
  if (length(missing)) {
    stop("Missing required columns in ", basename(path), ": ", paste(missing, collapse = ", "))
  }
  out
}

nearest_psd <- function(mat, floor_ratio = 1e-8) {
  mat <- (mat + t(mat)) / 2
  eig <- eigen(mat, symmetric = TRUE)
  floor_value <- max(max(abs(eig$values)) * floor_ratio, 1e-10)
  eig$values <- pmax(eig$values, floor_value)
  eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
}

coef_df <- read_input_csv(
  file.path(input_dir, "city_native_reduced_coefficients.csv"),
  c("model_id", "city", "activity", "component", "coefficient")
)
vcov_df <- read_input_csv(
  file.path(input_dir, "city_native_reduced_covariances.csv"),
  c("model_id", "city", "activity", "row_component", "col_component", "covariance")
)
basis_df <- read_input_csv(
  file.path(input_dir, "city_native_basis_definitions.csv"),
  c("model_id", "city", "activity", "basis_fun", "knots", "boundary_knots", "intercept")
)
grid <- read_input_csv(
  file.path(input_dir, "grid_row_environment_predictors.csv"),
  c("city", "grid10_id")
)

load_models <- function(activity) {
  defs <- basis_df[basis_df$activity == activity, , drop = FALSE]
  models <- lapply(seq_len(nrow(defs)), function(i) {
    d <- defs[i, ]
    csub <- coef_df[coef_df$model_id == d$model_id, , drop = FALSE]
    csub <- csub[order(csub$component), ]
    vsub <- vcov_df[vcov_df$model_id == d$model_id, , drop = FALSE]
    p <- nrow(csub)
    if (p == 0L || nrow(vsub) != p * p) {
      stop("Incomplete coefficient or covariance input for model_id: ", d$model_id)
    }
    v <- matrix(NA_real_, p, p)
    for (j in seq_len(nrow(vsub))) v[vsub$row_component[j], vsub$col_component[j]] <- vsub$covariance[j]
    list(
      model_id = d$model_id,
      city = d$city,
      activity = activity,
      coef = as.numeric(csub$coefficient),
      vcov = nearest_psd(v),
      basis_fun = d$basis_fun,
      knots = parse_numeric(d$knots),
      boundary_knots = parse_numeric(d$boundary_knots),
      intercept = isTRUE(as.logical(d$intercept))
    )
  })
  names(models) <- vapply(models, `[[`, character(1), "city")
  models
}

transform_model <- function(model) {
  basis <- onebasis(
    c(0, eval_points),
    fun = model$basis_fun,
    knots = model$knots,
    Boundary.knots = model$boundary_knots,
    intercept = model$intercept
  )
  cmat <- sweep(basis[-1, , drop = FALSE], 2L, basis[1, ], "-")
  list(
    coef = as.numeric(cmat %*% model$coef),
    vcov = nearest_psd(cmat %*% model$vcov %*% t(cmat)),
    transform = cmat
  )
}

build_grid_rows <- function(cities, metric, activity) {
  suffix <- if (metric == "mean") "_mean" else "_gini"
  x_cols <- paste0(predictor_bases, suffix)
  rows <- grid[grid$city %in% cities, c("city", "grid10_id", x_cols), drop = FALSE]
  for (col in x_cols) rows[[col]] <- suppressWarnings(as.numeric(rows[[col]]))
  rows <- rows[complete.cases(rows[, x_cols, drop = FALSE]), , drop = FALSE]
  means <- vapply(rows[x_cols], mean, numeric(1))
  sds <- vapply(rows[x_cols], sd, numeric(1))
  if (any(!is.finite(sds) | sds <= 0)) stop("Invalid predictor SD for ", metric, " / ", activity)
  z_names <- paste0("z_", predictor_bases)
  for (i in seq_along(x_cols)) rows[[z_names[i]]] <- (rows[[x_cols[i]]] - means[i]) / sds[i]
  scaling <- data.frame(
    metric = metric,
    activity = activity,
    predictor = predictor_bases,
    source_column = x_cols,
    mean = means,
    sd = sds,
    n_grid_rows = nrow(rows),
    n_cities = length(unique(rows$city)),
    stringsAsFactors = FALSE
  )
  list(rows = rows, z_names = z_names, scaling = scaling)
}

fit_representation <- function(models, rows_info, representation, metric, activity) {
  if (representation == "native_coefficients_direct") {
    outcomes <- lapply(models, function(m) list(coef = m$coef, vcov = m$vcov))
    component_names <- paste0("native_b", seq_along(outcomes[[1]]$coef))
  } else {
    outcomes <- lapply(models, transform_model)
    component_names <- paste0("logRR_CEHWI_", eval_points)
  }
  dims <- unique(vapply(outcomes, function(x) length(x$coef), integer(1)))
  if (length(dims) != 1L || dims != 3L) stop("Expected three outcome coordinates for ", metric, " / ", activity)

  rows <- rows_info$rows
  y <- t(vapply(rows$city, function(city) outcomes[[city]]$coef, numeric(3)))
  colnames(y) <- component_names
  s_list <- lapply(rows$city, function(city) outcomes[[city]]$vcov)
  formula <- as.formula(paste("y ~", paste(rows_info$z_names, collapse = " + ")))
  fit <- tryCatch(mvmeta(formula, data = rows, S = s_list, method = "reml"), error = function(e) NULL)
  fit_method <- "REML"
  if (is.null(fit)) {
    ridge <- max(1e-6, max(vapply(s_list, function(v) max(abs(v)), numeric(1))) * 0.01)
    s_list <- lapply(s_list, function(v) nearest_psd(v + diag(ridge, 3)))
    fit <- mvmeta(formula, data = rows, S = s_list, method = "reml")
    fit_method <- "REML_with_covariance_ridge"
  }

  beta_matrix <- coef(fit, format = "matrix")
  full_vcov <- vcov(fit)
  flat_names <- names(coef(fit))
  joint_rows <- list()
  effect_rows <- list()
  average_rows <- list()
  average_weights <- rep(1 / length(eval_points), length(eval_points))

  for (i in seq_along(predictor_bases)) {
    term <- rows_info$z_names[i]
    indices <- match(paste0(component_names, ".", term), flat_names)
    if (anyNA(indices)) stop("Could not locate coefficient block for ", term)
    theta <- as.numeric(beta_matrix[term, ])
    block <- nearest_psd(full_vcov[indices, indices, drop = FALSE])
    wald <- as.numeric(t(theta) %*% solve(block, theta))
    joint_p <- pchisq(wald, df = length(theta), lower.tail = FALSE)
    joint_rows[[i]] <- data.frame(
      representation = representation,
      metric = metric,
      activity_type = activity,
      predictor = predictor_bases[i],
      variable_only = unname(predictor_labels[predictor_bases[i]]),
      joint_wald = wald,
      joint_df = length(theta),
      joint_p = joint_p,
      significant_0_05 = joint_p < 0.05,
      n_grid_rows = nrow(rows),
      n_cities = length(unique(rows$city)),
      fit_method = fit_method,
      stringsAsFactors = FALSE
    )
    if (representation == "common_exposure_coordinates") {
      for (j in seq_along(eval_points)) {
        se <- sqrt(block[j, j])
        effect_rows[[length(effect_rows) + 1L]] <- data.frame(
          metric = metric,
          activity_type = activity,
          predictor = predictor_bases[i],
          variable_only = unname(predictor_labels[predictor_bases[i]]),
          exposure = eval_points[j],
          coefficient = theta[j],
          se = se,
          ci_low = theta[j] - 1.96 * se,
          ci_high = theta[j] + 1.96 * se,
          joint_p = joint_p,
          stringsAsFactors = FALSE
        )
      }
      avg <- sum(average_weights * theta)
      avg_se <- sqrt(as.numeric(t(average_weights) %*% block %*% average_weights))
      average_rows[[i]] <- data.frame(
        metric = metric,
        activity_type = activity,
        predictor = predictor_bases[i],
        variable_only = unname(predictor_labels[predictor_bases[i]]),
        coefficient = avg,
        se = avg_se,
        ci_low = avg - 1.96 * avg_se,
        ci_high = avg + 1.96 * avg_se,
        joint_wald = wald,
        joint_df = length(theta),
        joint_p = joint_p,
        significant = joint_p < 0.05,
        star_label = ifelse(joint_p < 0.001, "***", ifelse(joint_p < 0.01, "**", ifelse(joint_p < 0.05, "*", ""))),
        estimand = "Mean predictor-associated log-RR modification across CEHWI 2, 4 and 6 versus 0",
        n_grid_rows = nrow(rows),
        n_cities = length(unique(rows$city)),
        stringsAsFactors = FALSE
      )
    }
  }
  list(
    joint = do.call(rbind, joint_rows),
    effects = if (length(effect_rows)) do.call(rbind, effect_rows) else NULL,
    average = if (length(average_rows)) do.call(rbind, average_rows) else NULL
  )
}

all_joint <- list()
all_effects <- list()
all_average <- list()
all_scaling <- list()
city_theta_rows <- list()
city_vcov_rows <- list()

for (activity in activities) {
  models <- load_models(activity)
  if (length(models) != 63L) stop("Expected 63 models for ", activity, "; found ", length(models))
  transformed <- lapply(models, transform_model)
  for (city in names(models)) {
    model_id <- paste(city, activity, sep = "__")
    theta <- transformed[[city]]$coef
    vtheta <- transformed[[city]]$vcov
    city_theta_rows[[length(city_theta_rows) + 1L]] <- data.frame(
      model_id = model_id, city = city, activity_type = activity,
      exposure = eval_points, log_rr = theta, stringsAsFactors = FALSE
    )
    idx <- expand.grid(row_exposure = eval_points, col_exposure = eval_points)
    idx$covariance <- mapply(
      function(x, y) vtheta[match(x, eval_points), match(y, eval_points)],
      idx$row_exposure, idx$col_exposure
    )
    city_vcov_rows[[length(city_vcov_rows) + 1L]] <- cbind(
      data.frame(model_id = model_id, city = city, activity_type = activity, stringsAsFactors = FALSE), idx
    )
  }
  for (metric in c("mean", "gini")) {
    rows_info <- build_grid_rows(names(models), metric, activity)
    all_scaling[[length(all_scaling) + 1L]] <- rows_info$scaling
    native <- fit_representation(models, rows_info, "native_coefficients_direct", metric, activity)
    common <- fit_representation(models, rows_info, "common_exposure_coordinates", metric, activity)
    all_joint[[length(all_joint) + 1L]] <- native$joint
    all_joint[[length(all_joint) + 1L]] <- common$joint
    all_effects[[length(all_effects) + 1L]] <- common$effects
    all_average[[length(all_average) + 1L]] <- common$average
  }
}

joint_df <- do.call(rbind, all_joint)
effect_df <- do.call(rbind, all_effects)
average_df <- do.call(rbind, all_average)
scaling_df <- unique(do.call(rbind, all_scaling))

write.csv(do.call(rbind, city_theta_rows), file.path(result_dir, "city_common_coordinate_log_rr.csv"), row.names = FALSE)
write.csv(do.call(rbind, city_vcov_rows), file.path(result_dir, "city_common_coordinate_covariances.csv"), row.names = FALSE)
write.csv(joint_df, file.path(result_dir, "meta_regression_joint_tests_native_vs_common.csv"), row.names = FALSE)
write.csv(effect_df, file.path(result_dir, "meta_regression_effects_at_cehwi_2_4_6.csv"), row.names = FALSE)
write.csv(average_df, file.path(result_dir, "meta_regression_average_common_coordinate_effects.csv"), row.names = FALSE)
write.csv(scaling_df, file.path(result_dir, "predictor_standardization.csv"), row.names = FALSE)
write.csv(average_df[average_df$metric == "mean", ], file.path(result_dir, "fig5_forest_plot_data_mean.csv"), row.names = FALSE)
write.csv(average_df[average_df$metric == "gini", ], file.path(result_dir, "fig5_forest_plot_data_gini.csv"), row.names = FALSE)

native_all <- joint_df[joint_df$representation == "native_coefficients_direct" & joint_df$activity_type == "all", ]
common_all <- joint_df[joint_df$representation == "common_exposure_coordinates" & joint_df$activity_type == "all", ]
comparison <- merge(
  native_all[, c("metric", "predictor", "variable_only", "joint_p", "significant_0_05")],
  common_all[, c("metric", "predictor", "joint_p", "significant_0_05")],
  by = c("metric", "predictor"), suffixes = c("_native", "_common")
)
comparison$significance_changed <- comparison$significant_0_05_native != comparison$significant_0_05_common
write.csv(comparison, file.path(result_dir, "all_activity_joint_test_comparison.csv"), row.names = FALSE)
write.csv(comparison[comparison$significance_changed, ], file.path(result_dir, "all_activity_significance_changes.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(result_dir, "R_session_info.txt"))

cat("Completed common-coordinate Fig. 5 analysis. Changed all-activity decisions: ", sum(comparison$significance_changed), "/", nrow(comparison), "\n", sep = "")
