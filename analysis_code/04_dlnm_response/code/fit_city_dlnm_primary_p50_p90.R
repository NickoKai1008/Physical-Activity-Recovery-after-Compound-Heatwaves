# ========================================================================
# 两阶段DLNM分析: CEHWI对PA的非线性累积滞后效应 (V5完整气象版)
# ========================================================================
# 
# 方法参考: 
#   - Gasparrini et al. (Lancet Planetary Health)
#   - Global excess deaths associated with heatwaves in 2023 (Innovation)
# 
# 核心框架（Nature标准 + V5完整气象控制）:
# 1. 第一阶段 (城市级):
#    - GAM + quasi-Poisson + DLNM cross-basis
#    - 【V5升级】完整气象控制变量:
#      * 温度、降水、降雪、风速、日照（2010-2024年完整数据）
#      * 不在第一阶段加入社会经济变量（避免过度参数化）
#      * 格子固定效应 (fish_id_fac) 捕捉空间异质性
#    - 【V5数据更新】75个城市
#    - 输出: 
#      a) 每个城市的cumulative RR曲线 (main lag 0-11天)
#      b) 格子固定效应系数（空间异质性）
#      c) 【V4改进】智能CI可视化（对数Y轴+截断极端值）
#      d) 【V5新增】X轴截断至95th分位数（避免长尾数据压缩主体）
# 
# 2. 第二阶段 (国家/区域级):
#    - 【V4关键改进】Random-effects meta-regression
#    - 社会经济变量作为meta-predictors（解释城市间异质性）:
#      * NDVI, Population, FAR, BH, GDP, Unemployed Population, WS, Walkability
#      * 这些变量在第二阶段用城市级平均值
#      * 符合Nature标准：第一阶段估计曲线，第二阶段解释异质性
#    - 输出城市级协变量的Forest Plot
#    - 【新增】VIF检查，避免共线性问题
# 
# ========== 第一阶段模型公式（reduced two-stage版）==========
# 
# trip_count_scaled ~ 
# Submission-primary Stage-1 formula uses cb_cehwi + cb_temp, city-specific
# positive-exposure p50/p90 knots, phenotype-specific lag 0-7/0-11, and a
# natural cubic spline of continuous calendar time with seven degrees of
# freedom per observed year.
#   cb_cehwi +                              # DLNM cross-basis (主暴露)
#   # RH excluded: most city weather files do not contain usable RH.
#   ns(log_precip, df=3) +                  # log-transformed precipitation
#   ns(wind_speed, df=3) +                  # wind speed
#   ns(calendar_time, 7 df/year) +          # seasonality and long-term trend
#   dow_fac +                               # categorical day of week
#   fish_id_fac                             # 格子固定效应（捕捉空间异质性）
# 
# 【当前策略】参考两阶段DLNM文献：
#   1. 第一阶段同时控制 mean/apparent temperature cross-basis，用于分离普通高温效应与热浪特异效应
#   2. 只保留降水、风速、doy、year、dow、fish_id（RH仅做缺失诊断，不入模）
#   3. 先 crossreduce 累积 lag，再把 overall reduced spline coefficients 送入第二阶段
#   4. 社会经济/建成环境变量只在第二阶段作为 meta-predictors
#   5. 符合Nature/Lancet最佳实践
# 
# ========== 第二阶段模型公式（V4随机效应meta）==========
# 
# mvmeta(coef_matrix ~ 
#   NDVI_mean +                             # 城市平均植被指数
#   Pop_mean +                              # 城市平均人口密度
#   FAR_mean +                              # 城市平均容积率
#   BD_mean +                               # 城市平均建筑密度
#   GDP_mean +                              # 城市平均GDP
#   unemployed_pop_mean_scaled +            # 【新增】城市平均失业人口（2010-2023年均值）
#   WS_mean +                               # 城市平均风速
#   Walk_mean,                              # 城市平均步行指数
#   S = vcov_list,                          # 城市内方差（第一阶段估计）
#   method = "reml")                        # 随机效应（REML）
# 
# 【V4改进】随机效应meta-regression:
#   - 允许城市间真实效应存在变异（τ²）
#   - 社会经济变量解释部分城市间异质性
#   - 输出I²统计量（异质性程度）
#   - 【新增】VIF检查，避免meta-predictors共线性
# 
# DLNM cross-basis设置:
#   - 暴露维度 (CEHWI): natural cubic spline
#     * 投稿主分析使用跨城市共享结点与边界；未提供共享参数时才回退到城市分位数
#   - 滞后维度 (main lag 0-11天): natural cubic spline, 3 df
#   - 参照: CEHWI = 0 (非热浪日)
# 
# 【V2】社会经济变量处理:
#   - 第一阶段: 所有变量z-score标准化，在格子级模型中作为协变量
#   - 第二阶段: 计算城市平均值，在meta-regression中作为meta-predictors
#   - 理论: β_socioecon_stage1 捕捉"格子内空间差异"
#          β_socioecon_stage2 捕捉"城市间差异"
#          β_fish_id 捕捉"残差空间差异"（Unexplained Grid Effect）
# 
# ========================================================================

library(tidyverse)
library(lubridate)
library(mgcv)
library(dlnm)      # DLNM核心包
library(mvmeta)    # 第二阶段meta-regression
library(broom)
library(janitor)
library(stringr)

# Save every ggplot PNG together with an editable SVG twin. This is intentionally
# global so all existing stage-2 outputs (RR curves, AF forests, AF percentile
# summaries, and meta-predictor plots) stay in sync without hand-maintaining
# dozens of individual ggsave() calls.
.ggplot2_ggsave <- ggplot2::ggsave
save_plot_with_editable_svg <- function(filename, plot = ggplot2::last_plot(), ..., device = NULL) {
  .ggplot2_ggsave(filename = filename, plot = plot, ..., device = device)
  save_svg <- tolower(Sys.getenv("DLNM_SAVE_SVG", unset = "1")) %in% c("1", "true", "yes", "y", "on")
  file_ext <- tolower(tools::file_ext(filename))
  if (save_svg && identical(file_ext, "png")) {
    svg_file <- sub("\\.[Pp][Nn][Gg]$", ".svg", filename)
    svg_args <- list(filename = svg_file, plot = plot, ..., device = "svg")
    tryCatch(
      do.call(.ggplot2_ggsave, svg_args),
      error = function(e) {
        warning("SVG companion save failed for ", svg_file, ": ", conditionMessage(e), call. = FALSE)
      }
    )
  }
  invisible(filename)
}
ggsave <- save_plot_with_editable_svg

# Set font for Chinese characters (Windows)
if (.Platform$OS.type == "windows") {
  windowsFonts(SimHei = windowsFont("SimHei"))
}

# 【V5.2】内存管理设置（处理大数据集）
# 增加R的内存限制（Windows）
if (.Platform$OS.type == "windows") {
  memory.limit(size = 56000)  # 设置为56GB（根据你的系统RAM调整）
}

# 垃圾回收设置
gc()  # 清理内存
options(expressions = 5e5)  # 增加表达式嵌套限制

# ========== Configuration ==========
BASE_DIR <- Sys.getenv("HEATPA_DATA_ROOT", unset = file.path(getwd(), "external_data"))
TRIPS_DIR <- file.path(BASE_DIR, "output")
TEMP_DIR <- file.path(BASE_DIR, "5_Daily_52Cities/5_Daily_52Cities")

# Submission time-control specification. The primary model follows the
# environmental time-series convention of a fixed natural cubic spline for
# continuous calendar time (7 df per observed year). The archived
# year-factor + cyclic day-of-year specification remains available as a
# sensitivity analysis through DLNM_TIME_CONTROL_SPEC=cyclic_doy_year.
TIME_CONTROL_SPEC <- tolower(trimws(Sys.getenv(
  "DLNM_TIME_CONTROL_SPEC",
  unset = "continuous_time_ns"
)))
if (!TIME_CONTROL_SPEC %in% c("continuous_time_ns", "cyclic_doy_year")) {
  stop("DLNM_TIME_CONTROL_SPEC must be continuous_time_ns or cyclic_doy_year")
}
TIME_DF_PER_YEAR <- as.integer(Sys.getenv("DLNM_TIME_DF_PER_YEAR", unset = "7"))
WEATHER_NS_DF <- as.integer(Sys.getenv("DLNM_WEATHER_NS_DF", unset = "3"))
if (!is.finite(TIME_DF_PER_YEAR) || TIME_DF_PER_YEAR < 1L) {
  stop("DLNM_TIME_DF_PER_YEAR must be a positive integer")
}
if (!is.finite(WEATHER_NS_DF) || WEATHER_NS_DF < 1L) {
  stop("DLNM_WEATHER_NS_DF must be a positive integer")
}

build_fixed_ns_spec <- function(values, df = 3L, knots_from_positive = FALSE) {
  x <- as.numeric(values)
  x <- x[is.finite(x)]
  if (length(x) < 10L || length(unique(x)) < (df + 1L)) return(NULL)

  boundary <- range(x)
  knot_source <- if (knots_from_positive) x[x > boundary[1]] else x
  if (length(unique(knot_source)) < (df - 1L)) return(NULL)

  probs <- seq(0, 1, length.out = df + 1L)[-c(1L, df + 1L)]
  knots <- unique(as.numeric(stats::quantile(
    knot_source,
    probs = probs,
    na.rm = TRUE,
    names = FALSE,
    type = 8
  )))
  knots <- knots[knots > boundary[1] & knots < boundary[2]]
  if (length(knots) != (df - 1L)) return(NULL)
  list(knots = knots, boundary = boundary, df = df)
}

env_flag <- function(name, default = FALSE) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) return(default)
  tolower(value) %in% c("1", "true", "yes", "y", "on")
}

# Stage-1 resume is on by default. Set DLNM_STAGE1_FORCE_RERUN=1 to overwrite
# existing model RDS files, or DLNM_STAGE1_RESUME=0 to ignore checkpoints.
STAGE1_RESUME <- env_flag("DLNM_STAGE1_RESUME", TRUE)
STAGE1_FORCE_RERUN <- env_flag("DLNM_STAGE1_FORCE_RERUN", FALSE)

resolve_existing_path <- function(paths) {
  existing <- paths[file.exists(paths)]
  if (length(existing) > 0) existing[[1]] else paths[[1]]
}

resolve_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE))
  }

  ofiles <- vapply(
    sys.frames(),
    function(frame) if (!is.null(frame$ofile)) frame$ofile else NA_character_,
    character(1)
  )
  ofiles <- ofiles[!is.na(ofiles)]
  if (length(ofiles) > 0) {
    return(normalizePath(dirname(tail(ofiles, 1)), winslash = "/", mustWork = FALSE))
  }

  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

SCRIPT_DIR <- resolve_script_dir()
GRID10_ENV_PREDICTOR_FILE <- resolve_existing_path(c(
  file.path(SCRIPT_DIR, "grid10_environment_predictors.csv"),
  file.path(getwd(), "grid10_environment_predictors.csv"),
  file.path(BASE_DIR, "grid10_environment_predictors.csv")
))

# 【V5新增】使用1km精度grid_vars.gpkg（包含GDP、犯罪率、失业率、路口密度等）
GRID_VARS_FILE <- resolve_existing_path(c(
  file.path(BASE_DIR, "VAR", "grid_vars.gpkg")
))
CITY_BOUNDARY_FILE <- resolve_existing_path(c(
  file.path(BASE_DIR, "VAR", "City", "City.shp")
))

format_exposure_label <- function(indicator) {
  key <- tolower(indicator)
  if (key == "cehwi") return("CEHWI heatwave exposure")
  if (key == "exceeded_quantity") return("Exceeded Quantity heatwave exposure")
  paste0(toupper(indicator), " heatwave exposure")
}

build_rr_caption <- function(indicator,
                             has_pi = FALSE,
                             pi_caption = NULL,
                             ci_clipped = FALSE,
                             reliability = "STABLE",
                             include_percentiles = FALSE,
                             x_truncated = TRUE) {
  parts <- c(
    paste0("How to read: X-axis = ", format_exposure_label(indicator),
           "; Y-axis RR compares the PA outcome at each exposure level with the DLNM reference exposure."),
    "RR = 1 means no difference vs the reference exposure; RR > 1 means higher PA than reference; RR < 1 means lower PA than reference.",
    if (has_pi) "Dark gray band = 95% CI; light gray band = 95% prediction interval." else "Gray band = 95% CI.",
    if (include_percentiles) "Vertical lines mark the 25th, 75th, and 90th exposure percentiles." else NULL,
    if (x_truncated) "X-axis is truncated at the 98th percentile for readability." else NULL,
    if (ci_clipped) "Extreme CI values were clipped for plotting; use the csv output for exact values." else NULL,
    if (!is.null(pi_caption) && nzchar(pi_caption)) pi_caption else NULL,
    if (!is.null(reliability) && reliability != "STABLE") paste0("Model flag: ", reliability, " - interpret as exploratory.") else NULL
  )
  paste(parts[!is.na(parts) & nzchar(parts)], collapse = "\n")
}

build_partition_rr_subtitle <- function(partition_family,
                                        partition_name,
                                        indicator,
                                        model_type,
                                        n_cities,
                                        pooled_df = NULL,
                                        meta_model = NULL,
                                        reliability = "STABLE") {
  effect_text <- "Peak RR = N/A"
  if (!is.null(pooled_df) &&
      is.data.frame(pooled_df) &&
      all(c("cehwi", "rr") %in% names(pooled_df))) {
    pooled_df_valid <- pooled_df %>%
      filter(is.finite(cehwi), is.finite(rr))
    if (nrow(pooled_df_valid) > 0) {
      peak_idx <- which.max(abs(pooled_df_valid$rr - 1))[1]
      peak_rr <- pooled_df_valid$rr[peak_idx]
      peak_exposure <- pooled_df_valid$cehwi[peak_idx]
      effect_text <- paste0(
        "Peak RR = ", round(peak_rr, 3),
        " @ ", round(peak_exposure, 2),
        " (", sprintf("%+.1f", (peak_rr - 1) * 100), "%)"
      )
    }
  }
  
  i2_text <- "I² = N/A"
  heterogeneity_interp <- ""
  summary_obj <- tryCatch(summary(meta_model), error = function(e) NULL)
  qstat <- tryCatch(summary_obj$qstat, error = function(e) NULL)
  if (!is.null(qstat) &&
      !is.null(qstat$Q) &&
      !is.null(qstat$df) &&
      length(qstat$Q) > 0 &&
      length(qstat$df) > 0 &&
      is.finite(qstat$Q[1]) &&
      is.finite(qstat$df[1]) &&
      qstat$Q[1] > 0) {
    q_val <- qstat$Q[1]
    q_df <- qstat$df[1]
    i2_value <- if (q_val > q_df) ((q_val - q_df) / q_val) * 100 else 0
    i2_text <- paste0("I² = ", round(i2_value, 1), "%")
    heterogeneity_interp <- if (i2_value > 90) {
      " (Very High)"
    } else if (i2_value > 75) {
      " (High)"
    } else if (i2_value > 50) {
      " (Moderate)"
    } else if (i2_value > 25) {
      " (Low-Moderate)"
    } else {
      " (Low)"
    }
  }
  
  aic_text <- "AIC = N/A"
  aic_value <- tryCatch(AIC(meta_model), error = function(e) NA_real_)
  if (is.finite(aic_value)) {
    aic_text <- paste0("AIC = ", round(aic_value, 1))
  }
  
  metrics <- c(
    paste0("N cities = ", n_cities),
    effect_text,
    paste0(i2_text, heterogeneity_interp),
    aic_text,
    if (!is.null(reliability) && reliability != "STABLE") paste0("Flag = ", reliability) else NULL
  )
  
  paste0(
    partition_family, ": ", partition_name,
    " | Meta-regression: ", toupper(indicator),
    " vs PA | Model: ", toupper(model_type),
    " | Reference: ", toupper(indicator), " = 0\n",
    paste(metrics[!is.na(metrics) & nzchar(metrics)], collapse = "  |  ")
  )
}

build_rr_distribution_combined_plot <- function(rr_plot,
                                                hist_plot,
                                                caption_text,
                                                heights = c(2, 1),
                                                caption_size = 10) {
  rr_plot_clean <- rr_plot +
    labs(caption = NULL) +
    theme(plot.caption = element_blank())
  hist_plot_clean <- hist_plot +
    labs(caption = NULL) +
    theme(plot.caption = element_blank())
  
  patchwork::wrap_plots(rr_plot_clean, hist_plot_clean, ncol = 1, heights = heights) +
    patchwork::plot_annotation(
      caption = caption_text,
      theme = theme(
        plot.caption = element_text(
          size = caption_size,
          color = "gray40",
          hjust = 1,
          lineheight = 1.1,
          margin = margin(t = 8)
        ),
        plot.margin = margin(5, 10, 5, 10)
      )
    )
}

forest_direction_colors <- c(
  "Positive" = "#2166AC",
  "Negative" = "#B2182B",
  "Neutral" = "#7F7F7F"
)

forest_significance_stars <- function(p_values, blank_ns = TRUE) {
  ns_value <- if (blank_ns) "" else "ns"
  ifelse(
    is.na(p_values), "",
    ifelse(
      p_values < 0.001, "***",
      ifelse(
        p_values < 0.01, "**",
        ifelse(p_values < 0.05, "*", ns_value)
      )
    )
  )
}

forest_effect_direction <- function(values, neutral_zero = TRUE) {
  ifelse(
    is.na(values), "Neutral",
    ifelse(
      values > 0, "Positive",
      ifelse(values < 0, "Negative", ifelse(neutral_zero, "Neutral", "Positive"))
    )
  )
}

forest_star_offset <- function(xmin, xmax, frac = 0.018, fallback = 0.1) {
  span <- diff(range(c(xmin, xmax), na.rm = TRUE))
  if (!is.finite(span) || span <= 0) {
    span <- fallback
  }
  span * frac
}

control_term_label <- function(term_name) {
  case_when(
        grepl("relative_humidity", term_name) ~ "Relative Humidity",
    grepl("log_precip|precipitation", term_name) ~ "Precipitation",
    grepl("wind_speed", term_name) ~ "Wind Speed",
    grepl("\\bdoy\\b", term_name) ~ "Day-of-season adjustment",
        grepl("dow_fac", term_name, fixed = TRUE) ~ "Day of Week",
    grepl("has_snow", term_name, fixed = TRUE) ~ "Snow Indicator",
    grepl("year_fac", term_name) ~ paste0("Year ", str_replace(term_name, "year_fac", "")),
    TRUE ~ term_name
  )
}

detect_smooth_source_var <- function(term_name, available_names) {
  candidates <- case_when(
        grepl("relative_humidity", term_name) ~ "relative_humidity",
    grepl("log_precip", term_name) ~ "log_precip",
    grepl("precipitation", term_name) ~ "precipitation",
    grepl("wind_speed", term_name) ~ "wind_speed",
    grepl("\\bdoy\\b", term_name) ~ "doy",
    TRUE ~ NA_character_
  )
  if (is.na(candidates) || !candidates %in% available_names) {
    return(NULL)
  }
  candidates
}

compute_weekend_contrast <- function(m_gam) {
  if (is.null(m_gam)) return(NULL)
  coef_vec <- tryCatch(coef(m_gam), error = function(e) NULL)
  vc <- tryCatch(vcov(m_gam), error = function(e) NULL)
  if (is.null(coef_vec) || is.null(vc) || length(coef_vec) == 0) return(NULL)
  
  coef_names <- names(coef_vec)
  dow_terms <- grep("^dow_fac", coef_names, value = TRUE)
  if (length(dow_terms) == 0) return(NULL)
  
  weights <- setNames(rep(0, length(coef_vec)), coef_names)
  for (day_name in c("Tue", "Wed", "Thu", "Fri")) {
    day_term <- dow_terms[grepl(day_name, dow_terms, fixed = TRUE)]
    if (length(day_term) > 0) weights[day_term[1]] <- -1 / 5
  }
  for (day_name in c("Sat", "Sun")) {
    day_term <- dow_terms[grepl(day_name, dow_terms, fixed = TRUE)]
    if (length(day_term) > 0) weights[day_term[1]] <- 1 / 2
  }
  
  if (!any(weights != 0)) return(NULL)
  estimate <- sum(weights * coef_vec, na.rm = TRUE)
  se <- sqrt(as.numeric(t(weights) %*% vc %*% weights))
  if (!is.finite(se) || se <= 0) se <- NA_real_
  df_res <- tryCatch(m_gam$df.residual, error = function(e) NA_real_)
  p_value <- if (is.finite(se) && se > 0) {
    test_stat <- estimate / se
    if (is.finite(df_res) && df_res > 0) 2 * pt(-abs(test_stat), df = df_res) else 2 * pnorm(-abs(test_stat))
  } else {
    NA_real_
  }
  
  tibble(
    variable = "Weekend vs Weekday",
    raw_term = "weekend_mean_minus_weekday_mean_from_dow_fac",
    type = "Parametric",
    coefficient = estimate,
    se = se,
    ci_low = ifelse(is.finite(se), estimate - 1.96 * se, NA_real_),
    ci_high = ifelse(is.finite(se), estimate + 1.96 * se, NA_real_),
    edf = NA_real_,
    f_value = NA_real_,
    p_value = p_value,
    significant = is.finite(p_value) && p_value < 0.05,
    contrast_note = "Mean weekend effect minus mean weekday effect; model still controls full day-of-week fixed effects"
  )
}

extract_smooth_term_contrasts <- function(m_gam, smooth_table, probs = c(0.1, 0.9)) {
  if (is.null(smooth_table) || nrow(smooth_table) == 0) {
    return(tibble())
  }
  
  model_data <- tryCatch(model.frame(m_gam), error = function(e) NULL)
  if (is.null(model_data) && !is.null(m_gam$model)) {
    model_data <- as.data.frame(m_gam$model)
  }
  if (is.null(model_data) || nrow(model_data) == 0) {
    return(tibble())
  }
  
  smooth_terms <- rownames(smooth_table)
  smooth_terms <- smooth_terms[!grepl("cb_cehwi|cb_exceeded|fish_id|\\bdoy\\b", smooth_terms)]
  if (length(smooth_terms) == 0) {
    return(tibble())
  }
  
  term_pred <- tryCatch(
    predict(m_gam, type = "terms", terms = smooth_terms, se.fit = TRUE),
    error = function(e) NULL
  )
  if (is.null(term_pred) || is.null(term_pred$fit)) {
    return(tibble())
  }
  
  fit_mat <- as.matrix(term_pred$fit)
  se_mat <- as.matrix(term_pred$se.fit)
  smooth_rows <- list()
  
  for (term_name in smooth_terms) {
    source_var <- detect_smooth_source_var(term_name, names(model_data))
    if (is.null(source_var) || !term_name %in% colnames(fit_mat)) next
    
    term_values <- model_data[[source_var]]
    term_fit <- fit_mat[, term_name]
    term_se <- se_mat[, term_name]
    valid_idx <- is.finite(term_values) & is.finite(term_fit)
    if (sum(valid_idx) < 10) next
    
    low_cut <- tryCatch(quantile(term_values[valid_idx], probs[1], na.rm = TRUE, type = 8), error = function(e) NA_real_)
    high_cut <- tryCatch(quantile(term_values[valid_idx], probs[2], na.rm = TRUE, type = 8), error = function(e) NA_real_)
    contrast_note <- paste0(
      source_var, " P", round(probs[2] * 100),
      " vs P", round(probs[1] * 100),
      " term contrast"
    )
    if ((!is.finite(low_cut) || !is.finite(high_cut) || high_cut <= low_cut) &&
        grepl("log_precip|precipitation", term_name)) {
      positive_values <- term_values[valid_idx & term_values > 0]
      if (length(unique(positive_values)) >= 2) {
        low_cut <- 0
        high_cut <- tryCatch(quantile(positive_values, probs[2], na.rm = TRUE, type = 8), error = function(e) NA_real_)
        contrast_note <- paste0(
          source_var, " wet-day P", round(probs[2] * 100),
          " vs dry-day contrast"
        )
      }
    }
    if (!is.finite(low_cut) || !is.finite(high_cut) || high_cut <= low_cut) next
    
    low_idx <- valid_idx & term_values <= low_cut
    high_idx <- valid_idx & term_values >= high_cut
    if (sum(low_idx) < 3 || sum(high_idx) < 3) next
    
    estimate <- mean(term_fit[high_idx], na.rm = TRUE) - mean(term_fit[low_idx], na.rm = TRUE)
    se_est <- sqrt(
      mean(term_se[high_idx]^2, na.rm = TRUE) / sum(high_idx) +
        mean(term_se[low_idx]^2, na.rm = TRUE) / sum(low_idx)
    )
    if (!is.finite(estimate)) next
    if (!is.finite(se_est) || se_est <= 0) {
      se_est <- NA_real_
    }
    
    smooth_rows[[length(smooth_rows) + 1]] <- tibble(
      variable = control_term_label(term_name),
      raw_term = term_name,
      type = "Smooth",
      coefficient = estimate,
      se = se_est,
      ci_low = ifelse(is.finite(se_est), estimate - 1.96 * se_est, NA_real_),
      ci_high = ifelse(is.finite(se_est), estimate + 1.96 * se_est, NA_real_),
      edf = suppressWarnings(as.numeric(smooth_table[term_name, "edf"])),
      f_value = suppressWarnings(as.numeric(smooth_table[term_name, "F"])),
      p_value = suppressWarnings(as.numeric(smooth_table[term_name, "p-value"])),
      significant = suppressWarnings(as.numeric(smooth_table[term_name, "p-value"])) < 0.05,
      contrast_note = contrast_note
    )
  }
  
  if (length(smooth_rows) == 0) {
    return(tibble())
  }
  
  bind_rows(smooth_rows)
}

rr_plot_theme <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 2),
      plot.subtitle = element_text(size = base_size - 2, color = "gray30", lineheight = 1.2),
      axis.title = element_text(face = "bold", size = base_size - 2),
      plot.caption = element_text(size = base_size - 4, color = "gray40", hjust = 1, lineheight = 1.1),
      panel.grid = element_blank(),
      axis.line = element_line(color = "gray30", linewidth = 0.5),
      plot.margin = margin(5, 10, 5, 10)
    )
}

rr_hist_theme <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 2),
      plot.subtitle = element_text(size = base_size - 1, color = "gray30"),
      axis.title = element_text(face = "bold", size = base_size),
      plot.caption = element_text(size = base_size - 2, color = "gray40", hjust = 1),
      panel.grid = element_blank(),
      axis.line = element_line(color = "gray30", linewidth = 0.5),
      plot.margin = margin(5, 10, 5, 10)
    )
}

create_partition_meta_summary_row <- function(partition_family,
                                              partition_name,
                                              indicator,
                                              model_type,
                                              n_cities,
                                              n_cities_total = NA_integer_,
                                              model_status,
                                              stability_flag = NA_character_,
                                              meta_predictors_output = FALSE,
                                              af_output = FALSE,
                                              output_dir = NA_character_,
                                              status_note = NA_character_) {
  tibble(
    partition_family = partition_family,
    partition_name = partition_name,
    indicator = toupper(indicator),
    model_type = toupper(model_type),
    n_cities = as.integer(n_cities),
    n_cities_total = as.integer(n_cities_total),
    model_status = model_status,
    stability_flag = stability_flag,
    meta_predictors_output = meta_predictors_output,
    af_output = af_output,
    output_dir = output_dir,
    status_note = status_note
  )
}

meta_predictor_mode_label <- function(mode = NULL) {
  if (is.null(mode) || !nzchar(mode)) {
    mode <- get0("META_PREDICTOR_MODE", ifnotfound = "mean")
  }
  if (grepl("_NO_CRIME$", toupper(mode))) {
    return(toupper(mode))
  }
  mode_label <- if (identical(tolower(mode), "gini")) "GINI" else "MEAN"
  include_crime <- get0("META_INCLUDE_CRIME", ifnotfound = TRUE)
  if (!isTRUE(include_crime)) {
    mode_label <- paste0(mode_label, "_NO_CRIME")
  }
  mode_label
}

meta_predictor_output_dir <- function(parent_dir, mode = NULL, create = TRUE) {
  out_dir <- file.path(parent_dir, paste0("META_PREDICTORS_", meta_predictor_mode_label(mode)))
  if (isTRUE(create)) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_dir
}

meta_model_output_dir <- function(parent_dir, mode = NULL, create = TRUE) {
  out_dir <- file.path(parent_dir, paste0("META_MODEL_", meta_predictor_mode_label(mode)))
  if (isTRUE(create)) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_dir
}

national_af_output_dir <- function(parent_dir, mode = NULL, create = TRUE) {
  out_dir <- file.path(parent_dir, paste0("NATIONAL_AF_", meta_predictor_mode_label(mode)))
  if (isTRUE(create)) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_dir
}

safe_write_csv <- function(x, path, label = NULL) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tryCatch({
    readr::write_csv(x, path)
    TRUE
  }, error = function(e) {
    msg <- paste0(
      format(Sys.time()), " | write_csv failed | ",
      ifelse(is.null(label), basename(path), label), " | ",
      path, " | ", conditionMessage(e)
    )
    cat("      Warning: ", msg, "\n", sep = "")
    try(cat(msg, "\n", file = file.path(dirname(path), "write_errors.log"), append = TRUE), silent = TRUE)
    FALSE
  })
}

save_meta_model_artifact <- function(obj, path, label = "meta_model") {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  if (isTRUE(get0("SAVE_META_MODEL_RDS", ifnotfound = FALSE))) {
    readr::write_rds(obj, path)
    return(TRUE)
  }
  status_path <- file.path(dirname(path), paste0(tools::file_path_sans_ext(basename(path)), "_skipped.txt"))
  writeLines(
    c(
      paste0(label, ".rds was intentionally not saved."),
      "Reason: SAVE_META_MODEL_RDS is FALSE to avoid multi-GB duplicated meta-model artifacts.",
      "All plotting/model summaries are still written as CSV/PNG outputs.",
      "Set SAVE_META_MODEL_RDS <- TRUE only if the heavy RDS object is explicitly needed."
    ),
    status_path
  )
  FALSE
}

find_first_numeric_col <- function(df, candidates, min_unique = 6) {
  candidates <- intersect(candidates, names(df))
  for (nm in candidates) {
    vals <- suppressWarnings(as.numeric(df[[nm]]))
    vals <- vals[is.finite(vals)]
    if (length(vals) >= 10 && dplyr::n_distinct(vals) >= min_unique) {
      return(nm)
    }
  }
  NULL
}

TEMPERATURE_CONTROL_VERSION <- "source_join_v1"

temperature_control_candidates <- function() {
  c(
    "mean_temperature_c", "mean_temperature", "mean_temp_c", "tmean", "tavg",
    "temp_mean", "temp_avg", "temperature_mean", "daily_mean_temp",
    "apparent_temperature_c", "apparent_temperature", "heat_index_c", "heat_index"
  )
}

ensure_temperature_control_columns <- function(data, source_data = NULL) {
  if (is.null(data) || nrow(data) == 0) return(data)
  if (!is.null(find_first_numeric_col(data, temperature_control_candidates(), min_unique = 6))) {
    return(data)
  }
  if (is.null(source_data) || !"date" %in% names(data) || !"date" %in% names(source_data)) {
    return(data)
  }
  
  source_col <- find_first_numeric_col(source_data, temperature_control_candidates(), min_unique = 6)
  if (is.null(source_col)) return(data)
  
  temp_lookup <- tibble(
    date = as.Date(source_data$date),
    .temperature_control_value = suppressWarnings(as.numeric(source_data[[source_col]]))
  ) %>%
    filter(!is.na(date)) %>%
    group_by(date) %>%
    summarise(
      mean_temperature_c = {
        vals <- .temperature_control_value[is.finite(.temperature_control_value)]
        if (length(vals) == 0) NA_real_ else mean(vals)
      },
      .groups = "drop"
    )
  
  # Replace unusable stale columns instead of creating .x/.y join duplicates.
  if ("mean_temperature_c" %in% names(data)) {
    data$mean_temperature_c <- NULL
  }
  data$date <- as.Date(data$date)
  data <- data %>% left_join(temp_lookup, by = "date")
  cat("    - Temperature column joined into model data from source: ", source_col, "\n", sep = "")
  data
}

read_integer_choice <- function(prompt, valid_choices, env_var = NULL) {
  if (!is.null(env_var) && nzchar(env_var)) {
    env_value <- Sys.getenv(env_var, unset = "")
    if (nzchar(env_value)) {
      env_choice <- suppressWarnings(as.integer(env_value))
      if (!is.na(env_choice) && env_choice %in% valid_choices) {
        cat(prompt, env_choice, " [from ", env_var, "]\n", sep = "")
        return(env_choice)
      }
      cat("Invalid ", env_var, "=", env_value, "; falling back to interactive prompt.\n", sep = "")
    }
  }
  choice <- suppressWarnings(as.integer(readline(prompt = prompt)))
  while (is.na(choice) || !choice %in% valid_choices) {
    cat("Invalid choice; please enter one of: ", paste(valid_choices, collapse = "/"), "\n", sep = "")
    choice <- suppressWarnings(as.integer(readline(prompt = prompt)))
  }
  choice
}

resolve_script_file <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE))
  }
  ofiles <- vapply(sys.frames(), function(frame) {
    if (!is.null(frame$ofile)) frame$ofile else NA_character_
  }, character(1))
  ofiles <- ofiles[!is.na(ofiles)]
  if (length(ofiles) > 0) {
    return(normalizePath(tail(ofiles, 1), winslash = "/", mustWork = FALSE))
  }
  NA_character_
}

lag_effect_days_file <- function() {
  resolve_existing_path(c(
    file.path(SCRIPT_DIR, "ward_k4_12d_75city_lowess_multi_threshold_effect_days.csv"),
    file.path(getwd(), "ward_k4_12d_75city_lowess_multi_threshold_effect_days.csv"),
    file.path(BASE_DIR, "ward_k4_12d_75city_lowess_multi_threshold_effect_days.csv"),
    file.path(SCRIPT_DIR, "..", "data", "lag_assignment.csv")
  ))
}

normalize_city_code <- function(x) {
  x <- as.character(x)
  x <- stringr::str_trim(x)
  x <- stringr::str_replace_all(x, "\\s+", "_")
  x <- stringr::str_replace_all(x, "\\.", "")
  x
}

lag_days_to_max_lag <- function(days, fallback_days = 12L) {
  days <- suppressWarnings(as.numeric(days))
  if (length(days) == 0 || !is.finite(days)) days <- fallback_days
  days <- as.integer(round(days))
  if (is.na(days) || days <= 0) days <- fallback_days
  # "7-day" means lag 0-6; "12-day" means lag 0-11.
  max(1L, min(11L, days - 1L))
}

load_lag_effect_days_table <- function() {
  lag_file <- lag_effect_days_file()
  if (!file.exists(lag_file)) {
    cat("Lag effect-days file not found; dynamic lag scenarios will fall back to lag12: ", lag_file, "\n", sep = "")
    return(tibble())
  }
  tryCatch({
    read_csv(lag_file, show_col_types = FALSE) %>%
      clean_names() %>%
      mutate(
        city_code = normalize_city_code(city),
        strong_effect_days = suppressWarnings(as.numeric(strong_effect_days_fdr001)),
        group_median_strong_days = suppressWarnings(as.numeric(group_median_strong)),
        group_median_overall_days = suppressWarnings(as.numeric(group_median_overall))
      )
  }, error = function(e) {
    cat("Failed to read lag effect-days file; dynamic lag scenarios will fall back to lag12: ",
        conditionMessage(e), "\n", sep = "")
    tibble()
  })
}

LAG_EFFECT_DAYS_TABLE <- NULL
LAG_SCENARIO_KEY <- "lag_group_median_overall"
LAG_SCENARIO_LABEL <- "lag_group_median_overall"
LAG_SCENARIO_DESCRIPTION <- "Phenotype-specific lag window: C3 lag 0-7; C1/C2/C4 lag 0-11"
LAG_DAYS_CURRENT <- 12L

configure_lag_scenario <- function(scenario_key) {
  scenario_key <- tolower(scenario_key)
  scenario_key <- dplyr::case_when(
    scenario_key %in% c("lag7", "7", "fixed7") ~ "lag7",
    scenario_key %in% c("lag12", "12", "fixed12") ~ "lag12",
    scenario_key %in% c("city_strong", "lag_city_strong", "city") ~ "lag_city_strong",
    scenario_key %in% c("group_median_strong", "lag_group_median_strong", "group_median_strong") ~ "lag_group_median_strong",
    scenario_key %in% c("group_median_overall", "lag_group_median_overall", "group_median_overall", "group_overall") ~ "lag_group_median_overall",
    TRUE ~ "lag12"
  )
  LAG_SCENARIO_KEY <<- scenario_key
  LAG_SCENARIO_LABEL <<- scenario_key
  LAG_SCENARIO_DESCRIPTION <<- switch(
    scenario_key,
    lag7 = "Fixed lag 0-6 days (7-day sensitivity)",
    lag12 = "Fixed lag 0-11 days (12-day sensitivity/main lag12-compatible window)",
    lag_city_strong = "City-specific strong-effect lag days from ward k=4 lag12 LOWESS table",
    lag_group_median_strong = "Cluster median strong-effect lag days from ward k=4 lag12 LOWESS table",
    lag_group_median_overall = "Cluster median overall-effect lag days from ward k=4 lag12 LOWESS table",
    "Fixed lag 0-11 days"
  )
  if (scenario_key %in% c("lag_city_strong", "lag_group_median_strong", "lag_group_median_overall") &&
      (is.null(LAG_EFFECT_DAYS_TABLE) || nrow(LAG_EFFECT_DAYS_TABLE) == 0)) {
    LAG_EFFECT_DAYS_TABLE <<- load_lag_effect_days_table()
  }
  invisible(scenario_key)
}

resolve_lag_days_for_city <- function(city_code) {
  city_code <- normalize_city_code(city_code)
  if (identical(LAG_SCENARIO_KEY, "lag7")) return(7L)
  if (identical(LAG_SCENARIO_KEY, "lag12")) return(12L)
  if (is.null(LAG_EFFECT_DAYS_TABLE) || nrow(LAG_EFFECT_DAYS_TABLE) == 0) return(12L)
  lag_row <- LAG_EFFECT_DAYS_TABLE %>% filter(city_code == !!city_code) %>% slice(1)
  if (nrow(lag_row) == 0) return(12L)
  days <- switch(
    LAG_SCENARIO_KEY,
    lag_city_strong = lag_row$strong_effect_days[1],
    lag_group_median_strong = lag_row$group_median_strong_days[1],
    lag_group_median_overall = lag_row$group_median_overall_days[1],
    12
  )
  days <- suppressWarnings(as.numeric(days))
  if (!is.finite(days) || days <= 0) 12L else as.integer(round(days))
}

resolve_max_lag_for_city <- function(city_code) {
  lag_days_to_max_lag(resolve_lag_days_for_city(city_code), fallback_days = 12L)
}

set_active_lag_for_city <- function(city_code, verbose = TRUE) {
  LAG_DAYS_CURRENT <<- resolve_lag_days_for_city(city_code)
  MAX_LAG <<- resolve_max_lag_for_city(city_code)
  if (isTRUE(verbose)) {
    cat("  [Lag scenario] ", LAG_SCENARIO_LABEL, " | ", city_code,
        " uses ", LAG_DAYS_CURRENT, " effect day(s), lag 0-", MAX_LAG, "\n", sep = "")
  }
  invisible(MAX_LAG)
}

expected_stage1_max_lag <- function(model_result) {
  if (is.null(model_result)) return(NA_integer_)
  city_code <- model_result$city_code
  if (is.null(city_code) || length(city_code) == 0 || !nzchar(city_code[1])) {
    city_code <- model_result$city
  }
  if (is.null(city_code) || length(city_code) == 0 || !nzchar(city_code[1])) {
    return(as.integer(MAX_LAG))
  }
  resolve_max_lag_for_city(city_code[1])
}

calculate_temperature_control_contrast <- function(cb_temp,
                                                   m_gam,
                                                   temp_values,
                                                   temp_control_col,
                                                   max_lag_value) {
  if (is.null(cb_temp) || is.null(m_gam) || is.null(temp_values)) return(NULL)
  temp_values <- suppressWarnings(as.numeric(temp_values))
  temp_values <- temp_values[is.finite(temp_values)]
  if (length(temp_values) < 20 || dplyr::n_distinct(temp_values) < 6) return(NULL)
  temp_p10 <- suppressWarnings(as.numeric(quantile(temp_values, 0.10, na.rm = TRUE)))
  temp_p90 <- suppressWarnings(as.numeric(quantile(temp_values, 0.90, na.rm = TRUE)))
  if (!is.finite(temp_p10) || !is.finite(temp_p90) || temp_p90 <= temp_p10) return(NULL)
  temp_cp <- tryCatch(
    crosspred(cb_temp, model = m_gam, at = temp_p90, cen = temp_p10, cumul = TRUE),
    error = function(e) NULL
  )
  if (is.null(temp_cp) || is.null(temp_cp$allRRfit)) return(NULL)
  rr <- suppressWarnings(as.numeric(temp_cp$allRRfit[1]))
  rr_low <- suppressWarnings(as.numeric(temp_cp$allRRlow[1]))
  rr_high <- suppressWarnings(as.numeric(temp_cp$allRRhigh[1]))
  if (!is.finite(rr) || rr <= 0 || !is.finite(rr_low) || rr_low <= 0 || !is.finite(rr_high) || rr_high <= 0) {
    return(NULL)
  }
  estimate <- log(rr)
  ci_low <- log(min(rr_low, rr_high))
  ci_high <- log(max(rr_low, rr_high))
  se <- calc_se_from_ci(ci_low, ci_high)
  p_value <- if (is.finite(se) && se > 0) 2 * pnorm(-abs(estimate / se)) else NA_real_
  tibble(
    variable = "Mean/Apparent Temperature",
    raw_term = "cb_temp",
    type = "Cross-basis",
    coefficient = estimate,
    se = se,
    ci_low = ci_low,
    ci_high = ci_high,
    edf = NA_real_,
    f_value = NA_real_,
    p_value = p_value,
    significant = is.finite(p_value) && p_value < 0.05,
    contrast_note = paste0(
      temp_control_col, " cumulative cross-basis P90 vs P10; lag 0-",
      max_lag_value, "; RR=", round(rr, 3),
      " (95% CI ", round(exp(ci_low), 3), "-", round(exp(ci_high), 3), ")"
    )
  )
}

is_current_stage1_result <- function(model_result) {
  if (is.null(model_result)) return(FALSE)
  result_lag <- suppressWarnings(as.integer(model_result$max_lag))
  if (length(result_lag) != 1 || !is.finite(result_lag)) return(FALSE)
  expected_lag <- expected_stage1_max_lag(model_result)
  if (!identical(result_lag, as.integer(expected_lag))) return(FALSE)
  if (!identical(model_result$temperature_control_version, TEMPERATURE_CONTROL_VERSION)) return(FALSE)
  if (is.null(model_result$includes_temperature_control)) return(FALSE)
  if (isTRUE(model_result$includes_temperature_control) &&
      (is.null(model_result$temperature_control_contrast) ||
       nrow(model_result$temperature_control_contrast) == 0)) {
    return(FALSE)
  }
  TRUE
}

filter_current_stage1_results <- function(model_results, context_label = "Stage-2") {
  if (is.null(model_results) || length(model_results) == 0) return(model_results)
  valid_idx <- vapply(model_results, is_current_stage1_result, logical(1))
  if (any(!valid_idx)) {
    cat(
      "    Warning: removed ", sum(!valid_idx), " stale first-stage result(s) in ",
      context_label, " because their lag window does not match current lag scenario=",
      LAG_SCENARIO_LABEL, ". Rerun Stage-1 for these cities/models in this lag folder.\n", sep = ""
    )
  }
  model_results[valid_idx]
}

af_model_type_colors <- function() {
  c(
    "Composite" = "#D53E4F",
    "Day" = "#FF8C00",
    "Night" = "#9B59B6"
  )
}

activity_modality_colors <- function() {
  c(
    "all" = "#6B7280",
    "ride" = "#9E2A2B",
    "run" = "#E07A5F",
    "walk" = "#2A9D8F"
  )
}

activity_modality_labels <- function() {
  c(
    "all" = "All activity",
    "ride" = "Ride",
    "run" = "Run",
    "walk" = "Walk"
  )
}

activity_analysis_mode_label <- function(mode = NULL) {
  if (is.null(mode) || !nzchar(mode)) {
    mode <- get0("STAGE1_ACTIVITY_MODE", ifnotfound = "combined")
  }
  if (identical(tolower(mode), "activity_3plus1")) "ACTIVITY_3PLUS1" else "COMBINED"
}

activity_analysis_file_suffix <- function(mode = NULL) {
  mode_label <- activity_analysis_mode_label(mode)
  if (identical(mode_label, "COMBINED")) "" else paste0("_", mode_label)
}

analysis_output_suffix <- function(meta_mode = NULL, activity_mode = NULL) {
  paste0("_", meta_predictor_mode_label(meta_mode), activity_analysis_file_suffix(activity_mode))
}

is_activity_split_mode <- function(mode = NULL) {
  identical(activity_analysis_mode_label(mode), "ACTIVITY_3PLUS1")
}

stage1_activity_types <- function(mode = NULL) {
  if (is_activity_split_mode(mode)) {
    c("all", "ride", "run", "walk")
  } else {
    "all"
  }
}

model_base_type <- function(model_type) {
  model_type <- tolower(as.character(model_type))
  dplyr::case_when(
    str_detect(model_type, "^composite") ~ "composite",
    str_detect(model_type, "^day") ~ "day",
    str_detect(model_type, "^night") ~ "night",
    TRUE ~ model_type
  )
}

model_activity_type <- function(model_type) {
  model_type <- tolower(as.character(model_type))
  activity <- str_match(model_type, "^(composite|day|night)_(all|ride|run|walk)$")[, 3]
  ifelse(is.na(activity) | !nzchar(activity), "all", activity)
}

stage1_model_name <- function(base_model, activity = "all", mode = NULL) {
  base_model <- model_base_type(base_model)
  activity <- tolower(activity)
  if (is_activity_split_mode(mode)) {
    paste0(base_model, "_", activity)
  } else {
    base_model
  }
}

stage1_model_types <- function(mode = NULL) {
  bases <- c("composite", "day", "night")
  activities <- stage1_activity_types(mode)
  as.vector(unlist(lapply(bases, function(base) {
    vapply(activities, function(activity) stage1_model_name(base, activity, mode), character(1))
  }), use.names = FALSE))
}

normalize_stage1_model_name <- function(model_name) {
  model_name <- tolower(as.character(model_name))
  model_name <- str_replace_all(model_name, "dayonly", "day")
  model_name <- str_replace_all(model_name, "nightonly", "night")
  valid_split <- str_match(model_name, "^(composite|day|night)_(all|ride|run|walk)$")
  if (!any(is.na(valid_split))) return(model_name)
  model_base_type(model_name)
}

stage1_model_display_label <- function(model_type) {
  base_label <- str_to_title(model_base_type(model_type))
  activity <- model_activity_type(model_type)
  if (activity == "all" && !is_activity_split_mode()) return(base_label)
  paste0(base_label, " - ", activity_modality_labels()[activity])
}

stage1_model_color <- function(model_type) {
  activity <- model_activity_type(model_type)
  if (is_activity_split_mode() || str_detect(tolower(model_type), "_(all|ride|run|walk)$")) {
    return(unname(activity_modality_colors()[activity]))
  }
  base <- str_to_title(model_base_type(model_type))
  unname(af_model_type_colors()[base])
}

partition_has_meta_predictors <- function(output_dir, mode = NULL) {
  predictor_dir <- meta_predictor_output_dir(output_dir, mode, create = FALSE)
  required_files <- c(
    "city_level_covariates.csv",
    "city_level_covariates_SUMMARY.csv",
    "city_level_covariates_forest.png",
    "city_level_covariates_forest_FULL.png"
  )
  mode_files_ok <- all(file.exists(file.path(predictor_dir, required_files)))
  if (dir.exists(predictor_dir)) {
    return(mode_files_ok)
  }
  # Backward compatibility for old outputs generated before MEAN/GINI subfolders existed.
  all(file.exists(file.path(output_dir, required_files)))
}

partition_has_af_outputs <- function(output_dir) {
  any(file.exists(file.path(output_dir, c(
    "AF_forest.png",
    "AF_meta_summary.csv",
    "AF_summary.csv"
  ))))
}

count_partition_model_cities <- function(city_names, successful_cities, indicator, model_type) {
  sum(vapply(city_names, function(city_name) {
    city_key <- paste0(city_name, "_", indicator)
    if (!city_key %in% names(successful_cities)) {
      return(FALSE)
    }
    model_type %in% names(successful_cities[[city_key]])
  }, logical(1)))
}

write_meta_predictor_status <- function(output_dir, lines, mode = NULL) {
  predictor_dir <- meta_predictor_output_dir(output_dir, mode)
  mode_line <- paste0("Meta-predictor mode: ", meta_predictor_mode_label(mode))
  writeLines(c(mode_line, lines), file.path(predictor_dir, "meta_predictors_status.txt"))
  writeLines(
    c(
      mode_line,
      paste0("Detailed files are in: ", basename(predictor_dir)),
      lines
    ),
    file.path(output_dir, "meta_predictors_status.txt")
  )
}

compute_meta_vif <- function(city_covariates) {
  if (is.null(city_covariates) || ncol(city_covariates) == 0) {
    return(setNames(numeric(0), character(0)))
  }
  if (ncol(city_covariates) == 1) {
    return(setNames(1, names(city_covariates)))
  }
  
  vif_values <- setNames(rep(1, ncol(city_covariates)), names(city_covariates))
  for (var_name in names(city_covariates)) {
    other_vars <- setdiff(names(city_covariates), var_name)
    values <- city_covariates[[var_name]]
    
    if (length(other_vars) == 0) next
    if (sum(!is.na(values)) < 2) {
      vif_values[var_name] <- Inf
      next
    }
    
    sd_value <- suppressWarnings(sd(values, na.rm = TRUE))
    if (!is.finite(sd_value) || sd_value < 1e-8) {
      vif_values[var_name] <- Inf
      next
    }
    
    formula_vif <- as.formula(paste(var_name, "~", paste(other_vars, collapse = " + ")))
    vif_values[var_name] <- tryCatch({
      lm_vif <- lm(formula_vif, data = city_covariates)
      r_squared <- summary(lm_vif)$r.squared
      if (!is.finite(r_squared) || r_squared >= 0.999999) {
        Inf
      } else {
        1 / max(1 - r_squared, 1e-8)
      }
    }, error = function(e) Inf)
  }
  
  vif_values
}

reduce_meta_predictors <- function(city_covariates,
                                   cat_prefix = "      ",
                                   preferred_drop = c("Distance_to_Transit_mean", "FAR_mean",
                                                      "Distance_to_Transit_gini_mean", "FAR_gini_mean"),
                                   severe_vif_threshold = 10,
                                   target_vif_threshold = 15,
                                   min_vars_to_keep = 2) {
  if (is.null(city_covariates) || ncol(city_covariates) == 0) {
    return(list(data = city_covariates, vif = setNames(numeric(0), character(0)), removed = character(0)))
  }
  
  city_covariates <- as.data.frame(city_covariates)
  removed_vars <- character(0)
  
  constant_vars <- names(city_covariates)[vapply(city_covariates, function(x) {
    n_non_missing <- sum(!is.na(x))
    if (n_non_missing < 2) return(TRUE)
    sd_value <- suppressWarnings(sd(x, na.rm = TRUE))
    !is.finite(sd_value) || sd_value < 1e-8
  }, logical(1))]
  
  if (length(constant_vars) > 0) {
    cat(cat_prefix, "检测到常量/近常量协变量，自动移除:", paste(constant_vars, collapse = ", "), "\n")
    city_covariates <- city_covariates[, setdiff(names(city_covariates), constant_vars), drop = FALSE]
    removed_vars <- c(removed_vars, constant_vars)
  }
  
  if (ncol(city_covariates) <= 1) {
    vif_values <- compute_meta_vif(city_covariates)
    return(list(data = city_covariates, vif = vif_values, removed = removed_vars))
  }
  
  vif_values <- compute_meta_vif(city_covariates)
  cat(cat_prefix, "VIF值:\n")
  for (var_name in names(vif_values)) {
    vif_val <- vif_values[var_name]
    status <- if (is.infinite(vif_val) || vif_val > 10) {
      "⚠ 严重共线性"
    } else if (vif_val > 5) {
      "⚠ 中度共线性"
    } else {
      "✓"
    }
    cat(sprintf("%s  %s: %.2f %s\n", cat_prefix, var_name, vif_val, status))
  }
  
  if (any(vif_values > severe_vif_threshold, na.rm = TRUE)) {
    first_drop <- intersect(names(city_covariates), preferred_drop)
    if (length(first_drop) > 0) {
      cat(cat_prefix, "检测到严重共线性（VIF>", severe_vif_threshold,
          "），先移除 ", paste(first_drop, collapse = ", "), "\n", sep = "")
      city_covariates <- city_covariates[, setdiff(names(city_covariates), first_drop), drop = FALSE]
      removed_vars <- c(removed_vars, first_drop)
      if (ncol(city_covariates) > 1) {
        vif_values <- compute_meta_vif(city_covariates)
      } else {
        vif_values <- compute_meta_vif(city_covariates)
      }
    }
  }
  
  while (ncol(city_covariates) > min_vars_to_keep &&
         length(vif_values) > 0 &&
         any(vif_values >= target_vif_threshold, na.rm = TRUE)) {
    max_vif_var <- names(which.max(vif_values))[1]
    cat(cat_prefix, "max VIF 仍 ≥ ", target_vif_threshold,
        "，继续移除: ", max_vif_var,
        " (VIF = ", round(vif_values[max_vif_var], 2), ")\n", sep = "")
    city_covariates <- city_covariates[, setdiff(names(city_covariates), max_vif_var), drop = FALSE]
    removed_vars <- c(removed_vars, max_vif_var)
    vif_values <- compute_meta_vif(city_covariates)
    if (ncol(city_covariates) <= 1) break
  }
  
  cat(cat_prefix, "✓ 共线性处理完成，保留", ncol(city_covariates), "个变量\n")
  list(data = city_covariates, vif = vif_values, removed = unique(removed_vars))
}

META_INCLUDE_CRIME <- FALSE

META_PREDICTOR_BASES_ALL <- c(
  "NDVI_2023",
  "total_20_55",
  "GDP",
  "Crime",
  "Unemployment",
  "BD",
  "Urbanization_Rate",
  "Street_Intersection_Density",
  "Walkability_Index"
)

META_PREDICTOR_EXCLUDED_BASES <- c("total_all", "FAR", "Distance_to_Transit")

refresh_meta_predictor_bases <- function(include_crime = TRUE) {
  bases <- META_PREDICTOR_BASES_ALL
  if (!isTRUE(include_crime)) {
    bases <- setdiff(bases, "Crime")
  }
  bases
}

META_PREDICTOR_BASES_FINAL <- refresh_meta_predictor_bases(META_INCLUDE_CRIME)

standard_city_name <- function(x) {
  x %>%
    trimws() %>%
    str_replace_all("[^A-Za-z0-9]+", "_") %>%
    str_replace_all("^_+|_+$", "")
}

drop_excluded_meta_predictors <- function(df) {
  if (is.null(df) || ncol(df) == 0) return(df)
  excluded_bases <- META_PREDICTOR_EXCLUDED_BASES
  if (!isTRUE(get0("META_INCLUDE_CRIME", ifnotfound = TRUE))) {
    excluded_bases <- unique(c(excluded_bases, "Crime"))
  }
  excluded <- names(df)[vapply(names(df), function(nm) {
    any(vapply(excluded_bases, function(base) {
      grepl(paste0("^", base, "(_|$)"), nm)
    }, logical(1)))
  }, logical(1))]
  df[, setdiff(names(df), excluded), drop = FALSE]
}

screen_meta_predictors_report_only <- function(covariates, cat_prefix = "      ") {
  if (is.null(covariates) || ncol(covariates) == 0) {
    return(list(data = covariates, vif = setNames(numeric(0), character(0)), removed = character(0)))
  }

  covariates <- drop_excluded_meta_predictors(as.data.frame(covariates))
  removed_vars <- character(0)

  constant_vars <- names(covariates)[vapply(covariates, function(x) {
    n_non_missing <- sum(!is.na(x))
    if (n_non_missing < 2) return(TRUE)
    sd_value <- suppressWarnings(sd(as.numeric(x), na.rm = TRUE))
    !is.finite(sd_value) || sd_value < 1e-8
  }, logical(1))]

  if (length(constant_vars) > 0) {
    cat(cat_prefix, "Removed constant/near-constant predictors: ", paste(constant_vars, collapse = ", "), "\n", sep = "")
    covariates <- covariates[, setdiff(names(covariates), constant_vars), drop = FALSE]
    removed_vars <- c(removed_vars, constant_vars)
  }

  vif_values <- compute_meta_vif(covariates)
  if (length(vif_values) > 0) {
    cat(cat_prefix, "VIF report only (no automatic VIF-based predictor deletion):\n", sep = "")
    for (var_name in names(vif_values)) {
      vif_val <- vif_values[var_name]
      vif_label <- if (is.infinite(vif_val)) "Inf" else sprintf("%.2f", vif_val)
      cat(cat_prefix, "  ", var_name, ": ", vif_label, "\n", sep = "")
    }
  }

  list(data = covariates, vif = vif_values, removed = unique(removed_vars))
}

standardize_meta_covariates <- function(covariates) {
  if (is.null(covariates) || ncol(covariates) == 0) return(covariates)
  covariates <- as.data.frame(covariates)
  for (col in names(covariates)) {
    covariates[[col]] <- suppressWarnings(as.numeric(covariates[[col]]))
    sx <- suppressWarnings(sd(covariates[[col]], na.rm = TRUE))
    mx <- suppressWarnings(mean(covariates[[col]], na.rm = TRUE))
    if (is.finite(sx) && sx > 0) {
      covariates[[col]] <- (covariates[[col]] - mx) / sx
    }
  }
  covariates
}

grid10_predictor_columns_for_mode <- function(grid_df, mode = "mean") {
  suffix <- if (identical(mode, "gini")) "_gini" else "_mean"
  intersect(paste0(META_PREDICTOR_BASES_FINAL, suffix), names(grid_df))
}

load_grid10_environment_predictors <- function(mode = META_PREDICTOR_MODE) {
  if (!file.exists(GRID10_ENV_PREDICTOR_FILE)) {
    return(NULL)
  }

  grid_df <- tryCatch(
    read_csv(GRID10_ENV_PREDICTOR_FILE, show_col_types = FALSE),
    error = function(e) {
      cat("          Warning: failed to read grid10_environment_predictors.csv: ", conditionMessage(e), "\n", sep = "")
      NULL
    }
  )
  if (is.null(grid_df) || nrow(grid_df) == 0) return(NULL)

  if (!"city_standard" %in% names(grid_df)) {
    if ("city" %in% names(grid_df)) {
      grid_df$city_standard <- standard_city_name(grid_df$city)
    } else {
      return(NULL)
    }
  }

  x_cols <- grid10_predictor_columns_for_mode(grid_df, mode)
  if (length(x_cols) == 0) return(NULL)
  expected_x_cols <- paste0(META_PREDICTOR_BASES_FINAL, if (identical(mode, "gini")) "_gini" else "_mean")
  missing_x_cols <- setdiff(expected_x_cols, names(grid_df))
  if (length(missing_x_cols) > 0) {
    cat("          Warning: grid10_environment_predictors.csv is missing required stage-2 predictor column(s): ",
        paste(missing_x_cols, collapse = ", "),
        "; falling back to city/grid covariates rebuilt from grid_vars.gpkg.\n", sep = "")
    return(NULL)
  }

  id_cols <- intersect(c("city", "city_standard", "grid10_id", "grid10_lon", "grid10_lat", "analysis_unit", "n_grid1km"), names(grid_df))
  grid_df <- grid_df[, c(id_cols, x_cols), drop = FALSE]
  grid_df$city_standard <- standard_city_name(grid_df$city_standard)
  grid_df
}

add_grid10_missing_city_predictors <- function(df_city,
                                               df_grid = NULL,
                                               target_cities = CITY_LIST,
                                               vars = META_PREDICTOR_BASES_FINAL) {
  if (!file.exists(GRID10_ENV_PREDICTOR_FILE) || length(vars) == 0) {
    return(list(city = df_city, grid = df_grid, added = character()))
  }
  if (is.null(df_city)) df_city <- tibble(city = character())

  grid10_raw <- tryCatch(
    read_csv(GRID10_ENV_PREDICTOR_FILE, show_col_types = FALSE),
    error = function(e) NULL
  )
  if (is.null(grid10_raw) || nrow(grid10_raw) == 0 || !"city_standard" %in% names(grid10_raw)) {
    return(list(city = df_city, grid = df_grid, added = character()))
  }

  grid10_raw <- grid10_raw %>%
    mutate(city_standard = standard_city_name(city_standard))
  present <- standard_city_name(df_city$city)
  missing <- setdiff(standard_city_name(target_cities), present)
  fallback <- grid10_raw %>%
    filter(city_standard %in% missing, !is.na(n_grid1km), n_grid1km > 0)
  if (nrow(fallback) == 0) {
    return(list(city = df_city, grid = df_grid, added = character()))
  }

  mean_cols <- intersect(paste0(vars, "_mean"), names(fallback))
  if (length(mean_cols) == 0) {
    return(list(city = df_city, grid = df_grid, added = character()))
  }

  city_rows <- fallback %>%
    group_by(city = city_standard) %>%
    summarise(across(all_of(mean_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
  names(city_rows) <- sub("_mean$", "", names(city_rows))
  city_rows <- city_rows %>%
    filter(if_any(all_of(intersect(vars, names(.))), ~ is.finite(.x)))

  if (nrow(city_rows) == 0) {
    return(list(city = df_city, grid = df_grid, added = character()))
  }

  for (col in setdiff(names(df_city), names(city_rows))) city_rows[[col]] <- NA
  for (col in setdiff(names(city_rows), names(df_city))) df_city[[col]] <- NA
  df_city <- bind_rows(
    df_city %>% filter(!standard_city_name(city) %in% city_rows$city),
    city_rows[, names(df_city), drop = FALSE]
  )

  grid_rows <- fallback %>%
    transmute(city = city_standard, across(all_of(mean_cols)))
  names(grid_rows) <- sub("_mean$", "", names(grid_rows))
  if (is.null(df_grid)) {
    df_grid <- grid_rows
  } else {
    for (col in setdiff(names(df_grid), names(grid_rows))) grid_rows[[col]] <- NA
    for (col in setdiff(names(grid_rows), names(df_grid))) df_grid[[col]] <- NA
    df_grid <- bind_rows(
      df_grid %>% filter(!standard_city_name(city) %in% city_rows$city),
      grid_rows[, names(df_grid), drop = FALSE]
    )
  }

  list(city = df_city, grid = df_grid, added = city_rows$city)
}

build_grid10_partition_meta_inputs <- function(partition_socioecon_avg,
                                               coef_matrix,
                                               vcov_list,
                                               partition_results = NULL,
                                               mode = META_PREDICTOR_MODE) {
  city_names <- rownames(coef_matrix)
  if (is.null(city_names)) city_names <- paste0("city_", seq_len(nrow(coef_matrix)))
  city_names_std <- standard_city_name(city_names)

  grid_df <- load_grid10_environment_predictors(mode)
  source_label <- "grid10_environment_predictors.csv"

  if (!is.null(grid_df)) {
    x_cols <- grid10_predictor_columns_for_mode(grid_df, mode)
    grid_df <- grid_df %>%
      filter(city_standard %in% city_names_std) %>%
      arrange(match(city_standard, city_names_std))
    if (nrow(grid_df) == 0 || length(x_cols) == 0) {
      grid_df <- NULL
    }
  }

  if (is.null(grid_df)) {
    if (is.null(partition_socioecon_avg) || nrow(partition_socioecon_avg) == 0) {
      return(NULL)
    }

    cov_city <- partition_socioecon_avg
    cov_city$city_standard <- standard_city_name(cov_city$city)
    cov_city <- cov_city %>%
      filter(city_standard %in% city_names_std) %>%
      arrange(match(city_standard, city_names_std))
    x_cols <- setdiff(names(cov_city), c("city", "city_standard"))
    fallback_x_cols <- if (identical(mode, "gini")) {
      paste0(META_PREDICTOR_BASES_FINAL, "_gini_mean")
    } else {
      paste0(META_PREDICTOR_BASES_FINAL, "_mean")
    }
    x_cols <- intersect(x_cols, fallback_x_cols)
    if (length(x_cols) == 0) return(NULL)

    fallback_rows <- lapply(seq_len(nrow(cov_city)), function(i) {
      city_i <- cov_city$city[i]
      n_rep <- NA_integer_
      if (!is.null(partition_results) && city_i %in% names(partition_results)) {
        n_rep <- suppressWarnings(as.integer(partition_results[[city_i]]$n_grids))
      }
      if (!is.finite(n_rep) || n_rep < 1) {
        if (exists("df_socioecon_grid_global") && !is.null(df_socioecon_grid_global) &&
            "city" %in% names(df_socioecon_grid_global)) {
          n_rep <- sum(df_socioecon_grid_global$city == city_i, na.rm = TRUE)
        }
      }
      if (!is.finite(n_rep) || n_rep < 1) n_rep <- 1L
      cov_city[rep(i, n_rep), c("city", "city_standard", x_cols), drop = FALSE] %>%
        mutate(grid10_id = paste0(city_standard, "_pseudo_grid_", seq_len(n())))
    })
    grid_df <- bind_rows(fallback_rows)
    source_label <- "city covariates replicated to available city grids (fallback)"
  }

  x_cols <- setdiff(names(grid_df), c("city", "city_standard", "grid10_id", "grid10_lon", "grid10_lat", "analysis_unit", "n_grid1km", "predictor_status"))
  x_cols <- names(drop_excluded_meta_predictors(grid_df[, x_cols, drop = FALSE]))
  if (identical(mode, "gini")) {
    rename_map <- setNames(paste0(META_PREDICTOR_BASES_FINAL, "_gini_mean"), paste0(META_PREDICTOR_BASES_FINAL, "_gini"))
    for (old_name in intersect(names(rename_map), x_cols)) {
      names(grid_df)[names(grid_df) == old_name] <- rename_map[[old_name]]
      x_cols[x_cols == old_name] <- rename_map[[old_name]]
    }
  }

  x_cols <- intersect(x_cols, names(grid_df))
  x_cols <- x_cols[vapply(grid_df[, x_cols, drop = FALSE], is.numeric, logical(1))]
  if (length(x_cols) == 0) return(NULL)

  row_city_idx <- match(grid_df$city_standard, city_names_std)
  keep <- !is.na(row_city_idx)
  grid_df <- grid_df[keep, , drop = FALSE]
  row_city_idx <- row_city_idx[keep]
  if (nrow(grid_df) == 0) return(NULL)

  coef_matrix_grid <- coef_matrix[row_city_idx, , drop = FALSE]
  vcov_list_grid <- vcov_list[row_city_idx]
  row_ids <- if ("grid10_id" %in% names(grid_df)) as.character(grid_df$grid10_id) else paste0(grid_df$city_standard, "_grid_", seq_len(nrow(grid_df)))
  rownames(coef_matrix_grid) <- make.unique(paste0(grid_df$city_standard, "__", row_ids))

  covariates_grid <- as.data.frame(grid_df[, x_cols, drop = FALSE])
  rownames(covariates_grid) <- rownames(coef_matrix_grid)

  list(
    covariates = covariates_grid,
    coef_matrix = coef_matrix_grid,
    vcov_list = vcov_list_grid,
    source = source_label,
    n_grid_rows = nrow(covariates_grid),
    n_cities = length(unique(grid_df$city_standard)),
    matched_cities = unique(grid_df$city_standard)
  )
}

minimum_partition_meta_cities <- function(n_predictors) {
  max(8L, as.integer(n_predictors) + 2L)
}

annotate_meta_predictor_coefficients <- function(meta_coef_df) {
  meta_coef_df %>%
    mutate(
      variable_label = case_when(
        grepl("BD|Building_Density", coef_name) ~ "Building Density (City Avg)",
        grepl("FAR", coef_name) ~ "Floor Area Ratio (City Avg)",
        grepl("NDVI", coef_name) & grepl("_mean", coef_name) ~ "NDVI (City Avg)",
        grepl("total_20_55|total_20_mean|Pop_mean", coef_name) ~ "Population (City Avg)",
        grepl("GDP", coef_name) ~ "GDP (City Avg)",
        grepl("unemployed_pop_mean|Unemployed_Population_mean", coef_name) ~ "Unemployed Population (City Avg)",
        grepl("Crime", coef_name) ~ "Crime (City Avg)",
        grepl("Unemployment", coef_name) ~ "Unemployment (City Avg)",
        grepl("Urbanization_Rate", coef_name) ~ "Urbanization (City Avg)",
        grepl("Street_Intersection_Density", coef_name) ~ "Street Intersection (City Avg)",
        grepl("Distance_to_Transit", coef_name) ~ "Distance to Transit (City Avg)",
        grepl("Walkability_Index|Walk_mean", coef_name) ~ "Walkability (City Avg)",
        grepl("WS_mean", coef_name) ~ "Wind Speed (City Avg)",
        TRUE ~ coef_name
      ),
      variable_label = ifelse(grepl("_gini_", coef_name),
                              str_replace(variable_label, "\\(City Avg\\)", "(City Gini)"),
                              variable_label),
      predictor_name = case_when(
        grepl("NDVI", coef_name) ~ "1_NDVI",
        grepl("total_20_55|total_20_mean|Pop_mean", coef_name) ~ "2_Population",
        grepl("BD|Building_Density", coef_name) ~ "3_Building_Density",
        grepl("FAR", coef_name) ~ "4_Floor_Area_Ratio",
        grepl("GDP", coef_name) ~ "5_GDP",
        grepl("Crime", coef_name) ~ "6_Crime",
        grepl("Unemployment", coef_name) ~ "7_Unemployment",
        grepl("Urbanization_Rate", coef_name) ~ "8_Urbanization",
        grepl("Street_Intersection_Density", coef_name) ~ "9_Street_Intersection",
        grepl("Distance_to_Transit", coef_name) ~ "10_Distance_to_Transit",
        grepl("Walkability_Index|Walk_mean", coef_name) ~ "11_Walkability",
        grepl("unemployed_pop_mean|Unemployed_Population_mean", coef_name) ~ "12_Unemployed_Pop",
        grepl("WS_mean", coef_name) ~ "13_Wind_Speed",
        TRUE ~ "99_Other"
      ),
      variable_only = case_when(
        grepl("Intercept", coef_name) ~ "Intercept",
        grepl("NDVI", coef_name) & grepl("_mean", coef_name) ~ "NDVI",
        grepl("total_20_55|total_20_mean|Pop_mean", coef_name) ~ "Population (20-55)",
        grepl("GDP", coef_name) ~ "GDP",
        grepl("Crime", coef_name) ~ "Crime",
        grepl("Unemployment", coef_name) ~ "Unemployment",
        grepl("BD|Building_Density", coef_name) ~ "Building Density",
        grepl("FAR", coef_name) ~ "Floor Area Ratio",
        grepl("Urbanization_Rate", coef_name) ~ "Urbanization Rate",
        grepl("Street_Intersection_Density", coef_name) ~ "Street Intersection Density",
        grepl("Distance_to_Transit", coef_name) ~ "Distance to Transit",
        grepl("Walkability_Index|Walk_mean", coef_name) ~ "Walkability Index",
        grepl("unemployed_pop_mean|Unemployed_Population_mean", coef_name) ~ "Unemployed Population",
        grepl("WS_mean", coef_name) ~ "Wind Speed",
        TRUE ~ "Other"
      ),
      cb_lag = str_extract(coef_name, "(cb_cehwiv[0-9]+(\\.l[0-9]+)?|overall_spline[0-9]+|basis[0-9]+|b[0-9]+)"),
      cb_lag = ifelse(is.na(cb_lag), "overall_reduced", cb_lag),
      significant = ifelse(!is.na(p_value) & p_value < 0.05, "sig", "ns")
    )
}

extract_meta_predictor_coefficients <- function(mv_model, indicator, model_type, n_cities) {
  model_summary <- tryCatch(summary(mv_model), error = function(e) NULL)
  if (is.null(model_summary) || !"coefficients" %in% names(model_summary)) {
    return(NULL)
  }
  
  coef_table <- model_summary$coefficients
  if (is.null(coef_table)) {
    return(NULL)
  }
  if (is.null(dim(coef_table))) {
    coef_table <- matrix(coef_table, nrow = 1)
    colnames(coef_table) <- names(model_summary$coefficients)
    rownames(coef_table) <- "coef_1"
  }
  
  coef_names <- rownames(coef_table)
  if (is.null(coef_names)) {
    coef_names <- paste0("coef_", seq_len(nrow(coef_table)))
    rownames(coef_table) <- coef_names
  }
  
  socioecon_rows <- grep("_mean", coef_names)
  if (length(socioecon_rows) == 0) {
    return(NULL)
  }
  
  coef_df <- data.frame(
    indicator = indicator,
    model_type = model_type,
    n_cities = n_cities,
    coef_name = coef_names,
    stringsAsFactors = FALSE
  )
  
  coef_df$coefficient <- if ("Estimate" %in% colnames(coef_table)) coef_table[, "Estimate"] else coef_table[, 1]
  coef_df$se <- if ("Std. Error" %in% colnames(coef_table)) coef_table[, "Std. Error"] else if (ncol(coef_table) >= 2) coef_table[, 2] else NA_real_
  coef_df$z_value <- if ("z" %in% colnames(coef_table)) coef_table[, "z"] else if (ncol(coef_table) >= 3) coef_table[, 3] else NA_real_
  coef_df$p_value <- if ("Pr(>|z|)" %in% colnames(coef_table)) coef_table[, "Pr(>|z|)"] else if ("Pr(>|t|)" %in% colnames(coef_table)) coef_table[, "Pr(>|t|)"] else if (ncol(coef_table) >= 4) coef_table[, 4] else NA_real_
  coef_df$ci_low <- if ("ci.lb" %in% colnames(coef_table)) coef_table[, "ci.lb"] else if (ncol(coef_table) >= 5) coef_table[, 5] else coef_df$coefficient - 1.96 * coef_df$se
  coef_df$ci_high <- if ("ci.ub" %in% colnames(coef_table)) coef_table[, "ci.ub"] else if (ncol(coef_table) >= 6) coef_table[, 6] else coef_df$coefficient + 1.96 * coef_df$se
  
  annotate_meta_predictor_coefficients(coef_df[socioecon_rows, , drop = FALSE])
}

save_meta_predictor_outputs <- function(meta_coef_df,
                                        output_dir,
                                        full_title,
                                        full_subtitle,
                                        summary_title,
                                        summary_subtitle) {
  if (is.null(meta_coef_df) || nrow(meta_coef_df) == 0) {
    return(FALSE)
  }
  
  bundle_files <- file.path(output_dir, c(
    "city_level_covariates.csv",
    "city_level_covariates_SUMMARY.csv",
    "city_level_covariates_forest.png",
    "city_level_covariates_forest_FULL.png"
  ))
  unlink(bundle_files[file.exists(bundle_files)])
  
  meta_coef_df <- annotate_meta_predictor_coefficients(meta_coef_df)
  write_csv(meta_coef_df, file.path(output_dir, "city_level_covariates.csv"))
  
  full_df <- meta_coef_df %>%
    filter(variable_only != "Other", variable_only != "Intercept") %>%
    arrange(variable_only, predictor_name, cb_lag)
  
  if (nrow(full_df) == 0) {
    return(FALSE)
  }
  
  full_df <- full_df %>%
    mutate(
      cb_lag = factor(cb_lag, levels = rev(unique(cb_lag))),
      direction = forest_effect_direction(coefficient, neutral_zero = TRUE),
      star_label = forest_significance_stars(p_value)
    )
  star_nudge_full <- forest_star_offset(full_df$ci_low, full_df$ci_high, fallback = 0.05)
  full_subtitle_styled <- paste0(
    full_subtitle,
    "\nBlue=Positive, Red=Negative | * p<0.05, ** p<0.01, *** p<0.001"
  )
  
  plot_height_full <- max(10, 2 + n_distinct(full_df$variable_only) * 1.2 + nrow(full_df) * 0.18)
  p_full <- ggplot(full_df, aes(x = coefficient, y = cb_lag)) +
    geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 0.9) +
    geom_errorbarh(
      aes(xmin = ci_low, xmax = ci_high, color = direction),
      height = 0.18,
      linewidth = 0.8
    ) +
    geom_point(
      aes(color = direction),
      size = 2.8,
      shape = 19
    ) +
    geom_text(
      data = full_df %>% filter(nzchar(star_label)),
      aes(label = star_label),
      nudge_x = star_nudge_full,
      nudge_y = 0.22,
      size = 3.1,
      color = "black",
      fontface = "bold",
      show.legend = FALSE
    ) +
    facet_grid(variable_only ~ ., scales = "free_y", space = "free_y", switch = "y") +
    scale_color_manual(values = forest_direction_colors, guide = "none") +
    labs(
      title = full_title,
      subtitle = full_subtitle_styled,
      x = "Meta-Regression Coefficient",
      y = "DLNM basis term"
    ) +
    coord_cartesian(clip = "off") +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 15, hjust = 0),
      plot.subtitle = element_text(size = 10, color = "gray30", hjust = 0, lineheight = 1.2),
      strip.placement = "outside",
      strip.text.y.left = element_text(angle = 0, face = "bold", size = 11),
      axis.title.x = element_text(face = "bold", size = 12, margin = margin(t = 8)),
      axis.title.y = element_text(face = "bold", size = 11),
      axis.text.y = element_text(size = 9, color = "black"),
      axis.text.x = element_text(size = 10),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line.x = element_line(color = "black", linewidth = 0.9),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      plot.margin = margin(12, 12, 12, 12)
    )
  ggsave(file.path(output_dir, "city_level_covariates_forest_FULL.png"), p_full, width = 15, height = plot_height_full, dpi = 300)
  
  summary_df <- full_df %>%
    group_by(variable_only) %>%
    summarise(
      coefficient = sum(coefficient / (se^2 + 1e-10), na.rm = TRUE) / sum(1 / (se^2 + 1e-10), na.rm = TRUE),
      se = sqrt(1 / sum(1 / (se^2 + 1e-10), na.rm = TRUE)),
      z_value = coefficient / se,
      p_value = 2 * pnorm(-abs(coefficient / se)),
      ci_low = coefficient - 1.96 * se,
      ci_high = coefficient + 1.96 * se,
      significant = ifelse(p_value < 0.05, "sig", "ns"),
      n_coefs = n(),
      .groups = "drop"
    ) %>%
    mutate(
      direction = forest_effect_direction(coefficient, neutral_zero = TRUE),
      star_label = forest_significance_stars(p_value)
    )
  write_csv(summary_df, file.path(output_dir, "city_level_covariates_SUMMARY.csv"))
  star_nudge_summary <- forest_star_offset(summary_df$ci_low, summary_df$ci_high, fallback = 0.05)
  summary_subtitle_styled <- paste0(
    summary_subtitle,
    "\nBlue=Positive, Red=Negative | * p<0.05, ** p<0.01, *** p<0.001"
  )
  
  plot_height_summary <- max(6, 4 + nrow(summary_df) * 0.8)
  p_summary <- ggplot(summary_df, aes(x = coefficient, y = reorder(variable_only, coefficient))) +
    geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1.5) +
    geom_errorbarh(
      aes(xmin = ci_low, xmax = ci_high, color = direction),
      height = 0.35,
      linewidth = 1.6
    ) +
    geom_point(
      aes(color = direction),
      size = 8,
      shape = 19
    ) +
    geom_text(
      data = summary_df %>% filter(nzchar(star_label)),
      aes(label = star_label),
      nudge_x = star_nudge_summary,
      nudge_y = 0.24,
      size = 5,
      color = "black",
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(values = forest_direction_colors, guide = "none") +
    labs(
      title = summary_title,
      subtitle = summary_subtitle_styled,
      x = "Meta-Regression Coefficient (Standardized)",
      y = ""
    ) +
    coord_cartesian(clip = "off") +
    theme_minimal(base_size = 18) +
    theme(
      plot.title = element_text(face = "bold", size = 20, hjust = 0),
      plot.subtitle = element_text(size = 13, color = "gray30", hjust = 0, lineheight = 1.2),
      axis.title.x = element_text(face = "bold", size = 16, margin = margin(t = 12)),
      axis.text.y = element_text(size = 15, face = "bold", color = "black"),
      axis.text.x = element_text(size = 14),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line.x = element_line(color = "black", linewidth = 1.2),
      axis.line.y = element_line(color = "black", linewidth = 1.2),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2),
      plot.margin = margin(16, 16, 16, 16)
    )
  ggsave(file.path(output_dir, "city_level_covariates_forest.png"), p_summary, width = 15, height = plot_height_summary, dpi = 300)
  
  TRUE
}

run_partition_meta_predictors <- function(partition_socioecon_avg,
                                          coef_matrix,
                                          vcov_list,
                                          partition_meta_dir,
                                          partition_label,
                                          indicator,
                                          model_type,
                                          partition_results = NULL) {
  predictor_output_dir <- meta_predictor_output_dir(partition_meta_dir)
  # New partition meta-predictor path:
  # use 10km grid rows when available; otherwise replicate city covariates to
  # each city's available grid count. The old city-row path below is kept only
  # as unreachable legacy code after this block returns.
  grid_inputs <- build_grid10_partition_meta_inputs(
    partition_socioecon_avg = partition_socioecon_avg,
    coef_matrix = coef_matrix,
    vcov_list = vcov_list,
    partition_results = partition_results,
    mode = META_PREDICTOR_MODE
  )

  if (is.null(grid_inputs)) {
    cat("          Meta-predictors skipped: no grid-level or fallback covariate inputs were available.\n")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were skipped for this partition model.",
        "Reason: neither grid10_environment_predictors.csv nor fallback replicated city covariates could be constructed."
      )
    )
    return(list(success = FALSE, note = "meta_predictors skipped: no grid-level covariate inputs"))
  }

  coef_matrix_p <- grid_inputs$coef_matrix
  vcov_list_p <- grid_inputs$vcov_list
  city_covariates_p <- grid_inputs$covariates

  screen_result_p <- screen_meta_predictors_report_only(city_covariates_p, cat_prefix = "          ")
  city_covariates_p <- screen_result_p$data

  if (is.null(city_covariates_p) || ncol(city_covariates_p) == 0) {
    cat("          Meta-predictors skipped: no usable predictors after explicit exclusion/constant screening.\n")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were skipped for this partition model.",
        "Reason: no usable predictors remained after excluding total_all, FAR, Distance_to_Transit and constant columns."
      )
    )
    return(list(success = FALSE, note = "meta_predictors skipped: no predictors left"))
  }

  complete_rows_p <- complete.cases(city_covariates_p) &
    apply(coef_matrix_p, 1, function(x) all(is.finite(x))) &
    vapply(vcov_list_p, function(V) all(is.finite(V)), logical(1))

  if (sum(complete_rows_p) < nrow(city_covariates_p)) {
    cat("          Removed incomplete grid meta rows: ", nrow(city_covariates_p) - sum(complete_rows_p), "\n", sep = "")
    city_covariates_p <- city_covariates_p[complete_rows_p, , drop = FALSE]
    coef_matrix_p <- coef_matrix_p[complete_rows_p, , drop = FALSE]
    vcov_list_p <- vcov_list_p[complete_rows_p]
  }

  city_covariates_p <- standardize_meta_covariates(city_covariates_p)

  min_meta_rows_p <- minimum_partition_meta_cities(ncol(city_covariates_p))
  if (nrow(city_covariates_p) < min_meta_rows_p) {
    cat("          Meta-predictors skipped: grid rows too few (", nrow(city_covariates_p),
        " < ", min_meta_rows_p, " for ", ncol(city_covariates_p), " predictors)\n", sep = "")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were skipped for this partition model.",
        paste0("Reason: ", nrow(city_covariates_p), " complete grid rows were available for ",
               ncol(city_covariates_p), " predictors; current rule requires at least ", min_meta_rows_p, ".")
      )
    )
    return(list(success = FALSE, note = paste0("meta_predictors skipped: ", nrow(city_covariates_p), "/", min_meta_rows_p, " grid rows")))
  }

  meta_formula_p <- as.formula(
    paste("coef_matrix_p ~", paste(names(city_covariates_p), collapse = " + ")),
    env = environment()
  )
  mv_cov_p <- NULL
  fit_flag <- "STABLE"
  last_meta_error_p <- NA_character_

  mv_cov_p <- tryCatch({
    eval(bquote(
      mvmeta(.(meta_formula_p), data = city_covariates_p, S = vcov_list_p, method = "reml")
    ))
  }, error = function(e) {
    last_meta_error_p <<- conditionMessage(e)
    cat("          Meta-predictors REML failed: ", substr(last_meta_error_p, 1, 120), "\n", sep = "")
    NULL
  })

  if (is.null(mv_cov_p)) {
    cat("          Meta-predictors trying regularization...\n")
    max_eigs_p <- sapply(vcov_list_p, function(V) {
      tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
    })
    reg_str_p <- max(0.01, max(max_eigs_p, na.rm = TRUE) * 0.1)
    vcov_reg_p <- lapply(vcov_list_p, function(V) V + diag(reg_str_p, nrow(V)))
    mv_cov_p <- tryCatch({
      eval(bquote(
        mvmeta(.(meta_formula_p), data = city_covariates_p, S = vcov_reg_p, method = "reml")
      ))
    }, error = function(e) {
      last_meta_error_p <<- conditionMessage(e)
      NULL
    })
    if (!is.null(mv_cov_p)) {
      fit_flag <- "UNSTABLE_REG"
      cat("          Meta-predictors regularization succeeded; interpret as exploratory.\n")
    }
  }

  if (is.null(mv_cov_p)) {
    cat("          Meta-predictors trying fixed effects...\n")
    mv_cov_p <- tryCatch({
      eval(bquote(
        mvmeta(.(meta_formula_p), data = city_covariates_p, method = "fixed")
      ))
    }, error = function(e) {
      last_meta_error_p <<- conditionMessage(e)
      cat("          Meta-predictors all fitting strategies failed: ", substr(last_meta_error_p, 1, 160), "\n", sep = "")
      NULL
    })
    if (!is.null(mv_cov_p)) {
      fit_flag <- "HIGHLY_UNSTABLE_FIXED"
      cat("          Meta-predictors fixed effects succeeded; interpret as highly exploratory.\n")
    }
  }

  if (is.null(mv_cov_p)) {
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were not generated for this partition model.",
        paste0("Predictor source: ", grid_inputs$source),
        paste0("Grid rows attempted: ", nrow(city_covariates_p), "; cities represented: ", grid_inputs$n_cities),
        paste0("Reason: mvmeta failed under REML, regularization, and fixed-effects fallback.",
               if (!is.na(last_meta_error_p) && nzchar(last_meta_error_p)) paste0(" Last error: ", last_meta_error_p) else "")
      )
    )
    return(list(
      success = FALSE,
      note = if (!is.na(last_meta_error_p) && nzchar(last_meta_error_p)) {
        paste0("meta_predictors fitting failed: ", last_meta_error_p)
      } else {
        "meta_predictors fitting failed"
      }
    ))
  }

  meta_coef_df <- extract_meta_predictor_coefficients(
    mv_model = mv_cov_p,
    indicator = indicator,
    model_type = model_type,
    n_cities = grid_inputs$n_cities
  )

  if (is.null(meta_coef_df) || nrow(meta_coef_df) == 0) {
    cat("          Meta-predictors output skipped: fitted model had no extractable predictor coefficients.\n")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were not generated for this partition model.",
        paste0("Predictor source: ", grid_inputs$source),
        "Reason: mvmeta fit succeeded but no usable predictor coefficient table could be extracted from summary()."
      )
    )
    return(list(success = FALSE, note = "meta_predictors summary extraction failed"))
  }

  full_title <- paste0("Meta-Predictors (Grid-Level Inputs, ", meta_predictor_mode_label(), ") - ", partition_label, " | ",
                        toupper(indicator), " - ", toupper(model_type))
  full_subtitle <- paste0(
    "Predictor mode: ", meta_predictor_mode_label(), " | 10km/grid rows; city DLNM coefficients replicated to grids | ",
    nrow(meta_coef_df), " coefficients | ",
    ncol(city_covariates_p), " predictors | Fit flag: ", fit_flag
  )
  summary_title <- "Meta-Predictors: Built Environment Modifiers of Heatwave Effects"
  summary_subtitle <- paste0(
    partition_label, " | ", toupper(indicator), " - ", toupper(model_type),
    " | Predictor mode: ", meta_predictor_mode_label(),
    " | ", nrow(city_covariates_p), " grid rows / ", grid_inputs$n_cities, " cities | ",
    ncol(city_covariates_p), " predictors"
  )

  output_success <- save_meta_predictor_outputs(
    meta_coef_df = meta_coef_df,
    output_dir = predictor_output_dir,
    full_title = full_title,
    full_subtitle = full_subtitle,
    summary_title = summary_title,
    summary_subtitle = summary_subtitle
  )

  if (output_success) {
    mode_model_output_dir <- meta_model_output_dir(partition_meta_dir)
    conditional_rr_n <- tryCatch({
      generate_partition_conditional_rr(
        mv_model = mv_cov_p,
        coef_matrix = coef_matrix_p,
        covariates_std = city_covariates_p,
        partition_results = partition_results,
        output_dir = mode_model_output_dir,
        indicator = indicator,
        model_type = model_type,
        title_prefix = partition_label
      )
    }, error = function(e) {
      cat("          Conditional RR output skipped: ", conditionMessage(e), "\n", sep = "")
      0L
    })
    if (conditional_rr_n > 0) {
      cat("          Conditional RR curves saved in ", basename(mode_model_output_dir),
          " (", conditional_rr_n, " predictor plot(s)).\n", sep = "")
    }
    
    write_csv(
      data.frame(
        term = names(screen_result_p$vif),
        vif = as.numeric(screen_result_p$vif),
        stringsAsFactors = FALSE
      ),
      file.path(predictor_output_dir, "city_level_covariates_vif.csv")
    )
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were generated successfully for this partition model.",
        paste0("Predictor source: ", grid_inputs$source),
        "Inference unit: grid rows with city-level DLNM coefficients/vcov replicated to each grid.",
        paste0("Complete grid rows used: ", nrow(city_covariates_p)),
        paste0("Cities represented: ", grid_inputs$n_cities),
        paste0("Predictors retained: ", paste(names(city_covariates_p), collapse = ", ")),
        "Excluded before modeling: total_all, FAR, Distance_to_Transit.",
        paste0("Fit flag: ", fit_flag)
      )
    )
    cat("          Partition meta-predictors saved using grid-row inputs (", nrow(city_covariates_p),
        " rows / ", grid_inputs$n_cities, " cities).\n", sep = "")
    return(list(success = TRUE, note = paste0("meta_predictors generated [", fit_flag, "] with grid-row inputs")))
  }

  write_meta_predictor_status(
    partition_meta_dir,
    c(
      "Meta-predictors were not generated for this partition model.",
      "Reason: coefficient extraction succeeded but plotting/output bundle returned no files."
    )
  )
  return(list(success = FALSE, note = "meta_predictors output bundle returned empty"))

  if (is.null(partition_socioecon_avg) || nrow(partition_socioecon_avg) == 0) {
    cat("          ⚠ Meta-predictors 跳过：无城市社会经济数据\n")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were skipped for this partition model.",
        "Reason: city-level socioeconomic data were unavailable in the current run.",
        "No city_level_covariates*.csv or forest plots were generated for this partition."
      )
    )
    return(list(success = FALSE, note = "meta_predictors skipped: no city-level socioeconomic data"))
  }
  
  city_names_p <- rownames(coef_matrix)
  city_covariates_p <- partition_socioecon_avg %>%
    filter(city %in% city_names_p) %>%
    arrange(match(city, city_names_p)) %>%
    select(-city) %>%
    as.data.frame()
  keep_p <- city_names_p %in% partition_socioecon_avg$city
  
  if (sum(keep_p) < 3) {
    cat("          ⚠ Meta-predictors 跳过：匹配到协变量的城市不足 (", sum(keep_p), ")\n")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were skipped for this partition model.",
        paste0("Reason: only ", sum(keep_p), " cities had matched socioeconomic covariates after alignment.")
      )
    )
    return(list(success = FALSE, note = paste0("meta_predictors skipped: only ", sum(keep_p), " matched cities")))
  }
  
  if (nrow(city_covariates_p) != sum(keep_p)) {
    cat("          ⚠ Meta-predictors 跳过：协变量数据不完整 (", nrow(city_covariates_p), " != ", sum(keep_p), ")\n")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were skipped for this partition model.",
        paste0("Reason: matched covariate rows (", nrow(city_covariates_p), ") did not equal matched cities (", sum(keep_p), ").")
      )
    )
    return(list(success = FALSE, note = "meta_predictors skipped: incomplete covariate rows"))
  }
  
  if (nrow(city_covariates_p) != nrow(coef_matrix)) {
    coef_matrix_p <- coef_matrix[keep_p, , drop = FALSE]
    vcov_list_p <- vcov_list[keep_p]
  } else {
    coef_matrix_p <- coef_matrix
    vcov_list_p <- vcov_list
  }
  
  rownames(city_covariates_p) <- rownames(coef_matrix_p)
  vif_result_p <- reduce_meta_predictors(city_covariates_p, cat_prefix = "          ")
  city_covariates_p <- vif_result_p$data
  
  if (is.null(city_covariates_p) || ncol(city_covariates_p) == 0) {
    cat("          ⚠ Meta-predictors 跳过：常量/共线性筛选后已无可用变量\n")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were skipped for this partition model.",
        "Reason: all candidate covariates were removed after constant-column and collinearity screening."
      )
    )
    return(list(success = FALSE, note = "meta_predictors skipped: no predictors left after screening"))
  }
  
  min_meta_cities <- minimum_partition_meta_cities(ncol(city_covariates_p))
  if (nrow(city_covariates_p) < min_meta_cities) {
    cat("          ⚠ Meta-predictors 跳过：城市数不足 (", nrow(city_covariates_p),
        " < ", min_meta_cities, "，当前 ", ncol(city_covariates_p), " 个predictors)\n", sep = "")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were skipped for this partition model.",
        paste0("Reason: after screening there were ", ncol(city_covariates_p),
               " predictors but only ", nrow(city_covariates_p),
               " cities; current rule requires at least ", min_meta_cities, " cities.")
      )
    )
    return(list(success = FALSE, note = paste0("meta_predictors skipped: ", nrow(city_covariates_p), "/", min_meta_cities, " cities after screening")))
  }
  
  meta_formula_p <- as.formula(
    paste("coef_matrix_p ~", paste(names(city_covariates_p), collapse = " + ")),
    env = environment()
  )
  mv_cov_p <- NULL
  fit_flag <- "STABLE"
  last_meta_error_p <- NA_character_
  
  mv_cov_p <- tryCatch({
    eval(bquote(
      mvmeta(.(meta_formula_p), data = city_covariates_p, S = vcov_list_p, method = "reml")
    ))
  }, error = function(e) {
    last_meta_error_p <<- conditionMessage(e)
    cat("          ⚠ Meta-predictors REML失败: ", substr(last_meta_error_p, 1, 100), "\n")
    NULL
  })
  
  if (is.null(mv_cov_p)) {
    cat("          → Meta-predictors 尝试 regularization...\n")
    max_eigs <- sapply(vcov_list_p, function(V) {
      tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
    })
    reg_str <- max(0.01, max(max_eigs, na.rm = TRUE) * 0.1)
    vcov_reg_p <- lapply(vcov_list_p, function(V) V + diag(reg_str, nrow(V)))
    mv_cov_p <- tryCatch({
      eval(bquote(
        mvmeta(.(meta_formula_p), data = city_covariates_p, S = vcov_reg_p, method = "reml")
      ))
    }, error = function(e) {
      last_meta_error_p <<- conditionMessage(e)
      NULL
    })
    if (!is.null(mv_cov_p)) {
      fit_flag <- "UNSTABLE_REG"
      cat("          ⚠ Meta-predictors regularization 成功（结果仅供参考）\n")
    }
  }
  
  if (is.null(mv_cov_p)) {
    cat("          → Meta-predictors 尝试 fixed effects...\n")
    mv_cov_p <- tryCatch({
      eval(bquote(
        mvmeta(.(meta_formula_p), data = city_covariates_p, method = "fixed")
      ))
    }, error = function(e) {
      last_meta_error_p <<- conditionMessage(e)
      cat("          ✗ Meta-predictors 所有策略失败: ", substr(last_meta_error_p, 1, 120), "\n")
      NULL
    })
    if (!is.null(mv_cov_p)) {
      fit_flag <- "HIGHLY_UNSTABLE_FIXED"
      cat("          ⚠ Meta-predictors fixed effects 成功（高度不稳定，仅供参考）\n")
    }
  }
  
  if (is.null(mv_cov_p)) {
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were not generated for this partition model.",
        paste0(
          "Reason: mvmeta failed under REML, regularization, and fixed-effects fallback.",
          if (!is.na(last_meta_error_p) && nzchar(last_meta_error_p)) {
            paste0(" Last error: ", last_meta_error_p)
          } else {
            ""
          }
        )
      )
    )
    return(list(
      success = FALSE,
      note = if (!is.na(last_meta_error_p) && nzchar(last_meta_error_p)) {
        paste0("meta_predictors fitting failed: ", last_meta_error_p)
      } else {
        "meta_predictors fitting failed"
      }
    ))
  }
  
  meta_coef_df <- extract_meta_predictor_coefficients(
    mv_model = mv_cov_p,
    indicator = indicator,
    model_type = model_type,
    n_cities = nrow(coef_matrix_p)
  )
  
  if (is.null(meta_coef_df) || nrow(meta_coef_df) == 0) {
    cat("          ⚠ 跳过 Meta-predictors 输出：summary 可提取结果为空\n")
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were not generated for this partition model.",
        "Reason: mvmeta fit succeeded but no usable coefficient table could be extracted from summary()."
      )
    )
    return(list(success = FALSE, note = "meta_predictors summary extraction failed"))
  }
  
  full_title <- paste0("Meta-Predictors (Full Detail) - ", partition_label, " | ",
                       toupper(indicator), " - ", toupper(model_type))
  full_subtitle <- paste0(
    "Faceted by social variable to avoid overlap | ",
    nrow(meta_coef_df), " coefficients | ",
    ncol(city_covariates_p), " predictors | Fit flag: ", fit_flag
  )
  summary_title <- "Meta-Predictors: How City Characteristics Modify Heatwave Effects"
  summary_subtitle <- paste0(
    partition_label, " | ", toupper(indicator), " - ", toupper(model_type),
    " | ", ncol(city_covariates_p), " predictors after screening | ",
    nrow(meta_coef_df), " coefficient rows"
  )
  
  output_success <- save_meta_predictor_outputs(
    meta_coef_df = meta_coef_df,
    output_dir = predictor_output_dir,
    full_title = full_title,
    full_subtitle = full_subtitle,
    summary_title = summary_title,
    summary_subtitle = summary_subtitle
  )
  
  if (output_success) {
    write_meta_predictor_status(
      partition_meta_dir,
      c(
        "Meta-predictors were generated successfully for this partition model.",
        paste0("Cities used: ", nrow(city_covariates_p)),
        paste0("Predictors retained after screening: ", paste(names(city_covariates_p), collapse = ", ")),
        paste0("Fit flag: ", fit_flag)
      )
    )
    cat("          ✓ 分区 Meta-predictors 已保存（city_level_covariates*.csv + 森林图）\n")
    cat("          Meta-predictors files: ", basename(predictor_output_dir), "\n", sep = "")
    return(list(success = TRUE, note = paste0("meta_predictors generated [", fit_flag, "] in ", basename(predictor_output_dir))))
  }
  
  write_meta_predictor_status(
    partition_meta_dir,
    c(
      "Meta-predictors were not generated for this partition model.",
      "Reason: coefficient extraction succeeded but plotting/output bundle returned no files."
    )
  )
  list(success = FALSE, note = "meta_predictors output bundle returned empty")
}

# DLNM参数
run_additional_partition_meta_family <- function(partition_mapping,
                                                 family_key,
                                                 family_display_label,
                                                 output_prefix,
                                                 section_label,
                                                 successful_cities,
                                                 OUTPUT_DIR,
                                                 partition_socioecon_avg,
                                                 partition_meta_run_summary) {
  cat("  [", section_label, "] 按", family_display_label, "分区分析...\n\n", sep = "")
  
  family_summary_list <- list()
  
  for (partition_name in names(partition_mapping)) {
    partition_cities <- partition_mapping[[partition_name]]
    cat("    ══ ", family_display_label, ": ", partition_name, " （", length(partition_cities), " 个城市）══\n", sep = "")
    
    family_output_dir <- file.path(OUTPUT_DIR, paste0(output_prefix, "_", partition_name))
    dir.create(family_output_dir, showWarnings = FALSE, recursive = TRUE)
    
    partition_model_counts <- expand.grid(
      indicator = c("cehwi", "exceeded_quantity"),
      model_type = STAGE1_MODEL_TYPES,
      stringsAsFactors = FALSE
    ) %>%
      mutate(n_cities = mapply(
        function(ind_i, mtype_i) count_partition_model_cities(partition_cities, successful_cities, ind_i, mtype_i),
        indicator,
        model_type
      ))
    
    partition_any_cities <- partition_cities[vapply(partition_cities, function(city_name) {
      any(vapply(c("cehwi", "exceeded_quantity"), function(ind_i) {
        city_key <- paste0(city_name, "_", ind_i)
        city_key %in% names(successful_cities) &&
          any(STAGE1_MODEL_TYPES %in% names(successful_cities[[city_key]]))
      }, logical(1)))
    }, logical(1))]
    
    partition_has_any_model <- any(partition_model_counts$n_cities >= 3, na.rm = TRUE)
    
    base_partition_results <- list()
    for (city_name in partition_cities) {
      city_key <- paste0(city_name, "_cehwi")
      if (city_key %in% names(successful_cities)) {
        city_result <- successful_cities[[city_key]]
        base_model_for_partition <- stage1_model_name("composite", "all", STAGE1_ACTIVITY_MODE)
        if (base_model_for_partition %in% names(city_result)) {
          base_partition_results[[city_key]] <- city_result[[base_model_for_partition]]
        }
      }
    }
    
    base_partition_results <- as.list(partition_any_cities)
    names(base_partition_results) <- partition_any_cities
    
    if (partition_has_any_model) {
      cat("      - 成功纳入", length(base_partition_results), "个城市\n")
      cat("      ✓ 目录已创建:", family_output_dir, "\n")
      partition_info <- data.frame(
        partition_family = family_key,
        partition_name = partition_name,
        n_cities_total = length(partition_cities),
        n_cities_included = length(partition_any_cities),
        cities = paste(partition_any_cities, collapse = ", "),
        model_availability = paste(
          paste0(toupper(partition_model_counts$indicator), "_", toupper(partition_model_counts$model_type), "=", partition_model_counts$n_cities),
          collapse = "; "
        )
      )
      write_csv(partition_info, file.path(family_output_dir, "partition_info.csv"))
      family_summary_list[[partition_name]] <- partition_info
      partition_af_model_results <- list()
      
      cat("      → 【V6】开始完整meta-regression分析...\n")
      for (ind in c("cehwi", "exceeded_quantity")) {
        for (mtype in STAGE1_MODEL_TYPES) {
          partition_results <- list()
          for (city_name in partition_cities) {
            city_key <- paste0(city_name, "_", ind)
            if (city_key %in% names(successful_cities)) {
              city_result <- successful_cities[[city_key]]
              if (mtype %in% names(city_result)) {
                partition_results[[city_name]] <- city_result[[mtype]]
              }
            }
          }
          partition_results <- filter_current_stage1_results(
            partition_results,
            paste0(family_display_label, " ", partition_name, " ", toupper(ind), " ", toupper(mtype))
          )
          
          if (length(partition_results) < 3) {
            partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
              partition_family = family_key,
              partition_name = partition_name,
              indicator = ind,
              model_type = mtype,
              n_cities = length(partition_results),
              n_cities_total = length(partition_cities),
              model_status = "SKIPPED_LT3",
              output_dir = file.path(family_output_dir, paste0(ind, "_", mtype)),
              status_note = "Skipped before fitting because fewer than 3 cities had usable first-stage results."
            )
            next
          }
          
          if (is.null(partition_af_model_results[[ind]])) partition_af_model_results[[ind]] <- list()
          partition_af_model_results[[ind]][[mtype]] <- partition_results
          
          partition_meta_dir_base <- file.path(family_output_dir, paste0(ind, "_", mtype))
          partition_meta_dir <- partition_meta_dir_base
          dir.create(partition_meta_dir, showWarnings = FALSE, recursive = TRUE)
          summary_result <- list(
            model_status = "FAILED",
            stability_flag = NA_character_,
            meta_predictors_output = FALSE,
            af_output = FALSE,
            output_dir = partition_meta_dir,
            status_note = NA_character_
          )
          
          tryCatch({
            coef_list <- lapply(partition_results, function(x) x$coef)
            vcov_list <- lapply(partition_results, function(x) x$vcov)
            if (any(sapply(coef_list, is.null))) stop("部分城市的coef为空")
            if (any(sapply(vcov_list, is.null))) stop("部分城市的vcov为空")
            coef_matrix <- do.call(rbind, coef_list)
            
            mv_model_simple <- tryCatch(
              mvmeta(coef_matrix, S = vcov_list, method = "reml"),
              error = function(e) NULL
            )
            model_reliability <- "STABLE"
            
            if (is.null(mv_model_simple)) {
              max_eigs <- sapply(vcov_list, function(V) {
                tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
              })
              reg_str <- max(0.01, max(max_eigs, na.rm = TRUE) * 0.1)
              vcov_reg <- lapply(vcov_list, function(V) V + diag(reg_str, nrow(V)))
              mv_model_simple <- tryCatch(
                mvmeta(coef_matrix, S = vcov_reg, method = "reml"),
                error = function(e) NULL
              )
              if (!is.null(mv_model_simple)) model_reliability <- "UNSTABLE_REG"
            }
            
            if (is.null(mv_model_simple)) {
              mv_model_simple <- tryCatch(mvmeta(coef_matrix, method = "fixed"), error = function(e) NULL)
              if (!is.null(mv_model_simple)) model_reliability <- "HIGHLY_UNSTABLE_FIXED"
            }
            
            if (is.null(mv_model_simple)) stop("所有策略失败")
            
            pooled_coef <- coef(mv_model_simple)
            pooled_vcov <- vcov(mv_model_simple)
            cb_template <- partition_results[[1]]$cb
            if (is.null(cb_template)) stop("cb_template为空")
            
            part_ranges <- lapply(partition_results, function(x) x$cehwi_range)
            part_ranges <- part_ranges[!sapply(part_ranges, is.null)]
            if (length(part_ranges) > 0) {
              cehwi_range_p <- range(unlist(part_ranges), na.rm = TRUE)
              cehwi_max <- max(cehwi_range_p[2], 1)
            } else {
              all_exp <- unlist(lapply(partition_results, function(x) {
                d <- x$cehwi_data
                if (!is.null(d)) d[d > 0] else NULL
              }))
              cehwi_max <- if (length(all_exp) > 0) max(max(all_exp, na.rm = TRUE), 1) else 10
            }
            if (!is.finite(cehwi_max) || cehwi_max <= 0) cehwi_max <- 10
            
            pooled_cp <- crosspred(
              cb_template,
              coef = pooled_coef,
              vcov = pooled_vcov,
              model.link = "log",
              at = seq(0, cehwi_max, length.out = 500),
              cen = 0,
              cumul = TRUE
            )
            pooled_df <- data.frame(
              cehwi = pooled_cp$predvar,
              rr = pooled_cp$allRRfit,
              rr_low = pooled_cp$allRRlow,
              rr_high = pooled_cp$allRRhigh
            )
            
            subtitle_text <- build_partition_rr_subtitle(
              partition_family = family_display_label,
              partition_name = partition_name,
              indicator = ind,
              model_type = mtype,
              n_cities = length(partition_results),
              pooled_df = pooled_df,
              meta_model = mv_model_simple,
              reliability = model_reliability
            )
            
            line_color <- stage1_model_color(mtype)
            if (is.null(line_color) || is.na(line_color)) line_color <- "#D53E4F"
            display_bundle <- prepare_rr_curve_display(pooled_df)
            if (is.null(display_bundle)) stop("No finite pooled RR values available for plotting.")
            pooled_df_plot <- display_bundle$data
            part_ci_exploded <- isTRUE(display_bundle$ci_clipped)
            
            title_suffix <- ifelse(
              model_reliability == "STABLE", "",
              ifelse(model_reliability == "UNSTABLE_REG", " ⚠ UNSTABLE", " ⚠⚠ HIGHLY UNSTABLE")
            )
            part_caption <- build_rr_caption(
              indicator = ind,
              has_pi = FALSE,
              ci_clipped = part_ci_exploded,
              reliability = model_reliability,
              include_percentiles = FALSE,
              x_truncated = TRUE
            )
            
            p_pooled_rr <- ggplot(pooled_df_plot, aes(x = cehwi)) +
              geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8) +
              geom_ribbon(
                data = pooled_df_plot %>% filter(rr_ribbon_valid),
                aes(ymin = rr_low_clipped, ymax = rr_high_clipped),
                fill = "gray80",
                alpha = 0.45
              ) +
              geom_line(aes(y = rr_line_plot), color = line_color, linewidth = 2, na.rm = TRUE) +
              coord_cartesian(ylim = display_bundle$y_limits, clip = "on") +
              labs(
                title = paste0(
                  family_display_label, ": ", partition_name, " - ",
                  toupper(ind), " - ", toupper(mtype), title_suffix,
                  if (part_ci_exploded) " [Y-axis: RR-based]" else ""
                ),
                subtitle = subtitle_text,
                x = toupper(ind),
                y = "Relative Risk (RR)",
                caption = part_caption
              ) +
              rr_plot_theme(14)
            ggsave(file.path(partition_meta_dir, "pooled_RR_curve.png"), p_pooled_rr, width = 12, height = 8, dpi = 300)
            safe_write_csv(
              pooled_df_plot %>% filter(rr_display_outlier),
              file.path(partition_meta_dir, "pooled_RR_display_outliers.csv"),
              label = "partition pooled RR display outliers"
            )
            
            coef_df <- as.data.frame(coef_matrix)
            coef_df$reliability <- model_reliability
            safe_write_csv(coef_df, file.path(partition_meta_dir, "pooled_coefs.csv"), label = "partition pooled coefficients")
            pooled_df$reliability <- model_reliability
            safe_write_csv(pooled_df, file.path(partition_meta_dir, "pooled_RR_data.csv"), label = "partition pooled RR data")
            save_pooled_lag_response(
              partition_results,
              partition_meta_dir,
              indicator = ind,
              model_type = mtype,
              group_label = paste0(family_display_label, ": ", partition_name),
              reliability = model_reliability
            )
            
            all_cehwi_data <- unlist(lapply(partition_results, function(x) {
              if (!is.null(x$cehwi_data)) x$cehwi_data[x$cehwi_data > 0] else NULL
            }))
            if (!is.null(all_cehwi_data) && length(all_cehwi_data) > 10) {
              q25 <- quantile(all_cehwi_data, 0.25, na.rm = TRUE)
              q75 <- quantile(all_cehwi_data, 0.75, na.rm = TRUE)
              q90 <- quantile(all_cehwi_data, 0.90, na.rm = TRUE)
              q98 <- quantile(all_cehwi_data, 0.98, na.rm = TRUE)
              
              p_hist <- ggplot(data.frame(cehwi = all_cehwi_data), aes(x = cehwi)) +
                geom_histogram(bins = 30, fill = "#3B9AB2", alpha = 0.7, color = "white") +
                geom_vline(xintercept = q25, linetype = "dotted", color = "#E69F00", linewidth = 1) +
                geom_vline(xintercept = q75, linetype = "dotted", color = "#D55E00", linewidth = 1) +
                geom_vline(xintercept = q90, linetype = "dashed", color = "#CC0000", linewidth = 1.2) +
                labs(
                  title = paste0("Exposure Distribution: ", partition_name),
                  subtitle = paste0(family_display_label, " | ", toupper(ind), " - ", toupper(mtype), " (", length(partition_results), " cities)"),
                  x = toupper(ind),
                  y = "Frequency",
                  caption = paste0("25th: ", round(q25, 2), " | 75th: ", round(q75, 2), " | 90th: ", round(q90, 2))
                ) +
                rr_hist_theme(12) +
                coord_cartesian(xlim = c(0, q98))
              ggsave(file.path(partition_meta_dir, "exposure_distribution.png"), p_hist, width = 12, height = 4, dpi = 300)
              
              part_rr_caption <- build_rr_caption(
                indicator = ind,
                has_pi = FALSE,
                ci_clipped = part_ci_exploded,
                reliability = model_reliability,
                include_percentiles = TRUE,
                x_truncated = TRUE
              )
              p_pooled_rr_with_lines <- p_pooled_rr +
                coord_cartesian(xlim = c(0, q98)) +
                geom_vline(xintercept = q25, linetype = "dotted", color = "#E69F00", linewidth = 0.8) +
                geom_vline(xintercept = q75, linetype = "dotted", color = "#D55E00", linewidth = 0.8) +
                geom_vline(xintercept = q90, linetype = "dashed", color = "#CC0000", linewidth = 1) +
                labs(caption = part_rr_caption)
              ggsave(file.path(partition_meta_dir, "pooled_RR_curve_with_percentiles.png"), p_pooled_rr_with_lines, width = 12, height = 8, dpi = 300)
              p_combined <- build_rr_distribution_combined_plot(
                rr_plot = p_pooled_rr_with_lines,
                hist_plot = p_hist,
                caption_text = part_rr_caption,
                heights = c(2, 1),
                caption_size = 10
              )
              ggsave(file.path(partition_meta_dir, "pooled_RR_with_distribution.png"), p_combined, width = 12, height = 10, dpi = 300)
            }
            
            plot_partition_af_results(
              partition_results = partition_results,
              partition_meta_dir = partition_meta_dir,
              partition_label = paste0(family_display_label, ": ", partition_name),
              indicator = ind,
              model_type = mtype,
              percentile = "overall"
            )
            summary_result$af_output <- partition_has_af_outputs(partition_meta_dir)
            
            meta_predictor_result <- run_partition_meta_predictors(
              partition_socioecon_avg = partition_socioecon_avg,
              coef_matrix = coef_matrix,
              vcov_list = vcov_list,
              partition_meta_dir = partition_meta_dir,
              partition_label = paste0(family_display_label, ": ", partition_name),
              indicator = ind,
              model_type = mtype,
              partition_results = partition_results
            )
            
            summary_result$model_status <- "SUCCESS"
            summary_result$stability_flag <- model_reliability
            summary_result$meta_predictors_output <- partition_has_meta_predictors(partition_meta_dir)
            summary_result$output_dir <- partition_meta_dir
            note_parts <- c(
              if (!summary_result$meta_predictors_output) meta_predictor_result$note else NULL,
              if (!summary_result$af_output) "AF outputs missing" else NULL
            )
            summary_result$status_note <- if (length(note_parts) > 0) paste(note_parts, collapse = " | ") else NA_character_
          }, error = function(e) {
            summary_result <<- modifyList(summary_result, list(
              model_status = "FAILED",
              output_dir = partition_meta_dir,
              meta_predictors_output = partition_has_meta_predictors(partition_meta_dir),
              af_output = partition_has_af_outputs(partition_meta_dir),
              status_note = conditionMessage(e)
            ))
          })
          
          partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
            partition_family = family_key,
            partition_name = partition_name,
            indicator = ind,
            model_type = mtype,
            n_cities = length(partition_results),
            n_cities_total = length(partition_cities),
            model_status = summary_result$model_status,
            stability_flag = summary_result$stability_flag,
            meta_predictors_output = summary_result$meta_predictors_output,
            af_output = summary_result$af_output,
            output_dir = summary_result$output_dir,
            status_note = summary_result$status_note
          )
        }
      }
      save_partition_af_percentile_outputs(
        partition_model_results = partition_af_model_results,
        partition_output_dir = family_output_dir,
        partition_label = paste0(family_display_label, ": ", partition_name),
        partition_family = family_key,
        partition_name = partition_name,
        mode = meta_predictor_mode_label()
      )
    } else {
      for (ind in c("cehwi", "exceeded_quantity")) {
        for (mtype in STAGE1_MODEL_TYPES) {
          partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
            partition_family = family_key,
            partition_name = partition_name,
            indicator = ind,
            model_type = mtype,
            n_cities = count_partition_model_cities(partition_cities, successful_cities, ind, mtype),
            n_cities_total = length(partition_cities),
            model_status = "SKIPPED_PARTITION_LT3",
            output_dir = family_output_dir,
            status_note = "Entire partition skipped because fewer than 3 cities were available for the base partition result set."
          )
        }
      }
    }
  }
  
  list(
    partition_meta_run_summary = partition_meta_run_summary,
    family_summary_list = family_summary_list
  )
}

MAX_LAG <- 11  # Main lag window: lag 0-11 days, aligned with lag-12 PPML/DTW phenotypes.
SENSITIVITY_MAX_LAGS <- c(6, 4)  # Reserved for sensitivity reruns; main model uses MAX_LAG.
DOY_SPLINE_K <- 6  # Seasonal cyclic spline flexibility, closer to standard temperature-DLNM practice.
LAG_PROFILE_EXPOSURE_PERCENTILE <- 0.90  # Store lag-response at city-specific p90 exposure for Stage-2 pooling.
SAVE_META_MODEL_RDS <- FALSE  # Heavy meta_model.rds files are optional and can fill the disk on full partition runs.
REFERENCE_CEHWI <- 0  # 参照值：非热浪日
SCALE_FACTOR <- 100000  # 【关键】trip_count缩放因子（越大越稳定，防止RR爆炸）
                        # 10000: 中等城市，100000: 大城市（推荐），1000000: 超大城市
META_PREDICTOR_MODE <- "mean"  # mean: 城市均值；gini: 城市内1km网格不均衡度
STAGE1_ACTIVITY_MODE <- "combined"  # combined: current PA-summed model; activity_3plus1: All + ride/run/walk
df_socioecon_grid_global <- NULL

# ========== 阶段选择（V5.2细分版）==========
cat("\n", rep("=", 100), "\n", sep = "")
cat("两阶段DLNM分析: CEHWI对PA的非线性累积滞后效应\n")
cat(rep("=", 100), "\n\n", sep = "")

cat("请选择运行的阶段:\n")
cat("  【第一阶段 - 单城市DLNM】\n")
cat("    1. 运行单城市DLNM (75个城市，每个城市独立分析)\n")
cat("\n  【第一阶段 - 分区DLNM（已停用）】\n")
cat("    2. 旧分区 pooled DLNM 入口（已停用；避免巨型RDS和不稳定拟合）\n")
cat("       → 分区结果改由第二阶段基于单城市 reduced coefficients 汇总\n")
cat("\n  【第一阶段 - 单城市入口】\n")
cat("    3. 运行单城市DLNM（旧完整入口；分区一阶段会自动跳过）\n")
cat("\n  【第二阶段 - Meta-regression】\n")
cat("    4. 只运行第二阶段 (需要第一阶段已完成)\n")
cat("\n  【完整流程】\n")
cat("    5. 运行单城市 + 第二阶段Meta-regression（不跑分区一阶段）\n")

cat("\n  【仅重绘可视化】\n")
cat("    6. 只重绘所有已保存结果的可视化（不重新拟合模型）\n")
if (!nzchar(Sys.getenv("DLNM_STAGE_CHOICE", unset = ""))) {
  Sys.setenv(DLNM_STAGE_CHOICE = "1")
}
stage_choice <- read_integer_choice(
  prompt = "\nEnter stage option (1/2/3/4/5/6): ",
  valid_choices = 1:6,
  env_var = "DLNM_STAGE_CHOICE"
)

if (stage_choice %in% c(4, 5)) {
  stop(
    "The archived direct reduced-coefficient Stage 2 is disabled for the ",
    "city-specific p50/p90 primary analysis. Run ",
    "pool_city_specific_p50_p90_curves_pointwise.R after the city models finish."
  )
}

while (!stage_choice %in% 1:6) {
  cat("⚠ 无效选择，请重新输入！\n")
  stage_choice <- as.integer(readline(prompt = "请输入选项 (1/2/3/4/5/6): "))
}

# 根据选择设置运行标志
RUN_CITIES <- stage_choice %in% c(1, 3, 5)      # 运行单城市分析
RUN_PARTITIONS <- FALSE                         # 分区 pooled 一阶段DLNM已停用，避免巨型RDS和不稳定拟合
RUN_STAGE2 <- stage_choice %in% c(4, 5)         # 运行第二阶段
RUN_RERENDER_ONLY <- stage_choice == 6          # 仅重绘可视化

# 分区 pooled 一阶段DLNM已废弃；第二阶段仍会基于单城市结果输出 Zone/Cluster/Region/DTW 分区 meta。
if (RUN_CITIES || RUN_STAGE2 || RUN_RERENDER_ONLY) {
  cat("\n请选择一阶段PA模态结果版本:\n")
  cat("  1. 合并PA（现有版本：ride+run+walk 合并为一个PA结局）\n")
  cat("  2. 三模态+合并（新增：All + Ride + Run + Walk，单独保存不覆盖旧结果）\n")
  if (!nzchar(Sys.getenv("DLNM_ACTIVITY_CHOICE", unset = ""))) {
    Sys.setenv(DLNM_ACTIVITY_CHOICE = "2")
  }
  activity_choice <- read_integer_choice(
    prompt = "Enter activity-output option (1/2): ",
    valid_choices = c(1, 2),
    env_var = "DLNM_ACTIVITY_CHOICE"
  )
  while (is.na(activity_choice) || !activity_choice %in% c(1, 2)) {
    cat("⚠ 无效选择，请重新输入。\n")
    activity_choice <- suppressWarnings(as.integer(readline(prompt = "请输入选项 (1/2): ")))
  }
  STAGE1_ACTIVITY_MODE <- ifelse(activity_choice == 2, "activity_3plus1", "combined")
}

STAGE1_MODEL_TYPES <- stage1_model_types(STAGE1_ACTIVITY_MODE)
model_type_filter_env <- trimws(Sys.getenv("DLNM_MODEL_TYPE_FILTER", unset = ""))
if (nzchar(model_type_filter_env)) {
  requested_model_types <- trimws(strsplit(model_type_filter_env, ",", fixed = TRUE)[[1]])
  unknown_model_types <- setdiff(requested_model_types, STAGE1_MODEL_TYPES)
  if (length(unknown_model_types) > 0) {
    stop(
      "Unknown model type(s) in DLNM_MODEL_TYPE_FILTER: ",
      paste(unknown_model_types, collapse = ", ")
    )
  }
  STAGE1_MODEL_TYPES <- STAGE1_MODEL_TYPES[STAGE1_MODEL_TYPES %in% requested_model_types]
}

PARTITION_RUN_ZONE   <- FALSE
PARTITION_RUN_CLUSTER <- FALSE
PARTITION_RUN_REGION  <- FALSE
PARTITION_RUN_DTW3    <- FALSE
PARTITION_RUN_DTW4    <- FALSE
if (stage_choice %in% c(2, 3, 5)) {
  cat("\n  ⚠ 分区 pooled 第一阶段DLNM已停用：不再合并城市直接拟合巨大GAM/RDS。\n")
  cat("    分区结果将在第二阶段meta中由单城市reduced coefficients汇总生成。\n")
  if (stage_choice == 2) {
    cat("    当前只选择了旧的分区一阶段入口，因此本次不会运行模型；请选择 4 跑第二阶段分区meta，或 5 跑单城市+第二阶段。\n")
  }
}
if (RUN_PARTITIONS) {
  cat("\n  分区分析 - 选择种类:\n")
  cat("    0. 全部跑 (Climate Zone + City Cluster + Geographic Region + DTW k=3 + DTW k=4)\n")
  cat("    1. 只跑 Climate Zone\n")
  cat("    2. 只跑 City Cluster\n")
  cat("    3. 只跑 Geographic Region\n")
  cat("    4. 只跑 DTW Optimized k=3\n")
  cat("    5. 只跑 DTW Optimized k=4\n")
  cat("    （每种都包含 CEHWI 与 超出量 两个指标）\n")
  partition_choice <- suppressWarnings(as.integer(readline(prompt = "  请输入选项 (0/1/2/3/4/5): ")))
  while (is.na(partition_choice) || !partition_choice %in% 0:5) {
    cat("  ⚠ 无效选择，请重新输入！\n")
    partition_choice <- suppressWarnings(as.integer(readline(prompt = "  请输入选项 (0/1/2/3/4/5): ")))
  }
  PARTITION_RUN_ZONE   <- partition_choice %in% c(0, 1)
  PARTITION_RUN_CLUSTER <- partition_choice %in% c(0, 2)
  PARTITION_RUN_REGION  <- partition_choice %in% c(0, 3)
  PARTITION_RUN_DTW3    <- partition_choice %in% c(0, 4)
  PARTITION_RUN_DTW4    <- partition_choice %in% c(0, 5)
}

# 显示选择
cat("\n✓ 已选择:\n")
if (RUN_CITIES) cat("  ✓ 单城市DLNM分析\n")
if (RUN_PARTITIONS) {
  cat("  ✓ 分区DLNM分析")
  if (PARTITION_RUN_ZONE)   cat(" [Climate Zone]")
  if (PARTITION_RUN_CLUSTER) cat(" [City Cluster]")
  if (PARTITION_RUN_REGION)  cat(" [Geographic Region]")
  if (PARTITION_RUN_DTW3)    cat(" [DTW k=3]")
  if (PARTITION_RUN_DTW4)    cat(" [DTW k=4]")
  cat("\n")
}
if (RUN_STAGE2) cat("  ✓ 第二阶段Meta-regression\n")
if (RUN_RERENDER_ONLY) cat("  ✓ 仅重绘可视化（不重新拟合模型）\n")
cat("\n")

# ========== 时间段选择 ==========
cat("请选择分析的时间段:\n")
cat("  1. 全部时间段 (2010-2024)\n")
cat("  2. 早期 (2010-2015)\n")
cat("  3. 中期 (2015-2020)\n")
cat("  4. 晚期 (2020-2024)\n")
cat("  5. 中晚期 (2015-2024)\n")

if (!nzchar(Sys.getenv("DLNM_TIME_CHOICE", unset = ""))) {
  Sys.setenv(DLNM_TIME_CHOICE = "1")
}
time_choice <- read_integer_choice(
  prompt = "Enter time-window option (1/2/3/4/5): ",
  valid_choices = 1:5,
  env_var = "DLNM_TIME_CHOICE"
)

while (!time_choice %in% 1:5) {
  cat("⚠ 无效选择，请重新输入！\n")
  time_choice <- as.integer(readline(prompt = "请输入选项 (1/2/3/4/5): "))
}

if (time_choice == 1) {
  START_DATE <- as.Date("2010-01-01")
  END_DATE <- as.Date("2024-12-31")
  TIME_LABEL <- "2010_2024"
  TIME_DESC <- "全部时间段 (2010-2024)"
} else if (time_choice == 2) {
  START_DATE <- as.Date("2010-01-01")
  END_DATE <- as.Date("2015-12-31")
  TIME_LABEL <- "2010_2015"
  TIME_DESC <- "早期 (2010-2015)"
} else if (time_choice == 3) {
  START_DATE <- as.Date("2015-01-01")
  END_DATE <- as.Date("2020-12-31")
  TIME_LABEL <- "2015_2020"
  TIME_DESC <- "中期 (2015-2020)"
} else if (time_choice == 4) {
  START_DATE <- as.Date("2020-01-01")
  END_DATE <- as.Date("2024-12-31")
  TIME_LABEL <- "2020_2024"
  TIME_DESC <- "晚期 (2020-2024)"
} else {
  START_DATE <- as.Date("2015-01-01")
  END_DATE <- as.Date("2024-12-31")
  TIME_LABEL <- "2015_2024"
  TIME_DESC <- "中晚期 (2015-2024)"
}

cat("\n✓ 已选择:", TIME_DESC, "\n\n")

if (RUN_STAGE2) {
  cat("请选择第二阶段meta-predictors构建方式:\n")
  cat("  1. 城市均值（当前做法：1km网格变量先按城市求平均）\n")
  cat("  2. 城市内Gini（新增：每个城市内1km网格变量的不均衡度）\n")
  meta_pred_choice <- read_integer_choice(
    prompt = "Enter meta-predictor mode (1=mean, 2=gini): ",
    valid_choices = c(1, 2),
    env_var = "DLNM_META_PRED_CHOICE"
  )
  while (is.na(meta_pred_choice) || !meta_pred_choice %in% c(1, 2)) {
    cat("⚠ 无效选择，请重新输入。\n")
    meta_pred_choice <- suppressWarnings(as.integer(readline(prompt = "请输入选项 (1/2): ")))
  }
  META_PREDICTOR_MODE <- ifelse(meta_pred_choice == 2, "gini", "mean")
  cat("\nChoose whether to include Crime in stage-2 meta-predictors:\n")
  cat("  1. Include Crime (sensitivity option)\n")
  cat("  2. Exclude Crime (recommended/default; all stage-2 meta models drop Crime)\n")
  crime_choice <- read_integer_choice(
    prompt = "Enter Crime option (1=include, 2=exclude): ",
    valid_choices = c(1, 2),
    env_var = "DLNM_CRIME_CHOICE"
  )
  while (is.na(crime_choice) || !crime_choice %in% c(1, 2)) {
    cat("Invalid choice. Please enter 1 or 2.\n")
    crime_choice <- suppressWarnings(as.integer(readline(prompt = "Enter option (1/2): ")))
  }
  META_INCLUDE_CRIME <- crime_choice == 1
  META_PREDICTOR_BASES_FINAL <- refresh_meta_predictor_bases(META_INCLUDE_CRIME)
  cat("Crime predictor:", ifelse(META_INCLUDE_CRIME, "included", "excluded"), "\n")
  cat("Final stage-2 meta-predictor bases:", paste(META_PREDICTOR_BASES_FINAL, collapse = ", "), "\n")
  cat("Output label:", meta_predictor_mode_label(), "\n")
  cat("Stage-2 predictor value mode:", ifelse(META_PREDICTOR_MODE == "gini", "grid/city Gini", "mean values"), "\n\n")
  if (FALSE) {
  cat("\n✓ 第二阶段meta-predictors:", ifelse(META_PREDICTOR_MODE == "gini", "城市内Gini", "城市均值"), "\n\n")
}
}

if (RUN_CITIES || RUN_STAGE2 || RUN_RERENDER_ONLY) {
  lag_scenario_choices <- c(
    "lag7",
    "lag12",
    "lag_city_strong",
    "lag_group_median_strong",
    "lag_group_median_overall"
  )
  # Default to the manuscript-primary phenotype window.
  env_lag_scenario <- Sys.getenv(
    "DLNM_LAG_SCENARIO",
    unset = "lag_group_median_overall"
  )
  if (nzchar(env_lag_scenario)) {
    lag_scenario_key <- configure_lag_scenario(env_lag_scenario)
  } else {
    cat("\nChoose Stage-1 DLNM lag-window strategy:\n")
    cat("  0. Run all lag scenarios below, each in its own output folder\n")
    cat("  1. lag7: fixed lag 0-6 days\n")
    cat("  2. lag12: fixed lag 0-11 days\n")
    cat("  3. lag_city_strong: city-specific significant strong-effect days\n")
    cat("  4. lag_group_median_strong: group median strong-effect days\n")
    cat("  5. lag_group_median_overall: group median overall-effect days\n")
    lag_choice <- read_integer_choice(
      prompt = "Enter lag option (0/1/2/3/4/5): ",
      valid_choices = 0:5,
      env_var = "DLNM_LAG_CHOICE"
    )
    if (lag_choice == 0) {
      script_file <- resolve_script_file()
      if (is.na(script_file) || !file.exists(script_file)) {
        cat("  Warning: cannot relaunch the current script automatically; falling back to lag12.\n")
        lag_scenario_key <- configure_lag_scenario("lag12")
      } else {
        rscript_bin <- file.path(R.home("bin"), "Rscript.exe")
        if (!file.exists(rscript_bin)) rscript_bin <- "Rscript"
        child_env <- c(
          paste0("DLNM_STAGE_CHOICE=", stage_choice),
          paste0("DLNM_ACTIVITY_CHOICE=", get0("activity_choice", ifnotfound = 1L)),
          paste0("DLNM_TIME_CHOICE=", time_choice),
          paste0("DLNM_META_PRED_CHOICE=", get0("meta_pred_choice", ifnotfound = 1L)),
          paste0("DLNM_CRIME_CHOICE=", get0("crime_choice", ifnotfound = 1L))
        )
        cat("\nRunning all selected lag scenarios via Rscript children:\n")
        for (scenario_i in lag_scenario_choices) {
          cat("  -> ", scenario_i, "\n", sep = "")
          status_i <- system2(
            rscript_bin,
            args = shQuote(script_file),
            env = c(child_env, paste0("DLNM_LAG_SCENARIO=", scenario_i))
          )
          if (!identical(status_i, 0L)) {
            warning("Lag scenario failed: ", scenario_i, " (status ", status_i, ")")
          }
        }
        cat("\nAll requested lag scenarios finished.\n")
        quit(save = "no", status = 0)
      }
    } else {
      lag_scenario_key <- configure_lag_scenario(lag_scenario_choices[[lag_choice]])
    }
  }
  set_active_lag_for_city("GLOBAL", verbose = FALSE)
  cat("\nLag scenario selected: ", LAG_SCENARIO_LABEL, "\n", sep = "")
  cat("  ", LAG_SCENARIO_DESCRIPTION, "\n", sep = "")
  cat("  Default/current lag window: lag 0-", MAX_LAG, " (", LAG_DAYS_CURRENT, " effect day(s))\n\n", sep = "")
}

OUTPUT_BASE_DIR <- Sys.getenv(
  "DLNM_OUTPUT_BASE_DIR",
  unset = file.path(BASE_DIR, paste0("R_output_DLNM_two_stage_", TIME_LABEL))
)
OUTPUT_DIR <- file.path(OUTPUT_BASE_DIR, LAG_SCENARIO_LABEL)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ========== 【全局】定义城市列表 ==========

CITY_LIST <- c(
  "Abilene", "Amarillo", "Arlington", "Atlanta", "Aurora", "Austin",
  "Bakersfield", "Baltimore", "Boston",
  "Cape_Coral", "Chandler", "Charleston", "Charlotte", "Chicago", "Cincinnati",
  "Clearwater", "Cleveland", "Columbia", "Columbus", "Corpus_Christi",
  "Dallas", "Denver", "Detroit",
  "Fort_Worth", "Fresno",
  "Gilbert",
  "Henderson", "Hollywood", "Houston",
  "Indianapolis",
  "Jacksonville",
  "Kansas_City",
  "Las_Vegas", "Long_Beach", "Los_Angeles", "Louisville", "Lubbock",
  "Mesa", "Miami", "Milwaukee", "Minneapolis", "Miramar",
  "Nashville", "New_York", "Newark",
  "Oakland", "Oklahoma_City", "Orlando", "Overland_Park",
  "Palm_Bay", "Philadelphia", "Phoenix", "Pittsburgh", "Portland",
  "Raleigh", "Richmond", "Riverside",
  "Sacramento", "Salt_Lake_City", "San_Antonio", "San_Bernardino",
  "San_Diego", "San_Francisco", "San_Jose", "Santa_Ana", "Scottsdale", "Seattle",
  "St_Louis", "St_Petersburg",
  "Tallahassee", "Tampa", "Tucson",
  "Virginia_Beach", "Visalia",
  "Washington"
)

CITY_NAME_MAPPING <- list(
  "Cape_Coral" = "Cape Coral",
  "Corpus_Christi" = "Corpus Christi",
  "Fort_Worth" = "Fort Worth",
  "Kansas_City" = "Kansas City",
  "Las_Vegas" = "Las Vegas",
  "Long_Beach" = "Long Beach",
  "Los_Angeles" = "Los Angeles",
  "New_York" = "New York",
  "Oklahoma_City" = "Oklahoma City",
  "Overland_Park" = "Overland Park",
  "Palm_Bay" = "Palm Bay",
  "Salt_Lake_City" = "Salt Lake City",
  "San_Antonio" = "San Antonio",
  "San_Bernardino" = "San Bernardino",
  "San_Diego" = "San Diego",
  "San_Francisco" = "San Francisco",
  "San_Jose" = "San Jose",
  "Santa_Ana" = "Santa Ana",
  "St_Louis" = "St. Louis",
  "St_Petersburg" = "St. Petersburg",
  "Virginia_Beach" = "Virginia Beach"
)

DTW_CLUSTER_OPTIMIZED3_FILE <- file.path(BASE_DIR, "dtw_cluster_output_optimized", "cluster_city_list_optimized3.txt")
DTW_CLUSTER_OPTIMIZED4_FILE <- file.path(BASE_DIR, "dtw_cluster_output_optimized", "cluster_city_list_optimized4.txt")

normalize_external_cluster_city <- function(city_name) {
  city_name %>%
    str_trim() %>%
    str_remove_all("\\*") %>%
    str_remove_all("\\.$") %>%
    str_replace_all("\\.", "") %>%
    str_squish() %>%
    str_replace_all(" ", "_")
}

read_external_cluster_mapping <- function(file_path, scheme_label) {
  if (!file.exists(file_path)) {
    stop(sprintf("未找到%s聚类文件: %s", scheme_label, file_path))
  }
  
  lines <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
  mapping <- list()
  current_group <- NULL
  
  for (line in lines) {
    line_trim <- str_trim(line)
    if (!nzchar(line_trim) ||
        str_starts(line_trim, "###") ||
        str_starts(line_trim, "Outlier rule") ||
        str_starts(line_trim, "Handling") ||
        str_starts(line_trim, "Total") ||
        str_starts(line_trim, "Note:")) {
      next
    }
    
    cluster_match <- str_match(line_trim, "^Cluster\\s+(\\d+)\\b")
    if (!is.na(cluster_match[1, 2])) {
      current_group <- paste0("Cluster_", cluster_match[1, 2])
      if (is.null(mapping[[current_group]])) {
        mapping[[current_group]] <- character(0)
      }
      next
    }
    
    if (str_detect(line_trim, "^\\[not clustered - no heatwave events\\]")) {
      current_group <- "No_Heatwave"
      if (is.null(mapping[[current_group]])) {
        mapping[[current_group]] <- character(0)
      }
      next
    }
    
    if (is.null(current_group)) next
    
    city_tokens <- str_split(line_trim, ",")[[1]] %>%
      vapply(normalize_external_cluster_city, character(1)) %>%
      unique()
    city_tokens <- city_tokens[nzchar(city_tokens)]
    
    if (length(city_tokens) > 0) {
      mapping[[current_group]] <- unique(c(mapping[[current_group]], city_tokens))
    }
  }
  
  mapping <- mapping[lengths(mapping) > 0]
  
  if (length(mapping) == 0) {
    stop(sprintf("%s聚类文件解析失败，未读到任何城市分组: %s", scheme_label, file_path))
  }
  
  all_cities <- unlist(mapping, use.names = FALSE)
  unknown_cities <- setdiff(all_cities, CITY_LIST)
  if (length(unknown_cities) > 0) {
    stop(sprintf(
      "%s聚类文件中存在脚本无法识别的城市: %s",
      scheme_label,
      paste(sort(unique(unknown_cities)), collapse = ", ")
    ))
  }
  
  duplicated_cities <- names(which(table(all_cities) > 1))
  if (length(duplicated_cities) > 0) {
    stop(sprintf(
      "%s聚类文件中以下城市被重复分配到多个簇: %s",
      scheme_label,
      paste(sort(duplicated_cities), collapse = ", ")
    ))
  }
  
  missing_cities <- setdiff(CITY_LIST, all_cities)
  if (length(missing_cities) > 0) {
    stop(sprintf(
      "%s聚类文件未覆盖全部75个城市，缺少: %s",
      scheme_label,
      paste(sort(missing_cities), collapse = ", ")
    ))
  }
  
  mapping
}

# ========== 气候带分类（须覆盖全部75城，与CITY_LIST一致）==========
CLIMATE_ZONE_MAPPING <- list(
  "Temperate" = c("Arlington", "Atlanta", "Austin", "Baltimore", "Charleston", "Charlotte",
                  "Clearwater", "Columbia", "Corpus_Christi", "Dallas", "Fort_Worth",
                  "Houston", "Jacksonville", "Los_Angeles", "Louisville",
                  "Nashville", "Oakland", "Oklahoma_City", "Orlando",
                  "Palm_Bay", "Philadelphia", "Portland", "Raleigh", "Richmond",
                  "Sacramento", "San_Antonio", "San_Francisco", "San_Jose",
                  "Seattle", "St_Petersburg", "Tallahassee", "Tampa",
                  "Virginia_Beach", "Washington"),
  "Cold" = c("Boston", "Chicago", "Cincinnati", "Cleveland", "Columbus", "Detroit",
             "Indianapolis", "Kansas_City", "Milwaukee", "Minneapolis", "New_York",
             "Newark", "Pittsburgh", "St_Louis", "Overland_Park"),
  "Arid" = c("Abilene", "Amarillo", "Aurora", "Bakersfield", "Chandler", "Denver",
             "Fresno", "Gilbert", "Henderson", "Las_Vegas", "Long_Beach", "Lubbock",
             "Mesa", "Phoenix", "Riverside", "Salt_Lake_City", "San_Bernardino",
             "San_Diego", "Santa_Ana", "Scottsdale", "Tucson", "Visalia"),
  "Tropical" = c("Cape_Coral", "Hollywood", "Miami", "Miramar")
)

# ========== 【V4新增】7大城市聚类分类 ==========
CITY_CLUSTER_MAPPING <- list(
  "Texas_Southern_Hot" = c("Abilene", "Arlington", "Austin", "Corpus_Christi", "Dallas",
                           "Fort_Worth", "Henderson", "Houston", "Las_Vegas", "Louisville",
                           "Oklahoma_City", "Philadelphia", "San_Antonio",
                           "St_Petersburg", "Washington"),
  "East_Coast_Moderate" = c("Baltimore", "Charleston", "Clearwater", "Columbia",
                             "Jacksonville", "Nashville", "New_York", "Newark",
                             "Orlando", "Palm_Bay", "Raleigh", "Richmond",
                             "Sacramento", "Tallahassee", "Tampa", "Virginia_Beach"),
  "Northern_Cold" = c("Atlanta", "Boston", "Chicago", "Cincinnati", "Cleveland",
                       "Columbus", "Detroit", "Indianapolis", "Kansas_City",
                       "Milwaukee", "Minneapolis", "Pittsburgh", "St_Louis"),
  "West_Coast_Mediterranean" = c("Long_Beach", "Los_Angeles", "Oakland", "Portland",
                                  "San_Bernardino", "San_Diego", "San_Francisco",
                                  "San_Jose", "Santa_Ana", "Seattle"),
  "Arid_Desert" = c("Amarillo", "Aurora", "Bakersfield", "Chandler", "Denver",
                     "Fresno", "Gilbert", "Lubbock", "Mesa", "Overland_Park",
                     "Phoenix", "Riverside", "Salt_Lake_City", "Scottsdale",
                     "Tucson", "Visalia"),
  "Tropical" = c("Cape_Coral", "Hollywood", "Miami", "Miramar"),
  "Transition_Special" = c("Charlotte")
)

# ========== 【V5新增】地理区域分类（Geographic Region）==========
GEOGRAPHIC_REGION_MAPPING <- list(
  "Northeast" = c("Boston", "New_York", "Newark", "Philadelphia", "Pittsburgh"),
  
  "Southeast" = c("Atlanta", "Cape_Coral", "Charleston", "Charlotte", "Clearwater",
                  "Columbia", "Hollywood", "Jacksonville", "Louisville", "Miami",
                  "Miramar", "Nashville", "Orlando", "Palm_Bay", "Raleigh",
                  "Richmond", "St_Petersburg", "Tallahassee", "Tampa",
                  "Virginia_Beach", "Washington"),
  
  "South" = c("Baltimore", "St_Louis"),
  
  "Midwest" = c("Chicago", "Cincinnati", "Cleveland", "Columbus", "Detroit",
                "Indianapolis", "Kansas_City", "Milwaukee", "Minneapolis"),
  
  "Southwest" = c("Abilene", "Amarillo", "Arlington", "Austin", "Chandler",
                  "Corpus_Christi", "Dallas", "Fort_Worth", "Gilbert", "Houston",
                  "Lubbock", "Mesa", "Oklahoma_City", "Phoenix", "San_Antonio",
                  "Scottsdale", "Tucson"),
  
  "West" = c("Aurora", "Bakersfield", "Denver", "Fresno", "Henderson", "Las_Vegas",
             "Long_Beach", "Los_Angeles", "Oakland", "Overland_Park", "Portland",
             "Riverside", "Sacramento", "Salt_Lake_City", "San_Bernardino",
             "San_Diego", "San_Francisco", "San_Jose", "Santa_Ana", "Seattle",
             "Visalia")
)

DTW_CLUSTER_OPTIMIZED3_MAPPING <- read_external_cluster_mapping(
  DTW_CLUSTER_OPTIMIZED3_FILE,
  "DTW optimized k=3"
)

DTW_CLUSTER_OPTIMIZED4_MAPPING <- read_external_cluster_mapping(
  DTW_CLUSTER_OPTIMIZED4_FILE,
  "DTW optimized k=4"
)

city_filter_env <- trimws(Sys.getenv("DLNM_CITY_FILTER", unset = ""))
if (nzchar(city_filter_env)) {
  requested_cities <- trimws(strsplit(city_filter_env, ",", fixed = TRUE)[[1]])
  unknown_cities <- setdiff(requested_cities, CITY_LIST)
  if (length(unknown_cities) > 0) {
    stop("Unknown city name(s) in DLNM_CITY_FILTER: ", paste(unknown_cities, collapse = ", "))
  }
  CITY_LIST <- CITY_LIST[CITY_LIST %in% requested_cities]
}

indicator_filter_env <- trimws(Sys.getenv("DLNM_INDICATOR_FILTER", unset = ""))
INDICATOR_LIST <- if (nzchar(indicator_filter_env)) {
  trimws(strsplit(indicator_filter_env, ",", fixed = TRUE)[[1]])
} else {
  c("cehwi", "exceeded_quantity")
}
if (!all(INDICATOR_LIST %in% c("cehwi", "exceeded_quantity"))) {
  stop("DLNM_INDICATOR_FILTER must contain cehwi and/or exceeded_quantity")
}

DTW_CLUSTER_LAG12_MAPPING <- list(
  "Cluster_1" = c(
    "Arlington", "Bakersfield", "Baltimore", "Chandler", "Dallas",
    "Fort_Worth", "Fresno", "Houston", "Mesa", "New_York",
    "Oklahoma_City", "Phoenix", "Tucson", "Virginia_Beach", "Visalia",
    "Corpus_Christi", "Detroit", "Miami"
  ),
  "Cluster_2" = c(
    "Abilene", "Austin", "Boston", "Columbus", "Henderson",
    "Los_Angeles", "Minneapolis", "Philadelphia", "Sacramento",
    "San_Antonio", "San_Bernardino", "San_Jose", "Washington", "Newark"
  ),
  "Cluster_3" = c(
    "Atlanta", "Charleston", "Columbia", "Kansas_City", "Orlando",
    "Overland_Park", "Raleigh", "St_Louis", "St_Petersburg", "Tampa",
    "Cleveland"
  ),
  "Cluster_4" = c(
    "Amarillo", "Cape_Coral", "Charlotte", "Chicago", "Cincinnati",
    "Clearwater", "Gilbert", "Indianapolis", "Jacksonville", "Las_Vegas",
    "Louisville", "Lubbock", "Milwaukee", "Miramar", "Nashville",
    "Palm_Bay", "Richmond", "Riverside", "Scottsdale", "Tallahassee"
  ),
  "No_Heatwave" = c(
    "Aurora", "Denver", "Hollywood", "Long_Beach", "Oakland",
    "Pittsburgh", "Portland", "Salt_Lake_City", "San_Diego",
    "San_Francisco", "Santa_Ana", "Seattle"
  )
)

# ========== 【覆盖检查】三种分区分法须覆盖全部75城 ==========
.all_in <- function(mapping) unique(unlist(mapping))
.missing <- function(mapping) setdiff(CITY_LIST, .all_in(mapping))
stopifnot(
  "Climate Zone 未覆盖全部75城" = length(.missing(CLIMATE_ZONE_MAPPING)) == 0,
  "City Cluster 未覆盖全部75城" = length(.missing(CITY_CLUSTER_MAPPING)) == 0,
  "Geographic Region 未覆盖全部75城" = length(.missing(GEOGRAPHIC_REGION_MAPPING)) == 0,
  "DTW optimized k=3 未覆盖全部75城" = length(.missing(DTW_CLUSTER_OPTIMIZED3_MAPPING)) == 0,
  "DTW optimized k=4 未覆盖全部75城" = length(.missing(DTW_CLUSTER_OPTIMIZED4_MAPPING)) == 0,
  "DTW4lag12 未覆盖全部75城" = length(.missing(DTW_CLUSTER_LAG12_MAPPING)) == 0
)
# 第二阶段分区来自 df_socioecon_global 的 climate_zone/city_cluster/geographic_region（同上映射），故同样覆盖75城

# ========== 【V5】读取1km精度grid_vars.gpkg（仅用于第二阶段）==========
cat("\n[全局] 读取1km精度grid_vars.gpkg（用于第二阶段meta-regression）...\n")

if (file.exists(GRID_VARS_FILE)) {
  library(sf)
  
  # 读取gpkg文件
  cat("  读取grid_vars.gpkg...\n")
  grid_vars_sf <- st_read(GRID_VARS_FILE, quiet = TRUE)
  
  cat("  ✓ 已加载", nrow(grid_vars_sf), "个1km网格\n")
  cat("  ✓ 变量数:", ncol(grid_vars_sf) - 1, "（不含geometry）\n\n")
  
  # 读取城市边界（用于空间join）
  cat("  读取城市边界用于空间匹配...\n")
  if (!file.exists(CITY_BOUNDARY_FILE)) {
    cat("  Warning: city boundary file not found:", CITY_BOUNDARY_FILE, "\n")
    cat("  Socioeconomic meta-predictors will be skipped.\n")
    cat("  Stage 2 can still run without meta-predictors.\n\n")
    df_socioecon_global <- NULL
    USE_SOCIOECON_STAGE2 <- FALSE
    SOCIOECON_VARS_USED <- c()
  } else {
    city_boundaries <- st_read(CITY_BOUNDARY_FILE, quiet = TRUE)
    cat("  ✓ 已加载", nrow(city_boundaries), "个城市边界\n\n")
    
    # 计算1km网格的质心
    cat("  计算1km网格质心并执行空间join...\n")
    grid_vars_centroids <- st_centroid(grid_vars_sf)
    
    # 确保CRS一致
    if (st_crs(grid_vars_centroids) != st_crs(city_boundaries)) {
      grid_vars_centroids <- st_transform(grid_vars_centroids, st_crs(city_boundaries))
    }
    
    # 空间join：判断每个1km网格属于哪个城市
    grid_with_city <- st_join(grid_vars_centroids, city_boundaries["NAME"], left = FALSE)
    
    # 转换为data.frame
    grid_with_city_df <- grid_with_city %>%
      st_drop_geometry() %>%
      rename(city = NAME) %>%
      mutate(
        city = str_replace_all(city, "\\. ", "_"),  # "St. Louis" → "St_Louis"
        city = str_replace_all(city, " ", "_"),      # 其他空格 → 下划线
        city = str_replace_all(city, "\\.", "_")     # 剩余点号 → 下划线
      )
  
    cat("  ✓ 空间匹配完成\n")
    cat("  ✓ 匹配到", n_distinct(grid_with_city_df$city), "个城市\n")
    cat("  ✓ 有效网格数:", nrow(grid_with_city_df), "\n\n")
    
    # 筛选目标城市
    cat("  筛选目标城市的网格...\n")
    target_cities <- CITY_LIST
    
    grid_with_city_df <- grid_with_city_df %>%
      filter(city %in% target_cities)
    
    cat("  ✓ 筛选后剩余", nrow(grid_with_city_df), "个网格\n")
    cat("  ✓ 覆盖城市:", n_distinct(grid_with_city_df$city), "个\n\n")
    
    # 【V5.1】检查是否有网格匹配成功
    if (nrow(grid_with_city_df) == 0) {
      cat("  ✗ 错误：没有网格匹配到目标城市\n")
      cat("  可能原因：\n")
      cat("    1. CRS不一致（已尝试自动转换）\n")
      cat("    2. 城市边界文件与grid_vars.gpkg不匹配\n")
      cat("    3. grid_vars.gpkg的网格不在城市范围内\n")
      cat("  → 将跳过社会经济变量的加载\n\n")
      df_socioecon_global <- NULL
      USE_SOCIOECON_STAGE2 <- FALSE
      SOCIOECON_VARS_USED <- c()
    } else {
      # 【V5.1】检查城市覆盖率
      matched_cities <- n_distinct(grid_with_city_df$city)
      expected_cities <- length(target_cities)
      
      if (matched_cities < expected_cities) {
        missing_cities <- setdiff(target_cities, unique(grid_with_city_df$city))
        cat("  ⚠ 警告：部分城市没有匹配的网格\n")
        cat("    匹配城市:", matched_cities, "/", expected_cities, "\n")
        cat("    缺失城市 (前10个):", paste(head(missing_cities, 10), collapse=", "), "\n\n")
      }
      
      # 【关键】聚合到城市级：计算每个城市的平均值
      cat("  聚合到城市级（计算平均值）...\n")
      
      # 【V5.1】根据用户选择的时间段，动态计算逐年变量的平均值
      start_year <- year(START_DATE)
      end_year <- year(END_DATE)
      cat("    分析时间段:", start_year, "-", end_year, "\n")
      
      # 【V5.1】计算GDP的时间段平均值
      gdp_cols <- paste0("GDP_", start_year:end_year)
      gdp_cols_available <- gdp_cols[gdp_cols %in% names(grid_with_city_df)]
      cat("    GDP列:", length(gdp_cols_available), "年 (", paste(start_year:end_year, collapse=", "), ")\n")
      
      if (length(gdp_cols_available) > 0) {
        grid_with_city_df$GDP_mean <- rowMeans(grid_with_city_df[, gdp_cols_available, drop = FALSE], na.rm = TRUE)
      } else {
        cat("    ⚠ 警告: 未找到时间段内的GDP数据，将使用NA\n")
        grid_with_city_df$GDP_mean <- NA
      }
      
      # 【V5.1】计算Crime的时间段平均值
      crime_cols <- paste0("Crime_", start_year:end_year)
      crime_cols_available <- crime_cols[crime_cols %in% names(grid_with_city_df)]
      cat("    Crime列:", length(crime_cols_available), "年\n")
      
      if (length(crime_cols_available) > 0) {
        grid_with_city_df$Crime_mean <- rowMeans(grid_with_city_df[, crime_cols_available, drop = FALSE], na.rm = TRUE)
      } else {
        cat("    ⚠ 警告: 未找到时间段内的Crime数据，将使用NA\n")
        grid_with_city_df$Crime_mean <- NA
      }
      
      # 【V5.1】计算Unemployment的时间段平均值
      unemp_cols <- paste0("Unemployment_", start_year:end_year)
      unemp_cols_available <- unemp_cols[unemp_cols %in% names(grid_with_city_df)]
      cat("    Unemployment列:", length(unemp_cols_available), "年\n\n")
      
      if (length(unemp_cols_available) > 0) {
        grid_with_city_df$Unemployment_mean <- rowMeans(grid_with_city_df[, unemp_cols_available, drop = FALSE], na.rm = TRUE)
      } else {
        cat("    ⚠ 警告: 未找到时间段内的Unemployment数据，将使用NA\n")
        grid_with_city_df$Unemployment_mean <- NA
      }
    
    # 【V5.1】聚合到城市级（使用动态计算的时间段平均值）
    df_socioecon_global <- grid_with_city_df %>%
      group_by(city) %>%
      summarise(
        # 环境变量（单年）
        NDVI_2023 = mean(NDVI_2023, na.rm = TRUE),
        
        # 人口变量（单年，使用总人口20-55岁）
        total_20_55 = mean(total_20_5, na.rm = TRUE),
        
        # 【V5.1修改】经济变量（时间段平均）
        GDP = mean(GDP_mean, na.rm = TRUE),
        
        # 【V5.1修改】犯罪率（时间段平均）
        Crime = mean(Crime_mean, na.rm = TRUE),
        
        # 【V5.1修改】失业率（时间段平均）
        Unemployment = mean(Unemployment_mean, na.rm = TRUE),
        
        # 城市形态变量（单年/静态）
        BD = mean(BD, na.rm = TRUE),
        FAR = mean(FAR, na.rm = TRUE),
        
        # 城市化率（单年/静态）
        Urbanization_Rate = mean(Urbanization_Rate, na.rm = TRUE),
        
        # 路口密度（单年/静态）
        Street_Intersection_Density = mean(Street_Intersection_Density, na.rm = TRUE),
        
        # 到公交距离（单年/静态）
        Distance_to_Transit = mean(Distance_to_Transit, na.rm = TRUE),
        
        # 步行指数（单年/静态）
        Walkability_Index = mean(Walkability_Index, na.rm = TRUE),
        
        .groups = "drop"
      )
    
    cat("  ✓ 城市级聚合完成\n")
    cat("  ✓ 城市数:", nrow(df_socioecon_global), "\n")
    cat("  ✓ 变量数:", ncol(df_socioecon_global) - 1, "\n")
    cat("  变量: NDVI_2023, total_20_55,\n")
    cat("        GDP (", start_year, "-", end_year, "平均),\n", sep = "")
    cat("        Crime (", start_year, "-", end_year, "平均),\n", sep = "")
    cat("        Unemployment (", start_year, "-", end_year, "平均),\n", sep = "")
    cat("        BD, FAR, Urbanization_Rate, Street_Intersection_Density,\n")
    cat("        Distance_to_Transit, Walkability_Index\n")
    cat("  【V5.1】逐年变量已根据时间段平均\n")
    cat("  【修复】城市名称已标准化（空格→下划线）\n\n")
    
    # 【V5.1】Winsorize极端值（防止meta-regression不稳定）
    cat("  【V5.1】Winsorize极端值到99%分位数...\n")
      socioecon_vars <- META_PREDICTOR_BASES_FINAL
  
    df_socioecon_grid_global <- grid_with_city_df %>%
      transmute(
        city = city,
        NDVI_2023 = as.numeric(NDVI_2023),
        total_20_55 = as.numeric(total_20_5),
        GDP = as.numeric(GDP_mean),
        Crime = as.numeric(Crime_mean),
        Unemployment = as.numeric(Unemployment_mean),
        BD = as.numeric(BD),
        FAR = as.numeric(FAR),
        Urbanization_Rate = as.numeric(Urbanization_Rate),
        Street_Intersection_Density = as.numeric(Street_Intersection_Density),
        Distance_to_Transit = as.numeric(Distance_to_Transit),
        Walkability_Index = as.numeric(Walkability_Index)
      )
    cat("  ✓ 已保留城市内1km网格变量，用于可选Gini meta-predictors\n")

    grid10_fallback <- add_grid10_missing_city_predictors(
      df_city = df_socioecon_global,
      df_grid = df_socioecon_grid_global,
      target_cities = target_cities,
      vars = socioecon_vars
    )
    df_socioecon_global <- grid10_fallback$city
    df_socioecon_grid_global <- grid10_fallback$grid
    if (length(grid10_fallback$added) > 0) {
      cat("  ✓ grid10_environment_predictors fallback added missing city predictors: ",
          paste(grid10_fallback$added, collapse = ", "), "\n", sep = "")
      cat("  ✓ 城市数 after fallback:", nrow(df_socioecon_global), "\n")
    }
  
    for (var in socioecon_vars) {
      q99 <- quantile(df_socioecon_global[[var]], 0.99, na.rm = TRUE)
      q01 <- quantile(df_socioecon_global[[var]], 0.01, na.rm = TRUE)
      df_socioecon_global[[var]] <- pmin(pmax(df_socioecon_global[[var]], q01), q99)
    }
    
    # 标准化（用于第二阶段）
    for (var in socioecon_vars) {
      var_scaled <- paste0(var, "_scaled")
      df_socioecon_global[[var_scaled]] <- scale(df_socioecon_global[[var]])[,1]
    }
    
    if (all(c("total_20_55", "Unemployment") %in% names(df_socioecon_global))) {
      df_socioecon_global$Unemployed_Population <- pmax(df_socioecon_global$total_20_55, 0) *
        pmax(df_socioecon_global$Unemployment, 0) / 100
      q99_unemp <- quantile(df_socioecon_global$Unemployed_Population, 0.99, na.rm = TRUE)
      q01_unemp <- quantile(df_socioecon_global$Unemployed_Population, 0.01, na.rm = TRUE)
      df_socioecon_global$Unemployed_Population <- pmin(
        pmax(df_socioecon_global$Unemployed_Population, q01_unemp),
        q99_unemp
      )
      df_socioecon_global$Unemployed_Population_scaled <- scale(df_socioecon_global$Unemployed_Population)[,1]
      cat("  【V5.4】已基于 total_20_55 × Unemployment 推导城市级失业人口代理变量\n")
    }
    
    cat("  ✓ 变量已标准化\n")
    cat("  【V5说明】第一阶段不使用这些变量（避免过度参数化）\n")
    cat("  【V5优势】1km精度聚合到城市级，提供更准确的城市特征\n\n")
    
    # 【V5新增】添加分类信息（climate_zone, city_cluster, geographic_region）
    cat("  【V5】添加城市分类信息（气候带、聚类、地理区域）...\n")
    df_socioecon_global <- df_socioecon_global %>%
      mutate(
        climate_zone = case_when(
          city %in% CLIMATE_ZONE_MAPPING$Temperate ~ "Temperate",
          city %in% CLIMATE_ZONE_MAPPING$Cold ~ "Cold",
          city %in% CLIMATE_ZONE_MAPPING$Arid ~ "Arid",
          city %in% CLIMATE_ZONE_MAPPING$Tropical ~ "Tropical",
          TRUE ~ "Unknown"
        ),
        city_cluster = case_when(
          city %in% CITY_CLUSTER_MAPPING$Texas_Southern_Hot ~ "Texas_Southern_Hot",
          city %in% CITY_CLUSTER_MAPPING$East_Coast_Moderate ~ "East_Coast_Moderate",
          city %in% CITY_CLUSTER_MAPPING$Northern_Cold ~ "Northern_Cold",
          city %in% CITY_CLUSTER_MAPPING$West_Coast_Mediterranean ~ "West_Coast_Mediterranean",
          city %in% CITY_CLUSTER_MAPPING$Arid_Desert ~ "Arid_Desert",
          city %in% CITY_CLUSTER_MAPPING$Tropical ~ "Tropical",
          city %in% CITY_CLUSTER_MAPPING$Transition_Special ~ "Transition_Special",
          TRUE ~ "Unknown"
        ),
        geographic_region = case_when(
          city %in% GEOGRAPHIC_REGION_MAPPING$Northeast ~ "Northeast",
          city %in% GEOGRAPHIC_REGION_MAPPING$Southeast ~ "Southeast",
          city %in% GEOGRAPHIC_REGION_MAPPING$South ~ "South",
          city %in% GEOGRAPHIC_REGION_MAPPING$Midwest ~ "Midwest",
          city %in% GEOGRAPHIC_REGION_MAPPING$Southwest ~ "Southwest",
          city %in% GEOGRAPHIC_REGION_MAPPING$West ~ "West",
          TRUE ~ "Unknown"
        )
      )
    
    cat("  ✓ 分类信息已添加\n")
    cat("    - Climate Zone:", n_distinct(df_socioecon_global$climate_zone), "类\n")
    cat("    - City Cluster:", n_distinct(df_socioecon_global$city_cluster), "类\n")
    cat("    - Geographic Region:", n_distinct(df_socioecon_global$geographic_region), "类\n\n")
      
      USE_SOCIOECON_STAGE2 <- TRUE  # 只在第二阶段使用
      SOCIOECON_VARS_USED <- socioecon_vars
    }
  }
  
  # 【V5.1】确保变量被正确设置
  if (!exists("USE_SOCIOECON_STAGE2") || !USE_SOCIOECON_STAGE2) {
    df_socioecon_global <- NULL
    USE_SOCIOECON_STAGE2 <- FALSE
    SOCIOECON_VARS_USED <- c()
  }
} else {
  cat("  ⚠ 警告: 未找到grid_vars.gpkg文件\n")
  cat("    路径:", GRID_VARS_FILE, "\n")
  cat("    第二阶段将不包含社会经济meta-predictors\n\n")
  
  USE_SOCIOECON_STAGE2 <- FALSE
  df_socioecon_global <- NULL
  SOCIOECON_VARS_USED <- c()
}

# ===== V5 Repair: retry stage-2 socioecon loading if the earlier block failed =====
if ((!exists("USE_SOCIOECON_STAGE2") || !isTRUE(USE_SOCIOECON_STAGE2) || is.null(df_socioecon_global)) &&
    file.exists(GRID_VARS_FILE) &&
    file.exists(CITY_BOUNDARY_FILE)) {
  cat("\n[V5 repair] Rebuilding city-level stage-2 meta-predictors after failed initialization...\n")
  
  grid_vars_sf_repair <- st_read(GRID_VARS_FILE, quiet = TRUE)
  city_boundaries_repair <- st_read(CITY_BOUNDARY_FILE, quiet = TRUE)
  grid_centroids_repair <- st_centroid(grid_vars_sf_repair)
  
  if (st_crs(grid_centroids_repair) != st_crs(city_boundaries_repair)) {
    grid_centroids_repair <- st_transform(grid_centroids_repair, st_crs(city_boundaries_repair))
  }
  
  grid_with_city_repair <- st_join(grid_centroids_repair, city_boundaries_repair["NAME"], left = FALSE)
  grid_with_city_df_repair <- grid_with_city_repair %>%
    st_drop_geometry() %>%
    rename(city = NAME) %>%
    mutate(
      city = str_replace_all(city, "\\. ", "_"),
      city = str_replace_all(city, " ", "_"),
      city = str_replace_all(city, "\\.", "_")
    ) %>%
    filter(city %in% CITY_LIST)
  
  if (nrow(grid_with_city_df_repair) > 0) {
    start_year_repair <- year(START_DATE)
    end_year_repair <- year(END_DATE)
    
    gdp_cols_repair <- paste0("GDP_", start_year_repair:end_year_repair)
    gdp_cols_repair <- gdp_cols_repair[gdp_cols_repair %in% names(grid_with_city_df_repair)]
    grid_with_city_df_repair$GDP_mean <- if (length(gdp_cols_repair) > 0) {
      rowMeans(grid_with_city_df_repair[, gdp_cols_repair, drop = FALSE], na.rm = TRUE)
    } else {
      rep(NA_real_, nrow(grid_with_city_df_repair))
    }
    
    crime_cols_repair <- paste0("Crime_", start_year_repair:end_year_repair)
    crime_cols_repair <- crime_cols_repair[crime_cols_repair %in% names(grid_with_city_df_repair)]
    grid_with_city_df_repair$Crime_mean <- if (length(crime_cols_repair) > 0) {
      rowMeans(grid_with_city_df_repair[, crime_cols_repair, drop = FALSE], na.rm = TRUE)
    } else {
      rep(NA_real_, nrow(grid_with_city_df_repair))
    }
    
    unemp_cols_repair <- paste0("Unemployment_", start_year_repair:end_year_repair)
    unemp_cols_repair <- unemp_cols_repair[unemp_cols_repair %in% names(grid_with_city_df_repair)]
    grid_with_city_df_repair$Unemployment_mean <- if (length(unemp_cols_repair) > 0) {
      rowMeans(grid_with_city_df_repair[, unemp_cols_repair, drop = FALSE], na.rm = TRUE)
    } else {
      rep(NA_real_, nrow(grid_with_city_df_repair))
    }
    
    socioecon_vars_repair <- META_PREDICTOR_BASES_FINAL
    
    df_socioecon_grid_global <- grid_with_city_df_repair %>%
      transmute(
        city = city,
        NDVI_2023 = as.numeric(NDVI_2023),
        total_20_55 = as.numeric(total_20_5),
        GDP = as.numeric(GDP_mean),
        Crime = as.numeric(Crime_mean),
        Unemployment = as.numeric(Unemployment_mean),
        BD = as.numeric(BD),
        FAR = as.numeric(FAR),
        Urbanization_Rate = as.numeric(Urbanization_Rate),
        Street_Intersection_Density = as.numeric(Street_Intersection_Density),
        Distance_to_Transit = as.numeric(Distance_to_Transit),
        Walkability_Index = as.numeric(Walkability_Index)
      )
    
    df_socioecon_global <- grid_with_city_df_repair %>%
      group_by(city) %>%
      summarise(
        NDVI_2023 = mean(NDVI_2023, na.rm = TRUE),
        total_20_55 = mean(total_20_5, na.rm = TRUE),
        GDP = mean(GDP_mean, na.rm = TRUE),
        Crime = mean(Crime_mean, na.rm = TRUE),
        Unemployment = mean(Unemployment_mean, na.rm = TRUE),
        BD = mean(BD, na.rm = TRUE),
        FAR = mean(FAR, na.rm = TRUE),
        Urbanization_Rate = mean(Urbanization_Rate, na.rm = TRUE),
        Street_Intersection_Density = mean(Street_Intersection_Density, na.rm = TRUE),
        Distance_to_Transit = mean(Distance_to_Transit, na.rm = TRUE),
        Walkability_Index = mean(Walkability_Index, na.rm = TRUE),
        .groups = "drop"
      )

    grid10_fallback_repair <- add_grid10_missing_city_predictors(
      df_city = df_socioecon_global,
      df_grid = df_socioecon_grid_global,
      target_cities = CITY_LIST,
      vars = socioecon_vars_repair
    )
    df_socioecon_global <- grid10_fallback_repair$city
    df_socioecon_grid_global <- grid10_fallback_repair$grid
    if (length(grid10_fallback_repair$added) > 0) {
      cat("[V5 repair] grid10_environment_predictors fallback added missing city predictors: ",
          paste(grid10_fallback_repair$added, collapse = ", "), "\n", sep = "")
    }
    
    for (var in socioecon_vars_repair) {
      q99 <- quantile(df_socioecon_global[[var]], 0.99, na.rm = TRUE)
      q01 <- quantile(df_socioecon_global[[var]], 0.01, na.rm = TRUE)
      df_socioecon_global[[var]] <- pmin(pmax(df_socioecon_global[[var]], q01), q99)
      df_socioecon_global[[paste0(var, "_scaled")]] <- scale(df_socioecon_global[[var]])[, 1]
    }
    
    if (all(c("total_20_55", "Unemployment") %in% names(df_socioecon_global))) {
      df_socioecon_global$Unemployed_Population <- pmax(df_socioecon_global$total_20_55, 0) *
        pmax(df_socioecon_global$Unemployment, 0) / 100
      q99_unemp <- quantile(df_socioecon_global$Unemployed_Population, 0.99, na.rm = TRUE)
      q01_unemp <- quantile(df_socioecon_global$Unemployed_Population, 0.01, na.rm = TRUE)
      df_socioecon_global$Unemployed_Population <- pmin(
        pmax(df_socioecon_global$Unemployed_Population, q01_unemp),
        q99_unemp
      )
      df_socioecon_global$Unemployed_Population_scaled <- scale(df_socioecon_global$Unemployed_Population)[, 1]
    }
    
    df_socioecon_global <- df_socioecon_global %>%
      mutate(
        climate_zone = case_when(
          city %in% CLIMATE_ZONE_MAPPING$Temperate ~ "Temperate",
          city %in% CLIMATE_ZONE_MAPPING$Cold ~ "Cold",
          city %in% CLIMATE_ZONE_MAPPING$Arid ~ "Arid",
          city %in% CLIMATE_ZONE_MAPPING$Tropical ~ "Tropical",
          TRUE ~ "Unknown"
        ),
        city_cluster = case_when(
          city %in% CITY_CLUSTER_MAPPING$Texas_Southern_Hot ~ "Texas_Southern_Hot",
          city %in% CITY_CLUSTER_MAPPING$East_Coast_Moderate ~ "East_Coast_Moderate",
          city %in% CITY_CLUSTER_MAPPING$Northern_Cold ~ "Northern_Cold",
          city %in% CITY_CLUSTER_MAPPING$West_Coast_Mediterranean ~ "West_Coast_Mediterranean",
          city %in% CITY_CLUSTER_MAPPING$Arid_Desert ~ "Arid_Desert",
          city %in% CITY_CLUSTER_MAPPING$Tropical ~ "Tropical",
          city %in% CITY_CLUSTER_MAPPING$Transition_Special ~ "Transition_Special",
          TRUE ~ "Unknown"
        ),
        geographic_region = case_when(
          city %in% GEOGRAPHIC_REGION_MAPPING$Northeast ~ "Northeast",
          city %in% GEOGRAPHIC_REGION_MAPPING$Southeast ~ "Southeast",
          city %in% GEOGRAPHIC_REGION_MAPPING$South ~ "South",
          city %in% GEOGRAPHIC_REGION_MAPPING$Midwest ~ "Midwest",
          city %in% GEOGRAPHIC_REGION_MAPPING$Southwest ~ "Southwest",
          city %in% GEOGRAPHIC_REGION_MAPPING$West ~ "West",
          TRUE ~ "Unknown"
        )
      )
    
    USE_SOCIOECON_STAGE2 <- TRUE
    SOCIOECON_VARS_USED <- socioecon_vars_repair
    
    cat("[V5 repair] Rebuilt city-level meta-predictors for", nrow(df_socioecon_global), "cities using",
        length(SOCIOECON_VARS_USED), "base variables\n\n")
  } else {
    cat("[V5 repair] Rebuild failed: city boundaries and grid_vars.gpkg still did not match target cities; continuing without meta-predictors\n\n")
    df_socioecon_global <- NULL
    USE_SOCIOECON_STAGE2 <- FALSE
    SOCIOECON_VARS_USED <- c()
  }
}

# ========== 仅重绘可视化辅助函数 ==========

safe_read_csv_file <- function(file_path) {
  if (!file.exists(file_path)) return(NULL)
  tryCatch(
    read_csv(file_path, show_col_types = FALSE),
    error = function(e) NULL
  )
}

infer_saved_model_type <- function(file_name) {
  model_name <- str_extract(
    basename(file_name),
    "(composite|day|night|Composite|Day|Night)(_(all|ride|run|walk))?|DayOnly|NightOnly"
  )
  if (is.na(model_name) || is.null(model_name)) return("composite")
  normalize_stage1_model_name(model_name)
}

clean_saved_result_label <- function(label_text) {
  label_text %>%
    str_replace("_(Composite|Day|Night|DayOnly|NightOnly)$", "") %>%
    str_replace("_DLNM_STAGE1$", "")
}

rerender_stage1_result_dir <- function(result_dir) {
  rds_files <- list.files(result_dir, pattern = "_DLNM_result\\.rds$", full.names = TRUE)
  if (length(rds_files) == 0) return(0L)
  
  rerendered_n <- 0L
  overlay_results <- list()
  overlay_indicator <- if (grepl("_exceeded_quantity$", basename(result_dir), ignore.case = TRUE)) "exceeded_quantity" else "cehwi"
  overlay_city_name <- clean_saved_result_label(basename(result_dir))
  for (rds_file in rds_files) {
    result <- tryCatch(read_rds(rds_file), error = function(e) NULL)
    if (is.null(result)) next
    
    model_type <- infer_saved_model_type(rds_file)
    indicator <- if (!is.null(result$indicator)) result$indicator else {
      if (grepl("_exceeded_quantity$", basename(result_dir), ignore.case = TRUE)) "exceeded_quantity" else "cehwi"
    }
    display_name <- clean_saved_result_label(
      if (!is.null(result$city)) result$city else basename(result_dir)
    )
    overlay_city_name <- display_name
    overlay_indicator <- indicator
    result$model_name <- model_type
    result$model_type <- model_base_type(model_type)
    result$activity_type <- model_activity_type(model_type)
    result$activity_label <- unname(activity_modality_labels()[result$activity_type])
    overlay_results[[model_type]] <- result
    
    try(plot_city_rr_curve(result, result_dir, model_type = model_type), silent = TRUE)
    try(plot_control_variables_forest(result, result_dir, display_name, indicator, model_type = model_type), silent = TRUE)
    rerendered_n <- rerendered_n + 1L
  }
  try(save_city_activity_overlay_plots(overlay_results, result_dir, overlay_city_name, overlay_indicator), silent = TRUE)
  
  rerendered_n
}

collect_saved_stage1_results <- function(output_dir) {
  list(
    cehwi = load_stage1_results(output_dir, "cehwi"),
    exceeded_quantity = load_stage1_results(output_dir, "exceeded_quantity")
  )
}

get_partition_city_list <- function(partition_family, partition_name) {
  if (partition_family == "Zone" && exists("CLIMATE_ZONE_MAPPING") && partition_name %in% names(CLIMATE_ZONE_MAPPING)) {
    return(CLIMATE_ZONE_MAPPING[[partition_name]])
  }
  if (partition_family == "Cluster" && exists("CITY_CLUSTER_MAPPING") && partition_name %in% names(CITY_CLUSTER_MAPPING)) {
    return(CITY_CLUSTER_MAPPING[[partition_name]])
  }
  if (partition_family == "Region" && exists("GEOGRAPHIC_REGION_MAPPING") && partition_name %in% names(GEOGRAPHIC_REGION_MAPPING)) {
    return(GEOGRAPHIC_REGION_MAPPING[[partition_name]])
  }
  if (partition_family == "DTW_Optimized3" && exists("DTW_CLUSTER_OPTIMIZED3_MAPPING") && partition_name %in% names(DTW_CLUSTER_OPTIMIZED3_MAPPING)) {
    return(DTW_CLUSTER_OPTIMIZED3_MAPPING[[partition_name]])
  }
  if (partition_family == "DTW_Optimized4" && exists("DTW_CLUSTER_OPTIMIZED4_MAPPING") && partition_name %in% names(DTW_CLUSTER_OPTIMIZED4_MAPPING)) {
    return(DTW_CLUSTER_OPTIMIZED4_MAPPING[[partition_name]])
  }
  if (partition_family == "DTW4lag12" && exists("DTW_CLUSTER_LAG12_MAPPING") && partition_name %in% names(DTW_CLUSTER_LAG12_MAPPING)) {
    return(DTW_CLUSTER_LAG12_MAPPING[[partition_name]])
  }
  NULL
}

collect_exposure_values_from_stage1 <- function(stage1_results, indicator, model_type, city_subset = NULL) {
  result_list <- stage1_results[[indicator]]
  if (is.null(result_list) || length(result_list) == 0) return(numeric(0))
  
  values <- unlist(lapply(names(result_list), function(city_key) {
    city_name <- str_replace(city_key, paste0("_", indicator, "$"), "")
    if (!is.null(city_subset) && !city_name %in% city_subset) return(NULL)
    city_models <- result_list[[city_key]]
    if (is.null(city_models) || !model_type %in% names(city_models)) return(NULL)
    model_result <- city_models[[model_type]]
    if (!is.null(model_result$cehwi_data)) {
      return(model_result$cehwi_data[is.finite(model_result$cehwi_data) & model_result$cehwi_data > 0])
    }
    if (!is.null(model_result$pred_df) && "cehwi" %in% names(model_result$pred_df)) {
      return(model_result$pred_df$cehwi[is.finite(model_result$pred_df$cehwi) & model_result$pred_df$cehwi > 0])
    }
    NULL
  }))
  
  values[is.finite(values)]
}

build_saved_pooled_subtitle <- function(output_dir, indicator, model_type, n_cities, reliability = "STABLE", prefix_text = NULL) {
  model_eval <- safe_read_csv_file(file.path(output_dir, "model_evaluation.csv"))
  effect_text <- "Peak RR = N/A"
  i2_text <- "I² = N/A"
  aic_text <- "AIC = N/A"
  sig_text <- NULL
  
  if (!is.null(model_eval) && nrow(model_eval) > 0) {
    effect_text <- if (all(c("rr_at_max", "effect_pct") %in% names(model_eval)) &&
                       is.finite(model_eval$rr_at_max[1]) &&
                       is.finite(model_eval$effect_pct[1])) {
      paste0("Peak RR = ", round(model_eval$rr_at_max[1], 3), " (", sprintf("%+.1f", model_eval$effect_pct[1]), "%)")
    } else effect_text
    i2_text <- if ("I_squared" %in% names(model_eval) && is.finite(model_eval$I_squared[1])) {
      paste0("I² = ", round(model_eval$I_squared[1], 1), "%")
    } else i2_text
    aic_text <- if ("AIC" %in% names(model_eval) && is.finite(model_eval$AIC[1])) {
      paste0("AIC = ", round(model_eval$AIC[1], 1))
    } else aic_text
    if ("n_sig_coefs" %in% names(model_eval) && "n_coefs" %in% names(model_eval)) {
      sig_text <- paste0(model_eval$n_sig_coefs[1], "/", model_eval$n_coefs[1], " sig. coefs")
    }
  }
  
  metric_parts <- c(
    paste0("N cities = ", n_cities),
    effect_text,
    i2_text,
    aic_text,
    sig_text,
    if (!is.null(reliability) && reliability != "STABLE") paste0("Flag = ", reliability) else NULL
  )
  
  paste0(
    prefix_text,
    ifelse(is.null(prefix_text), "", "\n"),
    paste(metric_parts[!is.na(metric_parts) & nzchar(metric_parts)], collapse = "  |  ")
  )
}

prepare_rr_curve_display <- function(pooled_df,
                                     y_upper_cap = 10,
                                     y_upper_quantile = 0.95) {
  if (is.null(pooled_df) || nrow(pooled_df) == 0) return(NULL)
  required_cols <- c("cehwi", "rr", "rr_low", "rr_high")
  if (!all(required_cols %in% names(pooled_df))) return(NULL)
  
  rr_raw <- suppressWarnings(as.numeric(pooled_df$rr))
  rr_finite <- rr_raw[is.finite(rr_raw) & rr_raw >= 0]
  if (length(rr_finite) < 2) return(NULL)
  
  y_upper <- suppressWarnings(quantile(rr_finite, y_upper_quantile, na.rm = TRUE, names = FALSE))
  if (!is.finite(y_upper) || y_upper <= 0) {
    y_upper <- max(rr_finite, na.rm = TRUE)
  }
  if (!is.finite(y_upper) || y_upper <= 0) y_upper <- 1.5
  y_upper <- min(y_upper_cap, max(1.5, y_upper * 1.15))
  y_lower <- 0
  
  plot_df <- pooled_df %>%
    mutate(
      rr = suppressWarnings(as.numeric(rr)),
      rr_low = suppressWarnings(as.numeric(rr_low)),
      rr_high = suppressWarnings(as.numeric(rr_high)),
      rr_display_outlier = !is.finite(rr) | rr < y_lower | rr > y_upper,
      rr_line_plot = ifelse(rr_display_outlier, NA_real_, rr),
      rr_low_clipped = ifelse(is.finite(rr_low), pmax(rr_low, y_lower), NA_real_),
      rr_high_clipped = ifelse(is.finite(rr_high), pmin(rr_high, y_upper), y_upper),
      rr_ribbon_valid = is.finite(rr_low_clipped) & is.finite(rr_high_clipped) & rr_high_clipped >= rr_low_clipped
    )
  
  ci_clipped <- any(
    !is.finite(plot_df$rr_low) | !is.finite(plot_df$rr_high) |
      plot_df$rr_low < y_lower | plot_df$rr_high > y_upper,
    na.rm = TRUE
  ) || any(plot_df$rr_display_outlier, na.rm = TRUE)
  
  list(
    data = plot_df,
    y_limits = c(y_lower, y_upper),
    ci_clipped = ci_clipped,
    n_display_outliers = sum(plot_df$rr_display_outlier, na.rm = TRUE)
  )
}

rerender_saved_rr_bundle <- function(pooled_df,
                                     output_dir,
                                     title_text,
                                     subtitle_text,
                                     indicator,
                                     model_type,
                                     exposure_values = numeric(0),
                                     reliability = "STABLE") {
  if (is.null(pooled_df) || nrow(pooled_df) == 0) return(FALSE)
  
  pooled_df <- pooled_df %>% filter(is.finite(cehwi), is.finite(rr), is.finite(rr_low), is.finite(rr_high))
  if (nrow(pooled_df) == 0) return(FALSE)
  
  line_color <- stage1_model_color(model_type)
  if (is.null(line_color) || is.na(line_color)) line_color <- "#D53E4F"
  
  display_bundle <- prepare_rr_curve_display(pooled_df)
  if (is.null(display_bundle)) return(FALSE)
  pooled_df_plot <- display_bundle$data
  ci_exploded <- isTRUE(display_bundle$ci_clipped)
  
  p_rr <- ggplot(pooled_df_plot, aes(x = cehwi)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_ribbon(
      data = pooled_df_plot %>% filter(rr_ribbon_valid),
      aes(ymin = rr_low_clipped, ymax = rr_high_clipped),
      fill = "gray80",
      alpha = 0.45
    ) +
    geom_line(aes(y = rr_line_plot), color = line_color, linewidth = 2, na.rm = TRUE) +
    coord_cartesian(ylim = display_bundle$y_limits, clip = "on") +
    labs(title = title_text, subtitle = subtitle_text, x = toupper(indicator), y = "Relative Risk (RR)") +
    rr_plot_theme(14)
  
  ggsave(file.path(output_dir, "pooled_RR_curve.png"), p_rr, width = 12, height = 8, dpi = 300)
  safe_write_csv(
    pooled_df_plot %>% filter(rr_display_outlier),
    file.path(output_dir, "pooled_RR_display_outliers.csv"),
    label = "pooled RR display outliers"
  )
  
  if (length(exposure_values) > 10) {
    x_max <- quantile(exposure_values, 0.98, na.rm = TRUE)
    hist_df <- data.frame(cehwi = exposure_values) %>% filter(is.finite(cehwi), cehwi > 0, cehwi <= x_max)
    if (nrow(hist_df) > 0) {
      bin_width <- diff(range(hist_df$cehwi, na.rm = TRUE)) / 30
      if (!is.finite(bin_width) || bin_width <= 0) bin_width <- 1
      p_hist <- ggplot(hist_df, aes(x = cehwi)) +
        geom_histogram(binwidth = bin_width, fill = "gray70", color = "white", linewidth = 0.3) +
        labs(title = "Exposure Distribution", x = toupper(indicator), y = "Count") +
        rr_hist_theme(12)
      ggsave(file.path(output_dir, "exposure_distribution.png"), p_hist, width = 12, height = 4, dpi = 300)
      
      caption_text <- build_rr_caption(
        indicator = indicator,
        ci_clipped = ci_exploded,
        reliability = reliability,
        include_percentiles = FALSE,
        x_truncated = TRUE
      )
      p_combined <- build_rr_distribution_combined_plot(p_rr, p_hist, caption_text = caption_text)
      ggsave(file.path(output_dir, "pooled_RR_with_distribution.png"), p_combined, width = 12, height = 10, dpi = 300)
    }
  }
  
  TRUE
}

conditional_rr_predictor_label <- function(pred_var) {
  case_when(
    grepl("NDVI", pred_var) ~ "NDVI (Vegetation Index)",
    grepl("total_20", pred_var) ~ "Population (20-55 years)",
    grepl("GDP", pred_var) ~ "GDP",
    grepl("Crime", pred_var) ~ "Crime",
    grepl("Unemployment", pred_var) ~ "Unemployment",
    grepl("BD|Building_Density", pred_var) ~ "Building Density",
    grepl("Urbanization", pred_var) ~ "Urbanization Rate",
    grepl("Street_Intersection", pred_var) ~ "Street Intersection Density",
    grepl("Walkability", pred_var) ~ "Walkability Index",
    TRUE ~ pred_var
  )
}

conditional_rr_ylim <- function(pred_data) {
  rr_vals <- suppressWarnings(as.numeric(pred_data$rr))
  rr_vals <- rr_vals[is.finite(rr_vals) & rr_vals > 0]
  if (length(rr_vals) < 2) return(NULL)
  q <- suppressWarnings(quantile(rr_vals, c(0.02, 0.98), na.rm = TRUE, names = FALSE))
  if (any(!is.finite(q)) || diff(q) <= 0) {
    q <- range(rr_vals, na.rm = TRUE)
  }
  y_min <- min(q[1], 1, na.rm = TRUE)
  y_max <- max(q[2], 1, na.rm = TRUE)
  pad <- max(0.05, (y_max - y_min) * 0.12)
  c(max(0, y_min - pad), y_max + pad)
}

save_conditional_rr_outputs <- function(conditional_rr,
                                        output_dir,
                                        indicator = NULL,
                                        model_type = NULL,
                                        title_prefix = NULL,
                                        write_data = TRUE,
                                        show_ci = FALSE) {
  if (is.null(conditional_rr)) return(0L)
  conditional_rr_df <- if (is.list(conditional_rr) && !is.data.frame(conditional_rr)) {
    bind_rows(conditional_rr)
  } else {
    as.data.frame(conditional_rr)
  }
  if (is.null(conditional_rr_df) || nrow(conditional_rr_df) == 0 ||
      !"predictor" %in% names(conditional_rr_df)) return(0L)
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  if (isTRUE(write_data)) {
    safe_write_csv(conditional_rr_df, file.path(output_dir, "conditional_RR_curves.csv"), label = "conditional RR curves")
  }
  
  saved_n <- 0L
  for (pred_var in unique(conditional_rr_df$predictor)) {
    pred_data <- conditional_rr_df %>% filter(predictor == pred_var)
    if (nrow(pred_data) == 0) next
    
    pred_label <- conditional_rr_predictor_label(pred_var)
    y_limits <- conditional_rr_ylim(pred_data)
    curve_subtitle <- paste0(
      if (!is.null(title_prefix) && nzchar(title_prefix)) paste0(title_prefix, " | ") else "",
      if (!is.null(indicator) && !is.null(model_type)) paste0(toupper(indicator), " - ", toupper(model_type), " | ") else "",
      "Low/Mean/High are -1/0/+1 SD of the standardized meta-predictor. ",
      "Y-axis is scaled to the fitted RR curves; full CI values remain in conditional_RR_curves.csv."
    )
    
    p_cond <- ggplot(pred_data, aes(x = cehwi, y = rr, color = level, fill = level)) +
      geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8)
    if (isTRUE(show_ci) && all(c("rr_low", "rr_high") %in% names(pred_data))) {
      p_cond <- p_cond +
        geom_ribbon(aes(ymin = rr_low, ymax = rr_high), alpha = 0.08, color = NA)
    }
    p_cond <- p_cond +
      geom_line(linewidth = 1.8) +
      scale_color_manual(values = c("Low" = "#2C7BB6", "Mean" = "#555555", "Medium" = "#555555", "High" = "#D7191C")) +
      scale_fill_manual(values = c("Low" = "#2C7BB6", "Mean" = "#555555", "Medium" = "#555555", "High" = "#D7191C")) +
      labs(
        title = paste0("Conditional RR Curves - ", pred_label),
        subtitle = curve_subtitle,
        x = ifelse(is.null(indicator), "Exposure", toupper(indicator)),
        y = "Relative Risk (RR)",
        color = NULL,
        fill = NULL
      ) +
      rr_plot_theme(14) +
      theme(
        legend.position = "top",
        plot.subtitle = element_text(size = 10, color = "gray35", lineheight = 1.15)
      )
    if (!is.null(y_limits)) {
      p_cond <- p_cond + coord_cartesian(ylim = y_limits)
    }
    
    safe_pred_name <- str_replace_all(pred_var, "[^a-zA-Z0-9_]", "_")
    ggsave(file.path(output_dir, paste0("conditional_RR_", safe_pred_name, ".png")),
           p_cond, width = 14, height = 9, dpi = 300)
    saved_n <- saved_n + 1L
  }
  
  saved_n
}

generate_partition_conditional_rr <- function(mv_model,
                                              coef_matrix,
                                              covariates_std,
                                              partition_results,
                                              output_dir,
                                              indicator,
                                              model_type,
                                              title_prefix = NULL) {
  if (is.null(mv_model) || is.null(coef_matrix) || is.null(covariates_std) ||
      ncol(covariates_std) == 0 || is.null(partition_results) || length(partition_results) == 0) {
    return(0L)
  }
  
  cb_template <- tryCatch({
    first_with_cb <- partition_results[vapply(partition_results, function(x) !is.null(x$cb), logical(1))]
    if (length(first_with_cb) == 0) NULL else first_with_cb[[1]]$cb
  }, error = function(e) NULL)
  if (is.null(cb_template)) return(0L)
  
  exposure_values <- unlist(lapply(partition_results, function(x) {
    if (!is.null(x$cehwi_data)) return(x$cehwi_data[x$cehwi_data > 0])
    if (!is.null(x$cehwi_range)) return(x$cehwi_range)
    NULL
  }))
  exposure_values <- exposure_values[is.finite(exposure_values) & exposure_values >= 0]
  cehwi_max <- if (length(exposure_values) > 0) {
    max(quantile(exposure_values, 0.98, na.rm = TRUE), 1, na.rm = TRUE)
  } else {
    10
  }
  if (!is.finite(cehwi_max) || cehwi_max <= 0) cehwi_max <- 10
  cehwi_seq <- seq(0, cehwi_max, length.out = 300)
  
  predictor_names <- names(covariates_std)
  newdata_mean <- as.data.frame(as.list(rep(0, length(predictor_names))))
  names(newdata_mean) <- predictor_names
  
  conditional_rr_list <- list()
  levels_num <- c(-1, 0, 1)
  level_names <- c("Low", "Mean", "High")
  
  for (pred_var in predictor_names) {
    pred_curves <- list()
    for (i in seq_along(levels_num)) {
      newdata_cond <- newdata_mean
      newdata_cond[[pred_var]] <- levels_num[i]
      pred_cond <- tryCatch(
        predict(mv_model, newdata = newdata_cond, vcov = TRUE),
        error = function(e) NULL
      )
      if (is.null(pred_cond)) next
      
      coef_cond <- as.vector(pred_cond$fit)
      vcov_cond <- pred_cond$vcov
      if (length(dim(vcov_cond)) == 3) vcov_cond <- vcov_cond[1, , ]
      if (length(coef_cond) != ncol(coef_matrix)) next
      
      cp_cond <- tryCatch(
        crosspred(
          cb_template,
          coef = coef_cond,
          vcov = vcov_cond,
          model.link = "log",
          at = cehwi_seq,
          cen = get0("REFERENCE_CEHWI", ifnotfound = 0),
          cumul = TRUE
        ),
        error = function(e) NULL
      )
      if (is.null(cp_cond)) next
      
      pred_curves[[level_names[i]]] <- data.frame(
        cehwi = cehwi_seq,
        rr = cp_cond$allRRfit,
        rr_low = cp_cond$allRRlow,
        rr_high = cp_cond$allRRhigh,
        level = level_names[i],
        predictor = pred_var,
        stringsAsFactors = FALSE
      )
    }
    if (length(pred_curves) > 0) {
      conditional_rr_list[[pred_var]] <- bind_rows(pred_curves)
    }
  }
  
  if (length(conditional_rr_list) == 0) return(0L)
  
  saved_n <- save_conditional_rr_outputs(
    conditional_rr_list,
    output_dir = output_dir,
    indicator = indicator,
    model_type = model_type,
    title_prefix = title_prefix,
    write_data = TRUE,
    show_ci = FALSE
  )
  
  save_meta_model_artifact(
    list(
      model = mv_model,
      conditional_rr = conditional_rr_list,
      meta_predictor_mode = meta_predictor_mode_label(),
      indicator = indicator,
      model_type = model_type,
      title_prefix = title_prefix
    ),
    file.path(output_dir, "meta_model.rds"),
    label = "partition conditional RR meta model"
  )
  
  saved_n
}

rerender_saved_conditional_rr <- function(output_dir) {
  model_dirs <- list.dirs(output_dir, full.names = TRUE, recursive = FALSE)
  model_dirs <- model_dirs[grepl("^META_MODEL_(MEAN|GINI)(_NO_CRIME)?$", basename(model_dirs))]
  if (length(model_dirs) > 0) {
    return(sum(vapply(model_dirs, rerender_saved_conditional_rr, integer(1))))
  }
  
  conditional_rr_df <- safe_read_csv_file(file.path(output_dir, "conditional_RR_curves.csv"))
  if (is.null(conditional_rr_df) || nrow(conditional_rr_df) == 0) return(0L)
  save_conditional_rr_outputs(
    conditional_rr_df,
    output_dir = output_dir,
    title_prefix = basename(dirname(output_dir)),
    write_data = FALSE,
    show_ci = FALSE
  )
}

rerender_saved_meta_predictor_forests <- function(output_dir, title_prefix = NULL) {
  predictor_dirs <- list.dirs(output_dir, full.names = TRUE, recursive = FALSE)
  predictor_dirs <- predictor_dirs[grepl("^META_PREDICTORS_(MEAN|GINI)(_NO_CRIME)?$", basename(predictor_dirs))]
  if (length(predictor_dirs) > 0) {
    rendered <- vapply(predictor_dirs, function(dir_i) {
      rerender_saved_meta_predictor_forests(
        dir_i,
        title_prefix = paste0(ifelse(is.null(title_prefix), basename(output_dir), title_prefix),
                              " - ", basename(dir_i))
      )
    }, logical(1))
    return(any(rendered))
  }
  
  meta_coef_df <- safe_read_csv_file(file.path(output_dir, "city_level_covariates.csv"))
  if (is.null(meta_coef_df) || nrow(meta_coef_df) == 0) return(FALSE)
  
  title_prefix <- ifelse(is.null(title_prefix), basename(output_dir), title_prefix)
  save_meta_predictor_outputs(
    meta_coef_df = meta_coef_df,
    output_dir = output_dir,
    full_title = paste0(title_prefix, " - City-level Covariate Coefficients"),
    full_subtitle = "Re-rendered from saved model coefficients",
    summary_title = paste0(title_prefix, " - City-level Covariate Summary"),
    summary_subtitle = "Re-rendered from saved model coefficients"
  )
}

rerender_saved_partition_af <- function(output_dir, title_prefix = NULL) {
  af_df <- safe_read_csv_file(file.path(output_dir, "AF_summary.csv"))
  if (is.null(af_df) || nrow(af_df) == 0) return(FALSE)
  af_meta_summary <- safe_read_csv_file(file.path(output_dir, "AF_meta_summary.csv"))
  
  af_df <- af_df %>%
    mutate(
      p_value = if ("p_value" %in% names(af_df)) p_value else 2 * pnorm(-abs(af_pct / pmax(af_se, 1e-10))),
      direction = forest_effect_direction(af_pct, neutral_zero = TRUE),
      star_label = forest_significance_stars(p_value)
    ) %>%
    arrange(desc(af_pct)) %>%
    mutate(city = factor(city, levels = rev(unique(city))))
  
  af_subtitle <- paste0(ifelse(is.null(title_prefix), basename(output_dir), title_prefix), " | Re-rendered from saved AF summary")
  if (!is.null(af_meta_summary) && nrow(af_meta_summary) > 0) {
    af_subtitle <- paste0(
      af_subtitle,
      " | RE pooled AF = ",
      round(af_meta_summary$pooled_af[1], 2), "% (95% CI: ",
      round(af_meta_summary$pooled_low[1], 2), "% - ",
      round(af_meta_summary$pooled_high[1], 2), "%)"
    )
  }
  af_subtitle <- paste0(af_subtitle, "\nBlue=Positive, Red=Negative | * p<0.05, ** p<0.01, *** p<0.001")
  
  af_star_nudge <- forest_star_offset(af_df$af_low, af_df$af_high, fallback = 0.1)
  p_af <- ggplot(af_df, aes(x = af_pct, y = city)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_errorbarh(aes(xmin = af_low, xmax = af_high, color = direction), height = 0.24, linewidth = 1) +
    geom_point(aes(color = direction), size = 3.5, shape = 19) +
    geom_text(
      data = af_df %>% filter(nzchar(star_label)),
      aes(label = star_label),
      nudge_x = af_star_nudge,
      nudge_y = 0.22,
      size = 3.2,
      color = "black",
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(values = forest_direction_colors, guide = "none") +
    labs(
      title = paste0(ifelse(is.null(title_prefix), basename(output_dir), title_prefix), " - AF Forest"),
      subtitle = af_subtitle,
      x = "Attributable Fraction (%)",
      y = ""
    ) +
    coord_cartesian(clip = "off") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 11, color = "gray30", lineheight = 1.2),
      axis.text.y = element_text(size = 10),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  ggsave(file.path(output_dir, "AF_forest.png"), p_af, width = 10, height = max(6, nrow(af_df) * 0.3), dpi = 300)
  TRUE
}

rerender_saved_weekend_outputs <- function(output_dir) {
  weekend_output_dir <- file.path(output_dir, "WEEKEND_EFFECT_SUMMARY")
  weekend_df <- safe_read_csv_file(file.path(weekend_output_dir, "all_cities_weekend_grid_effects.csv"))
  weekend_summary <- safe_read_csv_file(file.path(weekend_output_dir, "weekend_effect_summary.csv"))
  if (is.null(weekend_df) || is.null(weekend_summary)) return(FALSE)
  
  weekend_only <- weekend_df %>% filter(variable == "Weekend_Effect")
  if (nrow(weekend_only) > 0) {
    for (indic in unique(weekend_only$indicator)) {
      for (mtype in unique(weekend_only$model_type)) {
        subset_data <- weekend_only %>% filter(indicator == indic, model_type == mtype) %>% arrange(desc(rr))
        if (nrow(subset_data) == 0) next
        plot_height <- max(8, 4 + nrow(subset_data) * 0.3)
        p_weekend <- ggplot(subset_data, aes(x = rr, y = reorder(city, rr))) +
          geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 1.2) +
          geom_errorbarh(aes(xmin = rr_low, xmax = rr_high, color = coefficient >= 0), height = 0.3, linewidth = 1) +
          geom_point(aes(color = coefficient >= 0), size = 3.5, shape = 19) +
          scale_color_manual(values = c("TRUE" = "#2166AC", "FALSE" = "#B2182B"), guide = "none") +
          labs(title = "Weekend Effect on Physical Activity", subtitle = paste0("Indicator: ", toupper(indic), " | Model: ", toupper(mtype)), x = "Relative Risk (RR)", y = "City") +
          theme_minimal(base_size = 14) +
          theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank())
        ggsave(file.path(weekend_output_dir, paste0("weekend_effect_forest_", indic, "_", mtype, ".png")), p_weekend, width = 12, height = plot_height, dpi = 300)
      }
    }
  }
  
  weekend_summary_plot <- weekend_summary %>%
    filter(variable == "Weekend_Effect") %>%
    mutate(
      rr_low = exp(mean_coef - 1.96 * se_pooled),
      rr_high = exp(mean_coef + 1.96 * se_pooled),
      model_label = paste0(toupper(indicator), " - ", toupper(model_type)),
      direction = ifelse(mean_rr >= 1, "Positive", "Negative")
    )
  
  if (nrow(weekend_summary_plot) > 0) {
    p_weekend_summary <- ggplot(weekend_summary_plot, aes(x = mean_rr, y = reorder(model_label, mean_rr))) +
      geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 1.2) +
      geom_errorbarh(aes(xmin = rr_low, xmax = rr_high, color = direction), height = 0.3, linewidth = 1.1) +
      geom_point(aes(color = direction), size = 4, shape = 19) +
      scale_color_manual(values = c("Positive" = "#2166AC", "Negative" = "#B2182B"), guide = "none") +
      labs(title = "Weekend Effect on Physical Activity - Overall Summary", subtitle = "Re-rendered from saved weekend-effect summaries", x = "Relative Risk (RR)", y = "Model") +
      theme_minimal(base_size = 16) +
      theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank())
    ggsave(file.path(weekend_output_dir, "weekend_effect_forest_SUMMARY.png"), p_weekend_summary, width = 14, height = 8, dpi = 300)
  }
  
  TRUE
}

add_pooled_af_significance <- function(pooled_af_summary, mode = NULL) {
  if (is.null(pooled_af_summary) || nrow(pooled_af_summary) == 0) return(pooled_af_summary)
  mode_label <- meta_predictor_mode_label(mode)
  pooled_af_summary <- pooled_af_summary %>%
    mutate(
      p_value = ifelse(
        is.finite(pooled_af) & is.finite(pooled_se) & pooled_se > 0,
        2 * pnorm(-abs(pooled_af / pooled_se)),
        NA_real_
      ),
      star_label = forest_significance_stars(p_value),
      effect_direction = forest_effect_direction(pooled_af, neutral_zero = TRUE),
      meta_predictor_mode = mode_label
    )
  pooled_af_summary
}

save_pooled_af_comparison_plot <- function(pooled_af_summary, output_dir, mode = NULL) {
  if (is.null(pooled_af_summary) || nrow(pooled_af_summary) == 0) return(FALSE)
  
  mode_label <- meta_predictor_mode_label(mode)
  mode_suffix <- analysis_output_suffix(mode)
  percentile_levels <- c("p25", "p50", "p75", "p90", "p95")
  percentile_labels <- c("25th", "50th", "75th", "90th", "95th")
  
  af_comparison <- add_pooled_af_significance(pooled_af_summary, mode = mode_label) %>%
    filter(percentile %in% percentile_levels) %>%
    mutate(
      percentile_label = factor(percentile, levels = percentile_levels, labels = percentile_labels),
      indicator_label = case_when(
        indicator == "cehwi" ~ "CEHWI",
        indicator == "exceeded_quantity" ~ "Exceeded quantity",
        TRUE ~ indicator
      ),
      model_label = factor(str_to_title(model_base_type(model_type)), levels = c("Composite", "Day", "Night")),
      activity_type = model_activity_type(model_type),
      activity_label = unname(activity_modality_labels()[activity_type])
    )
  if (nrow(af_comparison) == 0) return(FALSE)
  
  af_palette <- af_model_type_colors()
  activity_labels <- activity_modality_labels()
  activity_palette <- setNames(
    unname(activity_modality_colors()[names(activity_labels)]),
    unname(activity_labels)
  )
  model_offsets <- c("Composite" = 0.24, "Day" = 0, "Night" = -0.24)
  activity_offsets <- c("All activity" = 0.30, "Ride" = 0.10, "Run" = -0.10, "Walk" = -0.30)
  has_activity_split <- any(af_comparison$activity_type != "all", na.rm = TRUE)
  
  add_af_percentile_plot_coords <- function(plot_df) {
    x_range <- range(c(plot_df$pooled_low, plot_df$pooled_high, plot_df$pooled_af), na.rm = TRUE)
    x_span <- if (all(is.finite(x_range)) && diff(x_range) > 0) diff(x_range) else 10
    x_offset <- max(0.25, x_span * 0.018)
    out <- plot_df %>%
      mutate(
        percentile_index = match(percentile, percentile_levels),
        model_offset = unname(model_offsets[as.character(model_label)]),
        model_offset = ifelse(is.na(model_offset), 0, model_offset),
        activity_offset = unname(activity_offsets[as.character(activity_label)]),
        activity_offset = ifelse(is.na(activity_offset), 0, activity_offset),
        point_y = percentile_index + if (has_activity_split) activity_offset else model_offset,
        star_x = pooled_af + x_offset,
        star_y = point_y + 0.10
      )
    out
  }
  
  make_af_percentile_plot <- function(plot_df, title_suffix = NULL, facet_indicator = FALSE) {
    plot_df <- add_af_percentile_plot_coords(plot_df)
    
    color_var <- if (has_activity_split) "activity_label" else "model_label"
    legend_title <- if (has_activity_split) "PA modality" else NULL
    color_values <- if (has_activity_split) activity_palette else af_palette
    
    p <- ggplot(plot_df, aes(x = pooled_af, y = point_y, color = .data[[color_var]])) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray55", linewidth = 0.55) +
      geom_errorbarh(
        aes(xmin = pooled_low, xmax = pooled_high),
        height = 0.10,
        linewidth = 0.9
      ) +
      geom_point(size = 3.7, shape = 19) +
      geom_text(
        data = plot_df %>% filter(nzchar(star_label)),
        aes(x = star_x, y = star_y, label = star_label),
        size = 4.0,
        color = "black",
        fontface = "bold",
        show.legend = FALSE
      ) +
      scale_color_manual(values = color_values, name = legend_title) +
      scale_y_continuous(
        breaks = seq_along(percentile_labels),
        labels = percentile_labels,
        limits = c(0.55, length(percentile_labels) + 0.55),
        expand = expansion(mult = c(0.02, 0.02))
      ) +
      labs(
        title = paste0("Pooled Attributable Fraction by Exposure Intensity", title_suffix),
        subtitle = paste0(
          "Output set: ", mode_label,
          activity_analysis_file_suffix(),
          " | circles are pooled AF; horizontal bars are 95% CI; * p<0.05, ** p<0.01, *** p<0.001."
        ),
        x = "Pooled AF (%)",
        y = "Exposure percentile used for AF"
      ) +
      theme_minimal(base_size = 15) +
      theme(
        plot.title = element_text(face = "bold", size = 18, hjust = 0),
        plot.subtitle = element_text(size = 11, color = "gray35", hjust = 0),
        strip.text = element_text(face = "bold", size = 13),
        axis.title = element_text(face = "bold"),
        axis.text = element_text(color = "gray20"),
        legend.position = "top",
        panel.grid = element_blank(),
        axis.line = element_line(color = "gray35", linewidth = 0.35),
        plot.margin = margin(18, 22, 18, 18)
      )
    
    if (isTRUE(facet_indicator)) {
      if (has_activity_split) {
        p <- p + facet_grid(indicator_label ~ model_label, scales = "free_x")
      } else {
        p <- p + facet_wrap(~ indicator_label, ncol = 1, scales = "free_x")
      }
    } else if (has_activity_split) {
      p <- p + facet_wrap(~ model_label, nrow = 1, scales = "free_x")
    }
    p
  }
  
  p_af_comparison <- make_af_percentile_plot(af_comparison, title_suffix = "", facet_indicator = TRUE)
  af_comparison_csv <- add_af_percentile_plot_coords(af_comparison)
  
  safe_write_csv(
    af_comparison_csv,
    file.path(output_dir, paste0("AF_comparison_by_percentile_plot_data", mode_suffix, ".csv")),
    label = "AF comparison percentile plot data"
  )
  ggsave(
    file.path(output_dir, paste0("AF_comparison_by_percentile", mode_suffix, ".png")),
    p_af_comparison,
    width = if (has_activity_split) 16 else 13,
    height = if (has_activity_split) 8.5 else 9,
    dpi = 300,
    bg = "white"
  )
  
  for (ind_label in unique(af_comparison$indicator_label)) {
    indicator_df <- af_comparison %>% filter(indicator_label == ind_label)
    if (nrow(indicator_df) == 0) next
    indicator_file <- toupper(gsub("[^A-Za-z0-9]+", "_", ind_label))
    indicator_df_csv <- add_af_percentile_plot_coords(indicator_df)
    safe_write_csv(
      indicator_df_csv,
      file.path(output_dir, paste0("AF_comparison_by_percentile_", indicator_file, "_plot_data", mode_suffix, ".csv")),
      label = paste0("AF comparison percentile plot data - ", indicator_file)
    )
    p_indicator <- make_af_percentile_plot(indicator_df, title_suffix = paste0(" - ", ind_label), facet_indicator = FALSE)
    ggsave(
      file.path(output_dir, paste0("AF_comparison_by_percentile_", indicator_file, mode_suffix, ".png")),
      p_indicator,
      width = if (has_activity_split) 14 else 10.5,
      height = if (has_activity_split) 6.8 else 6.6,
      dpi = 300,
      bg = "white"
    )
  }
  
  TRUE
}

save_af_vs_rr_scatter_plot <- function(all_city_af, output_dir, mode = NULL) {
  if (is.null(all_city_af) || nrow(all_city_af) == 0) return(FALSE)
  
  mode_label <- meta_predictor_mode_label(mode)
  mode_suffix <- analysis_output_suffix(mode)
  has_activity_split_global <- is_activity_split_mode()
  
  af_rr_data <- all_city_af %>%
    filter(percentile == "p90", is.finite(rr), is.finite(af_pct), rr > 0) %>%
    mutate(
      indicator_label = case_when(
        indicator == "cehwi" ~ "CEHWI",
        indicator == "exceeded_quantity" ~ "Exceeded quantity",
        TRUE ~ indicator
      ),
      model_label = factor(str_to_title(model_base_type(model_type)), levels = c("Composite", "Day", "Night")),
      activity_type = model_activity_type(model_type),
      activity_label = unname(activity_modality_labels()[activity_type])
    )
  if (nrow(af_rr_data) == 0) return(FALSE)
  has_activity_split <- has_activity_split_global || any(af_rr_data$activity_type != "all", na.rm = TRUE)
  
  rr_cap <- suppressWarnings(as.numeric(quantile(af_rr_data$rr, 0.95, na.rm = TRUE)))
  af_caps <- suppressWarnings(as.numeric(quantile(af_rr_data$af_pct, c(0.025, 0.975), na.rm = TRUE)))
  if (!is.finite(rr_cap) || rr_cap <= 0) rr_cap <- max(af_rr_data$rr, na.rm = TRUE)
  if (any(!is.finite(af_caps)) || diff(af_caps) <= 0) af_caps <- range(af_rr_data$af_pct, na.rm = TRUE)
  
  af_rr_outliers <- af_rr_data %>%
    filter(rr > rr_cap | af_pct < af_caps[1] | af_pct > af_caps[2])
  af_rr_plot <- af_rr_data %>%
    filter(rr <= rr_cap, af_pct >= af_caps[1], af_pct <= af_caps[2])
  if (nrow(af_rr_plot) < 3) af_rr_plot <- af_rr_data
  
  label_df <- af_rr_plot %>%
    group_by(model_label, indicator_label, activity_label) %>%
    slice_max(order_by = abs(af_pct), n = if (has_activity_split) 1 else 2, with_ties = FALSE) %>%
    ungroup()
  
  rr_curve <- data.frame(
    rr = seq(max(0.001, min(af_rr_plot$rr, na.rm = TRUE)), rr_cap, length.out = 300)
  ) %>%
    mutate(af_pct = (rr - 1) / rr * 100) %>%
    filter(is.finite(af_pct), af_pct >= af_caps[1], af_pct <= af_caps[2])

  crop_note <- paste0(
    "Output set: ", mode_label,
    activity_analysis_file_suffix(),
    " | AF is mathematically derived from RR; ",
    nrow(af_rr_outliers), " outside-range point(s) saved to CSV."
  )
  af_rr_plot <- af_rr_plot %>%
    mutate(direction = forest_effect_direction(af_pct, neutral_zero = TRUE))
  
  activity_labels <- activity_modality_labels()
  activity_palette <- setNames(
    unname(activity_modality_colors()[names(activity_labels)]),
    unname(activity_labels)
  )
  
  p_af_vs_rr <- ggplot(af_rr_plot, aes(x = rr, y = af_pct)) +
    geom_line(data = rr_curve, aes(x = rr, y = af_pct), inherit.aes = FALSE,
              color = "gray70", linewidth = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray55", linewidth = 0.55) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray55", linewidth = 0.55)
  
  if (has_activity_split) {
    p_af_vs_rr <- p_af_vs_rr +
      geom_point(aes(color = activity_label), size = 2.7, alpha = 1, shape = 19) +
      facet_grid(model_label ~ indicator_label) +
      scale_color_manual(values = activity_palette, name = "PA modality")
  } else {
    p_af_vs_rr <- p_af_vs_rr +
      geom_point(aes(color = direction, shape = indicator_label), size = 2.8, alpha = 1) +
      facet_wrap(~ model_label, nrow = 1) +
      scale_color_manual(values = forest_direction_colors, guide = "none") +
      scale_shape_manual(values = c("CEHWI" = 16, "Exceeded quantity" = 17), name = NULL)
  }
  
  p_af_vs_rr <- p_af_vs_rr +
    coord_cartesian(xlim = c(0, rr_cap), ylim = af_caps, clip = "on") +
    labs(
      title = "AF vs RR Diagnostic at the 90th Exposure Percentile",
      subtitle = paste0(crop_note, " Panels separate heatwave indicator and PA model type; gray curve is the AF=(RR-1)/RR transform."),
      x = "Relative Risk (RR), display-cropped",
      y = "Attributable Fraction (%)"
    ) +
    theme_minimal(base_size = 15) +
    theme(
      plot.title = element_text(face = "bold", size = 18),
      plot.subtitle = element_text(size = 10.5, color = "gray35"),
      strip.text = element_text(face = "bold", size = 11),
      axis.title = element_text(face = "bold"),
      legend.position = "top",
      panel.spacing.x = grid::unit(if (has_activity_split) 24 else 18, "pt"),
      panel.spacing.y = grid::unit(if (has_activity_split) 18 else 8, "pt"),
      panel.grid = element_blank(),
      axis.line = element_line(color = "gray35", linewidth = 0.35),
      plot.margin = margin(18, 28, 18, 18)
    )
  
  if (requireNamespace("ggrepel", quietly = TRUE) && nrow(label_df) > 0) {
    p_af_vs_rr <- p_af_vs_rr +
      ggrepel::geom_text_repel(
        data = label_df,
        aes(label = city),
        size = 2.8,
        color = "black",
        min.segment.length = 0,
        segment.color = "gray45",
        segment.linewidth = 0.35,
        box.padding = 0.35,
        point.padding = 0.25,
        max.overlaps = Inf,
        show.legend = FALSE
      )
  } else if (nrow(label_df) > 0) {
    p_af_vs_rr <- p_af_vs_rr +
      geom_text(
        data = label_df,
        aes(label = city),
        size = 2.7,
        nudge_y = 0.5,
        check_overlap = TRUE,
        show.legend = FALSE
      )
  }
  
  safe_write_csv(af_rr_data, file.path(output_dir, paste0("AF_vs_RR_scatter_plot_data_all", mode_suffix, ".csv")), label = "AF vs RR scatter all data")
  safe_write_csv(af_rr_outliers, file.path(output_dir, paste0("AF_vs_RR_scatter_display_outliers", mode_suffix, ".csv")), label = "AF vs RR scatter display outliers")
  ggsave(
    file.path(output_dir, paste0("AF_vs_RR_scatter", mode_suffix, ".png")),
    p_af_vs_rr,
    width = if (has_activity_split) 15.5 else 15.5,
    height = if (has_activity_split) 9.2 else 5.8,
    dpi = 300,
    bg = "white"
  )
  TRUE
}

rerender_saved_national_af_outputs <- function(output_dir, mode = NULL) {
  mode_label <- meta_predictor_mode_label(mode)
  mode_suffix <- analysis_output_suffix(mode_label)
  national_af_dir <- national_af_output_dir(output_dir, mode = mode_label)
  
  read_national_af_csv <- function(stem) {
    candidates <- c(
      file.path(national_af_dir, paste0(stem, mode_suffix, ".csv")),
      file.path(national_af_dir, paste0(stem, ".csv")),
      file.path(output_dir, paste0(stem, mode_suffix, ".csv")),
      file.path(output_dir, paste0(stem, ".csv"))
    )
    for (candidate in candidates) {
      obj <- safe_read_csv_file(candidate)
      if (!is.null(obj) && nrow(obj) > 0) return(obj)
    }
    NULL
  }
  
  all_city_af <- read_national_af_csv("ALL_CITIES_AF")
  pooled_af_summary <- read_national_af_csv("POOLED_AF_SUMMARY")
  af_by_pred_summary <- read_national_af_csv("AF_by_predictor_summary")
  
  if (!is.null(af_by_pred_summary) && nrow(af_by_pred_summary) > 0) {
    af_by_pred_summary <- af_by_pred_summary %>%
      mutate(direction = forest_effect_direction(af_mean, neutral_zero = TRUE), star_label = forest_significance_stars(p_value))
    af_pred_star_nudge <- forest_star_offset(af_by_pred_summary$af_low_pooled, af_by_pred_summary$af_high_pooled, fallback = 0.1)
    p_af_forest <- ggplot(af_by_pred_summary, aes(x = af_mean, y = reorder(predictor_label, af_mean))) +
      geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1.2) +
      geom_errorbarh(aes(xmin = af_low_pooled, xmax = af_high_pooled, color = direction), height = 0.4, linewidth = 2) +
      geom_point(aes(color = direction), size = 8, shape = 19) +
      geom_text(
        data = af_by_pred_summary %>% filter(nzchar(star_label)),
        aes(label = star_label),
        nudge_x = af_pred_star_nudge,
        nudge_y = 0.24,
        size = 5,
        color = "black",
        fontface = "bold",
        show.legend = FALSE
      ) +
      scale_color_manual(values = forest_direction_colors, guide = "none") +
      labs(title = "Attributable Fraction by Meta-Predictor", subtitle = "Re-rendered from saved AF summaries\nBlue=Positive, Red=Negative | * p<0.05, ** p<0.01, *** p<0.001", x = "Attributable Fraction (%)", y = "") +
      coord_cartesian(clip = "off") +
      theme_minimal(base_size = 18) +
      theme(plot.title = element_text(face = "bold", size = 22, hjust = 0), plot.subtitle = element_text(size = 15, color = "gray30"), axis.text.y = element_text(size = 16, face = "bold"), axis.text.x = element_text(size = 15), axis.title.x = element_text(size = 18, face = "bold"), panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5))
    plot_height <- max(10, 5 + nrow(af_by_pred_summary) * 1.2)
    ggsave(file.path(national_af_dir, paste0("AF_forest_plot_by_predictor", mode_suffix, ".png")), p_af_forest, width = 16, height = plot_height, dpi = 300)
  }
  
  if (!is.null(all_city_af) && nrow(all_city_af) > 0 && !is.null(pooled_af_summary) && nrow(pooled_af_summary) > 0) {
    for (ind in c("cehwi", "exceeded_quantity")) {
      for (mtype in STAGE1_MODEL_TYPES) {
        af_subset <- all_city_af %>% filter(indicator == ind, model_type == mtype, percentile == "overall")
        af_meta_row <- pooled_af_summary %>% filter(indicator == ind, model_type == mtype, percentile == "overall")
        if (nrow(af_subset) < 3 || nrow(af_meta_row) == 0) next
        af_subset <- af_subset %>%
          arrange(desc(af_pct)) %>%
          mutate(city = factor(city, levels = rev(unique(city))), p_value = 2 * pnorm(-abs(af_pct / pmax(af_se, 1e-10))), direction = forest_effect_direction(af_pct, neutral_zero = TRUE), star_label = forest_significance_stars(p_value))
        af_city_star_nudge <- forest_star_offset(af_subset$af_low, af_subset$af_high, fallback = 0.1)
        p_af_city <- ggplot(af_subset, aes(x = af_pct, y = city)) +
          geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
          geom_errorbarh(aes(xmin = af_low, xmax = af_high, color = direction), height = 0.22, linewidth = 1) +
          geom_point(aes(color = direction), size = 3.5, shape = 19) +
          geom_text(
            data = af_subset %>% filter(nzchar(star_label)),
            aes(label = star_label),
            nudge_x = af_city_star_nudge,
            nudge_y = 0.22,
            size = 3.2,
            color = "black",
            fontface = "bold",
            show.legend = FALSE
          ) +
          scale_color_manual(values = forest_direction_colors, guide = "none") +
          labs(title = paste0("Attributable Fraction (AF) - ", toupper(ind), " - ", toupper(mtype)), subtitle = paste0(nrow(af_subset), " cities | Pooled AF = ", round(af_meta_row$pooled_af[1], 2), "% (95% CI ", round(af_meta_row$pooled_low[1], 2), ", ", round(af_meta_row$pooled_high[1], 2), ")\nBlue=Positive, Red=Negative | * p<0.05, ** p<0.01, *** p<0.001"), x = "Attributable Fraction (%)", y = "") +
          coord_cartesian(clip = "off") +
          theme_minimal(base_size = 14) +
          theme(plot.title = element_text(face = "bold", size = 16), axis.text.y = element_text(size = 10), panel.grid.major.y = element_blank(), panel.grid.minor = element_blank())
        ggsave(file.path(national_af_dir, paste0(toupper(ind), "_", toupper(mtype), "_AF_forest", mode_suffix, ".png")), p_af_city, width = 10, height = max(6, nrow(af_subset) * 0.3), dpi = 300)
      }
    }
    
    save_pooled_af_comparison_plot(pooled_af_summary, national_af_dir, mode = mode_label)
    save_af_vs_rr_scatter_plot(all_city_af, national_af_dir, mode = mode_label)
  }
  
  TRUE
}

rerender_saved_visualizations <- function(output_dir) {
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("仅重绘可视化：从已保存结果文件重建图表（不重新拟合模型）\n")
  cat(rep("=", 100), "\n\n", sep = "")
  
  stage1_results <- collect_saved_stage1_results(output_dir)
  top_dirs <- list.dirs(output_dir, recursive = FALSE, full.names = TRUE)
  stage1_city_dirs <- top_dirs[grepl("_(cehwi|exceeded_quantity)$", basename(top_dirs), ignore.case = TRUE)]
  stage1_partition_dirs <- top_dirs[grepl("_DLNM_STAGE1$", basename(top_dirs), ignore.case = TRUE)]
  
  city_plot_n <- sum(vapply(stage1_city_dirs, rerender_stage1_result_dir, integer(1)))
  partition_plot_n <- sum(vapply(stage1_partition_dirs, rerender_stage1_result_dir, integer(1)))
  cat("  ✓ 第一阶段单城市结果已重绘:", city_plot_n, "个模型结果\n")
  cat("  ✓ 第一阶段分区结果已重绘:", partition_plot_n, "个模型结果\n")
  
  pooled_dirs <- top_dirs[grepl("^POOLED_META_(cehwi|exceeded_quantity)_(composite|day|night)(_(all|ride|run|walk))?$", basename(top_dirs), ignore.case = TRUE)]
  for (pooled_dir in pooled_dirs) {
    dir_match <- str_match(basename(pooled_dir), "^POOLED_META_(cehwi|exceeded_quantity)_((composite|day|night)(_(all|ride|run|walk))?)$")
    if (any(is.na(dir_match))) next
    indicator <- dir_match[2]
    model_type <- dir_match[3]
    pooled_df <- safe_read_csv_file(file.path(pooled_dir, "pooled_RR_data.csv"))
    if (is.null(pooled_df)) {
      meta_obj <- tryCatch(read_rds(file.path(pooled_dir, "meta_model.rds")), error = function(e) NULL)
      pooled_df <- if (!is.null(meta_obj) && !is.null(meta_obj$pred_df)) meta_obj$pred_df else NULL
    }
    exposure_values <- collect_exposure_values_from_stage1(stage1_results, indicator, model_type)
    subtitle_text <- build_saved_pooled_subtitle(
      output_dir = pooled_dir,
      indicator = indicator,
      model_type = model_type,
      n_cities = length(stage1_results[[indicator]]),
      reliability = if (!is.null(pooled_df) && "reliability" %in% names(pooled_df)) pooled_df$reliability[1] else "STABLE",
      prefix_text = paste0("Meta-regression: ", toupper(indicator), " vs PA | Model: ", toupper(model_type), " | Reference: ", toupper(indicator), " = 0")
    )
    if (!is.null(pooled_df)) {
      rerender_saved_rr_bundle(
        pooled_df = pooled_df,
        output_dir = pooled_dir,
        title_text = paste0("Pooled RR Curve - ", toupper(indicator), " - ", toupper(model_type)),
        subtitle_text = subtitle_text,
        indicator = indicator,
        model_type = model_type,
        exposure_values = exposure_values,
        reliability = if ("reliability" %in% names(pooled_df)) pooled_df$reliability[1] else "STABLE"
      )
    }
    rerender_saved_conditional_rr(pooled_dir)
    rerender_saved_meta_predictor_forests(pooled_dir, title_prefix = paste0("Pooled Meta-regression - ", toupper(indicator), " - ", toupper(model_type)))
  }
  cat("  ✓ 第二阶段全国 pooled 图件已重绘\n")
  
  partition_summary_path <- file.path(output_dir, "PARTITION_META_SUMMARY", "partition_meta_run_summary.csv")
  partition_summary_df <- safe_read_csv_file(partition_summary_path)
  if (!is.null(partition_summary_df) && nrow(partition_summary_df) > 0) {
    partition_summary_unique <- partition_summary_df %>%
      filter(!is.na(output_dir), nzchar(output_dir), model_status == "SUCCESS") %>%
      group_by(partition_family, partition_name, indicator, model_type, output_dir) %>%
      summarise(n_cities = max(n_cities, na.rm = TRUE), stability_flag = dplyr::first(na.omit(stability_flag)), .groups = "drop")
    
    for (i in seq_len(nrow(partition_summary_unique))) {
      row_i <- partition_summary_unique[i, ]
      partition_dir <- row_i$output_dir[[1]]
      pooled_df <- safe_read_csv_file(file.path(partition_dir, "pooled_RR_data.csv"))
      city_subset <- get_partition_city_list(row_i$partition_family[[1]], row_i$partition_name[[1]])
      exposure_values <- collect_exposure_values_from_stage1(stage1_results, row_i$indicator[[1]], row_i$model_type[[1]], city_subset = city_subset)
      title_text <- paste0(row_i$partition_family[[1]], ": ", row_i$partition_name[[1]], " - ", toupper(row_i$indicator[[1]]), " - ", toupper(row_i$model_type[[1]]))
      subtitle_text <- build_partition_rr_subtitle(
        partition_family = row_i$partition_family[[1]],
        partition_name = row_i$partition_name[[1]],
        indicator = row_i$indicator[[1]],
        model_type = row_i$model_type[[1]],
        n_cities = row_i$n_cities[[1]],
        pooled_df = pooled_df,
        meta_model = NULL,
        reliability = ifelse(is.na(row_i$stability_flag[[1]]) || !nzchar(row_i$stability_flag[[1]]), "STABLE", row_i$stability_flag[[1]])
      )
      if (!is.null(pooled_df) && nrow(pooled_df) > 0) {
        rerender_saved_rr_bundle(
          pooled_df = pooled_df,
          output_dir = partition_dir,
          title_text = title_text,
          subtitle_text = subtitle_text,
          indicator = row_i$indicator[[1]],
          model_type = row_i$model_type[[1]],
          exposure_values = exposure_values,
          reliability = ifelse(is.na(row_i$stability_flag[[1]]) || !nzchar(row_i$stability_flag[[1]]), "STABLE", row_i$stability_flag[[1]])
        )
      }
      rerender_saved_conditional_rr(partition_dir)
      rerender_saved_meta_predictor_forests(partition_dir, title_prefix = title_text)
      rerender_saved_partition_af(partition_dir, title_prefix = title_text)
    }
    cat("  ✓ 第二阶段分区图件已按保存结果重绘\n")
  }
  
  rerender_saved_weekend_outputs(output_dir)
  national_af_dirs <- list.dirs(output_dir, recursive = FALSE, full.names = TRUE)
  national_af_dirs <- national_af_dirs[grepl("^NATIONAL_AF_(MEAN|GINI)(_NO_CRIME)?$", basename(national_af_dirs))]
  if (length(national_af_dirs) > 0) {
    invisible(vapply(
      basename(national_af_dirs),
      function(dir_name) rerender_saved_national_af_outputs(output_dir, mode = sub("^NATIONAL_AF_", "", dir_name)),
      logical(1)
    ))
  } else {
    rerender_saved_national_af_outputs(output_dir)
  }
  
  cat("\n✅ 仅重绘可视化完成！\n")
  cat("输出目录:", output_dir, "\n")
  cat("说明: 本次只读取已保存结果文件重建图件，未重新拟合任何模型。\n\n")
  
  invisible(TRUE)
}

# ========== 核心函数: 第一阶段 - 城市级DLNM ==========

fit_dlnm_stage1 <- function(data, city_name, indicator = "cehwi", min_obs = 100) {
  # 第一阶段：拟合城市级GAM + DLNM
  # 
  # 参数:
  #   data: 城市时间序列数据
  #   city_name: 城市名称
  #   indicator: 热浪指标 ("cehwi" 或 "exceeded_quantity")
  #   min_obs: 最小观测数
  
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("第一阶段: 城市级DLNM -", city_name, "\n")
  cat(rep("=", 80), "\n\n", sep = "")
  
  # 检查使用哪个暴露列（如果有cehwi_exposure就用它，否则用原始列）
  if ("cehwi_exposure" %in% names(data)) {
    cehwi_col <- "cehwi_exposure"
    cat("  使用合并后的暴露变量: cehwi_exposure\n")
  } else {
    if (indicator == "cehwi") {
      cehwi_col <- "composite_cehwi"
    } else {
      cehwi_col <- "composite_exceeded_quantity"
    }
  }
  
  # 【V4】不再在第一阶段合并社会经济数据（简化模型）
  # 社会经济变量（包括失业人口）将在第二阶段用城市平均值作为meta-predictors
  
  # 准备数据
  df_model <- data %>%
    filter(!is.na(trip_count), !is.na(.data[[cehwi_col]])) %>%
    mutate(
      cehwi = .data[[cehwi_col]],
      trip_count_scaled = trip_count / SCALE_FACTOR,  # 【优化1】缩放trip_count，避免数值过大
      fish_id_fac = as.factor(fish_id),
      lag_group = if ("city" %in% names(.)) {
        interaction(city, fish_id, drop = TRUE)
      } else {
        as.factor(fish_id)
      },
      date_num = as.numeric(date),
      year = year(date),
      year_fac = as.factor(year),
      doy = yday(date),  # day of year / day of season (1-366)
      dow = wday(date, week_start = 1),  # day of week (1=周一, 7=周日)
      dow_fac = factor(dow, levels = 1:7, labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
    ) %>%
    arrange(lag_group, date)

  observed_years <- dplyr::n_distinct(df_model$year[is.finite(df_model$year)])
  calendar_time_df <- as.integer(TIME_DF_PER_YEAR * observed_years)
  if (TIME_CONTROL_SPEC == "continuous_time_ns" && calendar_time_df < 3L) {
    stop("Continuous-time spline requires at least 3 total degrees of freedom")
  }
  
  # 【V5.1修复】检测是否为分区分析（多城市数据）
  has_city_effect <- "city" %in% names(df_model)
  if (has_city_effect) {
    # 【关键优化】不创建city_fac（避免与fish_id共线+内存爆炸）
    # 分区分析将使用 fish_id 随机效应代替固定效应
    cat("  【分区分析】检测到多城市数据（将使用内存优化方法）\n")
  }
  
  cat("  Trip count统计:\n")
  cat("    - 原始范围:", round(range(df_model$trip_count, na.rm = TRUE), 0), "\n")
  cat("    - 缩放后范围:", round(range(df_model$trip_count_scaled, na.rm = TRUE), 2), "\n")
  
  if (nrow(df_model) < min_obs) {
    cat("  ⚠ 样本量不足 (N =", nrow(df_model), ")\n")
    return(NULL)
  }
  
  cat("  样本量: N =", nrow(df_model), "\n")
  cat("  格子数:", n_distinct(df_model$fish_id), "\n")
  cat("  日期范围:", as.character(min(df_model$date)), "至", as.character(max(df_model$date)), "\n")
  cehwi_range <- range(df_model$cehwi, na.rm = TRUE)
  cat("  CEHWI范围:", round(cehwi_range, 2), "\n")
  
  # ========== 检查数据可用性 ==========
  
  # 检查CEHWI是否有变化（如果全是0，说明没有热浪）
  cehwi_sd <- sd(df_model$cehwi, na.rm = TRUE)
  cehwi_max <- max(df_model$cehwi, na.rm = TRUE)
  
  if (cehwi_max == 0 || cehwi_sd < 0.01) {
    cat("  ⚠ CEHWI无变化（全为0或接近0），该时段无热浪事件\n")
    cat("     建议选择其他时间段（如2015-2020）\n")
    return(NULL)
  }
  
  # 【V4修改】检查正值数量（筛选标准: ≥30天）
  n_positive_cehwi <- sum(df_model$cehwi > 0, na.rm = TRUE)
  pct_positive <- n_positive_cehwi / nrow(df_model) * 100
  cat("  CEHWI > 0的天数:", n_positive_cehwi, "(", round(pct_positive, 1), "%)\n")
  
  if (n_positive_cehwi < 30) {
    cat("  ⚠ 热浪天数不足 (<30天)，无法稳健估计DLNM\n")
    cat("    【V4标准】Nature级别研究要求至少30天热浪事件\n")
    return(NULL)
  }
  
  cat("  ✓ 热浪天数充足 (≥30天)，可以进行DLNM分析\n")
  
  # ========== 创建DLNM cross-basis ==========
  
  # 1. 确定暴露维度的内结点 (50%和90%分位数)
  # 只基于CEHWI > 0的值来确定内结点
  cehwi_positive <- df_model$cehwi[df_model$cehwi > 0]
  if (length(cehwi_positive) > 10) {
    cehwi_knots <- quantile(cehwi_positive, probs = c(0.5, 0.9), na.rm = TRUE)
  } else {
    # 如果正值太少，使用全部数据
    cehwi_knots <- quantile(df_model$cehwi, probs = c(0.5, 0.9), na.rm = TRUE)
  }
  cehwi_boundary_knots <- range(df_model$cehwi, na.rm = TRUE)
  cat("\n  CEHWI内结点:", round(cehwi_knots, 4),
      " (city-specific positive-exposure p50/p90 basis)\n")
  # This is the manuscript-primary basis. Shared/common knots are fitted only
  # by the separate supplementary sensitivity workflow.
  
  # 2. 创建cross-basis矩阵
  # - 暴露维度: natural cubic spline；投稿主分析使用跨城市共享结点与边界
  # - 滞后维度: natural cubic spline, 3 df；窗口由当前 lag scenario 决定
  
  tryCatch({
    # 创建cross-basis (crossbasis函数会自动处理滞后，不需要手动创建滞后矩阵)
    cb_cehwi <- crossbasis(
      df_model$cehwi,
      group = df_model$lag_group,
      lag = MAX_LAG,
      argvar = list(
        fun = "ns",
        knots = cehwi_knots,
        Boundary.knots = cehwi_boundary_knots
      ),
      arglag = list(fun = "ns", df = 3)  # 滞后: ns, 3 df
    )
    
    cat("  ✓ Cross-basis创建成功\n")
    cat("    - 暴露维度: natural spline, 内结点 =", round(cehwi_knots, 2), "\n")
    cat("    - 滞后维度: natural spline, df = 3, lag = 0-", MAX_LAG, "天\n")
    
    # ========== 拟合GAM + DLNM ==========
    
    cat("\n  拟合GAM + quasi-Poisson + DLNM...\n")
    
    # Temperature control cross-basis: separates heatwave-specific CEHWI effects
    # from the ordinary nonlinear temperature-response background.
    cb_temp <- NULL
    temp_control_col <- NULL
    temp_control_knots <- NA_real_
    temperature_control_contrast <- NULL
    temp_candidates <- temperature_control_candidates()
    temp_control_col <- find_first_numeric_col(df_model, temp_candidates, min_unique = 6)
    if (!is.null(temp_control_col)) {
      df_model$temp_control <- suppressWarnings(as.numeric(df_model[[temp_control_col]]))
      temp_missing <- sum(!is.finite(df_model$temp_control))
      temp_missing_pct <- 100 * temp_missing / nrow(df_model)
      if (temp_missing > 0 && temp_missing_pct <= 5) {
        temp_median <- median(df_model$temp_control, na.rm = TRUE)
        df_model$temp_control[!is.finite(df_model$temp_control)] <- temp_median
        cat("    - Temperature control: ", temp_control_col,
            " (", temp_missing, " missing values median-imputed; ",
            round(temp_missing_pct, 2), "%)\n", sep = "")
      }
      if (sum(!is.finite(df_model$temp_control)) == 0) {
        temp_control_knots <- unique(as.numeric(quantile(
          df_model$temp_control,
          probs = c(0.50, 0.90),
          na.rm = TRUE
        )))
        if (length(temp_control_knots) >= 2 &&
            dplyr::n_distinct(df_model$temp_control) >= 6) {
          cb_temp <- tryCatch(
            crossbasis(
              df_model$temp_control,
              group = df_model$lag_group,
              lag = MAX_LAG,
              argvar = list(fun = "ns", knots = temp_control_knots),
              arglag = list(fun = "ns", df = 3)
            ),
            error = function(e) {
              cat("    - Temperature cross-basis skipped: ", conditionMessage(e), "\n", sep = "")
              NULL
            }
          )
          if (!is.null(cb_temp)) {
            cat("    - Temperature cross-basis control: ", temp_control_col,
                " | knots p50/p90 = ", paste(round(temp_control_knots, 2), collapse = ", "),
                " | lag = 0-", MAX_LAG, "\n", sep = "")
          }
        } else {
          cat("    - Temperature control skipped: insufficient unique temperature values.\n")
        }
      } else {
        cat("    - Temperature control skipped: ", round(temp_missing_pct, 1),
            "% missing/non-finite values in ", temp_control_col, ".\n", sep = "")
      }
    } else {
      cat("    - Temperature control skipped: no usable mean/apparent temperature column found.\n")
    }

    # First-stage control logic: CEHWI cross-basis + mean/apparent-temperature cross-basis
    # + day-of-season + year + day-of-week, plus precipitation and wind speed.
    # 本研究额外保留 precipitation、wind_speed 与 fish_id FE/RE；mean/apparent temperature 用 cb_temp 控制。
    has_rh_diagnostic <- FALSE
    has_precip <- "precipitation" %in% names(df_model)
    has_wind <- "wind_speed" %in% names(df_model)
    includes_precipitation_control <- FALSE
    includes_wind_speed_control <- FALSE
    
    # 【V5】构建包含气象变量的模型公式
    formula_parts <- c("trip_count_scaled ~ cb_cehwi")
    if (!is.null(cb_temp)) {
      formula_parts <- c(formula_parts, "cb_temp")
    }
    weather_var_count <- 0
    time_control_term <- if (TIME_CONTROL_SPEC == "continuous_time_ns") {
      paste0("splines::ns(date_num, df = ", calendar_time_df, ")")
    } else {
      paste0("s(doy, k = ", DOY_SPLINE_K, ", bs = 'cc')")
    }
    cat(
      "    - Time control: ", TIME_CONTROL_SPEC,
      if (TIME_CONTROL_SPEC == "continuous_time_ns") {
        paste0(" (", TIME_DF_PER_YEAR, " df/year; ", calendar_time_df, " total df)")
      } else {
        paste0(" (cyclic doy k=", DOY_SPLINE_K, " + categorical year)")
      },
      "\n",
      sep = ""
    )
    
    cat("    【V5】检查并添加气象控制变量:\n")
    
    if (has_rh_diagnostic) {
      rh_values <- df_model$relative_humidity
      rh_missing <- sum(!is.finite(rh_values))
      rh_unique <- length(unique(rh_values[is.finite(rh_values)]))
      cat("      ⊘ RH仅诊断，不入模: missing/non-finite = ", rh_missing,
          ", finite unique = ", rh_unique, "\n", sep = "")
      cat("      ✓ 相对湿度 RH (relative_humidity)\n")
      weather_var_count <- weather_var_count + 1
    }
    
    # 添加降水量（对数变换，因为通常右偏）
    if (has_precip) {
      df_model$log_precip <- log(df_model$precipitation + 1)
      weather_precip_term <- if (TIME_CONTROL_SPEC == "continuous_time_ns") {
        precip_ns_spec <- build_fixed_ns_spec(
          df_model$log_precip,
          df = WEATHER_NS_DF,
          knots_from_positive = TRUE
        )
        if (is.null(precip_ns_spec)) {
          cat("      - Precipitation control skipped: insufficient finite variation for the prespecified natural spline.\n")
          NULL
        } else {
          precip_ns_knots <- precip_ns_spec$knots
          precip_ns_boundary <- precip_ns_spec$boundary
          "splines::ns(log_precip, knots = precip_ns_knots, Boundary.knots = precip_ns_boundary)"
        }
      } else {
        "s(log_precip, k = 3, bs = 'cr')"
      }
      if (!is.null(weather_precip_term)) {
        formula_parts <- c(formula_parts, weather_precip_term)
        includes_precipitation_control <- TRUE
      cat("      ✓ 降水量 (log_precip)\n")
        weather_var_count <- weather_var_count + 1
      }
    }
    
    # 添加风速
    if (has_wind) {
      weather_wind_term <- if (TIME_CONTROL_SPEC == "continuous_time_ns") {
        wind_ns_spec <- build_fixed_ns_spec(
          df_model$wind_speed,
          df = WEATHER_NS_DF,
          knots_from_positive = FALSE
        )
        if (is.null(wind_ns_spec)) {
          cat("      - Wind-speed control skipped: insufficient finite variation for the prespecified natural spline.\n")
          NULL
        } else {
          wind_ns_knots <- wind_ns_spec$knots
          wind_ns_boundary <- wind_ns_spec$boundary
          "splines::ns(wind_speed, knots = wind_ns_knots, Boundary.knots = wind_ns_boundary)"
        }
      } else {
        "s(wind_speed, k = 3, bs = 'cr')"
      }
      if (!is.null(weather_wind_term)) {
        formula_parts <- c(formula_parts, weather_wind_term)
        includes_wind_speed_control <- TRUE
      cat("      ✓ 风速 (wind_speed)\n")
        weather_var_count <- weather_var_count + 1
      }
    }
    
    if (weather_var_count == 0) {
      cat("      ⚠ 无气象数据，模型将不包含气象控制变量\n")
    }
    
    # 添加标准时间和空间控制变量
    # 【核心区分】单城市 vs 分区分析
    if (has_city_effect) {
      # ========== 分区合并分析（多城市数据）==========
      # 问题：373个格子 + 22个城市 → 内存爆炸
      # 解决方案：
      #   1. 不添加city_fac（避免与fish_id共线）
      #   2. fish_id改用随机效应（大幅减少内存）
      
      n_grids <- n_distinct(df_model$fish_id)
      cat("      【分区分析优化】检测到多城市数据（", n_grids, "个格子）\n", sep="")
      
      formula_parts <- c(formula_parts,
                         paste0("s(doy, k = ", DOY_SPLINE_K, ", bs = 'cc')"),   # 季节性（循环样条）
                         "year_fac",                    # 年份固定效应
                         "dow_fac",                     # 星期几固定效应
                         "s(fish_id, bs = 're')")       # 【关键】格子随机效应（内存优化）
      
      cat("      ✓ 格子效应: 随机效应 s(fish_id, bs='re')\n")
      cat("      ✓ 城市信息: 已通过fish_id嵌套关系捕捉（无需city_fac）\n")
      cat("      ℹ 内存优化: ~14GB → ~2GB\n")
      
    } else {
      # ========== 单城市分析（保持原样）==========
      formula_parts <- c(formula_parts,
                         paste0("s(doy, k = ", DOY_SPLINE_K, ", bs = 'cc')"),  # 季节性（循环样条）
                         "year_fac",                   # 年份固定效应
                         "dow_fac",                    # 星期几固定效应
                         "fish_id_fac")                # 格子固定效应（标准方法）
    }
    
    if (TIME_CONTROL_SPEC == "continuous_time_ns") {
      formula_parts <- vapply(
        formula_parts,
        function(term) {
          if (grepl("^s\\(doy,", term)) time_control_term else term
        },
        character(1)
      )
      formula_parts <- setdiff(formula_parts, "year_fac")
    }

    if ("year_fac" %in% names(df_model) && is.factor(df_model$year_fac)) {
      df_model$year_fac <- droplevels(df_model$year_fac)
    }
    if ("dow_fac" %in% names(df_model) && is.factor(df_model$dow_fac)) {
      df_model$dow_fac <- droplevels(df_model$dow_fac)
    }
    if ("fish_id_fac" %in% names(df_model) && is.factor(df_model$fish_id_fac)) {
      df_model$fish_id_fac <- droplevels(df_model$fish_id_fac)
    }
    
    dropped_constant_controls <- character(0)
    if ("year_fac" %in% formula_parts &&
        (!("year_fac" %in% names(df_model)) || dplyr::n_distinct(df_model$year_fac[!is.na(df_model$year_fac)]) <= 1)) {
      formula_parts <- setdiff(formula_parts, "year_fac")
      dropped_constant_controls <- c(dropped_constant_controls, "year_fac")
    }
    if ("dow_fac" %in% formula_parts &&
        (!("dow_fac" %in% names(df_model)) || dplyr::n_distinct(df_model$dow_fac[!is.na(df_model$dow_fac)]) <= 1)) {
      formula_parts <- setdiff(formula_parts, "dow_fac")
      dropped_constant_controls <- c(dropped_constant_controls, "dow_fac")
    }
    if ("fish_id_fac" %in% formula_parts &&
        (!("fish_id_fac" %in% names(df_model)) || dplyr::n_distinct(df_model$fish_id_fac[!is.na(df_model$fish_id_fac)]) <= 1)) {
      formula_parts <- setdiff(formula_parts, "fish_id_fac")
      dropped_constant_controls <- c(dropped_constant_controls, "fish_id_fac")
    }
    if ("s(fish_id, bs = 're')" %in% formula_parts &&
        (!("fish_id" %in% names(df_model)) || dplyr::n_distinct(df_model$fish_id[!is.na(df_model$fish_id)]) <= 1)) {
      formula_parts <- setdiff(formula_parts, "s(fish_id, bs = 're')")
      dropped_constant_controls <- c(dropped_constant_controls, "s(fish_id, bs = 're')")
    }
    if (length(dropped_constant_controls) > 0) {
      cat("      ⚠ 自动移除常量控制项: ", paste(unique(dropped_constant_controls), collapse = ", "), "\n", sep = "")
    }
    
    formula_str <- paste(formula_parts, collapse = " + ")
    
    cat("    [V6] Model terms:", length(formula_parts) - 1,
        "| weather smooths:", weather_var_count,
        "| temperature cross-basis:", ifelse(is.null(cb_temp), 0, 1), "\n")
    
    # 【诊断】检查数据特征
    cat("    【V4】数据特征:\n")
    cat("      - 格子数:", n_distinct(df_model$fish_id), "\n")
    cat("      - CEHWI最大值:", round(max(df_model$cehwi, na.rm=TRUE), 2), "\n")
    cat("      - Trip count非零比例:", round(mean(df_model$trip_count_scaled > 0, na.rm=TRUE)*100, 1), "%\n")
    cat("      - 自由度评估: ", nrow(df_model) - n_distinct(df_model$fish_id) - 15, "\n")
    cat("    【V4】模型简化: 不使用社会经济变量（避免过度参数化）\n")
    
    # 【V5.2】根据数据量调整收敛参数
    if (has_city_effect) {
      # 分区分析：放宽内层IRLS + 外层Newton，避免梯度尺度大导致死循环
      cat("    开始拟合GAM模型（分区分析，大数据集优化）...\n")
      cat("      ℹ 内层 epsilon/mgcv.tol=1e-05；外层 Newton conv.tol=1e4（避免卡住）\n")
      m_gam <- gam(
        as.formula(formula_str),
        data = df_model,
        family = quasipoisson(link = "log"),
        method = "REML",
        control = gam.control(
          maxit = 200,
          trace = TRUE,        # 显示迭代进度，避免“看起来卡住”
          epsilon = 1e-05,
          mgcv.tol = 1e-05,
          newton = list(conv.tol = 1e4)  # 外层牛顿容差：大数据下 max(|grad|) 大，必须放宽
        )
      )
      cat("      ✓ 模型拟合完成\n")
    } else {
      # 单城市分析：小数据集，保持严格收敛
      cat("    开始拟合GAM模型 (可能需要1-3分钟)...\n")
      m_gam <- gam(
        as.formula(formula_str),
        data = df_model,
        family = quasipoisson(link = "log"),
        method = "REML",
        control = gam.control(
          maxit = 200,
          trace = FALSE,
          epsilon = 1e-07,    # 严格收敛（小数据可以做到）
          mgcv.tol = 1e-07
        )
      )
    }
    
    s <- summary(m_gam)
    temperature_control_contrast <- calculate_temperature_control_contrast(
      cb_temp = cb_temp,
      m_gam = m_gam,
      temp_values = if ("temp_control" %in% names(df_model)) df_model$temp_control else NULL,
      temp_control_col = temp_control_col,
      max_lag_value = MAX_LAG
    )
    if (!is.null(temperature_control_contrast) && nrow(temperature_control_contrast) > 0) {
      cat("    - Temperature control contrast stored: ",
          temperature_control_contrast$contrast_note[1], "\n", sep = "")
    }
    r2 <- s$r.sq
    dev_explained <- s$dev.expl
    
    cat("  ✓ 模型拟合成功\n")
    cat("    - R² =", round(r2, 4), "\n")
    cat("    - Deviance explained:", round(dev_explained * 100, 1), "%\n")
    cat("    - Dispersion parameter:", round(s$dispersion, 2), "\n")
    cat("    - AIC =", round(AIC(m_gam), 1), "\n")
    
    # ========== 输出控制变量系数和显著性 ==========
    
    cat("\n  === 控制变量显著性 ===\n")
    
    # 1. 平滑项 (湿度、季节性等)
    smooth_terms <- s$s.table
    if (is.null(smooth_terms)) {
      smooth_terms <- matrix(numeric(0), nrow = 0L, ncol = 0L)
    }
    smooth_p <- NA
    overall_wald_chisq <- NA_real_
    overall_wald_df <- NA_integer_
    overall_wald_p <- NA_real_
    
    cat("    [平滑项]\n")
    if (nrow(smooth_terms) > 0) {
      for (i in 1:nrow(smooth_terms)) {
        term_name <- rownames(smooth_terms)[i]
        edf <- smooth_terms[i, "edf"]
        f_val <- smooth_terms[i, "F"]
        p_val <- smooth_terms[i, "p-value"]
        sig <- ifelse(p_val < 0.001, "***", 
               ifelse(p_val < 0.01, "**",
               ifelse(p_val < 0.05, "*", "ns")))
        
        cat(sprintf("      %s: edf=%.2f, F=%.2f, p=%s %s\n", 
                    term_name, edf, f_val, format.pval(p_val, digits=3), sig))
      }
    }
    
    # 2. 参数项 (年份、周末等固定效应)
    param_table <- s$p.table
    param_names <- rownames(param_table)
    
    cat("\n    [固定效应参数项]\n")
    
    if (nrow(param_table) > 0) {
      # 分类显示：截距、DLNM、全局控制变量、格子固定效应
      
      # A. 截距（全局）
      intercept_idx <- which(param_names == "(Intercept)")
      if (length(intercept_idx) > 0) {
        cat("      === 截距（全局，所有格子共享）===\n")
        for (i in intercept_idx) {
          coef <- param_table[i, "Estimate"]
          se <- param_table[i, "Std. Error"]
          t_val <- param_table[i, "t value"]
          p_val <- param_table[i, "Pr(>|t|)"]
          sig <- ifelse(p_val < 0.001, "***", 
                 ifelse(p_val < 0.01, "**",
                 ifelse(p_val < 0.05, "*", "ns")))
          cat(sprintf("        截距: %.4f (SE=%.4f, t=%.2f, p=%s) %s\n", 
                      coef, se, t_val, format.pval(p_val, digits=3), sig))
        }
      }
      
      # B. DLNM系数（仅显示前3个示例）
      cb_idx <- grep("cb_cehwi", param_names)
      if (length(cb_idx) > 0) {
        cat(sprintf("\n      === DLNM Cross-basis系数（共%d个，仅显示前3个示例）===\n", length(cb_idx)))
        for (i in cb_idx[1:min(3, length(cb_idx))]) {
          term_name <- param_names[i]
          coef <- param_table[i, "Estimate"]
          sig <- ifelse(param_table[i, "Pr(>|t|)"] < 0.001, "***", 
                 ifelse(param_table[i, "Pr(>|t|)"] < 0.01, "**",
                 ifelse(param_table[i, "Pr(>|t|)"] < 0.05, "*", "ns")))
          cat(sprintf("        %s: %.4f %s\n", term_name, coef, sig))
        }
        if (length(cb_idx) > 3) {
          cat(sprintf("        ... (其余%d个已省略，完整系数见CSV)\n", length(cb_idx) - 3))
        }
      }
      
      # C. 全局控制变量（年份、周末，所有格子共享）
      cat("\n      === 全局控制变量（所有格子共享）===\n")
      
      # 年份效应
      year_idx <- grep("year_fac", param_names)
      if (length(year_idx) > 0) {
        cat("        [年份固定效应]\n")
        for (i in year_idx) {
          term_name <- param_names[i]
          year_label <- str_replace(term_name, "year_fac", "")
          coef <- param_table[i, "Estimate"]
          se <- param_table[i, "Std. Error"]
          t_val <- param_table[i, "t value"]
          p_val <- param_table[i, "Pr(>|t|)"]
          sig <- ifelse(p_val < 0.001, "***", 
                 ifelse(p_val < 0.01, "**",
                 ifelse(p_val < 0.05, "*", "ns")))
          cat(sprintf("          年份%s: %.4f (SE=%.4f, t=%.2f, p=%s) %s\n", 
                      year_label, coef, se, t_val, format.pval(p_val, digits=3), sig))
        }
      }
      
      # 星期几效应
      dow_idx <- grep("dow_fac", param_names)
      if (length(dow_idx) > 0) {
        cat("        [星期几固定效应]\n")
        for (i in dow_idx) {
          term_name <- param_names[i]
          coef <- param_table[i, "Estimate"]
          se <- param_table[i, "Std. Error"]
          t_val <- param_table[i, "t value"]
          p_val <- param_table[i, "Pr(>|t|)"]
          sig <- ifelse(p_val < 0.001, "***", 
                 ifelse(p_val < 0.01, "**",
                 ifelse(p_val < 0.05, "*", "ns")))
          rr_change <- (exp(coef) - 1) * 100  # 转换为百分比变化
          cat(sprintf("          %s: %.4f (RR=%.3f, 变化%.1f%%, p=%s) %s\n", 
                      term_name,
                      coef, exp(coef), rr_change, format.pval(p_val, digits=3), sig))
        }
      }
      
      # D. 格子固定效应（每个格子独立的截距）
      fish_id_rows <- grep("fish_id_fac", param_names)
      if (length(fish_id_rows) > 0) {
        cat(sprintf("\n      === 格子固定效应（每个格子独立截距，共%d个）===\n", length(fish_id_rows)))
        cat("        （已省略显示，完整系数见CSV）\n")
      }
    }
    
    cat("\n")
    
    # 提取DLNM显著性（cross-basis在mgcv中为参数项，不会出现在平滑表）
    cat("    - 检测DLNM项...\n")
    cb_smooth_idx <- if (nrow(smooth_terms) > 0) which(grepl("cb_cehwi", rownames(smooth_terms))) else integer(0)
    cb_param_idx <- grep("cb_cehwi", names(coef(m_gam)))
    if (length(cb_smooth_idx) > 0 && cb_smooth_idx[1] <= nrow(smooth_terms)) {
      cb_smooth <- smooth_terms[cb_smooth_idx[1], ]
      edf <- cb_smooth["edf"]; f_val <- cb_smooth["F"]; smooth_p <- cb_smooth["p-value"]
      sig_mark <- ifelse(smooth_p < 0.001, "***", ifelse(smooth_p < 0.01, "**", ifelse(smooth_p < 0.05, "*", "ns")))
      cat("    - DLNM平滑项显著性: edf =", round(edf, 2), ", F =", round(f_val, 2),
          ", p =", format.pval(smooth_p, digits = 3), " ", sig_mark, "\n", sep = "")
    } else if (length(cb_param_idx) > 0) {
      # cross-basis 在 gam 中为参数化项（矩阵协变量），无平滑行；不误用第一个平滑项
      cat("    - DLNM为参数化 cross-basis（", length(cb_param_idx), " 个系数），显著性见上方「DLNM Cross-basis系数」\n", sep = "")
    } else {
      cat("    ⚠ 未找到DLNM系数\n")
    }
    
    # ========== 预测累积RR曲线 (main lag 0-11天累积) ==========
    
    cat("\n  预测累积RR曲线...\n")
    
    # 预测值序列：从0到CEHWI的max，确保包含有意义的范围
    # 使用更多预测点（500个）确保曲线和置信区间平滑，避免马赛克效应
    cehwi_max <- max(df_model$cehwi, na.rm = TRUE)
    # 如果max太小，至少预测到5
    if (cehwi_max < 5) {
      cehwi_seq <- seq(0, 5, length.out = 500)
    } else {
      cehwi_seq <- seq(0, cehwi_max, length.out = 500)
    }
    
    # 使用crosspred计算累积RR (main lag 0-11天累积)
    cp <- crosspred(
      cb_cehwi,
      m_gam,
      at = cehwi_seq,
      cen = REFERENCE_CEHWI,  # 参照: CEHWI = 0
      cumul = TRUE  # 累积RR (main lag 0-11天)
    )
    
    cat("  ✓ 累积RR曲线计算完成\n")
    cat("    - 参照值: CEHWI =", REFERENCE_CEHWI, "\n")
    cat("    - 累积滞后: 0-", MAX_LAG, "天\n")
    
    # 提取预测结果
    pred_df <- data.frame(
      cehwi = cehwi_seq,
      rr = cp$allRRfit,
      rr_low = cp$allRRlow,
      rr_high = cp$allRRhigh
    )
    
    # 【计算效应百分比】基于RR曲线从最小到最大CEHWI的变化
    effect_pct <- NA
    if (nrow(pred_df) >= 2) {
      min_cehwi_idx <- which.min(pred_df$cehwi)
      max_cehwi_idx <- which.max(pred_df$cehwi)
      rr_at_min <- pred_df$rr[min_cehwi_idx]
      rr_at_max <- pred_df$rr[max_cehwi_idx]
      
      # RR变化百分比
      if (is.finite(rr_at_min) && is.finite(rr_at_max) && rr_at_min > 0) {
        effect_pct <- (rr_at_max - rr_at_min) / rr_at_min * 100
        
        # 检查是否异常（>1000%或<-99%通常是数值不稳定）
        if (abs(effect_pct) > 1000) {
          cat("    - 效应变化: ", round(effect_pct, 1), "% ", 
              ifelse(effect_pct > 0, "↑", "↓"),
              " ⚠ 异常大（数值不稳定）\n", sep = "")
          cat("      (RR从", round(rr_at_min, 3), "到", round(rr_at_max, 3), ")\n", sep = "")
          cat("      → 建议: 增加样本量或检查数据质量\n")
        } else {
          cat("    - 效应变化: ", round(effect_pct, 1), "% ", 
              ifelse(effect_pct > 0, "↑", "↓"),
              " (RR从", round(rr_at_min, 3), "到", round(rr_at_max, 3), ")\n", sep = "")
        }
      }
    }
    
    # 【优化2】检查RR是否合理，标记异常值
    rr_range <- range(pred_df$rr, na.rm = TRUE)
    rr_high_range <- range(pred_df$rr_high, na.rm = TRUE)
    
    has_extreme <- any(pred_df$rr > 100 | pred_df$rr < 0.01 | pred_df$rr_high > 1000, na.rm = TRUE)
    
    cat("    - RR范围:", round(rr_range, 3), "\n")
    cat("    - RR_high最大值:", round(max(pred_df$rr_high, na.rm = TRUE), 3), "\n")
    
    if (has_extreme) {
      cat("    ⚠ 检测到极端RR值（>100 或 <0.01），可能因为:\n")
      cat("      1) 样本量不足\n")
      cat("      2) 模型过度离散\n")
      cat("      3) 热浪天数太少\n")
      cat("    → 建议: 增加样本量或选择其他时间段\n")
    }
    
    # ========== 提取用于第二阶段的系数 ==========
    # 参考两阶段DLNM文献做法：先把 lag 维度累积掉，再提取 reduced spline coefficients。
    # 这样二阶段处理的是 overall cumulative exposure-response，而不是完整 cross-basis 的 lag×var 系数。
    reduced_overall <- tryCatch(
      crossreduce(cb_cehwi, model = m_gam, type = "overall", cen = REFERENCE_CEHWI),
      error = function(e) {
        cat("    ⚠ crossreduce(type='overall')失败:", conditionMessage(e), "\n")
        NULL
      }
    )
    
    if (is.null(reduced_overall)) {
      return(NULL)
    }
    
    cb_coef <- coef(reduced_overall)
    cb_vcov <- vcov(reduced_overall)
    if (is.null(names(cb_coef)) || any(is.na(names(cb_coef))) || any(names(cb_coef) == "")) {
      names(cb_coef) <- paste0("overall_spline", seq_along(cb_coef))
    }
    reduced_basis_x <- seq(min(df_model$cehwi, na.rm = TRUE),
                           max(df_model$cehwi, na.rm = TRUE),
                           length.out = 100)
    reduced_basis <- onebasis(
      reduced_basis_x,
      fun = "ns",
      knots = cehwi_knots,
      Boundary.knots = cehwi_boundary_knots
    )
    colnames(reduced_basis) <- names(cb_coef)

    lag_profile_value <- suppressWarnings(as.numeric(quantile(
      cehwi_positive,
      probs = LAG_PROFILE_EXPOSURE_PERCENTILE,
      na.rm = TRUE
    )))
    lag_coef_p90 <- NULL
    lag_vcov_p90 <- NULL
    lag_basis_p90 <- NULL
    if (is.finite(lag_profile_value) && lag_profile_value > REFERENCE_CEHWI) {
      reduced_lag_p90 <- tryCatch(
        crossreduce(
          cb_cehwi,
          model = m_gam,
          type = "var",
          value = lag_profile_value,
          cen = REFERENCE_CEHWI
        ),
        error = function(e) {
          cat("    Warning: crossreduce(type='var') for lag profile failed: ",
              conditionMessage(e), "\n", sep = "")
          NULL
        }
      )
      if (!is.null(reduced_lag_p90)) {
        lag_coef_p90 <- coef(reduced_lag_p90)
        lag_vcov_p90 <- vcov(reduced_lag_p90)
        lag_basis_p90 <- reduced_lag_p90$basis
        if (is.null(names(lag_coef_p90)) ||
            any(is.na(names(lag_coef_p90))) ||
            any(names(lag_coef_p90) == "")) {
          names(lag_coef_p90) <- paste0("lag_spline", seq_along(lag_coef_p90))
        }
        if (!is.null(lag_basis_p90) && ncol(lag_basis_p90) == length(lag_coef_p90)) {
          colnames(lag_basis_p90) <- names(lag_coef_p90)
        }
        cat("    - Stored lag-response reduced coefficients at city p",
            round(100 * LAG_PROFILE_EXPOSURE_PERCENTILE), " exposure = ",
            round(lag_profile_value, 2), " (", length(lag_coef_p90), " coef)\n", sep = "")
      }
    }
    
    overall_wald <- tryCatch({
      if (length(cb_coef) == 0 || any(dim(cb_vcov) != c(length(cb_coef), length(cb_coef)))) {
        return(NULL)
      }
      vcov_inv <- tryCatch(
        solve(cb_vcov),
        error = function(e) MASS::ginv(cb_vcov)
      )
      wald_stat <- as.numeric(t(cb_coef) %*% vcov_inv %*% cb_coef)
      wald_df_local <- length(cb_coef)
      wald_p <- pchisq(wald_stat, df = wald_df_local, lower.tail = FALSE)
      list(chisq = wald_stat, df = wald_df_local, p = wald_p)
    }, error = function(e) {
      cat("    ⚠ DLNM整体Wald检验失败:", conditionMessage(e), "\n")
      NULL
    })
    
    if (!is.null(overall_wald)) {
      overall_wald_chisq <- overall_wald$chisq
      overall_wald_df <- overall_wald$df
      overall_wald_p <- overall_wald$p
    }
    
    cat("\n  提取第二阶段所需系数:\n")
    cat("    - Reduced cumulative spline系数数:", length(cb_coef), "\n")
    cat("    - 协方差矩阵维度:", dim(cb_vcov), "\n")
    if (!is.na(overall_wald_p)) {
      cat("    - DLNM整体Wald检验: Chi-square(", overall_wald_df, ") = ",
          round(overall_wald_chisq, 2), ", p = ",
          format.pval(overall_wald_p, digits = 3), "\n", sep = "")
    }
    
    # ========== 【V4修改】提取格子固定效应 + 周末效应 ==========
    
    cat("\n  === 格子固定效应 + 时间效应 ===\n")
    
    socioecon_coefs_df <- NULL  # 保留这个变量名（兼容性）
    all_coefs <- coef(m_gam)
    all_coef_names <- names(all_coefs)
    
    socioecon_coefs_list <- list()
    
    # 【V4】第一阶段不提取社会经济变量系数（因为没有使用）
    # 但保留数据结构，以便后续代码兼容
    
    # 2. 提取周末效应
    weekend_contrast <- compute_weekend_contrast(m_gam)
    if (!is.null(weekend_contrast) && nrow(weekend_contrast) > 0) {
      coef_val <- weekend_contrast$coefficient[1]
      se_val <- weekend_contrast$se[1]
      p_val <- weekend_contrast$p_value[1]
      ci_low <- weekend_contrast$ci_low[1]
      ci_high <- weekend_contrast$ci_high[1]
      rr <- exp(coef_val)
      rr_low <- exp(ci_low)
      rr_high <- exp(ci_high)
      rr_change_pct <- (rr - 1) * 100
      t_val <- ifelse(is.finite(se_val) && se_val > 0, coef_val / se_val, NA_real_)
      sig <- ifelse(is.na(p_val), "ns",
             ifelse(p_val < 0.001, "***",
             ifelse(p_val < 0.01, "**",
             ifelse(p_val < 0.05, "*", "ns"))))
      
      cat(sprintf("    Weekend vs Weekday: coef=%.4f, RR=%.3f (%+.1f%%), p=%s %s\n",
                  coef_val, rr, rr_change_pct, 
                  format.pval(p_val, digits=3), sig))
      
      socioecon_coefs_list[["weekend_effect"]] <- tibble(
        city = city_name,
        indicator = indicator,
        variable = "Weekend_Effect",
        coefficient = coef_val,
        se = se_val,
        t_value = t_val,
        p_value = p_val,
        ci_low = ci_low,
        ci_high = ci_high,
        rr = rr,
        rr_low = rr_low,
        rr_high = rr_high,
        rr_change_pct = rr_change_pct,
        significant = sig,
        var_order = 8
      )
    }
    
    # 3. 【V5.2修复】提取格子效应（空间异质性）- 支持固定效应和随机效应
    
    # 尝试提取固定效应（单城市分析）
    fish_id_idx <- grep("fish_id_fac", all_coef_names)
    
    # 尝试提取随机效应方差分量（分区分析）
    grid_sd <- NA
    grid_se <- NA
    n_grids <- NA
    effect_type <- "none"
    
    if (length(fish_id_idx) > 0) {
      # ========== 固定效应（单城市分析）==========
      grid_coefs <- all_coefs[fish_id_idx]
      grid_sd <- sd(grid_coefs, na.rm = TRUE)
      grid_var <- var(grid_coefs, na.rm = TRUE)
      grid_se <- grid_sd / sqrt(length(grid_coefs))
      n_grids <- length(grid_coefs)
      effect_type <- "fixed"
      
      cat(sprintf("    Grid Spatial Heterogeneity (固定效应): SD=%.4f (SE=%.4f, n_grids=%d)\n",
                  grid_sd, grid_se, n_grids))
      
    } else if ("gam.vcomp" %in% names(m_gam)) {
      # ========== 随机效应（分区分析）==========
      # 从gam.vcomp提取随机效应的方差分量
      vcomp <- m_gam$gam.vcomp
      fish_id_vcomp_idx <- grep("fish_id", names(vcomp))
      
      if (length(fish_id_vcomp_idx) > 0) {
        grid_var <- vcomp[fish_id_vcomp_idx[1]]  # 方差分量
        grid_sd <- sqrt(grid_var)                 # 标准差
        grid_se <- grid_sd * 0.1                  # 粗略估计SE（随机效应SE不易直接获得）
        n_grids <- n_distinct(df_model$fish_id)
        effect_type <- "random"
        
        cat(sprintf("    Grid Spatial Heterogeneity (随机效应): SD=%.4f (variance=%.4f, n_grids=%d)\n",
                    grid_sd, grid_var, n_grids))
      }
    }
    
    # 如果成功提取任一种效应，保存结果
    if (!is.na(grid_sd)) {
      # 置信区间
      ci_low_grid <- grid_sd - 1.96 * grid_se
      ci_high_grid <- grid_sd + 1.96 * grid_se
      
      cat(sprintf("      → 【V5.2】格子间空间异质性（%s效应，第二阶段将用社会经济变量解释）\n", 
                  ifelse(effect_type=="fixed", "固定", "随机")))
      
      socioecon_coefs_list[["grid_heterogeneity"]] <- tibble(
        city = city_name,
        indicator = indicator,
        variable = "Grid_Heterogeneity",
        coefficient = grid_sd,  # 使用标准差
        se = grid_se,
        t_value = NA,
        p_value = NA,
        ci_low = ci_low_grid,
        ci_high = ci_high_grid,
        rr = NA,
        rr_low = NA,
        rr_high = NA,
        rr_change_pct = NA,
        significant = "spatial",  # 特殊标记
        var_order = 9
      )
    }
    
    if (length(socioecon_coefs_list) > 0) {
      socioecon_coefs_df <- bind_rows(socioecon_coefs_list) %>%
        arrange(var_order)
    } else {
      cat("    【V4】仅保留核心控制变量\n")
    }
    
    cat("\n")
    
    # ========== 提取全局控制变量系数 ==========
    
    # 提取全局控制变量系数（年份、星期几等）
    all_coefs <- coef(m_gam)
    all_coef_names <- names(all_coefs)
    
    # 全局控制变量（非格子特定的）
    year_idx <- grep("year_fac", all_coef_names)
    dow_idx <- grep("dow_fac", all_coef_names)
    intercept_idx <- which(all_coef_names == "(Intercept)")
    
    global_coefs <- NULL
    if (length(c(intercept_idx, year_idx, dow_idx)) > 0) {
      global_idx <- c(intercept_idx, year_idx, dow_idx)
      
      # 从city_name中提取模型类型
      model_type_extracted <- ifelse(grepl("_", city_name),
                                     str_extract(city_name, "[^_]+$"),
                                     "Unknown")
      
      global_coefs <- tibble(
        city = city_name,
        indicator = indicator,
        model_type = model_type_extracted,
        variable_type = "global",  # 全局变量，所有格子共享
        variable_name = all_coef_names[global_idx],
        coefficient = all_coefs[global_idx],
        se = sqrt(diag(vcov(m_gam)))[global_idx],
        t_value = param_table[all_coef_names[global_idx], "t value"],
        p_value = param_table[all_coef_names[global_idx], "Pr(>|t|)"]
      )
      
      cat("    - 提取全局控制变量:", nrow(global_coefs), "个\n")
    }
    
    # ========== 提取格子效应 ==========
    # 【V5.2】区分固定效应（单城市）和随机效应（分区分析）
    
    # 尝试提取固定效应的系数
    fish_id_idx <- grep("fish_id_fac", all_coef_names)
    
    grid_coefs <- NULL
    if (length(fish_id_idx) > 0) {
      # ========== 固定效应（单城市）：提取每个格子的系数 ==========
      fish_id_coefs <- all_coefs[fish_id_idx]
      fish_id_names <- all_coef_names[fish_id_idx]
      
      # 解析格子ID（从"fish_id_facXXX"中提取XXX）
      grid_ids <- str_replace(fish_id_names, "fish_id_fac", "")
      
      # 从city_name中提取模型类型（例如"Atlanta_Day" -> "Day"）
      model_type_extracted <- ifelse(grepl("_", city_name),
                                     str_extract(city_name, "[^_]+$"),
                                     "Unknown")
      
      # 格子固定效应（相对于参考格子）
      grid_coefs <- tibble(
        city = city_name,
        indicator = indicator,
        model_type = model_type_extracted,
        variable_type = "grid_specific",  # 格子特定的截距
        variable_name = paste0("fish_id_", grid_ids),
        fish_id = grid_ids,
        coefficient = fish_id_coefs,
        se = sqrt(diag(vcov(m_gam)))[fish_id_idx],
        t_value = param_table[fish_id_names, "t value"],
        p_value = param_table[fish_id_names, "Pr(>|t|)"]
      )
      
      cat("    - 提取格子固定效应:", nrow(grid_coefs), "个格子\n")
      
    } else if ("gam.vcomp" %in% names(m_gam)) {
      # ========== 随机效应（分区分析）：只汇总方差，不提取单个格子 ==========
      vcomp <- m_gam$gam.vcomp
      fish_id_vcomp_idx <- grep("fish_id", names(vcomp))
      
      if (length(fish_id_vcomp_idx) > 0) {
        grid_var <- vcomp[fish_id_vcomp_idx[1]]
        grid_sd <- sqrt(grid_var)
        cat("    - 格子随机效应: SD =", round(grid_sd, 4), "(variance =", round(grid_var, 4), ")\n")
        cat("      （分区分析不提取单个格子系数，仅保留方差分量）\n")
      }
    }
    
    # 合并全局系数和格子系数
    all_model_coefs <- bind_rows(
      global_coefs,
      grid_coefs
    )
    
    # 【V7新增】计算城市级归因分数（Attributable Fraction, AF）
    # 参考：参考文献方法 - 基于城市的RR曲线和暴露分布
    af_city <- tryCatch({
      cehwi_positive <- df_model$cehwi[df_model$cehwi > 0]
      
      if (length(cehwi_positive) > 10 && nrow(pred_df) > 0) {
        af_transform <- function(rr_vals) {
          rr_vals <- pmax(0.01, pmin(100, rr_vals))
          af_vals <- (rr_vals - 1) / rr_vals * 100
          af_vals[is.na(af_vals) | is.infinite(af_vals)] <- NA_real_
          pmax(-100, pmin(100, af_vals))
        }
        
        # 1. 计算暴露分布（直方图）
        hist_breaks <- seq(min(cehwi_positive), max(cehwi_positive), length.out = 50)
        hist_result <- hist(cehwi_positive, breaks = hist_breaks, plot = FALSE)
        
        # 2. 获取直方图中点和频率
        x_midpoints <- hist_result$mids
        freq <- hist_result$counts / sum(hist_result$counts)
        
        # 3. 在每个暴露值上插值得到RR及其95%CI
        rr_at_x <- approx(pred_df$cehwi, pred_df$rr, xout = x_midpoints, rule = 2)$y
        rr_low_at_x <- approx(pred_df$cehwi, pred_df$rr_low, xout = x_midpoints, rule = 2)$y
        rr_high_at_x <- approx(pred_df$cehwi, pred_df$rr_high, xout = x_midpoints, rule = 2)$y
        
        af_at_x <- af_transform(rr_at_x)
        af_low_at_x <- af_transform(rr_low_at_x)
        af_high_at_x <- af_transform(rr_high_at_x)
        
        # 4. 加权求和得到总AF及其CI
        af_weighted <- sum(freq * af_at_x, na.rm = TRUE)
        af_weighted_low_raw <- sum(freq * af_low_at_x, na.rm = TRUE)
        af_weighted_high_raw <- sum(freq * af_high_at_x, na.rm = TRUE)
        af_weighted_low <- min(af_weighted_low_raw, af_weighted_high_raw, na.rm = TRUE)
        af_weighted_high <- max(af_weighted_low_raw, af_weighted_high_raw, na.rm = TRUE)
        af_weighted_se <- calc_se_from_ci(af_weighted_low, af_weighted_high)
        
        # 5. 计算不同分位数的AF（用于敏感性分析）
        q25 <- as.numeric(quantile(cehwi_positive, 0.25, na.rm = TRUE))
        q50 <- as.numeric(quantile(cehwi_positive, 0.50, na.rm = TRUE))
        q75 <- as.numeric(quantile(cehwi_positive, 0.75, na.rm = TRUE))
        q90 <- as.numeric(quantile(cehwi_positive, 0.90, na.rm = TRUE))
        q95 <- as.numeric(quantile(cehwi_positive, 0.95, na.rm = TRUE))
        
        rr_25 <- approx(pred_df$cehwi, pred_df$rr, xout = q25, rule = 2)$y
        rr_50 <- approx(pred_df$cehwi, pred_df$rr, xout = q50, rule = 2)$y
        rr_75 <- approx(pred_df$cehwi, pred_df$rr, xout = q75, rule = 2)$y
        rr_90 <- approx(pred_df$cehwi, pred_df$rr, xout = q90, rule = 2)$y
        rr_95 <- approx(pred_df$cehwi, pred_df$rr, xout = q95, rule = 2)$y
        
        rr_low_25 <- approx(pred_df$cehwi, pred_df$rr_low, xout = q25, rule = 2)$y
        rr_low_50 <- approx(pred_df$cehwi, pred_df$rr_low, xout = q50, rule = 2)$y
        rr_low_75 <- approx(pred_df$cehwi, pred_df$rr_low, xout = q75, rule = 2)$y
        rr_low_90 <- approx(pred_df$cehwi, pred_df$rr_low, xout = q90, rule = 2)$y
        rr_low_95 <- approx(pred_df$cehwi, pred_df$rr_low, xout = q95, rule = 2)$y
        
        rr_high_25 <- approx(pred_df$cehwi, pred_df$rr_high, xout = q25, rule = 2)$y
        rr_high_50 <- approx(pred_df$cehwi, pred_df$rr_high, xout = q50, rule = 2)$y
        rr_high_75 <- approx(pred_df$cehwi, pred_df$rr_high, xout = q75, rule = 2)$y
        rr_high_90 <- approx(pred_df$cehwi, pred_df$rr_high, xout = q90, rule = 2)$y
        rr_high_95 <- approx(pred_df$cehwi, pred_df$rr_high, xout = q95, rule = 2)$y
        
        af_25 <- af_transform(rr_25)
        af_50 <- af_transform(rr_50)
        af_75 <- af_transform(rr_75)
        af_90 <- af_transform(rr_90)
        af_95 <- af_transform(rr_95)
        
        af_low_25 <- af_transform(rr_low_25)
        af_low_50 <- af_transform(rr_low_50)
        af_low_75 <- af_transform(rr_low_75)
        af_low_90 <- af_transform(rr_low_90)
        af_low_95 <- af_transform(rr_low_95)
        
        af_high_25 <- af_transform(rr_high_25)
        af_high_50 <- af_transform(rr_high_50)
        af_high_75 <- af_transform(rr_high_75)
        af_high_90 <- af_transform(rr_high_90)
        af_high_95 <- af_transform(rr_high_95)
        
        list(
          af_overall = af_weighted,
          af_overall_low = min(af_weighted_low, af_weighted_high, na.rm = TRUE),
          af_overall_high = max(af_weighted_low, af_weighted_high, na.rm = TRUE),
          af_overall_se = af_weighted_se,
          cehwi_p25 = q25,
          cehwi_p50 = q50,
          cehwi_p75 = q75,
          cehwi_p90 = q90,
          cehwi_p95 = q95,
          rr_p25 = rr_25,
          rr_p50 = rr_50,
          rr_p75 = rr_75,
          rr_p90 = rr_90,
          rr_p95 = rr_95,
          rr_low_p25 = rr_low_25,
          rr_low_p50 = rr_low_50,
          rr_low_p75 = rr_low_75,
          rr_low_p90 = rr_low_90,
          rr_low_p95 = rr_low_95,
          rr_high_p25 = rr_high_25,
          rr_high_p50 = rr_high_50,
          rr_high_p75 = rr_high_75,
          rr_high_p90 = rr_high_90,
          rr_high_p95 = rr_high_95,
          af_p25 = af_25,
          af_p50 = af_50,
          af_p75 = af_75,
          af_p90 = af_90,
          af_p95 = af_95,
          af_p25_low = min(af_low_25, af_high_25, na.rm = TRUE),
          af_p50_low = min(af_low_50, af_high_50, na.rm = TRUE),
          af_p75_low = min(af_low_75, af_high_75, na.rm = TRUE),
          af_p90_low = min(af_low_90, af_high_90, na.rm = TRUE),
          af_p95_low = min(af_low_95, af_high_95, na.rm = TRUE),
          af_p25_high = max(af_low_25, af_high_25, na.rm = TRUE),
          af_p50_high = max(af_low_50, af_high_50, na.rm = TRUE),
          af_p75_high = max(af_low_75, af_high_75, na.rm = TRUE),
          af_p90_high = max(af_low_90, af_high_90, na.rm = TRUE),
          af_p95_high = max(af_low_95, af_high_95, na.rm = TRUE),
          af_p25_se = calc_se_from_ci(min(af_low_25, af_high_25, na.rm = TRUE), max(af_low_25, af_high_25, na.rm = TRUE)),
          af_p50_se = calc_se_from_ci(min(af_low_50, af_high_50, na.rm = TRUE), max(af_low_50, af_high_50, na.rm = TRUE)),
          af_p75_se = calc_se_from_ci(min(af_low_75, af_high_75, na.rm = TRUE), max(af_low_75, af_high_75, na.rm = TRUE)),
          af_p90_se = calc_se_from_ci(min(af_low_90, af_high_90, na.rm = TRUE), max(af_low_90, af_high_90, na.rm = TRUE)),
          af_p95_se = calc_se_from_ci(min(af_low_95, af_high_95, na.rm = TRUE), max(af_low_95, af_high_95, na.rm = TRUE)),
          n_exposure = length(cehwi_positive)
        )
      } else {
        NULL
      }
    }, error = function(e) {
      cat("    ⚠ AF计算失败: ", conditionMessage(e), "\n")
      NULL
    })

    full_cb_index <- grep("^cb_cehwi", names(coef(m_gam)))
    full_cb_coef <- coef(m_gam)[full_cb_index]
    full_cb_vcov <- vcov(m_gam)[full_cb_index, full_cb_index, drop = FALSE]
    save_full_attribution <- tolower(Sys.getenv(
      "DLNM_SAVE_FULL_ATTRIBUTION",
      unset = "0"
    )) %in% c("1", "true", "yes", "y", "on")
    daily_attribution_data <- if (save_full_attribution) {
      data.frame(
        date = df_model$date,
        fish_id = as.character(df_model$fish_id),
        lag_group = as.character(df_model$lag_group),
        trip_count = df_model$trip_count,
        exposure = df_model$cehwi
      )
    } else {
      NULL
    }
    
    return(list(
      model = m_gam,
      crosspred = cp,
      pred_df = pred_df,
      coef = cb_coef,
      vcov = cb_vcov,
      cb = reduced_basis,
      coef_type = "overall_cumulative_reduced",
      crossreduce = reduced_overall,
      city = city_name,
      indicator = indicator,
      n_obs = nrow(df_model),
      n_grids = n_distinct(df_model$fish_id),
      cehwi_range = range(df_model$cehwi, na.rm = TRUE),
      cehwi_data = df_model$cehwi[df_model$cehwi > 0],  # 【V6修复】保存暴露数据供AF和直方图使用
      cehwi_knots = cehwi_knots,
      cehwi_boundary_knots = cehwi_boundary_knots,
      cehwi_basis_version = "city_specific_positive_p50_p90_v1",
      full_cb_coef = full_cb_coef,
      full_cb_vcov = full_cb_vcov,
      full_cb_argvar = list(
        fun = "ns",
        knots = cehwi_knots,
        Boundary.knots = cehwi_boundary_knots
      ),
      full_cb_arglag = list(fun = "ns", df = 3),
      daily_attribution_data = daily_attribution_data,
      max_lag = MAX_LAG,
      doy_spline_k = DOY_SPLINE_K,
      time_control_spec = TIME_CONTROL_SPEC,
      time_df_per_year = TIME_DF_PER_YEAR,
      observed_years = observed_years,
      calendar_time_total_df = ifelse(
        TIME_CONTROL_SPEC == "continuous_time_ns",
        calendar_time_df,
        NA_integer_
      ),
      weather_spline_spec = ifelse(
        TIME_CONTROL_SPEC == "continuous_time_ns",
        paste0("natural_cubic_spline_df", WEATHER_NS_DF),
        "penalized_cubic_regression_spline_k3"
      ),
      precipitation_ns_knots = if (exists("precip_ns_knots", inherits = FALSE)) precip_ns_knots else NA_real_,
      precipitation_ns_boundary = if (exists("precip_ns_boundary", inherits = FALSE)) precip_ns_boundary else NA_real_,
      wind_ns_knots = if (exists("wind_ns_knots", inherits = FALSE)) wind_ns_knots else NA_real_,
      wind_ns_boundary = if (exists("wind_ns_boundary", inherits = FALSE)) wind_ns_boundary else NA_real_,
      includes_precipitation_control = includes_precipitation_control,
      includes_wind_speed_control = includes_wind_speed_control,
      fitted_formula = formula_str,
      temperature_control_col = temp_control_col,
      temperature_control_knots = temp_control_knots,
      temperature_control_version = TEMPERATURE_CONTROL_VERSION,
      includes_temperature_control = !is.null(cb_temp),
      temperature_control_contrast = temperature_control_contrast,
      lag_scenario = LAG_SCENARIO_KEY,
      lag_scenario_label = LAG_SCENARIO_LABEL,
      lag_days = LAG_DAYS_CURRENT,
      lag_profile_percentile = LAG_PROFILE_EXPOSURE_PERCENTILE,
      lag_profile_value = lag_profile_value,
      coef_lag_p90 = lag_coef_p90,
      vcov_lag_p90 = lag_vcov_p90,
      cb_lag_p90 = lag_basis_p90,
      lag_profile_type = "var_reduced_city_p90",
      summary = s,
      r2 = r2,
      dev_explained = dev_explained,
      smooth_p = smooth_p,
      overall_wald_chisq = overall_wald_chisq,
      overall_wald_df = overall_wald_df,
      overall_wald_p = overall_wald_p,
      effect_pct = effect_pct,
      all_coefs = all_model_coefs,  # 所有系数（全局+格子特定）
      socioecon_coefs = socioecon_coefs_df,  # 【V3新增】社会经济变量系数
      af = af_city  # 【V7新增】城市级归因分数
    ))
    
  }, error = function(e) {
    cat("  ✗ 模型失败:", conditionMessage(e), "\n")
    return(NULL)
  })
}

# ========== 辅助函数: 从RDS文件加载第一阶段结果 ==========

slim_stage1_result <- function(result) {
  if (is.null(result)) return(NULL)
  result$model <- NULL
  result$crosspred <- NULL
  result$crossreduce <- NULL
  result$summary <- NULL
  result
}

upgrade_stage1_result_to_reduced <- function(model_result) {
  if (is.null(model_result)) return(model_result)
  if (identical(model_result$coef_type, "overall_cumulative_reduced")) return(model_result)
  if (is.null(model_result$model) || is.null(model_result$cb)) return(model_result)
  
  reduced_overall <- tryCatch(
    crossreduce(model_result$cb, model = model_result$model, type = "overall", cen = REFERENCE_CEHWI),
    error = function(e) NULL
  )
  if (is.null(reduced_overall)) return(model_result)
  
  reduced_coef <- coef(reduced_overall)
  reduced_vcov <- vcov(reduced_overall)
  if (is.null(names(reduced_coef)) || any(is.na(names(reduced_coef))) || any(names(reduced_coef) == "")) {
    names(reduced_coef) <- paste0("overall_spline", seq_along(reduced_coef))
  }
  
  basis_x <- model_result$cehwi_data
  if (is.null(basis_x) || length(basis_x) < 2) {
    basis_x <- seq(model_result$cehwi_range[1], model_result$cehwi_range[2], length.out = 100)
  }
  knots_x <- model_result$cehwi_knots
  if (is.null(knots_x) || length(knots_x) == 0 || any(!is.finite(knots_x))) {
    knots_x <- quantile(basis_x[basis_x > 0], probs = c(0.5, 0.9), na.rm = TRUE)
  }
  
  reduced_basis <- tryCatch(
    onebasis(basis_x, fun = "ns", knots = knots_x),
    error = function(e) NULL
  )
  if (is.null(reduced_basis)) return(model_result)
  colnames(reduced_basis) <- names(reduced_coef)
  
  model_result$coef <- reduced_coef
  model_result$vcov <- reduced_vcov
  model_result$cb <- reduced_basis
  model_result$crossreduce <- reduced_overall
  model_result$coef_type <- "overall_cumulative_reduced"
  model_result
}

load_stage1_results <- function(output_dir, indicator, activity_mode = get0("STAGE1_ACTIVITY_MODE", ifnotfound = "combined")) {
  # 从保存的RDS文件中加载第一阶段的城市级结果
  # 用于第二阶段独立运行时
  
  cat("\n从RDS文件加载第一阶段结果...\n")
  
  city_results <- list()
  
  # 遍历所有城市文件夹
  city_dirs <- list.dirs(output_dir, recursive = FALSE, full.names = TRUE)
  
  for (city_dir in city_dirs) {
    dir_name <- basename(city_dir)
    
    # 检查是否是该指标的文件夹（例如：Chicago_cehwi）
    if (!grepl(paste0("_", indicator, "$"), dir_name)) {
      next
    }
    
    # 提取城市代码（例如：Chicago）
    city_code <- str_replace(dir_name, paste0("_", indicator), "")
    
    # 查找所有模型RDS文件
    rds_files <- list.files(city_dir, pattern = "_DLNM_result\\.rds$", full.names = TRUE)
    
    if (length(rds_files) == 0) {
      next
    }
    
    # 加载该城市的所有模型
    city_models <- list()
    for (rds_file in rds_files) {
      # 提取模型类型（例如：Chicago_composite_DLNM_result.rds -> composite）
      file_name <- basename(rds_file)
      model_name <- file_name
      model_name <- str_replace(model_name, paste0("^", fixed(city_code), "_"), "")
      model_name <- str_replace(model_name, "_DLNM_result\\.rds$", "")
      model_name <- normalize_stage1_model_name(model_name)
      if (!model_name %in% stage1_model_types(activity_mode)) {
        next
      }
      
      # 标准化模型名称
      model_name <- normalize_stage1_model_name(model_name)
      
      # 读取RDS文件
      tryCatch({
        model_result <- read_rds(rds_file)
        model_result <- upgrade_stage1_result_to_reduced(model_result)
        model_result$city_code <- city_code
        model_result$indicator <- indicator
        model_result$model_name <- model_name
        model_result$model_type <- model_base_type(model_name)
        model_result$activity_type <- model_activity_type(model_name)
        model_result$activity_label <- unname(activity_modality_labels()[model_result$activity_type])
        model_result$activity_analysis_mode <- activity_analysis_mode_label(activity_mode)
        if (!is_current_stage1_result(model_result)) {
          cat("  Warning: skip stale RDS (expected lag scenario=", LAG_SCENARIO_LABEL,
              ", city MAX_LAG=", expected_stage1_max_lag(model_result), "): ",
              city_code, "-", model_name, "\n", sep = "")
          next
        }
        if (!identical(model_result$coef_type, "overall_cumulative_reduced")) {
          cat("  ⚠ 跳过:", city_code, "-", model_name,
              "（旧RDS无法升级为crossreduce overall reduced coefficients；请重跑第一阶段）\n")
          next
        }
        city_models[[model_name]] <- model_result
        cat("  ✓ 加载:", city_code, "-", model_name, "\n")
      }, error = function(e) {
        cat("  ✗ 加载失败:", rds_file, "-", conditionMessage(e), "\n")
      })
    }
    
    # 保存到总列表
    if (length(city_models) > 0) {
      key <- paste0(city_code, "_", indicator)
      city_results[[key]] <- city_models
    }
  }
  
  cat("\n✓ 加载完成:", length(city_results), "个城市\n")
  
  if (exists("CITY_LIST", inherits = TRUE)) {
    loaded_city_codes <- unique(str_replace(names(city_results), paste0("_", indicator, "$"), ""))
    missing_city_codes <- setdiff(CITY_LIST, loaded_city_codes)
    if (length(missing_city_codes) > 0) {
      missing_file <- file.path(output_dir, paste0("stage1_missing_cities_", indicator, "_", LAG_SCENARIO_LABEL, ".csv"))
      missing_df <- tibble(
        indicator = indicator,
        lag_scenario = LAG_SCENARIO_LABEL,
        city = missing_city_codes,
        expected_city_count = length(CITY_LIST),
        loaded_city_count = length(loaded_city_codes),
        note = "No current-stage DLNM result RDS loaded. Check city log for no heatwave exposure, model failure, or stale lag scenario."
      )
      safe_write_csv(missing_df, missing_file, label = paste0("missing stage1 cities for ", indicator))
      cat("  Missing expected city/cities for ", toupper(indicator), ": ",
          paste(missing_city_codes, collapse = ", "),
          " (details saved to ", missing_file, ")\n", sep = "")
    }
  }
  
  return(city_results)
}

split_successful_cities_by_indicator <- function(successful_city_results) {
  if (is.null(successful_city_results) || length(successful_city_results) == 0) {
    return(list(cehwi = list(), exceeded_quantity = list()))
  }
  
  list(
    cehwi = successful_city_results[grepl("_cehwi$", names(successful_city_results))],
    exceeded_quantity = successful_city_results[grepl("_exceeded_quantity$", names(successful_city_results))]
  )
}

get_stage1_results_by_indicator <- function(successful_city_results, output_dir) {
  split_results <- split_successful_cities_by_indicator(successful_city_results)
  
  successful_cities_cehwi <- split_results$cehwi
  successful_cities_exceeded <- split_results$exceeded_quantity
  
  if (length(successful_cities_cehwi) == 0) {
    successful_cities_cehwi <- load_stage1_results(output_dir, "cehwi")
  }
  if (length(successful_cities_exceeded) == 0) {
    successful_cities_exceeded <- load_stage1_results(output_dir, "exceeded_quantity")
  }
  
  list(
    cehwi = successful_cities_cehwi,
    exceeded_quantity = successful_cities_exceeded
  )
}

calculate_gini <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x) & !is.na(x)]
  if (length(x) < 2) return(NA_real_)
  if (any(x < 0, na.rm = TRUE)) x <- x - min(x, na.rm = TRUE)
  if (sum(x, na.rm = TRUE) <= 0) return(0)
  x <- sort(x)
  n <- length(x)
  sum((2 * seq_len(n) - n - 1) * x) / (n * sum(x))
}

standardize_covariate_columns <- function(df, vars_to_scale) {
  for (var in vars_to_scale) {
    if (!var %in% names(df)) next
    q99 <- suppressWarnings(quantile(df[[var]], 0.99, na.rm = TRUE))
    q01 <- suppressWarnings(quantile(df[[var]], 0.01, na.rm = TRUE))
    if (is.finite(q01) && is.finite(q99) && q99 >= q01) {
      df[[var]] <- pmin(pmax(df[[var]], q01), q99)
    }
    scaled_name <- paste0(var, "_scaled")
    if (is.finite(sd(df[[var]], na.rm = TRUE)) && sd(df[[var]], na.rm = TRUE) > 0) {
      df[[scaled_name]] <- as.numeric(scale(df[[var]])[, 1])
    } else {
      df[[scaled_name]] <- NA_real_
    }
  }
  df
}

build_gini_city_covariates_table <- function(vars_used = SOCIOECON_VARS_USED) {
  if (!exists("df_socioecon_grid_global") || is.null(df_socioecon_grid_global) ||
      nrow(df_socioecon_grid_global) == 0 || length(vars_used) == 0) {
    return(NULL)
  }
  
  gini_vars <- intersect(intersect(vars_used, META_PREDICTOR_BASES_FINAL), names(df_socioecon_grid_global))
  if (length(gini_vars) == 0) return(NULL)
  
  gini_city <- df_socioecon_grid_global %>%
    group_by(city) %>%
    summarise(across(all_of(gini_vars), calculate_gini), .groups = "drop")
  
  names(gini_city)[names(gini_city) %in% gini_vars] <- paste0(gini_vars, "_gini")
  gini_scaled_vars <- paste0(gini_vars, "_gini")
  gini_city <- standardize_covariate_columns(gini_city, gini_scaled_vars)
  
  city_covariates <- gini_city %>%
    select(city, all_of(paste0(gini_scaled_vars, "_scaled"))) %>%
    distinct(city, .keep_all = TRUE)
  
  names(city_covariates)[-1] <- sub("_scaled$", "_mean", names(city_covariates)[-1])
  city_covariates <- bind_cols(city_covariates["city"], drop_excluded_meta_predictors(city_covariates[, -1, drop = FALSE]))
  as.data.frame(city_covariates)
}

build_city_covariates_table <- function(df_socioecon_city, vars_used = SOCIOECON_VARS_USED, mode = META_PREDICTOR_MODE) {
  if (is.null(df_socioecon_city) || nrow(df_socioecon_city) == 0 || length(vars_used) == 0) {
    return(NULL)
  }
  
  if (identical(mode, "gini")) {
    gini_covariates <- build_gini_city_covariates_table(vars_used)
    if (!is.null(gini_covariates) && nrow(gini_covariates) > 0) {
      cat("    【Meta-predictors】使用城市内1km网格Gini系数（已标准化）\n")
      return(gini_covariates)
    }
    cat("    Warning: Gini meta-predictors are unavailable; skipping Gini mode instead of silently using city means.\n")
    return(NULL)
    cat("    ⚠ Gini meta-predictors不可用，回退到城市均值\n")
    return(NULL)
  }
  
  vars_used <- intersect(vars_used, META_PREDICTOR_BASES_FINAL)
  scaled_cols <- paste0(vars_used, "_scaled")
  scaled_cols <- intersect(scaled_cols, names(df_socioecon_city))
  
  if (length(scaled_cols) == 0) {
    return(NULL)
  }
  
  city_covariates <- df_socioecon_city %>%
    select(city, all_of(scaled_cols)) %>%
    distinct(city, .keep_all = TRUE)
  
  names(city_covariates)[-1] <- sub("_scaled$", "_mean", names(city_covariates)[-1])
  
  if (FALSE && "Unemployed_Population_scaled" %in% names(df_socioecon_city)) {
    unemployed_proxy <- df_socioecon_city %>%
      select(city, Unemployed_Population_scaled) %>%
      distinct(city, .keep_all = TRUE) %>%
      rename(unemployed_pop_mean = Unemployed_Population_scaled)
    
    city_covariates <- city_covariates %>%
      left_join(unemployed_proxy, by = "city")
  }
  
  city_covariates <- bind_cols(city_covariates["city"], drop_excluded_meta_predictors(city_covariates[, -1, drop = FALSE]))
  as.data.frame(city_covariates)
}

calc_se_from_ci <- function(ci_low, ci_high) {
  ifelse(
    is.finite(ci_low) & is.finite(ci_high) & ci_high >= ci_low,
    abs(ci_high - ci_low) / 3.92,
    NA_real_
  )
}

fit_univariate_meta_af <- function(af_df) {
  af_valid <- af_df %>%
    filter(is.finite(af_pct), is.finite(af_se), af_se > 0)
  
  if (nrow(af_valid) < 3) {
    return(NULL)
  }
  
  yi <- matrix(af_valid$af_pct, ncol = 1)
  S_list <- lapply(af_valid$af_se^2, function(v) matrix(v, nrow = 1, ncol = 1))
  method_used <- "reml"
  
  meta_model <- tryCatch(
    mvmeta(yi, S = S_list, method = "reml"),
    error = function(e) NULL
  )
  
  if (is.null(meta_model)) {
    method_used <- "fixed"
    meta_model <- tryCatch(
      mvmeta(yi, S = S_list, method = "fixed"),
      error = function(e) NULL
    )
  }
  
  if (is.null(meta_model)) {
    return(NULL)
  }
  
  pooled_af <- as.numeric(coef(meta_model))[1]
  pooled_se <- sqrt(diag(vcov(meta_model)))[1]
  pooled_low <- pooled_af - 1.96 * pooled_se
  pooled_high <- pooled_af + 1.96 * pooled_se
  
  qstat <- tryCatch(summary(meta_model)$qstat, error = function(e) NULL)
  
  list(
    model = meta_model,
    data = af_valid,
    summary = data.frame(
      pooled_af = pooled_af,
      pooled_se = pooled_se,
      pooled_low = pooled_low,
      pooled_high = pooled_high,
      n_cities = nrow(af_valid),
      method = method_used,
      q_statistic = if (!is.null(qstat) && length(qstat$Q) > 0) qstat$Q[1] else NA_real_,
      q_df = if (!is.null(qstat) && length(qstat$df) > 0) qstat$df[1] else NA_real_,
      q_pvalue = if (!is.null(qstat) && length(qstat$pvalue) > 0) qstat$pvalue[1] else NA_real_
    )
  )
}

derive_lag_profile_at_exposure <- function(model_result, exposure_value, cen = REFERENCE_CEHWI) {
  if (is.null(model_result) ||
      is.null(model_result$coef) ||
      is.null(model_result$vcov) ||
      is.null(model_result$cehwi_knots) ||
      is.null(model_result$cehwi_range) ||
      !is.finite(exposure_value)) {
    return(NULL)
  }
  
  beta <- suppressWarnings(as.numeric(model_result$coef))
  V <- model_result$vcov
  if (!is.matrix(V) || length(beta) == 0 || any(!is.finite(beta)) || any(!is.finite(V))) {
    return(NULL)
  }
  
  var_dim <- if (!is.null(model_result$cb) && is.matrix(model_result$cb)) {
    ncol(model_result$cb)
  } else {
    length(model_result$cehwi_knots) + 1
  }
  if (!is.finite(var_dim) || var_dim <= 0 || length(beta) %% var_dim != 0) {
    return(NULL)
  }
  lag_dim <- length(beta) / var_dim
  if (lag_dim < 2 || any(dim(V) != c(length(beta), length(beta)))) {
    return(NULL)
  }
  
  exposure_range <- suppressWarnings(as.numeric(model_result$cehwi_range))
  if (length(exposure_range) != 2 || any(!is.finite(exposure_range)) || diff(exposure_range) <= 0) {
    return(NULL)
  }
  
  exposure_value <- max(min(exposure_value, exposure_range[2]), exposure_range[1])
  basis_value <- tryCatch(
    onebasis(exposure_value, fun = "ns", knots = model_result$cehwi_knots, Boundary.knots = exposure_range),
    error = function(e) NULL
  )
  basis_cen <- tryCatch(
    onebasis(max(min(cen, exposure_range[2]), exposure_range[1]), fun = "ns", knots = model_result$cehwi_knots, Boundary.knots = exposure_range),
    error = function(e) NULL
  )
  if (is.null(basis_value) || is.null(basis_cen) ||
      ncol(basis_value) != var_dim || ncol(basis_cen) != var_dim) {
    return(NULL)
  }
  
  var_contrast <- as.numeric(basis_value - basis_cen)
  transform_matrix <- matrix(0, nrow = lag_dim, ncol = length(beta))
  for (lag_i in seq_len(lag_dim)) {
    idx <- ((seq_len(var_dim) - 1) * lag_dim) + lag_i
    transform_matrix[lag_i, idx] <- var_contrast
  }
  
  lag_coef <- as.numeric(transform_matrix %*% beta)
  lag_vcov <- transform_matrix %*% V %*% t(transform_matrix)
  names(lag_coef) <- paste0("lag_spline", seq_along(lag_coef))
  colnames(lag_vcov) <- rownames(lag_vcov) <- names(lag_coef)
  
  list(
    coef = lag_coef,
    vcov = lag_vcov,
    exposure_value = exposure_value,
    source = "group_p90_projected_from_original_crossbasis"
  )
}

save_pooled_lag_response <- function(model_results,
                                     output_dir,
                                     indicator,
                                     model_type,
                                     group_label = "National",
                                     reliability = NA_character_) {
  if (is.null(model_results) || length(model_results) < 3) return(FALSE)
  
  target_exposure_values <- unlist(lapply(model_results, function(x) {
    if (is.null(x$cehwi_data)) return(NULL)
    x$cehwi_data[is.finite(x$cehwi_data) & x$cehwi_data > REFERENCE_CEHWI]
  }))
  target_exposure <- if (length(target_exposure_values) > 0) {
    as.numeric(quantile(target_exposure_values, probs = LAG_PROFILE_EXPOSURE_PERCENTILE, na.rm = TRUE))
  } else {
    NA_real_
  }
  
  projected_profiles <- lapply(model_results, derive_lag_profile_at_exposure, exposure_value = target_exposure)
  projected_idx <- vapply(projected_profiles, function(x) {
    !is.null(x) &&
      length(x$coef) > 0 &&
      all(is.finite(x$coef)) &&
      is.matrix(x$vcov) &&
      all(is.finite(x$vcov))
  }, logical(1))
  
  if (sum(projected_idx) >= 3) {
    lag_results <- model_results[projected_idx]
    lag_coef_list <- lapply(projected_profiles[projected_idx], `[[`, "coef")
    lag_vcov_list <- lapply(projected_profiles[projected_idx], `[[`, "vcov")
    lag_profile_values <- vapply(projected_profiles[projected_idx], `[[`, numeric(1), "exposure_value")
    lag_profile_source <- "group_p90_projected_from_original_crossbasis"
  } else {
    valid_idx <- vapply(model_results, function(x) {
      !is.null(x$coef_lag_p90) &&
        !is.null(x$vcov_lag_p90) &&
        length(x$coef_lag_p90) > 0 &&
        all(is.finite(x$coef_lag_p90)) &&
        is.matrix(x$vcov_lag_p90)
    }, logical(1))
    lag_results <- model_results[valid_idx]
    lag_coef_list <- lapply(lag_results, function(x) x$coef_lag_p90)
    lag_vcov_list <- lapply(lag_results, function(x) x$vcov_lag_p90)
    lag_profile_values <- suppressWarnings(as.numeric(vapply(
      lag_results,
      function(x) ifelse(is.null(x$lag_profile_value), NA_real_, x$lag_profile_value),
      numeric(1)
    )))
    lag_profile_source <- "city_p90_crossreduce_fallback"
  }
  
  if (length(lag_results) < 3) {
    writeLines(
      c(
        "Pooled lag-response was skipped.",
        "Reason: fewer than 3 cities had valid lag-profile coefficients.",
        "Fix: rerun Stage-1 with the current script so original cross-basis coefficients and lag-profile coefficients are saved."
      ),
      file.path(output_dir, "pooled_lag_response_p90_status.txt")
    )
    return(FALSE)
  }
  
  coef_lengths <- vapply(lag_coef_list, length, integer(1))
  common_length <- as.integer(names(sort(table(coef_lengths), decreasing = TRUE)[1]))
  keep_common <- coef_lengths == common_length
  lag_results <- lag_results[keep_common]
  lag_coef_list <- lag_coef_list[keep_common]
  lag_vcov_list <- lag_vcov_list[keep_common]
  lag_profile_values <- lag_profile_values[keep_common]
  if (length(lag_results) < 3) return(FALSE)
  
  lag_coef_matrix <- do.call(rbind, lag_coef_list)
  rownames(lag_coef_matrix) <- names(lag_results)
  
  for (i in seq_along(lag_vcov_list)) {
    V <- lag_vcov_list[[i]]
    if (!is.matrix(V) || any(!is.finite(V)) || any(dim(V) != c(common_length, common_length))) {
      lag_vcov_list[[i]] <- diag(1000, common_length)
    }
  }
  
  lag_meta <- tryCatch(
    mvmeta(lag_coef_matrix, S = lag_vcov_list, method = "reml"),
    error = function(e) NULL
  )
  method_used <- "reml"
  if (is.null(lag_meta)) {
    max_eigs <- sapply(lag_vcov_list, function(V) {
      tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
    })
    reg_strength <- max(0.001, max(max_eigs, na.rm = TRUE) * 0.05)
    lag_vcov_reg <- lapply(lag_vcov_list, function(V) V + diag(reg_strength, nrow(V)))
    lag_meta <- tryCatch(
      mvmeta(lag_coef_matrix, S = lag_vcov_reg, method = "reml"),
      error = function(e) NULL
    )
    method_used <- "reml_regularized"
  }
  if (is.null(lag_meta)) {
    lag_meta <- tryCatch(mvmeta(lag_coef_matrix, method = "fixed"), error = function(e) NULL)
    method_used <- "fixed"
  }
  if (is.null(lag_meta)) {
    writeLines(
      c("Pooled lag-response was skipped.", "Reason: mvmeta failed for lag-profile coefficients."),
      file.path(output_dir, "pooled_lag_response_p90_status.txt")
    )
    return(FALSE)
  }
  
  pooled_coef <- as.numeric(coef(lag_meta))
  pooled_vcov <- vcov(lag_meta)
  lags <- 0:MAX_LAG
  lag_basis <- tryCatch(
    onebasis(lags, fun = "ns", df = length(pooled_coef), intercept = TRUE),
    error = function(e) NULL
  )
  if (is.null(lag_basis) || ncol(lag_basis) != length(pooled_coef)) {
    lag_basis <- tryCatch(
      onebasis(lags, fun = "ns", df = length(pooled_coef)),
      error = function(e) NULL
    )
  }
  if (is.null(lag_basis) || ncol(lag_basis) != length(pooled_coef)) {
    writeLines(
      c("Pooled lag-response was skipped.", "Reason: could not rebuild the lag spline basis."),
      file.path(output_dir, "pooled_lag_response_p90_status.txt")
    )
    return(FALSE)
  }
  
  eta <- as.numeric(lag_basis %*% pooled_coef)
  eta_var <- rowSums((lag_basis %*% pooled_vcov) * lag_basis)
  eta_se <- sqrt(pmax(eta_var, 0))
  lag_df <- data.frame(
    group_label = group_label,
    indicator = indicator,
    model_type = model_type,
    lag = lags,
    rr = exp(eta),
    rr_low = exp(eta - 1.96 * eta_se),
    rr_high = exp(eta + 1.96 * eta_se),
    log_rr = eta,
    log_rr_se = eta_se,
    n_cities = length(lag_results),
    lag_profile_percentile = LAG_PROFILE_EXPOSURE_PERCENTILE,
    lag_profile_value_target = target_exposure,
    lag_profile_value_mean = mean(lag_profile_values, na.rm = TRUE),
    lag_profile_value_median = median(lag_profile_values, na.rm = TRUE),
    lag_profile_source = lag_profile_source,
    meta_method = method_used,
    reliability = reliability,
    stringsAsFactors = FALSE
  )
  
  safe_write_csv(lag_df, file.path(output_dir, "pooled_lag_response_p90.csv"), label = "pooled lag-response p90")
  
  y_fit_range <- range(lag_df$rr, 1, na.rm = TRUE)
  y_pad <- max(0.05, diff(y_fit_range) * 0.25)
  y_limits <- c(max(0, y_fit_range[1] - y_pad), y_fit_range[2] + y_pad)
  lag_df_plot <- lag_df %>%
    mutate(
      rr_low_clipped = pmax(rr_low, y_limits[1]),
      rr_high_clipped = pmin(rr_high, y_limits[2])
    )
  
  p_lag <- ggplot(lag_df_plot, aes(x = lag, y = rr)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.7) +
    geom_ribbon(aes(ymin = rr_low_clipped, ymax = rr_high_clipped), fill = "gray75", alpha = 0.45) +
    geom_line(color = stage1_model_color(model_type), linewidth = 1.5) +
    geom_point(color = stage1_model_color(model_type), size = 2.3) +
    coord_cartesian(ylim = y_limits) +
    labs(
      title = paste0("Pooled Lag-Response at p", round(100 * LAG_PROFILE_EXPOSURE_PERCENTILE), " Exposure"),
      subtitle = paste0(
        group_label, " | ", toupper(indicator), " - ", toupper(model_type),
        " | ", length(lag_results), " cities | CI clipped only for display"
      ),
      x = "Lag day",
      y = "Relative Risk (RR)",
      caption = "CSV keeps full CI. Lag profiles use group p90 exposure when original cross-basis coefficients are available; city-p90 profiles are fallback."
    ) +
    rr_plot_theme(13)
  
  ggsave(file.path(output_dir, "pooled_lag_response_p90.png"), p_lag, width = 10.5, height = 6.5, dpi = 300, bg = "white")
  TRUE
}

compute_missing_af_percentile <- function(model_result, prob) {
  pred_df <- model_result$pred_df
  exposure_values <- model_result$cehwi_data
  required_cols <- c("cehwi", "rr", "rr_low", "rr_high")
  if (is.null(pred_df) || !all(required_cols %in% names(pred_df)) || is.null(exposure_values)) {
    return(NULL)
  }
  exposure_positive <- exposure_values[is.finite(exposure_values) & exposure_values > 0]
  if (length(exposure_positive) <= 10) return(NULL)
  
  af_transform <- function(rr_vals) {
    rr_vals <- pmax(0.01, pmin(100, rr_vals))
    af_vals <- (rr_vals - 1) / rr_vals * 100
    af_vals[is.na(af_vals) | is.infinite(af_vals)] <- NA_real_
    pmax(-100, pmin(100, af_vals))
  }
  
  q_value <- as.numeric(quantile(exposure_positive, prob, na.rm = TRUE))
  rr <- approx(pred_df$cehwi, pred_df$rr, xout = q_value, rule = 2)$y
  rr_low <- approx(pred_df$cehwi, pred_df$rr_low, xout = q_value, rule = 2)$y
  rr_high <- approx(pred_df$cehwi, pred_df$rr_high, xout = q_value, rule = 2)$y
  af_value <- af_transform(rr)
  af_low_raw <- af_transform(rr_low)
  af_high_raw <- af_transform(rr_high)
  af_low <- min(af_low_raw, af_high_raw, na.rm = TRUE)
  af_high <- max(af_low_raw, af_high_raw, na.rm = TRUE)
  
  list(
    cehwi = q_value,
    rr = rr,
    rr_low = rr_low,
    rr_high = rr_high,
    af_pct = af_value,
    af_low = af_low,
    af_high = af_high,
    af_se = calc_se_from_ci(af_low, af_high)
  )
}

extract_city_af_records <- function(city_results, indicator) {
  if (is.null(city_results) || length(city_results) == 0) {
    return(data.frame())
  }
  
  percentile_specs <- list(
    overall = list(prefix = "af_overall", cehwi = NA_real_, rr = NA_real_, rr_low = NA_real_, rr_high = NA_real_),
    p25 = list(prefix = "af_p25", cehwi = "cehwi_p25", rr = "rr_p25", rr_low = "rr_low_p25", rr_high = "rr_high_p25"),
    p50 = list(prefix = "af_p50", cehwi = "cehwi_p50", rr = "rr_p50", rr_low = "rr_low_p50", rr_high = "rr_high_p50"),
    p75 = list(prefix = "af_p75", cehwi = "cehwi_p75", rr = "rr_p75", rr_low = "rr_low_p75", rr_high = "rr_high_p75"),
    p90 = list(prefix = "af_p90", cehwi = "cehwi_p90", rr = "rr_p90", rr_low = "rr_low_p90", rr_high = "rr_high_p90"),
    p95 = list(prefix = "af_p95", cehwi = "cehwi_p95", rr = "rr_p95", rr_low = "rr_low_p95", rr_high = "rr_high_p95")
  )
  
  af_records <- list()
  
  for (city_key in names(city_results)) {
    city_result <- city_results[[city_key]]
    
    for (mtype in names(city_result)) {
      model_result <- city_result[[mtype]]
      if (is.null(model_result$af)) next
      
      af_obj <- model_result$af
      city_name_clean <- sub("_(cehwi|exceeded_quantity)$", "", city_key, ignore.case = TRUE)
      
      for (percentile_name in names(percentile_specs)) {
        spec <- percentile_specs[[percentile_name]]
        prefix <- spec$prefix
        af_value <- af_obj[[prefix]]
        af_low <- af_obj[[paste0(prefix, "_low")]]
        af_high <- af_obj[[paste0(prefix, "_high")]]
        af_se <- af_obj[[paste0(prefix, "_se")]]
        fallback_af <- NULL
        
        if ((is.null(af_value) || is.null(af_low) || is.null(af_high)) &&
            percentile_name != "overall") {
          prob_value <- suppressWarnings(as.numeric(str_replace(percentile_name, "^p", "")) / 100)
          fallback_af <- compute_missing_af_percentile(model_result, prob_value)
          if (!is.null(fallback_af)) {
            af_value <- fallback_af$af_pct
            af_low <- fallback_af$af_low
            af_high <- fallback_af$af_high
            af_se <- fallback_af$af_se
          }
        }
        
        if (is.null(af_value) || is.null(af_low) || is.null(af_high)) next
        
        af_records[[length(af_records) + 1]] <- data.frame(
          city = city_name_clean,
          indicator = indicator,
          model_type = mtype,
          base_model_type = model_base_type(mtype),
          activity_type = model_activity_type(mtype),
          activity_label = unname(activity_modality_labels()[model_activity_type(mtype)]),
          percentile = percentile_name,
          cehwi = if (is.character(spec$cehwi) && !is.null(af_obj[[spec$cehwi]])) af_obj[[spec$cehwi]] else if (exists("fallback_af") && !is.null(fallback_af)) fallback_af$cehwi else spec$cehwi,
          rr = if (is.character(spec$rr) && !is.null(af_obj[[spec$rr]])) af_obj[[spec$rr]] else if (exists("fallback_af") && !is.null(fallback_af)) fallback_af$rr else spec$rr,
          rr_low = if (is.character(spec$rr_low) && !is.null(af_obj[[spec$rr_low]])) af_obj[[spec$rr_low]] else if (exists("fallback_af") && !is.null(fallback_af)) fallback_af$rr_low else spec$rr_low,
          rr_high = if (is.character(spec$rr_high) && !is.null(af_obj[[spec$rr_high]])) af_obj[[spec$rr_high]] else if (exists("fallback_af") && !is.null(fallback_af)) fallback_af$rr_high else spec$rr_high,
          af_pct = af_value,
          af_low = af_low,
          af_high = af_high,
          af_se = if (!is.null(af_se)) af_se else calc_se_from_ci(af_low, af_high),
          n_exposure = af_obj$n_exposure,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  
  if (length(af_records) == 0) {
    return(data.frame())
  }
  
  bind_rows(af_records)
}

build_partition_af_df <- function(partition_results, percentile = "overall") {
  prefix <- switch(
    percentile,
    "overall" = "af_overall",
    "p25" = "af_p25",
    "p50" = "af_p50",
    "p75" = "af_p75",
    "p90" = "af_p90",
    "p95" = "af_p95",
    "af_overall"
  )
  
  af_rows <- list()
  
  for (city_name in names(partition_results)) {
    model_result <- partition_results[[city_name]]
    af_obj <- model_result$af
    if (is.null(af_obj)) next
    
    af_value <- af_obj[[prefix]]
    af_low <- af_obj[[paste0(prefix, "_low")]]
    af_high <- af_obj[[paste0(prefix, "_high")]]
    af_se <- af_obj[[paste0(prefix, "_se")]]
    fallback_af <- NULL
    
    if ((is.null(af_value) || is.null(af_low) || is.null(af_high)) &&
        percentile != "overall") {
      prob_value <- suppressWarnings(as.numeric(str_replace(percentile, "^p", "")) / 100)
      fallback_af <- compute_missing_af_percentile(model_result, prob_value)
      if (!is.null(fallback_af)) {
        af_value <- fallback_af$af_pct
        af_low <- fallback_af$af_low
        af_high <- fallback_af$af_high
        af_se <- fallback_af$af_se
      }
    }
    
    if (is.null(af_value) || is.null(af_low) || is.null(af_high)) next
    
    af_rows[[length(af_rows) + 1]] <- data.frame(
      city = city_name,
      percentile = percentile,
      cehwi = if (!is.null(af_obj[[paste0("cehwi_", percentile)]])) af_obj[[paste0("cehwi_", percentile)]] else if (!is.null(fallback_af)) fallback_af$cehwi else NA_real_,
      rr = if (!is.null(af_obj[[paste0("rr_", percentile)]])) af_obj[[paste0("rr_", percentile)]] else if (!is.null(fallback_af)) fallback_af$rr else NA_real_,
      rr_low = if (!is.null(af_obj[[paste0("rr_low_", percentile)]])) af_obj[[paste0("rr_low_", percentile)]] else if (!is.null(fallback_af)) fallback_af$rr_low else NA_real_,
      rr_high = if (!is.null(af_obj[[paste0("rr_high_", percentile)]])) af_obj[[paste0("rr_high_", percentile)]] else if (!is.null(fallback_af)) fallback_af$rr_high else NA_real_,
      af_pct = af_value,
      af_low = af_low,
      af_high = af_high,
      af_se = if (!is.null(af_se)) af_se else calc_se_from_ci(af_low, af_high),
      n_exposure = af_obj$n_exposure,
      stringsAsFactors = FALSE
    )
  }
  
  if (length(af_rows) == 0) {
    return(data.frame())
  }
  
  bind_rows(af_rows)
}

save_partition_af_percentile_outputs <- function(partition_model_results,
                                                 partition_output_dir,
                                                 partition_label,
                                                 partition_family,
                                                 partition_name,
                                                 mode = NULL) {
  if (is.null(partition_model_results) || length(partition_model_results) == 0) return(FALSE)
  
  mode_label <- meta_predictor_mode_label(mode)
  mode_suffix <- analysis_output_suffix(mode)
  partition_af_dir <- file.path(partition_output_dir, paste0("AF_PERCENTILE_SUMMARY", mode_suffix))
  dir.create(partition_af_dir, showWarnings = FALSE, recursive = TRUE)
  
  percentile_levels <- c("overall", "p25", "p50", "p75", "p90", "p95")
  city_af_rows <- list()
  pooled_af_rows <- list()
  
  for (ind in c("cehwi", "exceeded_quantity")) {
    if (is.null(partition_model_results[[ind]])) next
    for (mtype in STAGE1_MODEL_TYPES) {
      partition_results <- partition_model_results[[ind]][[mtype]]
      if (is.null(partition_results) || length(partition_results) < 3) next
      
      for (pct in percentile_levels) {
        af_df <- build_partition_af_df(partition_results, percentile = pct)
        if (nrow(af_df) < 3) next
        
        af_df <- af_df %>%
          mutate(
            partition_family = partition_family,
            partition_name = partition_name,
            partition_label = partition_label,
            indicator = ind,
            model_type = mtype,
            p_value = 2 * pnorm(-abs(af_pct / pmax(af_se, 1e-10))),
            star_label = forest_significance_stars(p_value),
            direction = forest_effect_direction(af_pct, neutral_zero = TRUE),
            meta_predictor_mode = mode_label
          )
        city_af_rows[[length(city_af_rows) + 1]] <- af_df
        
        af_meta <- fit_univariate_meta_af(af_df)
        if (is.null(af_meta)) next
        
        pooled_row <- af_meta$summary[1, , drop = FALSE] %>%
          mutate(
            partition_family = partition_family,
            partition_name = partition_name,
            partition_label = partition_label,
            indicator = ind,
            model_type = mtype,
            percentile = pct
          )
        pooled_af_rows[[length(pooled_af_rows) + 1]] <- pooled_row
      }
    }
  }
  
  if (length(city_af_rows) == 0 || length(pooled_af_rows) == 0) {
    writeLines(
      c(
        "Partition AF percentile summary was skipped.",
        "Reason: fewer than 3 valid city-level AF estimates were available after filtering."
      ),
      file.path(partition_af_dir, "AF_percentile_summary_status.txt")
    )
    return(FALSE)
  }
  
  partition_city_af <- bind_rows(city_af_rows)
  partition_pooled_af <- bind_rows(pooled_af_rows) %>%
    add_pooled_af_significance(mode = mode_label)
  
  safe_write_csv(
    partition_city_af,
    file.path(partition_af_dir, paste0("PARTITION_ALL_CITIES_AF", mode_suffix, ".csv")),
    label = "partition city AF percentile data"
  )
  safe_write_csv(
    partition_pooled_af,
    file.path(partition_af_dir, paste0("PARTITION_POOLED_AF_SUMMARY", mode_suffix, ".csv")),
    label = "partition pooled AF percentile summary"
  )
  
  save_pooled_af_comparison_plot(partition_pooled_af, partition_af_dir, mode = mode_label)
  try(
    save_activity_pooled_rr_overlay(
      parent_dir = partition_output_dir,
      output_dir = partition_af_dir,
      file_prefix = "PARTITION_activity_3plus1_pooled_RR_overlay",
      title_prefix = partition_label,
      mode = mode_label,
      model_results_by_indicator = partition_model_results
    ),
    silent = TRUE
  )
  cat("          ✓ AF percentile summary saved: ", partition_af_dir, "\n", sep = "")
  TRUE
}

plot_partition_af_results <- function(partition_results, partition_meta_dir, partition_label, indicator, model_type, percentile = "overall") {
  partition_af <- build_partition_af_df(partition_results, percentile = percentile)
  
  if (nrow(partition_af) < 3) {
    cat("          ⚠ AF数据不足（<3城市），跳过AF森林图\n")
    return(NULL)
  }
  
  partition_af <- partition_af %>%
    arrange(desc(af_pct)) %>%
    mutate(
      city = factor(city, levels = city),
      p_value = 2 * pnorm(-abs(af_pct / pmax(af_se, 1e-10))),
      direction = forest_effect_direction(af_pct, neutral_zero = TRUE),
      star_label = forest_significance_stars(p_value)
    )
  
  af_meta <- fit_univariate_meta_af(partition_af)
  subtitle_text <- paste0(toupper(indicator), " - ", toupper(model_type), " (", nrow(partition_af), " cities)")
  
  if (!is.null(af_meta)) {
    af_meta_summary <- af_meta$summary %>%
      mutate(
        partition = partition_label,
        indicator = indicator,
        model_type = model_type,
        percentile = percentile
      )
    safe_write_csv(af_meta_summary, file.path(partition_meta_dir, "AF_meta_summary.csv"), label = "partition AF meta summary")
    subtitle_text <- paste0(
      subtitle_text,
      " | RE pooled AF = ",
      round(af_meta_summary$pooled_af, 2),
      "% (95% CI: ",
      round(af_meta_summary$pooled_low, 2),
      "% - ",
      round(af_meta_summary$pooled_high, 2),
      "%)"
    )
  }
  af_star_nudge <- forest_star_offset(partition_af$af_low, partition_af$af_high, fallback = 0.1)
  subtitle_text <- paste0(
    subtitle_text,
    "\nBlue=Positive, Red=Negative | * p<0.05, ** p<0.01, *** p<0.001"
  )
  
  p_af <- ggplot(partition_af, aes(x = af_pct, y = city)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_errorbarh(
      aes(xmin = af_low, xmax = af_high, color = direction),
      height = 0.22,
      linewidth = 1
    ) +
    geom_point(aes(color = direction), size = 3.5, shape = 19) +
    geom_text(
      data = partition_af %>% filter(nzchar(star_label)),
      aes(label = star_label),
      nudge_x = af_star_nudge,
      nudge_y = 0.22,
      size = 3.2,
      color = "black",
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(values = forest_direction_colors, guide = "none") +
    labs(
      title = paste0("AF - ", partition_label),
      subtitle = subtitle_text,
      x = "Attributable Fraction (%)",
      y = ""
    ) +
    coord_cartesian(clip = "off") +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  ggsave(file.path(partition_meta_dir, "AF_forest.png"), p_af, width = 10, height = max(6, nrow(partition_af) * 0.3), dpi = 300)
  safe_write_csv(partition_af, file.path(partition_meta_dir, "AF_summary.csv"), label = "partition AF city summary")
  cat("          ✓ AF森林图已生成 (", nrow(partition_af), " cities)\n")
  
  invisible(list(data = partition_af, meta = af_meta))
}

# ========== 【V3新增】可视化函数: Nature级别Forest Plot ==========

plot_forest_socioecon <- function(coef_data, output_path, title_text) {
  # Nature级别森林图：社会经济变量 + 时间效应贡献度
  
  if (is.null(coef_data) || nrow(coef_data) == 0) {
    return(NULL)
  }
  
  # 准备数据
  plot_data <- coef_data %>%
    mutate(
      # 按指定顺序设置标签（从上往下）
      variable_label = case_when(
        grepl("BD|Building_Density", variable) ~ "Building Density",
        grepl("FAR", variable) ~ "Floor Area Ratio",
        grepl("NDVI", variable) ~ "NDVI",
        grepl("total_20", variable) ~ "Population (20-55)",
        grepl("GDP", variable) ~ "GDP",
        grepl("Unemployed", variable) ~ "Unemployed Population",  # 【新增】
        grepl("WS_2020", variable) ~ "Wind Speed",
        grepl("Walkability", variable) ~ "Walkability Score",
        grepl("Weekend", variable) ~ "Weekend Effect",
        grepl("Unexplained", variable) ~ "Unexplained Grid Effect",
        TRUE ~ variable
      ),
      # 设置顺序（用于绘图，从上往下）
      plot_order = case_when(
        grepl("BH", variable) ~ 1,
        grepl("FAR", variable) ~ 2,
        grepl("NDVI", variable) ~ 3,
        grepl("total_20", variable) ~ 4,
        grepl("GDP", variable) ~ 5,
        grepl("Unemployed", variable) ~ 6,  # 【新增】失业人口
        grepl("WS_2020", variable) ~ 7,
        grepl("Walkability", variable) ~ 8,
        grepl("Weekend", variable) ~ 9,
        grepl("Unexplained", variable) ~ 10,
        TRUE ~ 99
      ),
      # 特殊处理：Unexplained Grid Effect用灰色，其他按正负
      direction = case_when(
        grepl("Unexplained", variable) ~ "Neutral",
        coefficient > 0 ~ "Positive",
        coefficient < 0 ~ "Negative",
        TRUE ~ "Neutral"
      ),
      star_label = case_when(
        grepl("Unexplained", variable) ~ "",
        TRUE ~ forest_significance_stars(p_value)
      )
    ) %>%
    arrange(plot_order)  # 按预定义顺序排列
  star_nudge_socioecon <- forest_star_offset(plot_data$ci_low, plot_data$ci_high, fallback = 0.1)
  
  # 【新增】检测极端值并警告
  extreme_coefs <- plot_data %>% filter(abs(coefficient) > 50 | abs(ci_high - ci_low) > 500)
  if (nrow(extreme_coefs) > 0) {
    cat("      ⚠ 检测到极端系数/CI值，可能因多重共线性或样本量不足\n")
    for (i in 1:nrow(extreme_coefs)) {
      cat(sprintf("        %s: coef=%.1f, CI=[%.1f, %.1f]\n",
                  extreme_coefs$variable_label[i],
                  extreme_coefs$coefficient[i],
                  extreme_coefs$ci_low[i],
                  extreme_coefs$ci_high[i]))
    }
  }
  
  # 【新增】对绘图范围进行智能调整（避免极端值压缩其他系数）
  # 仅用于Y轴范围，不改变数据
  non_residual_data <- plot_data %>% filter(!grepl("Unexplained", variable_label))
  if (nrow(non_residual_data) > 0) {
    coef_range <- range(non_residual_data$coefficient[is.finite(non_residual_data$coefficient)])
    ci_range <- range(c(non_residual_data$ci_low[is.finite(non_residual_data$ci_low)],
                        non_residual_data$ci_high[is.finite(non_residual_data$ci_high)]))
    
    # 如果CI范围过大（>100倍系数范围），进行截断
    if (diff(ci_range) > 100 * diff(coef_range) && diff(coef_range) > 0) {
      # 使用95%分位数作为边界
      ci_95 <- quantile(c(non_residual_data$ci_low, non_residual_data$ci_high), 
                        c(0.025, 0.975), na.rm = TRUE)
      plot_xlim <- c(ci_95[1] * 1.1, ci_95[2] * 1.1)
      cat("      → 绘图X轴范围调整为:", round(plot_xlim, 2), "\n")
    } else {
      plot_xlim <- NULL
    }
  } else {
    plot_xlim <- NULL
  }
  
  # 绘制（从上往下按预定义顺序）
  p <- ggplot(plot_data, aes(x = coefficient, y = reorder(variable_label, -plot_order))) +
    geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1) +
    geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, color = direction), 
                   height = 0.25, linewidth = 1) +
    geom_point(aes(color = direction), size = 5, shape = 19) +
    geom_text(
      data = plot_data %>% filter(nzchar(star_label)),
      aes(label = star_label),
      nudge_x = star_nudge_socioecon,
      nudge_y = 0.22,
      size = 4,
      color = "black",
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(values = forest_direction_colors, guide = "none") +
    labs(
      title = title_text,
      subtitle = "Standardized coefficients with 95% CI | Blue=Positive, Red=Negative, Gray=Residual SD | * p<0.05, ** p<0.01, *** p<0.001",
      x = "Coefficient (log scale)",
      y = ""
    ) +
    coord_cartesian(xlim = plot_xlim, clip = "off") +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 15, hjust = 0),
      plot.subtitle = element_text(size = 10, color = "gray30", hjust = 0),
      axis.title.x = element_text(face = "bold", size = 12),
      axis.text.y = element_text(size = 12, face = "bold", color = "black"),
      axis.text.x = element_text(size = 11),
      panel.grid = element_blank(),  # 移除所有网格线
      axis.line.x = element_line(color = "black", linewidth = 1),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
    )
  
  ggsave(output_path, p, width = 9, height = 6, dpi = 300)
  
  return(p)
}

# ========== 【V5新增】可视化函数: 控制变量森林图 ==========

plot_control_variables_forest <- function(model_result, output_dir, city_name, indicator, model_type = "composite") {
  # 提取GAM模型
  m_gam <- model_result$model
  if (is.null(m_gam)) return(NULL)
  
  s <- summary(m_gam)
  param_table <- s$p.table
  smooth_table <- s$s.table
  param_names <- rownames(param_table)
  
  # 用于存储结果的列表
  control_vars <- list()
  
  # 1. 提取平滑项（温度、湿度、降水等气象变量）
  if (nrow(smooth_table) > 0) {
    for (i in 1:nrow(smooth_table)) {
      term_name <- rownames(smooth_table)[i]
      
      # 跳过DLNM的cb_cehwi项
      if (grepl("cb_cehwi|cb_exceeded", term_name)) next
      
      edf <- smooth_table[i, "edf"]
      f_val <- smooth_table[i, "F"]
      p_val <- smooth_table[i, "p-value"]
      
      # 平滑项没有系数，用F值表示效应强度
      control_vars[[length(control_vars) + 1]] <- list(
        variable = term_name,
        type = "Smooth",
        coefficient = NA,
        se = NA,
        ci_low = NA,
        ci_high = NA,
        edf = edf,
        f_value = f_val,
        p_value = p_val,
        significant = p_val < 0.05
      )
    }
  }
  
  # 2. 提取参数项（星期几效应、年份效应等）
  if (nrow(param_table) > 0) {
    dow_idx <- grep("dow_fac", param_names, fixed = TRUE)
    if (length(dow_idx) > 0) {
      for (i in dow_idx) {
        coef <- param_table[i, "Estimate"]
        se <- param_table[i, "Std. Error"]
        p_val <- param_table[i, "Pr(>|t|)"]
        
        control_vars[[length(control_vars) + 1]] <- list(
          variable = "Day of Week Effect",
          type = "Parametric",
          coefficient = coef,
          se = se,
          ci_low = coef - 1.96 * se,
          ci_high = coef + 1.96 * se,
          edf = NA,
          f_value = NA,
          p_value = p_val,
          significant = p_val < 0.05
        )
      }
    }
    
    # 年份效应（只显示部分，避免太多）
    year_idx <- grep("year_fac", param_names)
    if (length(year_idx) > 0) {
      # 只取前3个和后3个年份
      year_to_show <- c(year_idx[1:min(3, length(year_idx))], 
                        year_idx[max(1, length(year_idx) - 2):length(year_idx)])
      year_to_show <- unique(year_to_show)
      
      for (i in year_to_show) {
        term_name <- param_names[i]
        year_label <- str_replace(term_name, "year_fac", "Year ")
        coef <- param_table[i, "Estimate"]
        se <- param_table[i, "Std. Error"]
        p_val <- param_table[i, "Pr(>|t|)"]
        
        control_vars[[length(control_vars) + 1]] <- list(
          variable = year_label,
          type = "Parametric",
          coefficient = coef,
          se = se,
          ci_low = coef - 1.96 * se,
          ci_high = coef + 1.96 * se,
          edf = NA,
          f_value = NA,
          p_value = p_val,
          significant = p_val < 0.05
        )
      }
    }
    
    # 其他气象参数项（如has_snow）
    snow_idx <- grep("has_snow", param_names, fixed = TRUE)
    if (length(snow_idx) > 0) {
      for (i in snow_idx) {
        coef <- param_table[i, "Estimate"]
        se <- param_table[i, "Std. Error"]
        p_val <- param_table[i, "Pr(>|t|)"]
        
        control_vars[[length(control_vars) + 1]] <- list(
          variable = "Snow Indicator",
          type = "Parametric",
          coefficient = coef,
          se = se,
          ci_low = coef - 1.96 * se,
          ci_high = coef + 1.96 * se,
          edf = NA,
          f_value = NA,
          p_value = p_val,
          significant = p_val < 0.05
        )
      }
    }
  }
  
  # 如果没有控制变量，跳过
  if (length(control_vars) == 0) {
    cat("    ⚠ 无控制变量，跳过森林图生成\n")
    return(NULL)
  }
  
  # 转换为data.frame
  df_control <- bind_rows(control_vars) %>%
    mutate(
      sig_mark = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ "ns"
      ),
      direction = forest_effect_direction(coefficient, neutral_zero = TRUE),
      star_label = forest_significance_stars(p_value),
      # 重新排序：平滑项在上，参数项在下
      order = ifelse(type == "Smooth", 1, 2)
    ) %>%
    arrange(order, desc(abs(ifelse(is.na(coefficient), f_value, coefficient))))
  
  # 绘制森林图
  # 对于平滑项（无系数），用F值表示效应强度；参数项用系数和CI
  
  # 分两个子图
  df_smooth <- df_control %>% filter(type == "Smooth")
  df_param <- df_control %>% filter(type == "Parametric")
  
  plots_list <- list()
  
  # 1. 平滑项（F值条形图）
  if (nrow(df_smooth) > 0) {
    p_smooth <- ggplot(df_smooth, aes(y = reorder(variable, f_value), x = f_value)) +
      geom_col(aes(fill = significant), alpha = 0.7, width = 0.6) +
      geom_text(aes(label = sig_mark, x = f_value + max(f_value) * 0.05), 
                hjust = 0, size = 4) +
      scale_fill_manual(values = c("TRUE" = "#E64B35", "FALSE" = "gray70"),
                        labels = c("TRUE" = "Significant", "FALSE" = "Not Significant")) +
      labs(
        title = "Smooth Terms (Nonlinear Effects)",
        subtitle = paste0("F-values with significance | ", city_name),
        x = "F-value",
        y = NULL,
        fill = "Significance"
      ) +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(face = "bold"),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "bottom"
      )
    
    plots_list$smooth <- p_smooth
  }
  
  # 2. 参数项（系数森林图）
  if (nrow(df_param) > 0) {
    param_star_nudge <- forest_star_offset(df_param$ci_low, df_param$ci_high, fallback = 0.05)
    p_param <- ggplot(df_param, aes(y = reorder(variable, coefficient), x = coefficient)) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
      geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, color = direction),
                     height = 0.3, linewidth = 0.8) +
      geom_point(aes(color = direction), size = 3.5, shape = 19) +
      geom_text(
        data = df_param %>% filter(nzchar(star_label)),
        aes(label = star_label),
        nudge_x = param_star_nudge,
        nudge_y = 0.2,
        hjust = 0,
        size = 4,
        color = "black",
        fontface = "bold"
      ) +
      scale_color_manual(values = forest_direction_colors, guide = "none") +
      labs(
        title = "Parametric Terms (Linear Effects)",
        subtitle = paste0("Coefficients with 95% CI | ", city_name, " | Blue=Positive, Red=Negative | * p<0.05, ** p<0.01, *** p<0.001"),
        x = "Coefficient Estimate",
        y = NULL
      ) +
      coord_cartesian(clip = "off") +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(face = "bold"),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank()
      )
    
    plots_list$param <- p_param
  }
  
  # 合并两个图（如果都存在）
  if (length(plots_list) == 2) {
    library(patchwork)
    p_combined <- plots_list$smooth / plots_list$param + 
      plot_layout(heights = c(1, 1.5))
    
    filename <- paste0(city_name, "_control_variables_", indicator, "_", model_type, ".png")
    ggsave(file.path(output_dir, filename), p_combined, 
           width = 12, height = max(8, nrow(df_control) * 0.4), dpi = 300)
    
  } else if (length(plots_list) == 1) {
    # 只有一个图
    p_single <- plots_list[[1]]
    filename <- paste0(city_name, "_control_variables_", indicator, "_", model_type, ".png")
    ggsave(file.path(output_dir, filename), p_single, 
           width = 12, height = max(6, nrow(df_control) * 0.5), dpi = 300)
  }
  
  # 保存CSV
  df_control_export <- df_control %>%
    select(variable, type, coefficient, se, ci_low, ci_high, edf, f_value, p_value, sig_mark)
  
  csv_filename <- paste0(city_name, "_control_variables_", indicator, "_", model_type, ".csv")
  write_csv(df_control_export, file.path(output_dir, csv_filename))
  cat("      - Cross-basis controls:", sum(df_control$type == 'Cross-basis'), "\n")
  
  cat("    ✓ 控制变量森林图已保存 (", model_type, ")\n")
  cat("      - 变量数:", nrow(df_control), "（Smooth:", nrow(df_smooth), "，Parametric:", nrow(df_param), "）\n")
  
  return(list(plot = if(length(plots_list) == 2) p_combined else plots_list[[1]], 
              data = df_control_export))
}

# ========== 可视化函数: 城市级RR曲线 ==========

plot_control_variables_forest <- function(model_result, output_dir, city_name, indicator, model_type = "composite") {
  m_gam <- model_result$model
  if (is.null(m_gam)) return(NULL)
  
  s <- summary(m_gam)
  param_table <- s$p.table
  smooth_table <- s$s.table
  param_names <- rownames(param_table)
  control_vars <- list()
  
  if (!is.null(model_result$temperature_control_contrast) &&
      nrow(model_result$temperature_control_contrast) > 0) {
    control_vars <- c(
      control_vars,
      split(model_result$temperature_control_contrast,
            seq_len(nrow(model_result$temperature_control_contrast)))
    )
  }
  
  smooth_df <- extract_smooth_term_contrasts(m_gam, smooth_table)
  if (nrow(smooth_df) > 0) {
    control_vars <- c(control_vars, split(smooth_df, seq_len(nrow(smooth_df))))
  }
  
  if (nrow(param_table) > 0) {
    weekend_contrast <- compute_weekend_contrast(m_gam)
    if (!is.null(weekend_contrast) && nrow(weekend_contrast) > 0) {
      control_vars <- c(control_vars, split(weekend_contrast, seq_len(nrow(weekend_contrast))))
    }
    
    snow_idx <- grep("has_snow", param_names, fixed = TRUE)
    if (length(snow_idx) > 0) {
      for (i in snow_idx) {
        coef <- param_table[i, "Estimate"]
        se <- param_table[i, "Std. Error"]
        p_val <- param_table[i, "Pr(>|t|)"]
        control_vars[[length(control_vars) + 1]] <- list(
          variable = "Snow Indicator",
          raw_term = param_names[i],
          type = "Parametric",
          coefficient = coef,
          se = se,
          ci_low = coef - 1.96 * se,
          ci_high = coef + 1.96 * se,
          edf = NA,
          f_value = NA,
          p_value = p_val,
          significant = p_val < 0.05,
          contrast_note = "Parametric coefficient"
        )
      }
    }
  }
  
  if (length(control_vars) == 0) {
    cat("    Control-variable forest plot status: no estimable terms.\n")
    return(NULL)
  }
  
  df_control <- bind_rows(control_vars) %>%
    mutate(
      sig_mark = forest_significance_stars(p_value, blank_ns = FALSE),
      direction = forest_effect_direction(coefficient, neutral_zero = TRUE),
      star_label = forest_significance_stars(p_value),
      order = dplyr::case_when(
        type == "Cross-basis" ~ 1,
        type == "Smooth" ~ 2,
        TRUE ~ 3
      ),
      display_label = dplyr::case_when(
        type == "Cross-basis" ~ paste0(variable, " [Cross-basis]"),
        type == "Smooth" ~ paste0(variable, " [Smooth]"),
        TRUE ~ variable
      )
    ) %>%
    arrange(order, desc(abs(coefficient))) %>%
    mutate(
      display_label = make.unique(as.character(display_label), sep = " | "),
      display_label = factor(display_label, levels = rev(unique(display_label)))
    )
  
  control_star_nudge <- forest_star_offset(df_control$ci_low, df_control$ci_high, fallback = 0.05)
  model_display <- stage1_model_display_label(model_type)
  subtitle_text <- paste0(
    toupper(indicator), " | ", model_display, " | ", city_name, "\n",
    "Blue=Positive, Red=Negative | Temperature is cumulative cross-basis P90-P10; weather smooths are P90-P10 contrasts; doy/year are adjustment terms not shown | * p<0.05, ** p<0.01, *** p<0.001"
  )
  
  p_control <- ggplot(df_control, aes(y = display_label, x = coefficient)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_errorbarh(
      aes(xmin = ci_low, xmax = ci_high, color = direction),
      height = 0.26,
      linewidth = 1
    ) +
    geom_point(aes(color = direction), size = 3.8, shape = 19) +
    geom_text(
      data = df_control %>% filter(nzchar(star_label)),
      aes(label = star_label),
      nudge_x = control_star_nudge,
      nudge_y = 0.22,
      hjust = 0,
      size = 4,
      color = "black",
      fontface = "bold"
    ) +
    scale_color_manual(values = forest_direction_colors, guide = "none") +
    labs(
      title = paste0(city_name, " - ", model_display, " - Control Variable Contributions"),
      subtitle = subtitle_text,
      x = "Effect Estimate on Linear Predictor",
      y = NULL
    ) +
    coord_cartesian(clip = "off") +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10, color = "gray30", lineheight = 1.15),
      axis.title = element_text(face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line.x = element_line(color = "black", linewidth = 0.5)
    )
  
  filename <- paste0(city_name, "_control_variables_", indicator, "_", model_type, ".png")
  ggsave(
    file.path(output_dir, filename),
    p_control,
    width = 12,
    height = max(6, nrow(df_control) * 0.5),
    dpi = 300
  )
  
  df_control_export <- df_control %>%
    select(variable, raw_term, type, contrast_note, coefficient, se, ci_low, ci_high, edf, f_value, p_value, sig_mark)
  
  csv_filename <- paste0(city_name, "_control_variables_", indicator, "_", model_type, ".csv")
  write_csv(df_control_export, file.path(output_dir, csv_filename))
  cat("      - Cross-basis controls:", sum(df_control$type == 'Cross-basis'), "\n")
  
  cat("    Control-variable forest plot saved (", model_type, ").\n")
  cat("      - 变量数:", nrow(df_control), "（Smooth:", sum(df_control$type == 'Smooth'), "，Parametric:", sum(df_control$type == 'Parametric'), "）\n")
  
  return(list(plot = p_control, data = df_control_export))
}

plot_city_rr_curve <- function(result, output_dir, model_type = "composite") {
  # model_type: "composite", "day", "night"
  if (is.null(result)) return(NULL)
  
  pred_df <- result$pred_df
  city_name <- result$city
  indicator <- result$indicator
  base_model_type <- model_base_type(model_type)
  model_display <- stage1_model_display_label(model_type)
  r2 <- ifelse(is.null(result$r2), NA, result$r2)
  # smooth_p：仅当 DLNM 在平滑表中有单独一行时才有意义；cross-basis 为参数项时为 NA。
  # 对参数化 cross-basis，优先使用整体 Wald 检验 p 值。
  smooth_p <- ifelse(is.null(result$smooth_p) || is.na(result$smooth_p), NA, result$smooth_p)
  overall_wald_p <- ifelse(is.null(result$overall_wald_p) || is.na(result$overall_wald_p), NA, result$overall_wald_p)
  effect_pct <- ifelse(is.null(result$effect_pct), NA, result$effect_pct)
  
  # 显著性标记：优先展示DLNM整体Wald检验，其次才是平滑项p值
  if (!is.na(overall_wald_p)) {
    if (overall_wald_p < 0.001) {
      sig_mark <- "***"
      sig_text <- "Wald p<0.001***"
    } else if (overall_wald_p < 0.01) {
      sig_mark <- "**"
      sig_text <- paste0("Wald p=", format.pval(overall_wald_p, digits = 3), "**")
    } else if (overall_wald_p < 0.05) {
      sig_mark <- "*"
      sig_text <- paste0("Wald p=", format.pval(overall_wald_p, digits = 3), "*")
    } else {
      sig_mark <- "ns"
      sig_text <- paste0("Wald p=", format.pval(overall_wald_p, digits = 3), " (ns)")
    }
  } else if (is.na(smooth_p)) {
    sig_mark <- ""
    sig_text <- "p=NA"
  } else if (smooth_p < 0.001) {
    sig_mark <- "***"
    sig_text <- "p<0.001***"
  } else if (smooth_p < 0.01) {
    sig_mark <- "**"
    sig_text <- paste0("p=", format.pval(smooth_p, digits = 3), "**")
  } else if (smooth_p < 0.05) {
    sig_mark <- "*"
    sig_text <- paste0("p=", format.pval(smooth_p, digits = 3), "*")
  } else {
    sig_mark <- "ns"
    sig_text <- paste0("p=", format.pval(smooth_p, digits = 3), " (ns)")
  }
  
  # Debug: 确保显著性被正确获取
  cat("    [Plot] DLNM整体Wald p值 =",
      ifelse(is.na(overall_wald_p), "NA", format.pval(overall_wald_p, digits = 3)),
      "| 平滑项p值 =",
      ifelse(is.na(smooth_p), "NA", format.pval(smooth_p, digits = 3)),
      ", 标记 =", sig_mark, "\n")
  
  # 【颜色方案】根据model_type选择颜色
  line_colors <- c(
    "composite" = "#D53E4F",  # 红色
    "day" = "#FF8C00",         # 橙色
    "night" = "#9B59B6"        # 紫色
  )
  line_color <- stage1_model_color(model_type)
  if (is.na(line_color) || !nzchar(line_color)) {
    line_color <- "#1F4E79"
  }
  
  # 【V4优化】智能检测极端RR值并处理
  rr_max <- max(pred_df$rr_high, na.rm = TRUE)
  rr_min <- min(pred_df$rr_low, na.rm = TRUE)
  rr_curve_max <- max(pred_df$rr, na.rm = TRUE)
  rr_curve_min <- min(pred_df$rr, na.rm = TRUE)
  
  # 【Nature标准】根据CI范围智能决定是否用对数Y轴
  # 如果CI跨越超过2个数量级，使用对数轴
  use_log_scale <- (rr_max / rr_min > 100) || (rr_max > 10) || (rr_min < 0.1)
  
  # 【V4截断标准】更严格的截断（Nature标准：99%分位数）
  # 计算99%分位数作为截断阈值
  ci_99_high <- quantile(pred_df$rr_high[is.finite(pred_df$rr_high)], 0.99, na.rm = TRUE)
  ci_01_low <- quantile(pred_df$rr_low[is.finite(pred_df$rr_low)], 0.01, na.rm = TRUE)
  
  # 只有当极端值超过99%分位数的2倍时才截断
  has_extreme_high <- rr_max > ci_99_high * 2
  has_extreme_low <- rr_min < ci_01_low / 2
  
  if (has_extreme_high || has_extreme_low) {
    cat("    【V4】检测到极端置信区间，智能截断（Nature标准）\n")
    cat("      原始CI范围: [", round(rr_min, 3), ",", round(rr_max, 3), "]\n")
    cat("      截断阈值: [", round(ci_01_low / 2, 3), ",", round(ci_99_high * 2, 3), "]\n")
    
    pred_df_plot <- pred_df %>%
      mutate(
        rr_low_clipped = pmax(rr_low, ci_01_low / 2),
        rr_high_clipped = pmin(rr_high, ci_99_high * 2)
      )
  } else {
    # 正常情况：完全保留
    pred_df_plot <- pred_df %>%
      mutate(
        rr_low_clipped = rr_low,
        rr_high_clipped = rr_high
      )
  }
  
  # 构建统计信息文本（放在subtitle）
  effect_text <- if (is.na(effect_pct)) {
    "N/A"
  } else if (abs(effect_pct) > 1000) {
    "Unstable"
  } else {
    paste0(round(effect_pct, 1), "%", ifelse(effect_pct > 0, "↑", "↓"))
  }
  
  # 构建完整的subtitle（包含模型信息和统计信息）
  # 第一行：模型参数
  # 第二行：统计信息（R²、显著性、效应）
  subtitle_text <- paste0(
    "DLNM: ", toupper(indicator), " vs PA (scaled by ", SCALE_FACTOR, ") | Lag 0-", MAX_LAG, " days | Reference: CEHWI = 0\n",
    "R² = ", ifelse(is.na(r2), "N/A", round(r2, 3)), 
    "  |  ", sig_text,
    "  |  Effect: ", effect_text
  )
  
  # 绘制累积RR曲线（不再需要图内标注）
  p <- ggplot(pred_df_plot, aes(x = cehwi, y = rr)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_ribbon(aes(ymin = rr_low_clipped, ymax = rr_high_clipped), 
                fill = "gray80", alpha = 0.5) +  # 统一浅灰色置信区间
    geom_line(color = line_color, linewidth = 1.5) +
    labs(
      title = paste0(city_name, " - ", model_display, " - Cumulative Exposure-Response Curve"),
      subtitle = subtitle_text,
      x = paste0(toupper(indicator)),
      y = "Relative Risk (RR)",
      caption = ifelse(has_extreme_high || has_extreme_low, 
                       "Note: Extreme CI values (>100 or <0.01) truncated for visualization", "")
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(face = "bold"),
      plot.caption = element_text(size = 9, color = "gray40", hjust = 1),
      panel.grid = element_blank(),  # 去掉所有网格线
      axis.line = element_line(color = "gray30", linewidth = 0.5)  # 添加坐标轴线
    )
  
  # 【V4优化】智能Y轴刻度（Nature标准）
  ci_min <- min(pred_df_plot$rr_low_clipped, na.rm = TRUE)
  ci_max <- max(pred_df_plot$rr_high_clipped, na.rm = TRUE)
  
  if (use_log_scale) {
    # 【Nature标准】对数Y轴 + 合理breaks
    cat("    【V4】使用对数Y轴（CI跨越多个数量级）\n")
    cat("      CI范围:", round(ci_min, 3), "-", round(ci_max, 3), "\n")
    
    # 智能选择breaks（只标注关键刻度）
    possible_breaks <- c(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
    y_breaks <- possible_breaks[possible_breaks >= ci_min * 0.5 & 
                                 possible_breaks <= ci_max * 2]
    # 确保包含1（参照线）
    if (!1 %in% y_breaks && ci_min < 1 && ci_max > 1) {
      y_breaks <- sort(c(y_breaks, 1))
    }
    
    # Y轴范围留余量
    y_lim_min <- max(0.005, ci_min * 0.5)
    y_lim_max <- min(200, ci_max * 2)
    
    p <- p + 
      scale_y_log10(
        breaks = y_breaks,
        labels = as.character(y_breaks),
        limits = c(y_lim_min, y_lim_max)
      ) +
      labs(y = "Relative Risk (RR, log scale)")
  } else {
    # 线性Y轴（正常情况）
    cat("    【V4】使用线性Y轴（CI范围合理）\n")
    
    y_min <- max(0, ci_min - (ci_max - ci_min) * 0.1)
    y_max <- ci_max + (ci_max - ci_min) * 0.1
    y_range <- y_max - y_min
    
    # 智能选择间隔
    if (y_range < 1) {
      y_breaks <- seq(floor(y_min * 10) / 10, ceiling(y_max * 10) / 10, by = 0.2)
    } else if (y_range < 3) {
      y_breaks <- seq(floor(y_min * 2) / 2, ceiling(y_max * 2) / 2, by = 0.5)
    } else {
      y_breaks <- seq(floor(y_min), ceiling(y_max), by = 1)
    }
    
    # 确保包含1（参照线）
    if (!1 %in% y_breaks && 1 >= y_min && 1 <= y_max) {
      y_breaks <- sort(c(y_breaks, 1))
    }
    
    p <- p + 
      scale_y_continuous(
        breaks = y_breaks,
        limits = c(y_min, y_max)
      )
  }
  
  # 保存RR曲线文件名包含model_type
  filename_rr <- paste0(city_name, "_DLNM_RR_curve_", indicator, "_", model_type, ".png")
  ggsave(file.path(output_dir, filename_rr), p, width = 10, height = 7, dpi = 300)
  
  cat("    ✓ RR曲线图已保存 (", model_type, ")\n")
  
  # ========== 【V6新增】生成城市级暴露分布直方图和AF ==========
  if (!is.null(result$cehwi_data) && length(result$cehwi_data) > 10) {
    tryCatch({
      cehwi_data <- result$cehwi_data[result$cehwi_data > 0]
      cehwi_data <- cehwi_data[!is.na(cehwi_data)]
      
      if (length(cehwi_data) > 10) {
        # 1. 计算百分位数
        p25 <- quantile(cehwi_data, 0.25, na.rm = TRUE)
        p50 <- quantile(cehwi_data, 0.50, na.rm = TRUE)
        p75 <- quantile(cehwi_data, 0.75, na.rm = TRUE)
        p90 <- quantile(cehwi_data, 0.90, na.rm = TRUE)
        p95 <- quantile(cehwi_data, 0.95, na.rm = TRUE)
        p98 <- quantile(cehwi_data, 0.98, na.rm = TRUE)  # 【V5.1】98th分位数用于X轴截断
        
        cat("    【V5.1】单城市RR_with_dist组合图X轴将截断至98th分位数:", round(p98, 2), "\n")
        
        # 2. 生成直方图（与RR曲线X轴对齐）
        hist_df <- data.frame(cehwi = cehwi_data)
        
        p_hist <- ggplot(hist_df, aes(x = cehwi)) +
          geom_histogram(fill = line_color, alpha = 0.6, bins = 30, color = "white") +
          geom_vline(xintercept = p25, linetype = "dotted", color = "gray40", linewidth = 0.8) +
          geom_vline(xintercept = p75, linetype = "dotted", color = "gray40", linewidth = 0.8) +
          geom_vline(xintercept = p90, linetype = "dotted", color = "gray40", linewidth = 0.8) +
          annotate("text", x = p25, y = Inf, label = "25th", vjust = 2, size = 3, color = "gray40") +
          annotate("text", x = p75, y = Inf, label = "75th", vjust = 2, size = 3, color = "gray40") +
          annotate("text", x = p90, y = Inf, label = "90th", vjust = 2, size = 3, color = "gray40") +
          labs(
            x = paste0(toupper(indicator)),
            y = "Frequency (Heatwave Days)",
            caption = paste0("Exposure distribution: n=", length(cehwi_data), " heatwave days")
          ) +
          theme_minimal(base_size = 10) +
          theme(
            axis.title = element_text(face = "bold"),
            panel.grid = element_blank(),
            axis.line = element_line(color = "gray30", linewidth = 0.5)
          ) +
          # 【V5.1】与RR曲线X轴一致，截断至98th
          coord_cartesian(xlim = c(0, p98))
        
        # 保存直方图
        filename_hist <- paste0(city_name, "_exposure_dist_", indicator, "_", model_type, ".png")
        ggsave(file.path(output_dir, filename_hist), p_hist, width = 10, height = 3, dpi = 300)
        
        cat("    ✓ 直方图已保存 (", model_type, ")\n")
        
        # 3. 生成组合图（RR曲线在上，直方图在下）
        library(patchwork)
        # 【V5.1】为RR曲线添加98th截断
        p_with_xlim <- p + coord_cartesian(xlim = c(0, p98))
        p_combined <- p_with_xlim / p_hist + plot_layout(heights = c(2, 1))
        
        filename_combined <- paste0(city_name, "_RR_with_dist_", indicator, "_", model_type, ".png")
        ggsave(file.path(output_dir, filename_combined), p_combined, width = 10, height = 9, dpi = 300)
        
        cat("    ✓ 组合图已保存 (", model_type, ")\n")
        
        # 4. 计算城市级AF @ 75th, 90th, 95th百分位数
        # 找到最接近这些百分位数的索引
        idx_p25 <- which.min(abs(pred_df$cehwi - p25))
        idx_p50 <- which.min(abs(pred_df$cehwi - p50))
        idx_p75 <- which.min(abs(pred_df$cehwi - p75))
        idx_p90 <- which.min(abs(pred_df$cehwi - p90))
        idx_p95 <- which.min(abs(pred_df$cehwi - p95))
        
        # 提取RR @ 这些百分位数
        rr_at_p25 <- pred_df$rr[idx_p25]
        rr_at_p50 <- pred_df$rr[idx_p50]
        rr_at_p75 <- pred_df$rr[idx_p75]
        rr_at_p90 <- pred_df$rr[idx_p90]
        rr_at_p95 <- pred_df$rr[idx_p95]
        
        rr_low_at_p25 <- pred_df$rr_low[idx_p25]
        rr_low_at_p50 <- pred_df$rr_low[idx_p50]
        rr_low_at_p75 <- pred_df$rr_low[idx_p75]
        rr_low_at_p90 <- pred_df$rr_low[idx_p90]
        rr_low_at_p95 <- pred_df$rr_low[idx_p95]
        
        rr_high_at_p25 <- pred_df$rr_high[idx_p25]
        rr_high_at_p50 <- pred_df$rr_high[idx_p50]
        rr_high_at_p75 <- pred_df$rr_high[idx_p75]
        rr_high_at_p90 <- pred_df$rr_high[idx_p90]
        rr_high_at_p95 <- pred_df$rr_high[idx_p95]
        
        # 计算AF（AF = (RR - 1) / RR * 100%）
        af_p25 <- (rr_at_p25 - 1) / rr_at_p25 * 100
        af_p50 <- (rr_at_p50 - 1) / rr_at_p50 * 100
        af_p75 <- (rr_at_p75 - 1) / rr_at_p75 * 100
        af_p90 <- (rr_at_p90 - 1) / rr_at_p90 * 100
        af_p95 <- (rr_at_p95 - 1) / rr_at_p95 * 100
        
        af_low_p25 <- (rr_low_at_p25 - 1) / rr_low_at_p25 * 100
        af_low_p50 <- (rr_low_at_p50 - 1) / rr_low_at_p50 * 100
        af_low_p75 <- (rr_low_at_p75 - 1) / rr_low_at_p75 * 100
        af_low_p90 <- (rr_low_at_p90 - 1) / rr_low_at_p90 * 100
        af_low_p95 <- (rr_low_at_p95 - 1) / rr_low_at_p95 * 100
        
        af_high_p25 <- (rr_high_at_p25 - 1) / rr_high_at_p25 * 100
        af_high_p50 <- (rr_high_at_p50 - 1) / rr_high_at_p50 * 100
        af_high_p75 <- (rr_high_at_p75 - 1) / rr_high_at_p75 * 100
        af_high_p90 <- (rr_high_at_p90 - 1) / rr_high_at_p90 * 100
        af_high_p95 <- (rr_high_at_p95 - 1) / rr_high_at_p95 * 100
        
        # 创建AF汇总DataFrame
        af_df <- data.frame(
          city = city_name,
          indicator = indicator,
          model_type = model_type,
          percentile = c("25th", "50th", "75th", "90th", "95th"),
          cehwi_value = c(p25, p50, p75, p90, p95),
          rr = c(rr_at_p25, rr_at_p50, rr_at_p75, rr_at_p90, rr_at_p95),
          rr_low = c(rr_low_at_p25, rr_low_at_p50, rr_low_at_p75, rr_low_at_p90, rr_low_at_p95),
          rr_high = c(rr_high_at_p25, rr_high_at_p50, rr_high_at_p75, rr_high_at_p90, rr_high_at_p95),
          af_pct = c(af_p25, af_p50, af_p75, af_p90, af_p95),
          af_low = c(af_low_p25, af_low_p50, af_low_p75, af_low_p90, af_low_p95),
          af_high = c(af_high_p25, af_high_p50, af_high_p75, af_high_p90, af_high_p95)
        )
        
        # 保存AF汇总
        filename_af <- paste0(city_name, "_AF_summary_", indicator, "_", model_type, ".csv")
        write_csv(af_df, file.path(output_dir, filename_af))
        
        cat("    ✓ AF已计算: 90th = ", round(af_p90, 2), "% (95% CI: ", 
            round(af_low_p90, 2), "% - ", round(af_high_p90, 2), "%)\n")
      }
    }, error = function(e) {
      cat("    ⚠ 直方图/AF生成失败:", conditionMessage(e), "\n")
    })
  } else {
    cat("    ⚠ 暴露数据不足，跳过直方图和AF生成\n")
  }
  
  return(p)
}

save_city_activity_overlay_plots <- function(all_results, city_output_dir, city_name, indicator) {
  if (!is_activity_split_mode(STAGE1_ACTIVITY_MODE) || is.null(all_results) || length(all_results) == 0) {
    return(invisible(FALSE))
  }
  
  activity_types <- c("all", "ride", "run", "walk")
  activity_labels <- activity_modality_labels()
  activity_colors <- activity_modality_colors()
  saved_n <- 0L
  
  for (base_model in c("composite", "day", "night")) {
    model_names <- vapply(activity_types, function(activity) {
      stage1_model_name(base_model, activity, STAGE1_ACTIVITY_MODE)
    }, character(1))
    tryCatch({
      available <- model_names[model_names %in% names(all_results)]
      if (length(available) < 2) stop("fewer than two activity-specific model results are available")
      
      rr_df <- bind_rows(lapply(available, function(model_name) {
        model_result <- all_results[[model_name]]
        if (is.null(model_result$pred_df) || nrow(model_result$pred_df) == 0) return(NULL)
        activity <- model_activity_type(model_name)
        model_result$pred_df %>%
          mutate(
            model_name = model_name,
            base_model = base_model,
            activity_type = activity,
            activity_label = unname(activity_labels[activity]),
            rr_low_plot = pmax(rr_low, 0.01),
            rr_high_plot = pmin(rr_high, 10),
            rr_plot = pmin(pmax(rr, 0.01), 10)
          )
      }))
      if (is.null(rr_df) || nrow(rr_df) == 0) stop("pooled prediction data are empty")
      
      exposure_df <- bind_rows(lapply(available, function(model_name) {
        model_result <- all_results[[model_name]]
        exposure_values <- model_result$cehwi_data
        if (is.null(exposure_values) || length(exposure_values) == 0) return(NULL)
        activity <- model_activity_type(model_name)
        tibble(
          exposure = exposure_values[is.finite(exposure_values) & exposure_values > 0],
          activity_type = activity,
          activity_label = unname(activity_labels[activity])
        )
      }))
      
      y_upper <- suppressWarnings(quantile(rr_df$rr_high_plot, 0.95, na.rm = TRUE))
      if (!is.finite(y_upper) || y_upper < 1.5) y_upper <- 2
      y_upper <- min(10, max(2, y_upper))
      
      all_rr_df <- rr_df %>% filter(activity_type == "all")
      modality_rr_df <- rr_df %>% filter(activity_type %in% c("ride", "run", "walk"))
      if (nrow(all_rr_df) == 0) stop("All-activity curve is missing")
      if (nrow(modality_rr_df) == 0) stop("activity-specific curves are missing")
      modality_palette <- setNames(
        unname(activity_colors[c("ride", "run", "walk")]),
        unname(activity_labels[c("ride", "run", "walk")])
      )
      
      p_rr <- ggplot() +
        geom_hline(yintercept = 1, linetype = "dashed", color = "gray55", linewidth = 0.55) +
        geom_ribbon(
          data = all_rr_df,
          aes(x = cehwi, ymin = rr_low_plot, ymax = rr_high_plot),
          fill = "#D1D5DB",
          alpha = 0.45,
          color = NA
        ) +
        geom_line(
          data = all_rr_df,
          aes(x = cehwi, y = rr_plot),
          color = "#6B7280",
          linewidth = 1.1,
          linetype = "longdash"
        ) +
        geom_line(
          data = modality_rr_df,
          aes(x = cehwi, y = rr_plot, color = activity_label),
          linewidth = 1.65
        ) +
        scale_color_manual(values = modality_palette, breaks = names(modality_palette)) +
        coord_cartesian(ylim = c(0, y_upper), clip = "off") +
        labs(
          title = paste0(city_name, " - ", str_to_title(base_model), " - Activity-specific cumulative RR"),
          subtitle = "Gray dashed line and ribbon = All activity estimate and 95% CI; colored thick lines = Ride, Run, Walk.",
          x = NULL,
          y = "Relative Risk (RR)",
          color = "PA modality"
        ) +
        rr_plot_theme(base_size = 14) +
        theme(legend.position = "top")
      
      hist_df <- exposure_df %>%
        filter(activity_type %in% c("ride", "run", "walk"), is.finite(exposure), exposure > 0)
      # Do not draw modality-stacked exposure histograms from model_result$cehwi_data:
      # the heatwave exposure series is shared by PA outcomes, so reusing it by
      # activity duplicates the same exposure distribution and creates a false
      # ride/run/walk composition. True composition histograms are regenerated by
      # the Nature-ready post-processing script from the raw PA trip records.
      if (FALSE && nrow(hist_df) > 0) {
        p_hist <- ggplot(hist_df, aes(x = exposure, fill = activity_label)) +
          geom_histogram(
            aes(y = after_stat(count / sum(count))),
            bins = 35,
            position = "stack",
            color = "white",
            linewidth = 0.2,
            alpha = 0.9
          ) +
          scale_fill_manual(values = modality_palette, breaks = names(modality_palette)) +
          scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
          labs(x = toupper(indicator), y = "Share of records", fill = "PA modality") +
          rr_hist_theme(base_size = 12) +
          theme(legend.position = "none")
        p_out <- p_rr / p_hist + patchwork::plot_layout(heights = c(4.0, 1.25), guides = "collect")
      } else {
        p_out <- p_rr
      }
      
      out_base <- paste0(city_name, "_ACTIVITY_3PLUS1_RR_overlay_", indicator, "_", base_model)
      write_csv(rr_df, file.path(city_output_dir, paste0(out_base, "_rr_plot_data.csv")))
      if (exists("hist_df") && nrow(hist_df) > 0) {
        write_csv(hist_df, file.path(city_output_dir, paste0(out_base, "_hist_plot_data.csv")))
      }
      write_csv(rr_df, file.path(city_output_dir, paste0(out_base, ".csv")))
      ggsave(
        file.path(city_output_dir, paste0(out_base, ".png")),
        p_out,
        width = 12.5,
        height = 8.4,
        dpi = 300,
        bg = "white"
      )
      saved_n <<- saved_n + 1L
      NULL
    }, error = function(e) {
      cat("    Warning: Activity 3+1 overlay failed for ", base_model, ": ", conditionMessage(e), "\n", sep = "")
      NULL
    })
  }
  
  if (saved_n > 0) {
    cat("    ✓ Activity 3+1 RR overlay plots saved: ", saved_n, "\n", sep = "")
  }
  invisible(saved_n > 0)
}

create_stage1_master_rr_panels <- function(stage1_results, output_dir) {
  if (is.null(stage1_results) || length(stage1_results) == 0) return(0L)
  master_dir <- file.path(output_dir, "MASTER_VISUALS", "STAGE1_CITY_RR")
  dir.create(master_dir, showWarnings = FALSE, recursive = TRUE)
  
  rr_records <- list()
  exposure_records <- list()
  for (city_key in names(stage1_results)) {
    city_models <- stage1_results[[city_key]]
    indicator <- ifelse(grepl("_exceeded_quantity$", city_key), "exceeded_quantity", "cehwi")
    city_clean <- sub("_(cehwi|exceeded_quantity)$", "", city_key)
    for (model_type in names(city_models)) {
      model_result <- city_models[[model_type]]
      if (is.null(model_result$pred_df) || nrow(model_result$pred_df) == 0) next
      pred_i <- model_result$pred_df %>%
        mutate(
          city = city_clean,
          indicator = indicator,
          model_type = model_type,
          rr_low_plot = pmax(rr_low, 0.01),
          rr_high_plot = pmin(rr_high, 10),
          rr_plot = pmin(pmax(rr, 0.01), 10)
        )
      rr_records[[length(rr_records) + 1]] <- pred_i
      
      exposure_values <- model_result$cehwi_data
      if (!is.null(exposure_values) && length(exposure_values) > 0) {
        exposure_records[[length(exposure_records) + 1]] <- tibble(
          city = city_clean,
          indicator = indicator,
          model_type = model_type,
          exposure = exposure_values[is.finite(exposure_values) & exposure_values > 0]
        )
      }
    }
  }
  
  if (length(rr_records) == 0) return(0L)
  rr_all <- bind_rows(rr_records) %>% filter(is.finite(cehwi), is.finite(rr_plot))
  exposure_all <- if (length(exposure_records) > 0) bind_rows(exposure_records) else tibble()
  saved_n <- 0L
  
  for (ind in unique(rr_all$indicator)) {
    for (mtype in unique(rr_all$model_type)) {
      plot_df <- rr_all %>% filter(indicator == ind, model_type == mtype)
      if (nrow(plot_df) == 0) next
      n_city <- n_distinct(plot_df$city)
      facet_cols <- if (n_city > 48) 6L else if (n_city > 24) 5L else 4L
      y_upper <- quantile(plot_df$rr_high_plot, 0.95, na.rm = TRUE)
      if (!is.finite(y_upper) || y_upper < 1.5) y_upper <- 2
      y_upper <- min(10, max(2, y_upper))
      
      p_rr <- ggplot(plot_df, aes(x = cehwi, y = rr_plot)) +
        geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.4) +
        geom_ribbon(aes(ymin = rr_low_plot, ymax = rr_high_plot), fill = "gray82", alpha = 0.75, color = NA) +
        geom_line(color = stage1_model_color(mtype), linewidth = 0.7) +
        facet_wrap(~ city, scales = "free_x", ncol = facet_cols) +
        coord_cartesian(ylim = c(0, y_upper), clip = "off") +
        labs(
          title = paste0("Stage 1 City-Specific Cumulative RR: ", toupper(ind), " - ", stage1_model_display_label(mtype)),
          subtitle = paste0(n_city, " cities | Reduced cumulative DLNM curves | CI clipped only for display when extreme"),
          x = NULL,
          y = "RR"
        ) +
        theme_minimal(base_size = 16) +
        theme(
          plot.title = element_text(face = "bold", size = 22),
          plot.subtitle = element_text(size = 14, color = "gray35"),
          strip.text = element_text(face = "bold", size = 12),
          axis.text = element_text(size = 10),
          axis.title.y = element_text(face = "bold", size = 14),
          panel.grid = element_blank(),
          axis.line = element_line(color = "gray35", linewidth = 0.35)
        )
      
      exp_df <- exposure_all %>% filter(indicator == ind, model_type == mtype, is.finite(exposure), exposure > 0)
      if (nrow(exp_df) > 0) {
        p_hist <- ggplot(exp_df, aes(x = exposure)) +
          geom_histogram(bins = 35, fill = "#5B8DB8", color = "white", linewidth = 0.25, alpha = 0.9) +
          labs(x = "Exposure distribution across included city-grids", y = "Count") +
          theme_minimal(base_size = 16) +
          theme(panel.grid = element_blank(), axis.title = element_text(face = "bold"))
        p_out <- p_rr / p_hist + patchwork::plot_layout(heights = c(4, 1))
      } else {
        p_out <- p_rr
      }
      
      out_file <- file.path(master_dir, paste0("MASTER_stage1_RR_", ind, "_", mtype, ".png"))
      plot_height <- min(48, max(12, ceiling(n_city / facet_cols) * 2.7 + ifelse(nrow(exp_df) > 0, 4, 2)))
      plot_width <- if (facet_cols >= 6) 24 else 22
      ggsave(out_file, p_out, width = plot_width, height = plot_height, dpi = 300, bg = "white")
      saved_n <- saved_n + 1L
    }
  }
  cat("  ✓ Stage-1大师合并RR图已保存:", saved_n, "张\n")
  saved_n
}

safe_create_stage1_master_rr_panels <- function(stage1_results, output_dir) {
  tryCatch(
    create_stage1_master_rr_panels(stage1_results, output_dir),
    error = function(e) {
      master_dir <- file.path(output_dir, "MASTER_VISUALS", "STAGE1_CITY_RR")
      dir.create(master_dir, showWarnings = FALSE, recursive = TRUE)
      msg <- conditionMessage(e)
      writeLines(
        c(
          "Stage-1 master RR panel generation failed.",
          paste0("Error: ", msg),
          "The city-level stage-1 model RDS/CSV/PNG files are still usable.",
          "You can regenerate this master visualization later from saved RDS files."
        ),
        file.path(master_dir, "MASTER_stage1_RR_error.txt")
      )
      cat("  Warning: Stage-1 master RR panel generation failed: ", msg, "\n", sep = "")
      0L
    }
  )
}

write_stage1_failure_report <- function(failed_cities, output_dir) {
  if (is.null(failed_cities) || length(failed_cities) == 0) {
    return(invisible(NULL))
  }
  city_indicator <- names(failed_cities)
  indicator <- ifelse(grepl("_exceeded_quantity$", city_indicator), "exceeded_quantity", "cehwi")
  city_code <- sub("_(cehwi|exceeded_quantity)$", "", city_indicator)
  log_file <- file.path(output_dir, city_indicator, paste0(city_code, "_DLNM_log.txt"))
  failure_df <- tibble(
    city_indicator = city_indicator,
    city_code = city_code,
    indicator = indicator,
    failure_reason = unname(unlist(failed_cities, use.names = FALSE)),
    log_file = ifelse(file.exists(log_file), log_file, NA_character_)
  )
  write_csv(failure_df, file.path(output_dir, "stage1_failed_city_indicator_summary.csv"))
  cat("  Stage-1 failed city-indicator combinations:\n")
  for (i in seq_len(nrow(failure_df))) {
    cat("    - ", failure_df$city_indicator[i], ": ", failure_df$failure_reason[i], "\n", sep = "")
  }
  invisible(failure_df)
}

create_stage2_master_rr_panels <- function(output_dir) {
  pooled_dirs <- list.dirs(output_dir, recursive = FALSE, full.names = TRUE)
  pooled_dirs <- pooled_dirs[grepl("^POOLED_META_(cehwi|exceeded_quantity)_(composite|day|night)(_(all|ride|run|walk))?$", basename(pooled_dirs), ignore.case = TRUE)]
  if (length(pooled_dirs) == 0) return(FALSE)
  
  rr_records <- list()
  for (pooled_dir in pooled_dirs) {
    rr_candidates <- file.path(pooled_dir, c("pooled_RR_data.csv", "pooled_RR_curve.csv"))
    rr_file <- rr_candidates[file.exists(rr_candidates)][1]
    if (is.na(rr_file) || !file.exists(rr_file)) next
    dir_match <- str_match(basename(pooled_dir), "^POOLED_META_(cehwi|exceeded_quantity)_((composite|day|night)(_(all|ride|run|walk))?)$")
    if (any(is.na(dir_match))) next
    rr_df <- tryCatch(read_csv(rr_file, show_col_types = FALSE), error = function(e) NULL)
    if (is.null(rr_df) || nrow(rr_df) == 0) next
    model_name <- normalize_stage1_model_name(dir_match[3])
    rr_records[[length(rr_records) + 1]] <- rr_df %>%
      mutate(
        indicator = dir_match[2],
        indicator_label = case_when(
          indicator == "cehwi" ~ "CEHWI",
          indicator == "exceeded_quantity" ~ "Exceeded quantity",
          TRUE ~ indicator
        ),
        model_type = model_name,
        base_model = model_base_type(model_name),
        base_model_label = factor(str_to_title(base_model), levels = c("Composite", "Day", "Night")),
        activity_type = model_activity_type(model_name),
        activity_label = unname(activity_modality_labels()[activity_type]),
        model_label = stage1_model_display_label(model_name),
        rr_low_plot = pmax(rr_low, 0.01),
        rr_high_plot = pmin(rr_high, 10),
        rr_plot = pmin(pmax(rr, 0.01), 10)
      )
  }
  
  if (length(rr_records) == 0) return(FALSE)
  rr_all <- bind_rows(rr_records) %>% filter(is.finite(cehwi), is.finite(rr_plot))
  if (nrow(rr_all) == 0) return(FALSE)
  
  master_dir <- file.path(output_dir, "MASTER_VISUALS", "STAGE2_META_RR")
  dir.create(master_dir, showWarnings = FALSE, recursive = TRUE)
  y_upper <- quantile(rr_all$rr_high_plot, 0.95, na.rm = TRUE)
  if (!is.finite(y_upper) || y_upper < 1.5) y_upper <- 2
  y_upper <- min(10, max(2, y_upper))
  
  has_activity_split <- any(rr_all$activity_type != "all", na.rm = TRUE)
  # R-stage exposure objects do not contain true ride/run/walk composition by
  # exposure bin. Keep the stage-2 R master plot line-only; the corrected
  # stacked modal histograms are produced after stage 2 by the Python
  # Nature-ready true-composition suite.
  if (FALSE && has_activity_split) {
    activity_labels <- activity_modality_labels()
    activity_palette <- setNames(
      unname(activity_modality_colors()[names(activity_labels)]),
      unname(activity_labels)
    )
    p_master <- ggplot(rr_all, aes(x = cehwi, y = rr_plot)) +
      geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.5) +
      geom_ribbon(
        data = rr_all %>% filter(activity_type == "all"),
        aes(ymin = rr_low_plot, ymax = rr_high_plot),
        fill = "#D1D5DB",
        alpha = 0.55,
        color = NA
      ) +
      geom_line(aes(color = activity_label, group = activity_label), linewidth = 1.05) +
      scale_color_manual(values = activity_palette, name = "PA modality") +
      facet_grid(indicator_label ~ base_model_label, scales = "free_x") +
      coord_cartesian(ylim = c(0, y_upper), clip = "on") +
      labs(
        title = "Stage 2 Pooled Meta-Regression RR Curves: All + Ride + Run + Walk",
        subtitle = "Gray ribbon shows the combined-activity 95% CI; modality lines are point estimates. CI clipped only for display when extreme.",
        x = "Exposure",
        y = "Relative Risk (RR)"
      )
  } else {
    p_master <- ggplot(rr_all, aes(x = cehwi, y = rr_plot)) +
      geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.5) +
      geom_ribbon(aes(ymin = rr_low_plot, ymax = rr_high_plot), fill = "gray82", alpha = 0.75, color = NA) +
      geom_line(color = "#1F4E79", linewidth = 1.1) +
      facet_grid(indicator_label ~ base_model_label, scales = "free_x") +
      coord_cartesian(ylim = c(0, y_upper), clip = "on") +
      labs(
        title = "Stage 2 Pooled Meta-Regression RR Curves",
        subtitle = "Overall cumulative exposure-response after first-stage lag reduction | CI clipped only for display when extreme",
        x = "Exposure",
        y = "Relative Risk (RR)"
      )
  }
  
  p_master <- p_master +
    theme_minimal(base_size = 18) +
    theme(
      plot.title = element_text(face = "bold", size = 24),
      plot.subtitle = element_text(size = 14, color = "gray35"),
      strip.text = element_text(face = "bold", size = 14),
      legend.position = "top",
      panel.spacing = grid::unit(16, "pt"),
      panel.grid = element_blank(),
      axis.title = element_text(face = "bold")
    )
  
  hist_df <- NULL
  if (has_activity_split) {
    stage1_results_for_hist <- tryCatch(collect_saved_stage1_results(output_dir), error = function(e) NULL)
    if (!is.null(stage1_results_for_hist)) {
      hist_records <- list()
      for (ind in c("cehwi", "exceeded_quantity")) {
        for (base_model in c("composite", "day", "night")) {
          for (activity in c("ride", "run", "walk")) {
            model_name <- stage1_model_name(base_model, activity, STAGE1_ACTIVITY_MODE)
            exposure_values <- collect_exposure_values_from_stage1(stage1_results_for_hist, ind, model_name)
            exposure_values <- exposure_values[is.finite(exposure_values) & exposure_values > 0]
            if (length(exposure_values) == 0) next
            hist_records[[length(hist_records) + 1]] <- tibble(
              exposure = exposure_values,
              indicator = ind,
              indicator_label = case_when(
                ind == "cehwi" ~ "CEHWI",
                ind == "exceeded_quantity" ~ "Exceeded quantity",
                TRUE ~ ind
              ),
              base_model = base_model,
              base_model_label = factor(str_to_title(base_model), levels = c("Composite", "Day", "Night")),
              activity_type = activity,
              activity_label = unname(activity_modality_labels()[activity])
            )
          }
        }
      }
      if (length(hist_records) > 0) {
        hist_df <- bind_rows(hist_records)
        p_hist <- ggplot(hist_df, aes(x = exposure, fill = activity_label)) +
          geom_histogram(
            aes(y = after_stat(count / sum(count))),
            bins = 35,
            position = "stack",
            color = "white",
            linewidth = 0.18,
            alpha = 0.9
          ) +
          scale_fill_manual(values = activity_palette, breaks = names(activity_palette), guide = "none") +
          scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
          facet_grid(indicator_label ~ base_model_label, scales = "free_x") +
          labs(x = "Exposure distribution by PA modality", y = "Share") +
          theme_minimal(base_size = 12) +
          theme(
            strip.text = element_blank(),
            panel.spacing = grid::unit(16, "pt"),
            panel.grid = element_blank(),
            axis.title = element_text(face = "bold"),
            axis.text.x = element_text(size = 8)
          )
        p_master <- p_master / p_hist + patchwork::plot_layout(heights = c(4.2, 1.25), guides = "collect")
      }
    }
  }
  
  out_suffix <- analysis_output_suffix()
  write_csv(rr_all, file.path(master_dir, paste0("MASTER_stage2_pooled_RR_all_models_plot_data", out_suffix, ".csv")))
  if (!is.null(hist_df) && nrow(hist_df) > 0) {
    write_csv(hist_df, file.path(master_dir, paste0("MASTER_stage2_pooled_RR_all_models_hist_plot_data", out_suffix, ".csv")))
  }
  ggsave(
    file.path(master_dir, paste0("MASTER_stage2_pooled_RR_all_models", out_suffix, ".png")),
    p_master,
    width = 18,
    height = if (has_activity_split && !is.null(hist_df) && nrow(hist_df) > 0) 14 else if (has_activity_split) 12 else 11,
    dpi = 300,
    bg = "white"
  )
  cat("  ✓ Stage-2大师合并RR图已保存: MASTER_stage2_pooled_RR_all_models.png\n")
  TRUE
}

# ========== 【V5新增】读取城市气象数据 ==========
save_activity_pooled_rr_overlay <- function(parent_dir,
                                            output_dir,
                                            file_prefix = "activity_3plus1_pooled_RR_overlay",
                                            title_prefix = NULL,
                                            mode = NULL,
                                            model_results_by_indicator = NULL) {
  if (!is_activity_split_mode()) return(FALSE)
  child_dirs <- list.dirs(parent_dir, recursive = FALSE, full.names = TRUE)
  if (length(child_dirs) == 0) return(FALSE)
  
  rr_records <- list()
  for (rr_dir in child_dirs) {
    dir_match <- str_match(
      basename(rr_dir),
      "^(cehwi|exceeded_quantity)_((composite|day|night)(_(all|ride|run|walk))?)(?:_.*)?$"
    )
    if (all(is.na(dir_match))) next
    model_name <- normalize_stage1_model_name(dir_match[3])
    rr_candidates <- file.path(rr_dir, c("pooled_RR_data.csv", "pooled_RR_curve.csv"))
    rr_file <- rr_candidates[file.exists(rr_candidates)][1]
    if (is.na(rr_file) || !file.exists(rr_file)) next
    rr_df <- tryCatch(read_csv(rr_file, show_col_types = FALSE), error = function(e) NULL)
    if (is.null(rr_df) || nrow(rr_df) == 0) next
    
    rr_records[[length(rr_records) + 1]] <- rr_df %>%
      mutate(
        indicator = dir_match[2],
        indicator_label = case_when(
          indicator == "cehwi" ~ "CEHWI",
          indicator == "exceeded_quantity" ~ "Exceeded quantity",
          TRUE ~ indicator
        ),
        model_type = model_name,
        base_model = model_base_type(model_name),
        base_model_label = factor(str_to_title(base_model), levels = c("Composite", "Day", "Night")),
        activity_type = model_activity_type(model_name),
        activity_label = unname(activity_modality_labels()[activity_type])
      )
  }
  
  if (length(rr_records) == 0) return(FALSE)
  rr_all <- bind_rows(rr_records) %>%
    filter(is.finite(cehwi), is.finite(rr), rr >= 0)
  if (nrow(rr_all) == 0 || !any(rr_all$activity_type != "all", na.rm = TRUE)) return(FALSE)
  
  hist_records <- list()
  # Do not derive PA-modality histograms from duplicated exposure vectors in
  # first-stage model objects. The correct modality composition requires raw PA
  # trip counts by exposure bin and is handled by the Nature-ready post-process.
  if (FALSE && !is.null(model_results_by_indicator) && length(model_results_by_indicator) > 0) {
    for (ind in c("cehwi", "exceeded_quantity")) {
      if (is.null(model_results_by_indicator[[ind]])) next
      for (base_model in c("composite", "day", "night")) {
        for (activity in c("ride", "run", "walk")) {
          model_name <- stage1_model_name(base_model, activity, STAGE1_ACTIVITY_MODE)
          result_list <- model_results_by_indicator[[ind]][[model_name]]
          if (is.null(result_list) || length(result_list) == 0) next
          exposure_values <- unlist(lapply(result_list, function(x) {
            if (!is.null(x$cehwi_data)) x$cehwi_data else NULL
          }))
          exposure_values <- exposure_values[is.finite(exposure_values) & exposure_values > 0]
          if (length(exposure_values) == 0) next
          hist_records[[length(hist_records) + 1]] <- tibble(
            exposure = exposure_values,
            indicator = ind,
            indicator_label = case_when(
              ind == "cehwi" ~ "CEHWI",
              ind == "exceeded_quantity" ~ "Exceeded quantity",
              TRUE ~ ind
            ),
            base_model = base_model,
            base_model_label = factor(str_to_title(base_model), levels = c("Composite", "Day", "Night")),
            activity_type = activity,
            activity_label = unname(activity_modality_labels()[activity])
          )
        }
      }
    }
  }
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  # Use panel-specific display limits. CEHWI and exceeded quantity can have very
  # different RR ranges; a single global y-axis compresses one exposure family and
  # makes the AF_PERCENTILE_SUMMARY overlay figures misleading.
  panel_limits <- rr_all %>%
    filter(is.finite(rr), rr >= 0) %>%
    group_by(indicator_label, base_model_label) %>%
    summarise(
      y_upper = suppressWarnings(as.numeric(quantile(rr, 0.95, na.rm = TRUE))),
      .groups = "drop"
    ) %>%
    mutate(
      y_upper = ifelse(!is.finite(y_upper) | y_upper < 1.5, 1.5, y_upper),
      # Keep DTW small-cluster plots readable: extreme modal RR values are
      # retained in CSV and omitted from display rather than capped into artifacts.
      y_upper = pmin(4, pmax(1.5, y_upper * 1.15)),
      y_lower = 0
    )
  
  rr_all <- rr_all %>%
    left_join(panel_limits, by = c("indicator_label", "base_model_label")) %>%
    mutate(
      y_lower = ifelse(is.finite(y_lower), y_lower, 0),
      y_upper = ifelse(is.finite(y_upper), y_upper, 1.5)
    ) %>%
    mutate(
      rr_display_outlier = !is.finite(rr) | rr < y_lower | rr > y_upper,
      rr_line_plot = ifelse(rr_display_outlier, NA_real_, rr),
      rr_low_plot = ifelse(is.finite(rr_low), pmax(rr_low, y_lower), NA_real_),
      rr_high_plot = ifelse(is.finite(rr_high), pmin(rr_high, y_upper), NA_real_),
      rr_ribbon_valid = is.finite(rr_low_plot) & is.finite(rr_high_plot) & rr_high_plot >= rr_low_plot
    )
  
  activity_labels <- activity_modality_labels()
  activity_palette <- setNames(
    unname(activity_modality_colors()[names(activity_labels)]),
    unname(activity_labels)
  )
  title_text <- if (is.null(title_prefix) || !nzchar(title_prefix)) {
    "Pooled RR Curves: All + Ride + Run + Walk"
  } else {
    paste0(title_prefix, " - Pooled RR Curves: All + Ride + Run + Walk")
  }
  
  p_overlay <- ggplot(rr_all, aes(x = cehwi)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.45) +
    geom_ribbon(
      data = rr_all %>% filter(activity_type == "all", rr_ribbon_valid),
      aes(ymin = rr_low_plot, ymax = rr_high_plot),
      fill = "#D1D5DB",
      alpha = 0.55,
      color = NA
    ) +
    geom_line(
      data = rr_all %>% filter(is.finite(rr_line_plot)),
      aes(y = rr_line_plot, color = activity_label,
          group = interaction(indicator_label, base_model_label, activity_label, drop = TRUE)),
      linewidth = 1.15,
      na.rm = TRUE
    ) +
    scale_color_manual(values = activity_palette, name = "PA modality") +
    facet_grid(indicator_label ~ base_model_label, scales = "free") +
    coord_cartesian(clip = "on") +
    labs(
      title = title_text,
      subtitle = "Gray ribbon shows the combined-activity 95% CI; modality lines are point estimates. Extreme line segments are omitted for display and kept in CSV.",
      x = "Exposure",
      y = "Relative Risk (RR)"
    ) +
    theme_minimal(base_size = 15) +
    theme(
      plot.title = element_text(face = "bold", size = 18),
      plot.subtitle = element_text(size = 11, color = "gray35"),
      strip.text = element_text(face = "bold", size = 11),
      legend.position = "top",
      panel.spacing = grid::unit(14, "pt"),
      panel.grid = element_blank(),
      axis.title = element_text(face = "bold")
    )
  
  if (length(hist_records) > 0) {
    hist_df <- bind_rows(hist_records)
    p_hist <- ggplot(hist_df, aes(x = exposure, fill = activity_label)) +
      geom_histogram(
        aes(y = after_stat(count / sum(count))),
        bins = 35,
        position = "stack",
        color = "white",
        linewidth = 0.18,
        alpha = 0.9
      ) +
      scale_fill_manual(values = activity_palette, breaks = names(activity_palette), guide = "none") +
      scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
      facet_grid(indicator_label ~ base_model_label, scales = "free_x") +
      labs(x = "Exposure distribution by PA modality", y = "Share") +
      theme_minimal(base_size = 12) +
      theme(
        strip.text = element_blank(),
        panel.spacing = grid::unit(14, "pt"),
        panel.grid = element_blank(),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text(size = 8)
      )
    p_overlay <- p_overlay / p_hist + patchwork::plot_layout(heights = c(4.0, 1.25), guides = "collect")
  }
  
  out_suffix <- analysis_output_suffix(mode)
  safe_write_csv(rr_all, file.path(output_dir, paste0(file_prefix, "_plot_data", out_suffix, ".csv")), label = "activity pooled RR overlay plot data")
  safe_write_csv(
    rr_all %>% filter(rr_display_outlier),
    file.path(output_dir, paste0(file_prefix, "_display_outliers", out_suffix, ".csv")),
    label = "activity pooled RR overlay display outliers"
  )
  if (length(hist_records) > 0) {
    safe_write_csv(bind_rows(hist_records), file.path(output_dir, paste0(file_prefix, "_hist_plot_data", out_suffix, ".csv")), label = "activity pooled RR overlay histogram data")
  }
  ggsave(
    file.path(output_dir, paste0(file_prefix, out_suffix, ".png")),
    p_overlay,
    width = 15,
    height = if (length(hist_records) > 0) 11.5 else 9.5,
    dpi = 300,
    bg = "white"
  )
  TRUE
}

load_weather_data <- function(city_name, start_date, end_date) {
  # 【V6修复】规范化城市名（先去点号，再空格→下划线）
  # "St. Louis" → "St Louis" → "St_Louis" ✓
  # "San Antonio" → "San Antonio" → "San_Antonio" ✓
  city_name_normalized <- gsub("\\.", "", city_name)  # 先去掉点号
  city_name_normalized <- gsub(" ", "_", city_name_normalized)  # 再空格→下划线
  
  # 城市名称到气象文件的映射（处理空格和特殊命名）
  weather_city_mapping <- list(
    "Fort_Worth" = "Fort_Worth_TX",
    "Kansas_City" = "Kansas_City_MO",
    "Las_Vegas" = "Las_Vegas_NV",
    "Long_Beach" = "Long_Beach_CA",
    "Los_Angeles" = "Los_Angeles_CA",
    "New_York" = "New_York_NY",
    "Overland_Park" = "Overland_Park_KS",
    "Salt_Lake_City" = "Salt_Lake_City_UT",
    "San_Antonio" = "San_Antonio_TX",
    "San_Bernardino" = "San_Bernardino_CA",
    "San_Diego" = "San_Diego_CA",
    "San_Francisco" = "San_Francisco_CA",
    "San_Jose" = "San_Jose_CA",
    "Santa_Ana" = "Santa_Ana_CA",
    "St_Louis" = "St_Louis_MO",
    "St_Petersburg" = "St_Petersburg_FL",
    "Virginia_Beach" = "Virginia_Beach_VA",
    # 新增城市
    "Bakersfield" = "Bakersfield_CA",
    "Charleston" = "Charleston_SC",
    "Columbia" = "Columbia_SC",
    "Fresno" = "Fresno_CA",
    "Tucson" = "Tucson_AZ"
  )
  
  # 其他城市的州缩写（根据实际文件）
  default_state_mapping <- list(
    "Abilene" = "TX", "Amarillo" = "TX", "Arlington" = "TX",
    "Atlanta" = "GA", "Aurora" = "CO", "Austin" = "TX", "Baltimore" = "MD",
    "Boston" = "MA", "Cape_Coral" = "FL", "Chandler" = "AZ",
    "Charlotte" = "NC", "Chicago" = "IL", "Cincinnati" = "OH",
    "Clearwater" = "FL", "Cleveland" = "OH", "Columbus" = "OH",
    "Corpus_Christi" = "TX", "Dallas" = "TX", "Denver" = "CO",
    "Detroit" = "MI", "Gilbert" = "AZ", "Henderson" = "NV",
    "Hollywood" = "FL", "Houston" = "TX", "Indianapolis" = "IN",
    "Jacksonville" = "FL", "Louisville" = "KY", "Lubbock" = "TX",
    "Mesa" = "AZ", "Miami" = "FL", "Milwaukee" = "WI", "Minneapolis" = "MN",
    "Miramar" = "FL", "Nashville" = "TN", "Newark" = "NJ", "Oakland" = "CA",
    "Oklahoma_City" = "OK", "Orlando" = "FL", "Palm_Bay" = "FL",
    "Philadelphia" = "PA", "Phoenix" = "AZ", "Pittsburgh" = "PA", "Portland" = "OR",
    "Raleigh" = "NC", "Richmond" = "VA", "Riverside" = "CA", "Sacramento" = "CA",
    "Scottsdale" = "AZ", "Seattle" = "WA", "Tallahassee" = "FL",
    "Tampa" = "FL", "Visalia" = "CA", "Washington" = "DC"
  )
  
  # 【V6修复】使用规范化后的城市名进行匹配
  # 确定气象文件名前缀
  if (city_name_normalized %in% names(weather_city_mapping)) {
    weather_file_prefix <- weather_city_mapping[[city_name_normalized]]
  } else if (city_name_normalized %in% names(default_state_mapping)) {
    weather_file_prefix <- paste0(city_name_normalized, "_", default_state_mapping[[city_name_normalized]])
  } else {
    cat("  ⚠ 未找到城市", city_name, "(规范化:", city_name_normalized, ")的气象文件映射\n")
    return(NULL)
  }
  
  weather_file <- file.path(BASE_DIR, "VAR", "US_Weather_Data_2010_2024", 
                            paste0(weather_file_prefix, "_2010_2024.csv"))
  
  if (!file.exists(weather_file)) {
    cat("  ⚠ 气象数据文件不存在:", weather_file, "\n")
    return(NULL)
  }
  
  weather_data <- tryCatch({
    df_raw <- read_csv(weather_file, show_col_types = FALSE) %>%
      mutate(date = as.Date(date)) %>%
      filter(date >= start_date & date <= end_date)
    
    precip_col <- intersect(c("precipitation", "prcp"), names(df_raw))[1]
    wind_col <- intersect(c("wind_speed", "wspd"), names(df_raw))[1]
    rh_col <- intersect(c("relative_humidity", "RH", "rh", "rhum", "humidity"), names(df_raw))[1]
    rh_day_col <- intersect(c("rhum_day", "relative_humidity_day", "humidity_day"), names(df_raw))[1]
    rh_night_col <- intersect(c("rhum_night", "relative_humidity_night", "humidity_night"), names(df_raw))[1]
    
    get_num_col <- function(data, col_name, default = NA_real_) {
      if (is.na(col_name) || !col_name %in% names(data)) {
        return(rep(default, nrow(data)))
      }
      as.numeric(data[[col_name]])
    }
    
    precip_vec <- get_num_col(df_raw, precip_col, default = 0)
    wind_vec <- get_num_col(df_raw, wind_col, default = 0)
    rh_vec <- get_num_col(df_raw, rh_col)
    rh_day_vec <- get_num_col(df_raw, rh_day_col)
    rh_night_vec <- get_num_col(df_raw, rh_night_col)
    
    cat("  气象列映射:",
        "relative_humidity =", ifelse(is.na(rh_col), "NA", rh_col), "|",
        "RH_day =", ifelse(is.na(rh_day_col), "NA", rh_day_col), "|",
        "RH_night =", ifelse(is.na(rh_night_col), "NA", rh_night_col), "|",
        "precipitation =", ifelse(is.na(precip_col), "NA", precip_col), "|",
        "wind_speed =", ifelse(is.na(wind_col), "NA", wind_col), "\n")
    
    # RH = relative humidity，相对湿度；若缺失则回退到昼夜均值。
    df_processed <- tibble(
      date = df_raw$date,
      relative_humidity = case_when(
        !is.na(rh_vec) ~ rh_vec,
        !is.na(rh_day_vec) & !is.na(rh_night_vec) ~ (rh_day_vec + rh_night_vec) / 2,
        !is.na(rh_day_vec) ~ rh_day_vec,
        !is.na(rh_night_vec) ~ rh_night_vec,
        TRUE ~ NA_real_
      ),
      precipitation = ifelse(is.na(precip_vec), 0, precip_vec),
      wind_speed = ifelse(is.na(wind_vec), 0, wind_vec)
    )
    rh_missing_raw <- sum(!is.finite(df_processed$relative_humidity))
    rh_unique_raw <- length(unique(df_processed$relative_humidity[is.finite(df_processed$relative_humidity)]))
    cat("  RH diagnostic summary: missing/non-finite = ",
        rh_missing_raw, "/", nrow(df_processed),
        ", finite unique = ", rh_unique_raw, "\n", sep = "")
    
    # 再次检查RH是否还有NA（用整体均值填充）
    if (any(is.na(df_processed$relative_humidity))) {
      rh_mean <- mean(df_processed$relative_humidity, na.rm = TRUE)
      if (!is.finite(rh_mean)) rh_mean <- 0
      df_processed <- df_processed %>%
        mutate(relative_humidity = ifelse(is.na(relative_humidity), rh_mean, relative_humidity))
    }
    
    return(df_processed %>% select(date, precipitation, wind_speed))
    
  }, error = function(e) {
    cat("  ✗ 读取气象数据失败:", e$message, "\n")
    return(NULL)
  })
  
  return(weather_data)
}

# ========== 【V5.3修复】构建完整日序列面板（补齐0出行日） ==========
build_complete_daily_panel <- function(df_long, hw_all, start_date, end_date, by_activity = FALSE) {
  activity_levels <- c("ride", "run", "walk")
  if (isTRUE(by_activity) && "trip_type" %in% names(df_long)) {
    trip_daily <- df_long %>%
      filter(trip_type %in% activity_levels) %>%
      group_by(fish_id, date, day_night, trip_type) %>%
      summarise(trip_count = sum(trip_count, na.rm = TRUE), .groups = "drop")
  } else {
    trip_daily <- df_long %>%
      group_by(fish_id, date, day_night) %>%
      summarise(trip_count = sum(trip_count, na.rm = TRUE), .groups = "drop")
  }
  
  hw_daily <- hw_all %>%
    filter(date >= start_date & date <= end_date) %>%
    distinct(fish_id, date, .keep_all = TRUE)
  
  if (isTRUE(by_activity) && "trip_type" %in% names(df_long)) {
    panel_daily <- tidyr::expand_grid(
      hw_daily,
      day_night = c("day", "night"),
      trip_type = activity_levels
    ) %>%
      left_join(trip_daily, by = c("fish_id", "date", "day_night", "trip_type")) %>%
      mutate(trip_count = dplyr::coalesce(trip_count, 0)) %>%
      arrange(fish_id, date, day_night, trip_type)
  } else {
    panel_daily <- tidyr::expand_grid(
      hw_daily,
      day_night = c("day", "night")
    ) %>%
      left_join(trip_daily, by = c("fish_id", "date", "day_night")) %>%
      mutate(trip_count = dplyr::coalesce(trip_count, 0)) %>%
      arrange(fish_id, date, day_night)
  }
  
  return(panel_daily)
}

# ========== 主分析函数: 城市级 ==========

stage1_model_rds_path <- function(city_output_dir, city_code, model_name) {
  file.path(city_output_dir, paste0(city_code, "_", model_name, "_DLNM_result.rds"))
}

append_stage1_status <- function(city_code, indicator, model_name, status, note = "") {
  status_file <- file.path(OUTPUT_DIR, "stage1_resume_status.csv")
  row <- tibble(
    time = as.character(Sys.time()),
    city_code = city_code,
    indicator = indicator,
    model_name = model_name,
    status = status,
    note = note
  )
  if (!file.exists(status_file)) {
    write_csv(row, status_file)
  } else {
    write.table(
      row,
      file = status_file,
      sep = ",",
      row.names = FALSE,
      col.names = FALSE,
      append = TRUE,
      qmethod = "double",
      fileEncoding = "UTF-8"
    )
  }
  invisible(status_file)
}

load_completed_stage1_model <- function(city_output_dir, city_code, indicator, model_name) {
  if (!isTRUE(STAGE1_RESUME) || isTRUE(STAGE1_FORCE_RERUN)) return(NULL)
  rds_file <- stage1_model_rds_path(city_output_dir, city_code, model_name)
  if (!file.exists(rds_file)) return(NULL)
  if (is.na(file.info(rds_file)$size) || file.info(rds_file)$size <= 0) return(NULL)
  
  model_result <- tryCatch(read_rds(rds_file), error = function(e) NULL)
  if (is.null(model_result)) {
    append_stage1_status(city_code, indicator, model_name, "RDS_READ_FAILED", rds_file)
    return(NULL)
  }
  model_result <- tryCatch(upgrade_stage1_result_to_reduced(model_result), error = function(e) model_result)
  model_result$city_code <- city_code
  model_result$indicator <- indicator
  model_result$model_name <- normalize_stage1_model_name(model_name)
  model_result$model_type <- model_base_type(model_name)
  model_result$activity_type <- model_activity_type(model_name)
  model_result$activity_label <- unname(activity_modality_labels()[model_result$activity_type])
  model_result$activity_analysis_mode <- activity_analysis_mode_label(STAGE1_ACTIVITY_MODE)

  if (!is_current_stage1_result(model_result)) {
    append_stage1_status(
      city_code, indicator, model_name, "RDS_STALE_REDO",
      paste0(rds_file, " | expected lag scenario=", LAG_SCENARIO_LABEL,
             " city MAX_LAG=", expected_stage1_max_lag(model_result))
    )
    return(NULL)
  }
  
  has_reduced <- identical(model_result$coef_type, "overall_cumulative_reduced") &&
    !is.null(model_result$coef) &&
    !is.null(model_result$vcov) &&
    !is.null(model_result$cb)
  if (!has_reduced) {
    append_stage1_status(city_code, indicator, model_name, "RDS_INCOMPLETE_REDO", rds_file)
    return(NULL)
  }
  
  append_stage1_status(city_code, indicator, model_name, "SKIPPED_EXISTING_RDS", rds_file)
  model_result
}

fit_or_resume_city_model <- function(city_output_dir,
                                     city_code,
                                     city_name,
                                     indicator,
                                     model_name,
                                     data,
                                     fit_label,
                                     min_obs,
                                     source_data = NULL) {
  normalized_model_name <- normalize_stage1_model_name(model_name)
  if (!normalized_model_name %in% STAGE1_MODEL_TYPES) {
    append_stage1_status(city_code, indicator, model_name, "SKIPPED_MODEL_FILTER", fit_label)
    return(NULL)
  }
  existing <- load_completed_stage1_model(city_output_dir, city_code, indicator, model_name)
  if (!is.null(existing)) {
    cat("    [Resume] Existing RDS found, skip fitting: ", city_code, " - ",
        toupper(indicator), " - ", toupper(model_name), "\n", sep = "")
    return(existing)
  }
  if (is.null(source_data)) {
    source_data <- tryCatch(get0("df", envir = parent.frame(), inherits = FALSE), error = function(e) NULL)
  }
  data <- ensure_temperature_control_columns(data, source_data = source_data)
  
  append_stage1_status(city_code, indicator, model_name, "STARTED", fit_label)
  result <- fit_dlnm_stage1(data, fit_label, indicator = indicator, min_obs = min_obs)
  if (is.null(result)) {
    append_stage1_status(city_code, indicator, model_name, "FAILED_OR_SKIPPED", fit_label)
    return(NULL)
  }
  append_stage1_status(city_code, indicator, model_name, "FITTED", fit_label)
  result
}

prepare_activity_stage1_df <- function(df, activity_key = "all") {
  activity_key <- tolower(activity_key)
  if (!"trip_type" %in% names(df)) return(df)
  
  if (activity_key == "all") {
    group_cols <- setdiff(names(df), c("trip_type", "trip_count"))
    return(
      df %>%
        group_by(across(all_of(group_cols))) %>%
        summarise(trip_count = sum(trip_count, na.rm = TRUE), .groups = "drop") %>%
        mutate(trip_type = "all") %>%
        relocate(trip_type, .after = day_night)
    )
  }
  
  df %>% filter(trip_type == activity_key)
}

analyze_city_dlnm <- function(city_code, city_name, trips_file, temp_dir, indicator = "cehwi") {
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("城市DLNM分析:", city_name, "(", city_code, ") - 指标:", toupper(indicator), "\n")
  cat(rep("=", 100), "\n\n", sep = "")
  
  city_output_dir <- file.path(OUTPUT_DIR, paste0(city_code, "_", indicator))
  dir.create(city_output_dir, showWarnings = FALSE, recursive = TRUE)
  
  if (isTRUE(STAGE1_RESUME) && !isTRUE(STAGE1_FORCE_RERUN)) {
    cat("  [Resume] Keeping existing city outputs; completed model RDS files will be skipped.\n")
  } else if (isTRUE(STAGE1_FORCE_RERUN)) {
    cat("  [Resume] Force rerun enabled; existing RDS files will be overwritten model-by-model.\n")
  }
  
  log_file <- file(file.path(city_output_dir, paste0(city_code, "_DLNM_log.txt")), "a")
  
  tryCatch({
    sink(log_file, type = "output", split = TRUE)
    
    cat("\n两阶段DLNM分析 (第一阶段) -", city_name, "\n")
    cat("时间:", as.character(Sys.time()), "\n")
    cat("分析时间段:", TIME_DESC, "\n\n")
    
    # Load data
    cat("[1/4] 加载数据...\n")
    df_raw <- read_csv(trips_file, show_col_types = FALSE)
    id_col <- names(df_raw)[1]
    trip_cols <- names(df_raw)[str_detect(names(df_raw), "\\d{4}/\\d{1,2}/\\d{1,2}_")]
    
    df_long <- df_raw %>%
      pivot_longer(cols = all_of(trip_cols), names_to = "column", values_to = "trip_count") %>%
      mutate(
        date = str_extract(column, "\\d{4}/\\d{1,2}/\\d{1,2}"),
        date = as.Date(date, format = "%Y/%m/%d"),
        trip_type = str_extract(column, "(walk|run|ride)"),
        day_night = str_extract(column, "(day|night)")
      ) %>%
      filter(!is.na(date), !is.na(trip_type), !is.na(day_night))
    
    names(df_long)[names(df_long) == id_col] <- "fish_id"
    
    cat("  Grids:", nrow(df_raw), "| 记录数:", nrow(df_long), "\n\n")
    
    # Load temperature/heatwave data
    cat("[2/4] 加载温度/热浪数据...\n")
    hw_files <- list.files(temp_dir, pattern = "_temperature__95\\.csv$", full.names = TRUE)
    
    hw_all <- map_dfr(hw_files, function(f) {
      df <- read_csv(f, show_col_types = FALSE, col_types = cols())
      df %>% mutate(fish_id = as.numeric(str_extract(basename(f), "\\d+")))
    })
    
    hw_all <- hw_all %>%
      rename(date = 1) %>%
      mutate(date = as.Date(date)) %>%
      clean_names()
    
    cat("  热浪记录:", nrow(hw_all), "\n\n")
    
    # 【V5修复】加载气象数据
    cat("[2.5/4] 加载气象数据...\n")
    # 直接传递城市名，让load_weather_data函数处理文件名映射
    df_weather <- load_weather_data(city_name, START_DATE, END_DATE)
    
    if (!is.null(df_weather)) {
      cat("  ✓ 气象数据已加载:", nrow(df_weather), "行\n")
      cat("  变量:", paste(setdiff(names(df_weather), "date"), collapse = ", "), "\n")
      
      # 合并气象数据到热浪数据
      hw_all <- hw_all %>%
        left_join(df_weather, by = "date")
      
      cat("  ✓ 气象数据已合并到热浪数据\n\n")
    } else {
      cat("  ⚠ 警告: 气象数据未加载，模型将不包含气象控制变量\n\n")
    }
    
    # ========== 确定可用的CEHWI列 ==========
    cat("  可用列（前30个）:", paste(head(names(hw_all), 30), collapse = ", "), "\n\n")
    
    # 确定列名
    if (indicator == "cehwi") {
      composite_col <- "composite_cehwi"
      daytime_col <- "daytime_cehwi"
      nighttime_col <- "nighttime_cehwi"
    } else {
      composite_col <- "composite_exceeded_quantity"
      daytime_col <- "daytime_exceeded_quantity"
      nighttime_col <- "nighttime_exceeded_quantity"
    }
    
    # 先筛选时间，再检查数据
    hw_all_filtered <- hw_all %>%
      filter(date >= START_DATE & date <= END_DATE)
    
    # 检查composite是否有非0值
    n_composite_positive <- sum(hw_all_filtered[[composite_col]] > 0, na.rm = TRUE)
    pct_composite <- n_composite_positive / nrow(hw_all_filtered) * 100
    
    cat("  检查热浪数据可用性:\n")
    cat("    ", composite_col, "> 0:", n_composite_positive, "(", round(pct_composite, 1), "%)\n")
    
    analysis_mode <- NULL
    cehwi_col_to_use <- NULL
    
    if (n_composite_positive >= 10) {
      # 有composite热浪 → 做3个模型（Composite/Day/Night）
      analysis_mode <- "composite"
      cehwi_col_to_use <- composite_col
      cat("  ✓ 分析模式: Composite (将做3个模型: Composite + Day + Night)\n")
      cat("    - 都使用", composite_col, "作为暴露变量\n")
      cat("    - Composite: 全天PA | Day: 昼间PA | Night: 夜间PA\n\n")
    } else {
      # composite不可用，检查day/night
      cat("  ⚠ Composite热浪不足，检查Day/Night...\n")
      
      n_day_positive <- sum(hw_all_filtered[[daytime_col]] > 0, na.rm = TRUE)
      n_night_positive <- sum(hw_all_filtered[[nighttime_col]] > 0, na.rm = TRUE)
      
      cat("    ", daytime_col, "> 0:", n_day_positive, "\n")
      cat("    ", nighttime_col, "> 0:", n_night_positive, "\n")
      
      if (n_day_positive >= 10 && n_night_positive >= 10) {
        # 两者都有，选多的那个
        if (n_day_positive >= n_night_positive) {
          analysis_mode <- "day_only"
          cehwi_col_to_use <- daytime_col
          cat("  ✓ 分析模式: Day Only (昼间热浪)\n\n")
        } else {
          analysis_mode <- "night_only"
          cehwi_col_to_use <- nighttime_col
          cat("  ✓ 分析模式: Night Only (夜间热浪)\n\n")
        }
      } else if (n_day_positive >= 10) {
        analysis_mode <- "day_only"
        cehwi_col_to_use <- daytime_col
        cat("  ✓ 分析模式: Day Only (昼间热浪)\n\n")
      } else if (n_night_positive >= 10) {
        analysis_mode <- "night_only"
        cehwi_col_to_use <- nighttime_col
        cat("  ✓ 分析模式: Night Only (夜间热浪)\n\n")
      } else {
        cat("  ✗ 所有类型热浪均不足，跳过该城市\n")
        cat("     建议选择其他时间段（如2015-2020或2010-2024）\n")
        return(NULL)
      }
    }
    
    # Merge
    cat("[3/4] 合并数据...\n")
    df <- build_complete_daily_panel(
      df_long, hw_all, START_DATE, END_DATE,
      by_activity = is_activity_split_mode(STAGE1_ACTIVITY_MODE)
    )
    
    cat("  合并后记录数:", nrow(df), "\n")
    
    # 【V6新逻辑】检查气象数据（只使用3个变量）
    weather_vars <- c("precipitation", "wind_speed")
    has_weather <- any(weather_vars %in% names(hw_all))
    
    if (has_weather) {
      available_weather_vars <- intersect(weather_vars, names(hw_all))
      cat("  【V6】气象数据已包含在数据中:\n")
      cat("    变量:", paste(available_weather_vars, collapse = ", "), "\n")
    } else {
      cat("  ⚠ 警告: 数据中未包含气象变量，模型将不包含气象控制变量\n")
    }
    cat("  时间范围:", as.character(min(df$date)), "至", as.character(max(df$date)), "\n\n")
    
    # Fit DLNM based on analysis_mode
    cat("[4/4] 拟合DLNM模型...\n")
    cat("  分析模式:", analysis_mode, "\n")
    cat("  使用指标:", cehwi_col_to_use, "\n\n")
    
    # 【V6新逻辑】提前定义气象变量（只使用3个）
    weather_vars <- c("precipitation", "wind_speed")
    available_weather_vars <- intersect(weather_vars, names(df))
    
    all_results <- list()
    
    save_city_model_result <- function(result, model_name) {
      if (is.null(result)) return(NULL)
      model_name <- normalize_stage1_model_name(model_name)
      activity_type <- model_activity_type(model_name)
      result$model_name <- model_name
      result$model_type <- model_base_type(model_name)
      result$activity_type <- activity_type
      result$activity_label <- unname(activity_modality_labels()[activity_type])
      result$activity_analysis_mode <- activity_analysis_mode_label(STAGE1_ACTIVITY_MODE)
      result$city_code <- city_code
      result$indicator <- indicator
      result$lag_scenario <- LAG_SCENARIO_KEY
      result$lag_scenario_label <- LAG_SCENARIO_LABEL
      result$lag_days <- LAG_DAYS_CURRENT
      result$max_lag <- MAX_LAG
      
      # Save the compact RDS first so an interrupted plotting/CSV step can resume
      # from the fitted model instead of refitting the GAM.
      slim_result <- slim_stage1_result(result)
      write_rds(slim_result, stage1_model_rds_path(city_output_dir, city_code, model_name))
      append_stage1_status(city_code, indicator, model_name, "SAVED_RDS", stage1_model_rds_path(city_output_dir, city_code, model_name))
      if (env_flag("DLNM_SKIP_STAGE1_SIDE_OUTPUTS", default = FALSE)) {
        rm(result)
        gc(verbose = FALSE)
        return(slim_result)
      }
      
      plot_city_rr_curve(result, city_output_dir, model_type = model_name)
      tryCatch(
        plot_control_variables_forest(result, city_output_dir, city_name,
                                      indicator = indicator, model_type = model_name),
        error = function(e) {
          cat("    ⚠ 控制变量森林图生成失败，但模型结果继续保存: ", conditionMessage(e), "\n", sep = "")
        }
      )
      
      write_csv(result$pred_df, file.path(city_output_dir, paste0(city_code, "_", model_name, "_DLNM_pred.csv")))
      if (!is.null(result$all_coefs)) {
        write_csv(result$all_coefs, file.path(city_output_dir, paste0(city_code, "_", model_name, "_coefficients.csv")))
        cat("    - 系数CSV已保存\n")
      }
      if (!is.null(result$socioecon_coefs)) {
        write_csv(result$socioecon_coefs,
                  file.path(city_output_dir, paste0(city_code, "_", model_name, "_weekend_grid_effects.csv")))
        cat("    - Weekend Effect & Grid Heterogeneity已保存\n")
      }
      
      rm(result)
      gc(verbose = FALSE)
      slim_result
    }
    
    if (analysis_mode == "composite" && is_activity_split_mode(STAGE1_ACTIVITY_MODE)) {
      cat("  Activity-mode version: ACTIVITY_3PLUS1 (All + Ride + Run + Walk)\n")
      activity_types <- stage1_activity_types(STAGE1_ACTIVITY_MODE)
      activity_labels <- activity_modality_labels()
      
      summarise_exprs <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        cehwi_exposure = expr(first(.data[[composite_col]]))
      )
      for (var in available_weather_vars) {
        summarise_exprs[[var]] <- expr(first(!!sym(var)))
      }
      
      summarise_exprs_day <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        composite_exposure = expr(first(.data[[composite_col]])),
        specific_exposure = expr(first(.data[[daytime_col]]))
      )
      for (var in available_weather_vars) {
        summarise_exprs_day[[var]] <- expr(first(!!sym(var)))
      }
      
      summarise_exprs_night <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        composite_exposure = expr(first(.data[[composite_col]])),
        specific_exposure = expr(first(.data[[nighttime_col]]))
      )
      for (var in available_weather_vars) {
        summarise_exprs_night[[var]] <- expr(first(!!sym(var)))
      }
      
      for (activity_key in activity_types) {
        activity_df <- prepare_activity_stage1_df(df, activity_key)
        if (nrow(activity_df) == 0) next
        activity_label <- unname(activity_labels[activity_key])
        cat("\n  [Activity] ", activity_label, "\n", sep = "")
        
        model_name <- stage1_model_name("composite", activity_key, STAGE1_ACTIVITY_MODE)
        df_composite <- activity_df %>%
          group_by(fish_id, date) %>%
          summarise(!!!summarise_exprs, .groups = "drop")
        result_composite <- fit_or_resume_city_model(
          city_output_dir = city_output_dir,
          city_code = city_code,
          city_name = city_name,
          indicator = indicator,
          model_name = model_name,
          data = df_composite,
          fit_label = paste0(city_name, "_Composite_", activity_label),
          min_obs = 100
        )
        if (!is.null(result_composite)) {
          all_results[[model_name]] <- save_city_model_result(result_composite, model_name)
          rm(result_composite)
          gc(verbose = FALSE)
        }
        
        model_name <- stage1_model_name("day", activity_key, STAGE1_ACTIVITY_MODE)
        df_day_combined <- activity_df %>%
          filter(day_night == "day") %>%
          group_by(fish_id, date) %>%
          summarise(!!!summarise_exprs_day, .groups = "drop") %>%
          mutate(
            source = case_when(
              composite_exposure > 0 ~ "composite",
              specific_exposure > 0 ~ "daytime",
              TRUE ~ "none"
            ),
            cehwi_exposure = case_when(
              composite_exposure > 0 ~ composite_exposure,
              specific_exposure > 0 ~ specific_exposure,
              TRUE ~ 0
            )
          ) %>%
          select(-composite_exposure, -specific_exposure)
        result_day <- fit_or_resume_city_model(
          city_output_dir = city_output_dir,
          city_code = city_code,
          city_name = city_name,
          indicator = indicator,
          model_name = model_name,
          data = df_day_combined,
          fit_label = paste0(city_name, "_Day_", activity_label),
          min_obs = 50
        )
        if (!is.null(result_day)) {
          all_results[[model_name]] <- save_city_model_result(result_day, model_name)
          rm(result_day)
          gc(verbose = FALSE)
        }
        
        model_name <- stage1_model_name("night", activity_key, STAGE1_ACTIVITY_MODE)
        df_night_combined <- activity_df %>%
          filter(day_night == "night") %>%
          group_by(fish_id, date) %>%
          summarise(!!!summarise_exprs_night, .groups = "drop") %>%
          mutate(
            source = case_when(
              composite_exposure > 0 ~ "composite",
              specific_exposure > 0 ~ "nighttime",
              TRUE ~ "none"
            ),
            cehwi_exposure = case_when(
              composite_exposure > 0 ~ composite_exposure,
              specific_exposure > 0 ~ specific_exposure,
              TRUE ~ 0
            )
          ) %>%
          select(-composite_exposure, -specific_exposure)
        result_night <- fit_or_resume_city_model(
          city_output_dir = city_output_dir,
          city_code = city_code,
          city_name = city_name,
          indicator = indicator,
          model_name = model_name,
          data = df_night_combined,
          fit_label = paste0(city_name, "_Night_", activity_label),
          min_obs = 50
        )
        if (!is.null(result_night)) {
          all_results[[model_name]] <- save_city_model_result(result_night, model_name)
          rm(result_night)
          gc(verbose = FALSE)
        }
      }
      
    } else if (analysis_mode == "composite") {
      # 模式1: Composite - 做3个模型
      cat("  [1/3] Composite模型 (全天PA, 用", composite_col, ")...\n")
      
      # 构建summarise表达式
      summarise_exprs <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        cehwi_exposure = expr(first(.data[[composite_col]]))
      )
      
      # 【V6新逻辑】添加气象变量（数据已完整填充，不需要na.omit）
      for (var in available_weather_vars) {
        summarise_exprs[[var]] <- expr(first(!!sym(var)))
      }
      
      df_composite <- df %>%
        group_by(fish_id, date) %>%
        summarise(!!!summarise_exprs, .groups = "drop")
      result_composite <- fit_or_resume_city_model(
        city_output_dir = city_output_dir,
        city_code = city_code,
        city_name = city_name,
        indicator = indicator,
        model_name = "composite",
        data = df_composite,
        fit_label = paste0(city_name, "_Composite"),
        min_obs = 100
      )
      if (!is.null(result_composite)) {
        all_results$composite <- save_city_model_result(result_composite, "composite")
        rm(result_composite)
        gc(verbose = FALSE)
      }
      
      cat("\n  [2/3] Day模型 (昼间PA, 合并", composite_col, "+", daytime_col, "数据)...\n")
      
      # 【V7修复】Day模型保留全部昼间观测，非热浪日暴露记为0
      summarise_exprs_day <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        composite_exposure = expr(first(.data[[composite_col]])),
        specific_exposure = expr(first(.data[[daytime_col]]))
      )
      
      for (var in available_weather_vars) {
        summarise_exprs_day[[var]] <- expr(first(!!sym(var)))
      }
      
      df_day_combined <- df %>%
        filter(day_night == "day") %>%
        group_by(fish_id, date) %>%
        summarise(!!!summarise_exprs_day, .groups = "drop") %>%
        mutate(
          source = case_when(
            composite_exposure > 0 ~ "composite",
            specific_exposure > 0 ~ "daytime",
            TRUE ~ "none"
          ),
          cehwi_exposure = case_when(
            composite_exposure > 0 ~ composite_exposure,
            specific_exposure > 0 ~ specific_exposure,
            TRUE ~ 0
          )
        ) %>%
        select(-composite_exposure, -specific_exposure)
      
      cat("    合并后Day数据: composite来源", sum(df_day_combined$source == "composite", na.rm = TRUE),
          "天 + daytime来源", sum(df_day_combined$source == "daytime", na.rm = TRUE),
          "天 + 非热浪对照", sum(df_day_combined$source == "none", na.rm = TRUE),
          "天 = 总计", nrow(df_day_combined), "天\n")
      
      result_day <- fit_or_resume_city_model(
        city_output_dir = city_output_dir,
        city_code = city_code,
        city_name = city_name,
        indicator = indicator,
        model_name = "day",
        data = df_day_combined,
        fit_label = paste0(city_name, "_Day"),
        min_obs = 50
      )
      if (!is.null(result_day)) {
        all_results$day <- save_city_model_result(result_day, "day")
        rm(result_day)
        gc(verbose = FALSE)
      }
      
      cat("\n  [3/3] Night模型 (夜间PA, 合并", composite_col, "+", nighttime_col, "数据)...\n")
      
      # 【V7修复】Night模型保留全部夜间观测，非热浪日暴露记为0
      summarise_exprs_night <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        composite_exposure = expr(first(.data[[composite_col]])),
        specific_exposure = expr(first(.data[[nighttime_col]]))
      )
      
      for (var in available_weather_vars) {
        summarise_exprs_night[[var]] <- expr(first(!!sym(var)))
      }
      
      df_night_combined <- df %>%
        filter(day_night == "night") %>%
        group_by(fish_id, date) %>%
        summarise(!!!summarise_exprs_night, .groups = "drop") %>%
        mutate(
          source = case_when(
            composite_exposure > 0 ~ "composite",
            specific_exposure > 0 ~ "nighttime",
            TRUE ~ "none"
          ),
          cehwi_exposure = case_when(
            composite_exposure > 0 ~ composite_exposure,
            specific_exposure > 0 ~ specific_exposure,
            TRUE ~ 0
          )
        ) %>%
        select(-composite_exposure, -specific_exposure)
      
      cat("    合并后Night数据: composite来源", sum(df_night_combined$source == "composite", na.rm = TRUE),
          "天 + nighttime来源", sum(df_night_combined$source == "nighttime", na.rm = TRUE),
          "天 + 非热浪对照", sum(df_night_combined$source == "none", na.rm = TRUE),
          "天 = 总计", nrow(df_night_combined), "天\n")
      
      result_night <- fit_or_resume_city_model(
        city_output_dir = city_output_dir,
        city_code = city_code,
        city_name = city_name,
        indicator = indicator,
        model_name = "night",
        data = df_night_combined,
        fit_label = paste0(city_name, "_Night"),
        min_obs = 50
      )
      if (!is.null(result_night)) {
        all_results$night <- save_city_model_result(result_night, "night")
        rm(result_night)
        gc(verbose = FALSE)
      }
      
    } else if (analysis_mode == "day_only" && is_activity_split_mode(STAGE1_ACTIVITY_MODE)) {
      activity_types <- stage1_activity_types(STAGE1_ACTIVITY_MODE)
      activity_labels <- activity_modality_labels()
      summarise_exprs_day_only <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        cehwi_exposure = expr(first(.data[[cehwi_col_to_use]]))
      )
      for (var in available_weather_vars) {
        summarise_exprs_day_only[[var]] <- expr(first(!!sym(var)))
      }
      for (activity_key in activity_types) {
        activity_df <- prepare_activity_stage1_df(df, activity_key)
        activity_label <- unname(activity_labels[activity_key])
        model_name <- stage1_model_name("day", activity_key, STAGE1_ACTIVITY_MODE)
        df_day <- activity_df %>%
          filter(day_night == "day") %>%
          group_by(fish_id, date) %>%
          summarise(!!!summarise_exprs_day_only, .groups = "drop")
        result_day <- fit_or_resume_city_model(
          city_output_dir = city_output_dir,
          city_code = city_code,
          city_name = city_name,
          indicator = indicator,
          model_name = model_name,
          data = df_day,
          fit_label = paste0(city_name, "_DayOnly_", activity_label),
          min_obs = 50
        )
        if (!is.null(result_day)) {
          all_results[[model_name]] <- save_city_model_result(result_day, model_name)
          rm(result_day)
          gc(verbose = FALSE)
        }
      }
      
    } else if (analysis_mode == "day_only") {
      # 模式2: 只做Day
      cat("  Day Only模型 (用", cehwi_col_to_use, ")...\n")
      
      # 【V5修复】保留气象变量 - Day Only模型
      summarise_exprs_day_only <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        cehwi_exposure = expr(first(.data[[cehwi_col_to_use]]))
      )
      
      # 【V6新逻辑】添加气象变量
      for (var in available_weather_vars) {
        summarise_exprs_day_only[[var]] <- expr(first(!!sym(var)))
      }
      
      df_day <- df %>%
        filter(day_night == "day") %>%
        group_by(fish_id, date) %>%
        summarise(!!!summarise_exprs_day_only, .groups = "drop")
      
      # Debug: 检查cehwi_exposure是否有值
      n_positive <- sum(df_day$cehwi_exposure > 0, na.rm = TRUE)
      cat("    传入DLNM的数据: N =", nrow(df_day), ", cehwi_exposure > 0:", n_positive, "天\n\n")
      
      result_day <- fit_or_resume_city_model(
        city_output_dir = city_output_dir,
        city_code = city_code,
        city_name = city_name,
        indicator = indicator,
        model_name = "day",
        data = df_day,
        fit_label = paste0(city_name, "_DayOnly"),
        min_obs = 50
      )
      if (!is.null(result_day)) {
        all_results$day <- save_city_model_result(result_day, "day")
        rm(result_day)
        gc(verbose = FALSE)
      }
      
    } else if (analysis_mode == "night_only" && is_activity_split_mode(STAGE1_ACTIVITY_MODE)) {
      activity_types <- stage1_activity_types(STAGE1_ACTIVITY_MODE)
      activity_labels <- activity_modality_labels()
      summarise_exprs_night_only <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        cehwi_exposure = expr(first(.data[[cehwi_col_to_use]]))
      )
      for (var in available_weather_vars) {
        summarise_exprs_night_only[[var]] <- expr(first(!!sym(var)))
      }
      for (activity_key in activity_types) {
        activity_df <- prepare_activity_stage1_df(df, activity_key)
        activity_label <- unname(activity_labels[activity_key])
        model_name <- stage1_model_name("night", activity_key, STAGE1_ACTIVITY_MODE)
        df_night <- activity_df %>%
          filter(day_night == "night") %>%
          group_by(fish_id, date) %>%
          summarise(!!!summarise_exprs_night_only, .groups = "drop")
        result_night <- fit_or_resume_city_model(
          city_output_dir = city_output_dir,
          city_code = city_code,
          city_name = city_name,
          indicator = indicator,
          model_name = model_name,
          data = df_night,
          fit_label = paste0(city_name, "_NightOnly_", activity_label),
          min_obs = 50
        )
        if (!is.null(result_night)) {
          all_results[[model_name]] <- save_city_model_result(result_night, model_name)
          rm(result_night)
          gc(verbose = FALSE)
        }
      }
      
    } else if (analysis_mode == "night_only") {
      # 模式3: 只做Night
      cat("  Night Only模型 (用", cehwi_col_to_use, ")...\n")
      
      # 【V5修复】保留气象变量 - Night Only模型
      summarise_exprs_night_only <- list(
        trip_count = expr(sum(trip_count, na.rm = TRUE)),
        cehwi_exposure = expr(first(.data[[cehwi_col_to_use]]))
      )
      
      # 【V6新逻辑】添加气象变量
      for (var in available_weather_vars) {
        summarise_exprs_night_only[[var]] <- expr(first(!!sym(var)))
      }
      
      df_night <- df %>%
        filter(day_night == "night") %>%
        group_by(fish_id, date) %>%
        summarise(!!!summarise_exprs_night_only, .groups = "drop")
      
      # Debug: 检查cehwi_exposure是否有值
      n_positive <- sum(df_night$cehwi_exposure > 0, na.rm = TRUE)
      cat("    传入DLNM的数据: N =", nrow(df_night), ", cehwi_exposure > 0:", n_positive, "天\n\n")
      
      result_night <- fit_or_resume_city_model(
        city_output_dir = city_output_dir,
        city_code = city_code,
        city_name = city_name,
        indicator = indicator,
        model_name = "night",
        data = df_night,
        fit_label = paste0(city_name, "_NightOnly"),
        min_obs = 50
      )
      if (!is.null(result_night)) {
        all_results$night <- save_city_model_result(result_night, "night")
        rm(result_night)
        gc(verbose = FALSE)
      }
    }
    
    # 保存所有结果
    if (FALSE && length(all_results) > 0) {
      for (model_name in names(all_results)) {
        result <- all_results[[model_name]]
        
        # 确定model_type用于颜色选择
        model_type_for_plot <- if (grepl("composite", model_name, ignore.case = TRUE)) {
          "composite"
        } else if (grepl("day", model_name, ignore.case = TRUE)) {
          "day"
        } else if (grepl("night", model_name, ignore.case = TRUE)) {
          "night"
        } else {
          "composite"  # 默认
        }
        
        write_rds(result, file.path(city_output_dir, paste0(city_code, "_", model_name, "_DLNM_result.rds")))
        plot_city_rr_curve(result, city_output_dir, model_type = model_type_for_plot)
        
        # 【V5新增】生成控制变量森林图
        tryCatch(
          plot_control_variables_forest(result, city_output_dir, city_name,
                                        indicator = indicator, model_type = model_type_for_plot),
          error = function(e) {
            cat("    ⚠ 控制变量森林图生成失败，但模型结果继续保存: ", conditionMessage(e), "\n", sep = "")
          }
        )
        
        write_csv(result$pred_df, file.path(city_output_dir, paste0(city_code, "_", model_name, "_DLNM_pred.csv")))
        
        # 保存模型系数（全局+格子）
        if (!is.null(result$all_coefs)) {
          write_csv(result$all_coefs, file.path(city_output_dir, paste0(city_code, "_", model_name, "_coefficients.csv")))
          cat("    - 系数CSV已保存\n")
        }
        
        # 【V5修改】保存Weekend Effect和Grid Heterogeneity（不绘制forest plot，留给第二阶段汇总）
        if (!is.null(result$socioecon_coefs)) {
          write_csv(result$socioecon_coefs, 
                    file.path(city_output_dir, paste0(city_code, "_", model_name, "_weekend_grid_effects.csv")))
          cat("    - Weekend Effect & Grid Heterogeneity已保存（第二阶段将汇总）\n")
        }
      }
      cat("\n  ✓ 所有模型结果已保存\n")
    }
    
    cat("\n", rep("=", 100), "\n", sep = "")
    cat("✅", city_name, "DLNM分析完成!\n")
    if (length(all_results) > 0) {
      cat("  成功模型数:", length(all_results), "/", ifelse(analysis_mode == "composite", 3, 1), "\n")
      cat("  模型类型:", paste(names(all_results), collapse = ", "), "\n")
    } else {
      cat("  ⚠ 所有模型都失败了\n")
    }
    cat(rep("=", 100), "\n\n", sep = "")
    
    try(save_city_activity_overlay_plots(all_results, city_output_dir, city_name, indicator), silent = TRUE)
    
    return(all_results)
    
  }, error = function(e) {
    cat("\n✗ 错误:", conditionMessage(e), "\n")
    return(NULL)
  }, finally = {
    while (sink.number() > 0) sink()
    try(close(log_file), silent = TRUE)
  })
}

# ========== 【V5.1新增】核心函数: 分区DLNM分析 ==========

analyze_partition_dlnm <- function(partition_name, partition_type, city_list, indicator = "cehwi") {
  # 【V5.1】对分区（Climate Zone/City Cluster/Geographic Region）进行第一阶段DLNM分析
  # 
  # 参数:
  #   partition_name: 分区名称（如"Temperate", "Texas_Southern_Hot", "Northeast"）
  #   partition_type: 分区类型（"zone", "cluster", "region"）
  #   city_list: 该分区包含的城市列表
  #   indicator: 热浪指标 ("cehwi" 或 "exceeded_quantity")
  #
  # 输出:
  #   与单个城市相同的完整输出（RR曲线、控制变量森林图、日志等）
  
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("【V5.1】分区DLNM分析 - ", partition_name, " (", partition_type, ")\n", sep = "")
  cat(rep("=", 100), "\n\n", sep = "")
  cat("  包含城市:", length(city_list), "个\n")
  cat("  城市列表:", paste(city_list, collapse = ", "), "\n\n")
  
  # 创建输出目录
  partition_dir_name <- switch(partition_type,
    "zone" = paste0("ZONE_", partition_name, "_DLNM_STAGE1"),
    "cluster" = paste0("CLUSTER_", partition_name, "_DLNM_STAGE1"),
    "region" = paste0("REGION_", partition_name, "_DLNM_STAGE1"),
    "dtw3" = paste0("DTW3_", partition_name, "_DLNM_STAGE1"),
    "dtw4" = paste0("DTW4_", partition_name, "_DLNM_STAGE1"),
    paste0("PARTITION_", partition_name, "_DLNM_STAGE1")
  )
  
  partition_output_dir <- file.path(OUTPUT_DIR, partition_dir_name)
  dir.create(partition_output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # 创建日志文件
  log_file <- file.path(partition_output_dir, paste0(partition_name, "_DLNM_log.txt"))
  
  tryCatch({
    sink(log_file, type = "output", split = TRUE)
    
    cat("\n【V5.1】分区DLNM分析 (第一阶段) -", partition_name, "\n")
    cat("分区类型:", partition_type, "\n")
    cat("时间:", as.character(Sys.time()), "\n")
    cat("分析时间段:", TIME_DESC, "\n\n")
    
    # 【1/5】加载并合并所有城市的数据
    cat("[1/5] 加载并合并分区内城市数据...\n")
    
    all_cities_data <- list()
    successful_loads <- 0
    
    for (city_safe_name in city_list) {
      city_folder_name <- ifelse(city_safe_name %in% names(CITY_NAME_MAPPING),
                                  CITY_NAME_MAPPING[[city_safe_name]],
                                  city_safe_name)
      
      trips_file <- file.path(TRIPS_DIR, paste0("trips_", city_safe_name, "_gridid_advanced.csv"))
      temp_dir <- file.path(TEMP_DIR, city_folder_name)
      
      if (!file.exists(trips_file) || !dir.exists(temp_dir)) {
        cat("  ⚠ 跳过", city_safe_name, "- 文件不存在\n")
        next
      }
      
      tryCatch({
        # 加载trip数据
        df_raw <- read_csv(trips_file, show_col_types = FALSE)
        id_col <- names(df_raw)[1]
        trip_cols <- names(df_raw)[str_detect(names(df_raw), "\\d{4}/\\d{1,2}/\\d{1,2}_")]
        
        df_long <- df_raw %>%
          pivot_longer(cols = all_of(trip_cols), names_to = "column", values_to = "trip_count") %>%
          mutate(
            date = str_extract(column, "\\d{4}/\\d{1,2}/\\d{1,2}"),
            date = as.Date(date, format = "%Y/%m/%d"),
            trip_type = str_extract(column, "(walk|run|ride)"),
            day_night = str_extract(column, "(day|night)")
          ) %>%
          filter(!is.na(date), !is.na(trip_type), !is.na(day_night))
        
        names(df_long)[names(df_long) == id_col] <- "fish_id"
        
        # 加载热浪数据
        hw_files <- list.files(temp_dir, pattern = "_temperature__95\\.csv$", full.names = TRUE)
        
        hw_all <- map_dfr(hw_files, function(f) {
          df <- read_csv(f, show_col_types = FALSE, col_types = cols())
          df %>% mutate(fish_id = as.numeric(str_extract(basename(f), "\\d+")))
        })
        
        hw_all <- hw_all %>%
          rename(date = 1) %>%
          mutate(date = as.Date(date)) %>%
          clean_names()
        
        # 【V6修复】加载气象数据（使用city_folder_name）
        # 直接传递城市名，让load_weather_data函数处理文件名映射
        df_weather <- load_weather_data(city_folder_name, START_DATE, END_DATE)
        
        if (is.null(df_weather)) {
          cat("      ⚠ 警告:", city_folder_name, "的气象数据未加载\n")
        }
        
        # 确定使用的暴露列
        if (indicator == "cehwi") {
          composite_col <- "composite_cehwi"
        } else {
          composite_col <- "composite_exceeded_quantity"
        }
        
        # 筛选时间并检查数据
        hw_all_filtered <- hw_all %>%
          filter(date >= START_DATE & date <= END_DATE)
        
        n_composite_positive <- sum(hw_all_filtered[[composite_col]] > 0, na.rm = TRUE)
        
        if (n_composite_positive < 5) {
          cat("  ⚠ 跳过", city_safe_name, "- 热浪数据不足\n")
          next
        }
        
        # 合并数据（补齐完整日历面板，避免DLNM lag跨缺失日）
        df_merged <- build_complete_daily_panel(df_long, hw_all_filtered, START_DATE, END_DATE)
        
        if (!is.null(df_weather)) {
          df_merged <- df_merged %>%
            left_join(df_weather, by = "date")
        }
        
        # 添加城市标识（用于模型中的城市固定效应）
        df_merged$city <- city_safe_name
        
        all_cities_data[[city_safe_name]] <- df_merged
        successful_loads <- successful_loads + 1
        cat("  ✓", city_safe_name, "- 加载成功\n")
        
      }, error = function(e) {
        cat("  ✗", city_safe_name, "- 加载失败:", conditionMessage(e), "\n")
      })
    }
    
    if (successful_loads < 2) {
      cat("\n  ✗ 失败: 成功加载的城市不足2个\n")
      sink()
      return(NULL)
    }
    
    cat("\n  ✓ 成功加载", successful_loads, "个城市数据\n")
    
    # 合并所有城市数据
    df_partition <- bind_rows(all_cities_data)
    cat("  总记录数:", nrow(df_partition), "\n\n")
    
    # 【2/5】准备分区级数据并拟合模型
    cat("[2/5] 拟合分区DLNM模型...\n")
    
    # 为composite、day、night分别拟合模型
    results <- list()
    
    for (model_name in STAGE1_MODEL_TYPES) {
      cat("\n  ────────────────────────────────────────\n")
      cat("  模型类型:", toupper(model_name), "\n")
      cat("  ────────────────────────────────────────\n\n")
      
      # 筛选对应的数据
      if (model_name == "composite") {
        df_model_input <- df_partition
      } else if (model_name == "day") {
        df_model_input <- df_partition %>% filter(day_night == "day")
      } else {
        df_model_input <- df_partition %>% filter(day_night == "night")
      }
      
      # 调用fit_dlnm_stage1（需要修改以支持城市固定效应）
      result <- fit_dlnm_stage1(df_model_input, partition_name, indicator = indicator, min_obs = 500)
      
      if (!is.null(result)) {
        results[[model_name]] <- result
        cat("  ✓", toupper(model_name), "模型拟合成功\n")
      } else {
        cat("  ✗", toupper(model_name), "模型拟合失败\n")
      }
    }
    
    if (length(results) == 0) {
      cat("\n  ✗ 所有模型均拟合失败\n")
      sink()
      return(NULL)
    }
    
    # 【3/5】生成RR曲线
    cat("\n[3/5] 生成RR曲线可视化...\n")
    
    for (model_name in names(results)) {
      result <- results[[model_name]]
      
      model_type_for_plot <- switch(model_name,
        "composite" = "composite",
        "day" = "day",
        "night" = "night",
        "composite"
      )
      
      plot_city_rr_curve(result, partition_output_dir, model_type = model_type_for_plot)
      cat("  ✓", toupper(model_name), "RR曲线已生成\n")
    }
    
    # 【4/5】生成控制变量森林图
    cat("\n[4/5] 生成控制变量森林图...\n")
    
    for (model_name in names(results)) {
      result <- results[[model_name]]
      
      model_type_for_plot <- switch(model_name,
        "composite" = "composite",
        "day" = "day",
        "night" = "night",
        "composite"
      )
      
      # 【V5.1】调用控制变量森林图函数
      tryCatch(
        plot_control_variables_forest(result, partition_output_dir, partition_name,
                                      indicator = indicator, model_type = model_type_for_plot),
        error = function(e) {
          cat("  ⚠ 控制变量森林图生成失败，但模型结果继续保存: ", conditionMessage(e), "\n", sep = "")
        }
      )
      cat("  ✓", toupper(model_name), "控制变量森林图已生成\n")
    }
    
    # 【5/5】保存结果文件
    cat("\n[5/5] 保存结果文件...\n")
    
    for (model_name in names(results)) {
      result <- results[[model_name]]
      
      # 保存RDS
      write_rds(result, file.path(partition_output_dir, 
                                   paste0(partition_name, "_", model_name, "_DLNM_result.rds")))
      
      # 保存预测数据
      write_csv(result$pred_df, file.path(partition_output_dir, 
                                           paste0(partition_name, "_", model_name, "_DLNM_pred.csv")))
      
      # 保存系数
      if (!is.null(result$all_coefs)) {
        write_csv(result$all_coefs, file.path(partition_output_dir, 
                                               paste0(partition_name, "_", model_name, "_coefficients.csv")))
      }
    }
    
    cat("\n  ✓ 所有文件已保存至:", partition_output_dir, "\n")
    
    sink()
    
    cat("✓", partition_name, "(", indicator, ") 分析完成 -", length(results), "个模型\n\n")
    
    return(results)
    
  }, error = function(e) {
    sink()
    cat("✗", partition_name, "分析失败:", conditionMessage(e), "\n\n")
    return(NULL)
  })
}

# ========== 第一阶段: 城市级DLNM分析 ==========

successful_cities <- list()
failed_cities <- list()
af_list <- list()  # 【V6新增】存储所有模型的AF数据

# ========== 【Part 1】单城市DLNM分析 ==========

if (RUN_RERENDER_ONLY) {
  rerender_saved_visualizations(OUTPUT_DIR)
}

if (RUN_CITIES) {
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("第一阶段 Part 1: 单城市DLNM分析\n")
  cat(rep("=", 100), "\n\n", sep = "")
  
  # 两个指标都分析
  for (indicator in INDICATOR_LIST) {
  cat("\n", rep("-", 80), "\n", sep = "")
  cat("指标:", toupper(indicator), "\n")
  cat(rep("-", 80), "\n", sep = "")
  
  for (city_safe_name in CITY_LIST) {
    set_active_lag_for_city(city_safe_name, verbose = TRUE)
    
    city_folder_name <- ifelse(city_safe_name %in% names(CITY_NAME_MAPPING),
                                CITY_NAME_MAPPING[[city_safe_name]],
                                city_safe_name)
    
    trips_file <- file.path(TRIPS_DIR, paste0("trips_", city_safe_name, "_gridid_advanced.csv"))
    temp_dir <- file.path(TEMP_DIR, city_folder_name)
    
    if (file.exists(trips_file) && dir.exists(temp_dir)) {
      tryCatch({
        result <- analyze_city_dlnm(city_safe_name, city_folder_name, 
                                    trips_file, temp_dir, indicator = indicator)
        if (!is.null(result) && length(result) > 0) {
          # result现在是一个list，包含composite/day/night模型
          key_name <- paste0(city_safe_name, "_", indicator)
          successful_cities[[key_name]] <- result
        } else {
          key_name <- paste0(city_safe_name, "_", indicator)
          failed_cities[[key_name]] <- "模型拟合失败"
        }
      }, error = function(e) {
        cat("✗", city_folder_name, "(", indicator, ") 失败:", conditionMessage(e), "\n\n")
        key_name <- paste0(city_safe_name, "_", indicator)
        failed_cities[[key_name]] <- paste0("运行错误: ", conditionMessage(e))
      })
    } else {
      cat("⚠ 跳过", city_safe_name, "- 文件不存在\n")
    }
  }
  }  # 结束指标循环
  
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("✅ 第一阶段（单城市）完成！\n")
  cat("  成功城市数:", length(successful_cities), "\n")
  cat("  失败城市数:", length(failed_cities), "\n")
  cat("  Note: these counts are city-indicator combinations (city × CEHWI/exceeded_quantity), not unique city counts.\n")
  write_stage1_failure_report(failed_cities, OUTPUT_DIR)
  safe_create_stage1_master_rr_panels(successful_cities, OUTPUT_DIR)
  cat(rep("=", 100), "\n\n", sep = "")
  
} else {
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("⊘ 跳过单城市分析（用户选择）\n")
  cat(rep("=", 100), "\n\n", sep = "")
}

# ========== 【Part 2】分区DLNM分析 ==========

if (RUN_PARTITIONS) {
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("第一阶段 Part 2: 分区DLNM分析（Climate Zone / City Cluster / Geographic Region / DTW k=3 / DTW k=4）\n")
  cat(rep("=", 100), "\n\n", sep = "")
  
  cat("  说明：按所选种类进行分区DLNM分析（每种均含 CEHWI + 超出量）\n")
  cat("  - 合并分区内所有城市的数据\n")
  cat("  - 格子随机效应 s(fish_id, bs='re') 控制空间异质性\n")
  cat("  - 生成与单城市相同的完整输出（RR曲线、控制变量森林图等）\n\n")
  
  # 存储分区结果
  successful_partitions <- list()
  failed_partitions <- list()
  
  # 检查是否有社会经济数据（用于分区映射）
  if (!is.null(df_socioecon_global)) {
    
    # ========== [1/3] Climate Zone 分区（可选）==========
    if (PARTITION_RUN_ZONE) {
      cat("\n", rep("-", 80), "\n", sep = "")
      cat("[1/5] 按Climate Zone分区分析 (CEHWI + 超出量)\n")
      cat(rep("-", 80), "\n\n", sep = "")
      
      for (indicator in INDICATOR_LIST) {
        cat("\n  指标:", toupper(indicator), "\n\n")
        
        for (zone in names(CLIMATE_ZONE_MAPPING)) {
          zone_cities <- CLIMATE_ZONE_MAPPING[[zone]]
          
          tryCatch({
            result <- analyze_partition_dlnm(
              partition_name = zone,
              partition_type = "zone",
              city_list = zone_cities,
              indicator = indicator
            )
            
            if (!is.null(result) && length(result) > 0) {
              key_name <- paste0("ZONE_", zone, "_", indicator)
              successful_partitions[[key_name]] <- result
            } else {
              key_name <- paste0("ZONE_", zone, "_", indicator)
              failed_partitions[[key_name]] <- "模型拟合失败"
            }
          }, error = function(e) {
            cat("✗ ZONE", zone, "(", indicator, ") 失败:", conditionMessage(e), "\n\n")
            key_name <- paste0("ZONE_", zone, "_", indicator)
            failed_partitions[[key_name]] <- paste0("运行错误: ", conditionMessage(e))
          })
        }
      }
      
      cat("\n  ✓ Climate Zone分区分析完成\n")
    }
    
    # ========== [2/3] City Cluster 分区（可选）==========
    if (PARTITION_RUN_CLUSTER) {
      cat("\n", rep("-", 80), "\n", sep = "")
      cat("[2/5] 按City Cluster分区分析 (CEHWI + 超出量)\n")
      cat(rep("-", 80), "\n\n", sep = "")
      
      for (indicator in INDICATOR_LIST) {
        cat("\n  指标:", toupper(indicator), "\n\n")
        
        for (cluster in names(CITY_CLUSTER_MAPPING)) {
          cluster_cities <- CITY_CLUSTER_MAPPING[[cluster]]
          
          tryCatch({
            result <- analyze_partition_dlnm(
              partition_name = cluster,
              partition_type = "cluster",
              city_list = cluster_cities,
              indicator = indicator
            )
            
            if (!is.null(result) && length(result) > 0) {
              key_name <- paste0("CLUSTER_", cluster, "_", indicator)
              successful_partitions[[key_name]] <- result
            } else {
              key_name <- paste0("CLUSTER_", cluster, "_", indicator)
              failed_partitions[[key_name]] <- "模型拟合失败"
            }
          }, error = function(e) {
            cat("✗ CLUSTER", cluster, "(", indicator, ") 失败:", conditionMessage(e), "\n\n")
            key_name <- paste0("CLUSTER_", cluster, "_", indicator)
            failed_partitions[[key_name]] <- paste0("运行错误: ", conditionMessage(e))
          })
        }
      }
      
      cat("\n  ✓ City Cluster分区分析完成\n")
    }
    
    # ========== [3/3] Geographic Region 分区（可选）==========
    if (PARTITION_RUN_REGION) {
      cat("\n", rep("-", 80), "\n", sep = "")
      cat("[3/5] 按Geographic Region分区分析 (CEHWI + 超出量)\n")
      cat(rep("-", 80), "\n\n", sep = "")
      
      for (indicator in INDICATOR_LIST) {
        cat("\n  指标:", toupper(indicator), "\n\n")
        
        for (region in names(GEOGRAPHIC_REGION_MAPPING)) {
          region_cities <- GEOGRAPHIC_REGION_MAPPING[[region]]
          
          tryCatch({
            result <- analyze_partition_dlnm(
              partition_name = region,
              partition_type = "region",
              city_list = region_cities,
              indicator = indicator
            )
            
            if (!is.null(result) && length(result) > 0) {
              key_name <- paste0("REGION_", region, "_", indicator)
              successful_partitions[[key_name]] <- result
            } else {
              key_name <- paste0("REGION_", region, "_", indicator)
              failed_partitions[[key_name]] <- "模型拟合失败"
            }
          }, error = function(e) {
            cat("✗ REGION", region, "(", indicator, ") 失败:", conditionMessage(e), "\n\n")
            key_name <- paste0("REGION_", region, "_", indicator)
            failed_partitions[[key_name]] <- paste0("运行错误: ", conditionMessage(e))
          })
        }
      }
      
      cat("\n  ✓ Geographic Region分区分析完成\n")
    }
    
    # 汇总分区分析结果
    cat("\n", rep("=", 100), "\n", sep = "")
    if (PARTITION_RUN_DTW3) {
      cat("\n", rep("-", 80), "\n", sep = "")
      cat("[4/5] 按DTW Optimized k=3分区分析 (CEHWI + 超出量)\n")
      cat(rep("-", 80), "\n\n", sep = "")
      
      for (indicator in INDICATOR_LIST) {
        cat("\n  指标:", toupper(indicator), "\n\n")
        
        for (cluster in names(DTW_CLUSTER_OPTIMIZED3_MAPPING)) {
          cluster_cities <- DTW_CLUSTER_OPTIMIZED3_MAPPING[[cluster]]
          tryCatch({
            result <- analyze_partition_dlnm(
              partition_name = cluster,
              partition_type = "dtw3",
              city_list = cluster_cities,
              indicator = indicator
            )
            
            if (!is.null(result) && length(result) > 0) {
              key_name <- paste0("DTW3_", cluster, "_", indicator)
              successful_partitions[[key_name]] <- result
            } else {
              key_name <- paste0("DTW3_", cluster, "_", indicator)
              failed_partitions[[key_name]] <- "模型拟合失败"
            }
          }, error = function(e) {
            cat("✗ DTW3", cluster, "(", indicator, ") 失败:", conditionMessage(e), "\n\n")
            key_name <- paste0("DTW3_", cluster, "_", indicator)
            failed_partitions[[key_name]] <- paste0("运行错误: ", conditionMessage(e))
          })
        }
      }
      
      cat("\n  ✓ DTW Optimized k=3分区分析完成\n")
    }
    
    if (PARTITION_RUN_DTW4) {
      cat("\n", rep("-", 80), "\n", sep = "")
      cat("[5/5] 按DTW Optimized k=4分区分析 (CEHWI + 超出量)\n")
      cat(rep("-", 80), "\n\n", sep = "")
      
      for (indicator in INDICATOR_LIST) {
        cat("\n  指标:", toupper(indicator), "\n\n")
        
        for (cluster in names(DTW_CLUSTER_OPTIMIZED4_MAPPING)) {
          cluster_cities <- DTW_CLUSTER_OPTIMIZED4_MAPPING[[cluster]]
          tryCatch({
            result <- analyze_partition_dlnm(
              partition_name = cluster,
              partition_type = "dtw4",
              city_list = cluster_cities,
              indicator = indicator
            )
            
            if (!is.null(result) && length(result) > 0) {
              key_name <- paste0("DTW4_", cluster, "_", indicator)
              successful_partitions[[key_name]] <- result
            } else {
              key_name <- paste0("DTW4_", cluster, "_", indicator)
              failed_partitions[[key_name]] <- "模型拟合失败"
            }
          }, error = function(e) {
            cat("✗ DTW4", cluster, "(", indicator, ") 失败:", conditionMessage(e), "\n\n")
            key_name <- paste0("DTW4_", cluster, "_", indicator)
            failed_partitions[[key_name]] <- paste0("运行错误: ", conditionMessage(e))
          })
        }
      }
      
      cat("\n  ✓ DTW Optimized k=4分区分析完成\n")
    }
    
    cat("✅ 【V5.1】分区DLNM分析完成！\n")
    cat("  成功分区数:", length(successful_partitions), "\n")
    cat("  失败分区数:", length(failed_partitions), "\n")
    cat("  - Climate Zone: ", sum(str_detect(names(successful_partitions), "^ZONE_")), "个\n")
    cat("  - City Cluster: ", sum(str_detect(names(successful_partitions), "^CLUSTER_")), "个\n")
    cat("  - Geographic Region: ", sum(str_detect(names(successful_partitions), "^REGION_")), "个\n")
    cat("  - DTW Optimized k=3: ", sum(str_detect(names(successful_partitions), "^DTW3_")), "个\n")
    cat("  - DTW Optimized k=4: ", sum(str_detect(names(successful_partitions), "^DTW4_")), "个\n")
    
  } else {
    cat("  ⚠ 未加载社会经济数据，跳过分区分析\n")
    cat("    （分区映射需要社会经济数据文件）\n")
  }
  
  # 【V4注释】第一阶段不使用社会经济变量，此代码块已禁用
  # 社会经济变量仅在第二阶段作为meta-predictors使用
  if (FALSE) {  # 【V4】已禁用第一阶段社会经济系数汇总
    cat("\n[V3] 汇总社会经济变量系数...\n")
    
    all_socioecon_coefs_list <- list()
    
    for (city_key in names(successful_cities)) {
      city_results <- successful_cities[[city_key]]
      
      for (model_type in names(city_results)) {
        result <- city_results[[model_type]]
        
        if (!is.null(result$socioecon_coefs)) {
          key <- paste0(city_key, "_", model_type)
          all_socioecon_coefs_list[[key]] <- result$socioecon_coefs %>%
            mutate(model_type = model_type)
        }
      }
    }
    
    if (length(all_socioecon_coefs_list) > 0) {
      df_all_socioecon <- bind_rows(all_socioecon_coefs_list)
      
      # 保存汇总CSV
      all_socioecon_file <- file.path(OUTPUT_DIR, 
                                       paste0("ALL_socioecon_coefficients_", TIME_LABEL, ".csv"))
      write_csv(df_all_socioecon, all_socioecon_file)
      cat("  ✓ 所有城市社会经济系数已保存:", all_socioecon_file, "\n")
      cat("    - 总记录数:", nrow(df_all_socioecon), "\n")
      cat("    - 城市数:", n_distinct(df_all_socioecon$city), "\n\n")
      
      # 按气候区域汇总
      cat("  生成按气候区域的汇总Forest Plot...\n")
      
      # 添加气候区域信息
      df_all_socioecon_with_zone <- df_all_socioecon %>%
        mutate(
          city_code = str_extract(city, "^[^_]+"),
          climate_zone = case_when(
            city_code %in% CLIMATE_ZONE_MAPPING$Temperate ~ "Temperate",
            city_code %in% CLIMATE_ZONE_MAPPING$Cold ~ "Cold",
            city_code %in% CLIMATE_ZONE_MAPPING$Arid ~ "Arid",
            city_code %in% CLIMATE_ZONE_MAPPING$Tropical ~ "Tropical",
            TRUE ~ "Unknown"
          )
        )
      
      # 按变量、气候区域、模型类型汇总（取平均）
      df_zone_avg <- df_all_socioecon_with_zone %>%
        group_by(climate_zone, model_type, variable) %>%
        summarise(
          # 对Unexplained Grid Effect，使用均方根；其他用平均
          coefficient = ifelse(
            any(grepl("Unexplained", variable)),
            sqrt(mean(coefficient^2, na.rm = TRUE)),  # RMS for SD
            mean(coefficient, na.rm = TRUE)
          ),
          # 【修复】SE合并使用Meta-analysis标准方法
          se = ifelse(
            any(grepl("Unexplained", variable)),
            sqrt(mean(se^2, na.rm = TRUE)),  # For SD, use RMS
            1 / sqrt(sum(1 / (se^2 + 1e-10), na.rm = TRUE))  # Inverse variance weighting
          ),
          p_value = median(p_value, na.rm = TRUE),
          n_cities = n(),
          .groups = "drop"
        ) %>%
        mutate(
          ci_low = coefficient - 1.96 * se,
          ci_high = coefficient + 1.96 * se
        )
      
      # 为每个气候区域和模型类型绘制Forest Plot
      for (zone in unique(df_zone_avg$climate_zone)) {
        if (zone == "Unknown") next
        
        for (mtype in unique(df_zone_avg$model_type)) {
          zone_model_data <- df_zone_avg %>% 
            filter(climate_zone == zone, model_type == mtype)
          
          if (nrow(zone_model_data) == 0) next
          
          zone_output_path <- file.path(OUTPUT_DIR, 
                                         paste0("ZONE_", zone, "_", mtype, "_socioecon_forest_", TIME_LABEL, ".png"))
          
          plot_forest_socioecon(
            zone_model_data,
            zone_output_path,
            paste0(zone, " Zone - ", toupper(mtype), " Model\nSocioeconomic Variables (", 
                   unique(zone_model_data$n_cities), " cities)")
          )
          
          cat("    ✓", zone, "-", mtype, "Forest Plot已保存\n")
        }
      }
      
      # 【V4新增】按城市聚类（7大类）汇总
      cat("\n  生成按城市聚类（7大类）的汇总Forest Plot...\n")
      
      # 添加城市聚类信息
      df_all_socioecon_with_cluster <- df_all_socioecon %>%
        mutate(
          city_code = str_extract(city, "^[^_]+"),
          city_cluster = case_when(
            city_code %in% CITY_CLUSTER_MAPPING$Texas_Southern_Hot ~ "Texas_Southern_Hot",
            city_code %in% CITY_CLUSTER_MAPPING$East_Coast_Moderate ~ "East_Coast_Moderate",
            city_code %in% CITY_CLUSTER_MAPPING$Northern_Cold ~ "Northern_Cold",
            city_code %in% CITY_CLUSTER_MAPPING$West_Coast_Mediterranean ~ "West_Coast_Mediterranean",
            city_code %in% CITY_CLUSTER_MAPPING$Arid_Desert ~ "Arid_Desert",
            city_code %in% CITY_CLUSTER_MAPPING$Tropical ~ "Tropical",
            city_code %in% CITY_CLUSTER_MAPPING$Transition_Special ~ "Transition_Special",
            TRUE ~ "Unknown"
          )
        )
      
      # 按变量、聚类、模型类型汇总
      df_cluster_avg <- df_all_socioecon_with_cluster %>%
        filter(city_cluster != "Unknown") %>%
        group_by(city_cluster, model_type, variable) %>%
        summarise(
          coefficient = ifelse(
            any(grepl("Unexplained", variable)),
            sqrt(mean(coefficient^2, na.rm = TRUE)),
            mean(coefficient, na.rm = TRUE)
          ),
          se = ifelse(
            any(grepl("Unexplained", variable)),
            sqrt(mean(se^2, na.rm = TRUE)),
            1 / sqrt(sum(1 / (se^2 + 1e-10), na.rm = TRUE))
          ),
          p_value = median(p_value, na.rm = TRUE),
          n_cities = n(),
          .groups = "drop"
        ) %>%
        mutate(
          ci_low = coefficient - 1.96 * se,
          ci_high = coefficient + 1.96 * se
        )
      
      # 为每个聚类和模型类型绘制Forest Plot
      for (cluster in unique(df_cluster_avg$city_cluster)) {
        for (mtype in unique(df_cluster_avg$model_type)) {
          cluster_model_data <- df_cluster_avg %>% 
            filter(city_cluster == cluster, model_type == mtype)
          
          if (nrow(cluster_model_data) == 0) next
          
          cluster_output_path <- file.path(OUTPUT_DIR, 
                                            paste0("CLUSTER_", cluster, "_", mtype, "_socioecon_forest_", TIME_LABEL, ".png"))
          
          plot_forest_socioecon(
            cluster_model_data,
            cluster_output_path,
            paste0(cluster, " Cluster - ", toupper(mtype), " Model\nSocioeconomic Variables (", 
                   unique(cluster_model_data$n_cities), " cities)")
          )
          
          cat("    ✓", cluster, "-", mtype, "Forest Plot已保存\n")
        }
      }
      
      # 总体汇总（按模型类型分别汇总）
      cat("\n  生成总体汇总Forest Plot...\n")
      
      df_overall_avg <- df_all_socioecon %>%
        group_by(model_type, variable) %>%
        summarise(
          # 对Unexplained Grid Effect，使用均方根；其他用平均
          coefficient = ifelse(
            any(grepl("Unexplained", variable)),
            sqrt(mean(coefficient^2, na.rm = TRUE)),  # RMS for SD
            mean(coefficient, na.rm = TRUE)
          ),
          # 【修复】SE合并使用Meta-analysis标准方法
          se = ifelse(
            any(grepl("Unexplained", variable)),
            sqrt(mean(se^2, na.rm = TRUE)),  # For SD, use RMS
            1 / sqrt(sum(1 / (se^2 + 1e-10), na.rm = TRUE))  # Inverse variance weighting
          ),
          p_value = median(p_value, na.rm = TRUE),
          n_total = n(),
          .groups = "drop"
        ) %>%
        mutate(
          ci_low = coefficient - 1.96 * se,
          ci_high = coefficient + 1.96 * se
        )
      
      # 为每个模型类型生成Forest Plot
      for (mtype in unique(df_overall_avg$model_type)) {
        overall_model_data <- df_overall_avg %>% filter(model_type == mtype)
        
        overall_output_path <- file.path(OUTPUT_DIR, 
                                          paste0("OVERALL_", mtype, "_socioecon_forest_", TIME_LABEL, ".png"))
        
        plot_forest_socioecon(
          overall_model_data,
          overall_output_path,
          paste0("Overall - ", toupper(mtype), " Model\nSocioeconomic Variables (", 
                 n_distinct(df_all_socioecon$city), " cities)")
        )
        
        cat("    ✓ 总体", mtype, "Forest Plot已保存\n")
      }
      
      cat("\n")
      
      # 保存气候区域、聚类和总体汇总的CSV（分模型类型）
      write_csv(df_zone_avg, 
                file.path(OUTPUT_DIR, paste0("ZONE_by_model_socioecon_summary_", TIME_LABEL, ".csv")))
      write_csv(df_cluster_avg, 
                file.path(OUTPUT_DIR, paste0("CLUSTER_by_model_socioecon_summary_", TIME_LABEL, ".csv")))
      write_csv(df_overall_avg, 
                file.path(OUTPUT_DIR, paste0("OVERALL_by_model_socioecon_summary_", TIME_LABEL, ".csv")))
      
      cat("  ✓ 汇总统计CSV已保存（按气候带、聚类、模型类型分开）\n")
    }
  }
  
  cat(rep("=", 100), "\n\n", sep = "")
  
} else {
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("⊘ 跳过分区分析（用户选择）\n")
  cat(rep("=", 100), "\n\n", sep = "")
}

# ========== 保存所有模型系数 ==========

if (RUN_CITIES && length(successful_cities) > 0) {
  cat("\n保存单城市模型系数...\n")
  
  all_model_coefficients <- list()
  
  for (city_key in names(successful_cities)) {
    city_results <- successful_cities[[city_key]]
    
    # 遍历每个模型类型（composite/day/night）
    for (model_type in names(city_results)) {
      result <- city_results[[model_type]]
      
      if (!is.null(result$all_coefs)) {
        key <- paste0(city_key, "_", model_type)
        all_model_coefficients[[key]] <- result$all_coefs
      }
    }
  }

if (length(all_model_coefficients) > 0) {
  # 合并所有系数
  df_all_coefs <- bind_rows(all_model_coefficients)
  
  # 保存完整系数表
  all_coefs_file <- file.path(OUTPUT_DIR, 
                               paste0("model_coefficients_ALL_", TIME_LABEL, ".csv"))
  write_csv(df_all_coefs, all_coefs_file)
  
  cat("  ✓ 所有模型系数已保存:", all_coefs_file, "\n")
  cat("    - 总记录数:", nrow(df_all_coefs), "\n")
  cat("    - 城市数:", n_distinct(df_all_coefs$city), "\n")
  cat("    - 指标:", paste(unique(df_all_coefs$indicator), collapse = ", "), "\n")
  cat("    - 模型类型:", paste(unique(df_all_coefs$model_type), collapse = ", "), "\n")
  
  # 分别保存全局系数和格子系数（便于查看）
  df_global_coefs <- df_all_coefs %>% filter(variable_type == "global")
  df_grid_coefs <- df_all_coefs %>% filter(variable_type == "grid_specific")
  
  if (nrow(df_global_coefs) > 0) {
    global_coefs_file <- file.path(OUTPUT_DIR, 
                                    paste0("model_coefficients_GLOBAL_", TIME_LABEL, ".csv"))
    write_csv(df_global_coefs, global_coefs_file)
    cat("  ✓ 全局系数已保存:", global_coefs_file, "\n")
    cat("    - 包含: 截距、年份效应、周末效应（所有格子共享）\n")
  }
  
  if (nrow(df_grid_coefs) > 0) {
    grid_coefs_file <- file.path(OUTPUT_DIR, 
                                  paste0("model_coefficients_GRID_", TIME_LABEL, ".csv"))
    write_csv(df_grid_coefs, grid_coefs_file)
    cat("  ✓ 格子固定效应已保存:", grid_coefs_file, "\n")
    cat("    - 包含: 每个格子的独立截距\n")
  }
} else {
  cat("  ⚠ 没有提取到模型系数\n")
}

} else {
  cat("\n⊘ 跳过单城市系数保存（未运行单城市分析）\n")
}

cat(rep("=", 100), "\n\n", sep = "")

# ========== 第二阶段: Meta-regression汇总 ==========

if (RUN_STAGE2) {
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("第二阶段: Multivariate Meta-Regression\n")
  cat(rep("=", 100), "\n\n", sep = "")
  
  if (!dir.exists(OUTPUT_DIR)) {
    cat("\n✗ 错误: 输出目录不存在:", OUTPUT_DIR, "\n")
    cat("  请先运行第一阶段，或检查时间段选择是否正确\n\n")
    stop("第二阶段需要第一阶段的结果")
  }
  
  if (!RUN_CITIES) {
    cat("检测到未运行单城市分析，正在从RDS文件加载结果...\n")
  } else {
    cat("检测到第一阶段结果已在内存中，优先直接复用；如有缺失再从输出目录补载...\n")
  }
  
  stage1_results_split <- get_stage1_results_by_indicator(successful_cities, OUTPUT_DIR)
  successful_cities_cehwi <- stage1_results_split$cehwi
  successful_cities_exceeded <- stage1_results_split$exceeded_quantity
  successful_cities <- c(successful_cities_cehwi, successful_cities_exceeded)
  
  if (length(successful_cities) == 0) {
    cat("\n✗ 错误: 未找到任何第一阶段结果\n")
    cat("  请先运行第一阶段分析\n\n")
    stop("第二阶段需要第一阶段的结果")
  }
  
  cat("\n✓ 已准备", length(successful_cities), "个城市结果\n")
  cat("  - CEHWI:", length(successful_cities_cehwi), "个\n")
  cat("  - Exceeded Quantity:", length(successful_cities_exceeded), "个\n\n")
  
  # 按指标和模型类型分组进行meta-regression
  for (indicator in INDICATOR_LIST) {
    indicator_cities <- successful_cities[grepl(paste0("_", indicator, "$"), names(successful_cities))]
  
  if (length(indicator_cities) < 3) {
    cat("⚠ 指标", toupper(indicator), "的成功城市不足3个，跳过meta-regression\n\n")
    next
  }
  
  cat("\n", rep("-", 80), "\n", sep = "")
  cat("指标:", toupper(indicator), "| 纳入城市数:", length(indicator_cities), "\n")
  cat(rep("-", 80), "\n\n", sep = "")
  
  # 按模型类型分组（composite/day/night）
  for (model_type in STAGE1_MODEL_TYPES) {
    # 提取该模型类型的结果
    model_results <- list()
    for (city_key in names(indicator_cities)) {
      city_result <- indicator_cities[[city_key]]
      if (model_type %in% names(city_result)) {
        model_results[[city_key]] <- city_result[[model_type]]
      }
    }
    model_results <- filter_current_stage1_results(
      model_results,
      paste0("national ", toupper(indicator), " ", toupper(model_type))
    )
    
    if (length(model_results) < 3) {
      cat("  ⚠", toupper(model_type), "模型的城市不足3个 (", length(model_results), ")，跳过\n\n")
      next
    }
    
    cat("  [", toupper(model_type), "模型] 纳入", length(model_results), "个城市\n")
    
    # 【修复】提前定义输出目录，供 Step 2 内 conditional_RR 保存使用，避免「找不到对象 pooled_output_dir」
    pooled_output_dir <- file.path(OUTPUT_DIR, paste0("POOLED_META_", indicator, "_", model_type))
    dir.create(pooled_output_dir, showWarnings = FALSE, recursive = TRUE)
    pooled_mode_output_dir <- meta_model_output_dir(pooled_output_dir)
    
    tryCatch({
      # 提取所有城市的系数和协方差矩阵
      coef_list <- lapply(model_results, function(x) x$coef)
      vcov_list <- lapply(model_results, function(x) x$vcov)
      
      # 【数据验证】检查是否有NULL或空值
      valid_idx <- sapply(coef_list, function(x) !is.null(x) && length(x) > 0)
      
      if (sum(valid_idx) < 3) {
        cat("    ⚠ 有效系数不足3个城市，跳过\n\n")
        next
      }
      
      if (sum(valid_idx) < length(valid_idx)) {
        cat("    ⚠ 移除", length(valid_idx) - sum(valid_idx), "个无效城市\n")
        coef_list <- coef_list[valid_idx]
        vcov_list <- vcov_list[valid_idx]
        model_results <- model_results[valid_idx]
      }
      
      # 【数据验证】检查所有城市的系数维度是否一致
      coef_lengths <- sapply(coef_list, length)
      if (length(unique(coef_lengths)) > 1) {
        cat("    ⚠ 警告：城市间系数维度不一致：", unique(coef_lengths), "\n")
        # 找到最常见的维度
        common_length <- as.numeric(names(sort(table(coef_lengths), decreasing = TRUE)[1]))
        valid_dim_idx <- coef_lengths == common_length
        cat("      保留", sum(valid_dim_idx), "个维度为", common_length, "的城市\n")
        coef_list <- coef_list[valid_dim_idx]
        vcov_list <- vcov_list[valid_dim_idx]
        model_results <- model_results[valid_dim_idx]
      }
      
      # 【修复】将list转换为矩阵（mvmeta要求格式）
      # 每行 = 一个城市，每列 = 一个cross-basis系数
      coef_matrix <- do.call(rbind, coef_list)
      rownames(coef_matrix) <- names(model_results)
      
      cat("    准备Meta-regression数据:\n")
      cat("      - 城市数:", nrow(coef_matrix), "\n")
      cat("      - 系数数:", ncol(coef_matrix), "\n")
      
      # 【V4新增】准备城市级社会经济协变量（用于meta-regression）
      city_covariates <- NULL
      if (USE_SOCIOECON_STAGE2 && !is.null(df_socioecon_global)) {
        cat("    【V4】准备城市级社会经济协变量（meta-predictors）...\n")
        
        # 提取城市名称（从rownames，格式如"Atlanta_cehwi"或"Fort_Worth_cehwi"）
        # 正确方法：去掉指标后缀（_cehwi或_exceeded_quantity）
        city_names_in_meta <- sub("_(cehwi|exceeded_quantity)$", "", rownames(coef_matrix))
        
        # 计算每个城市的社会经济变量平均值（动态）
        city_socioecon_avg <- build_city_covariates_table(df_socioecon_global, SOCIOECON_VARS_USED)
        if (is.null(city_socioecon_avg) || nrow(city_socioecon_avg) == 0) {
          cat("      ⚠ 城市级协变量不可用，meta-predictors 将跳过\n")
          city_covariates <- NULL
        }
        
        # 【新增】如果有失业人口数据，计算城市级平均失业人口
        if (FALSE && exists("df_unemployed_global") && !is.null(df_unemployed_global)) {
          cat("      【新增】计算城市级平均失业人口...\n")
          
          # 先将df_socioecon_global与df_unemployed_global合并（按fish_id）
          # 然后计算每个城市的平均失业人口（跨年份和网格）
          city_unemployed_avg <- df_socioecon_global %>%
            select(fish_id, city) %>%
            left_join(df_unemployed_global, by = "fish_id") %>%
            group_by(city) %>%
            summarise(
              unemployed_pop_mean = mean(unemployed_pop, na.rm = TRUE),
              .groups = "drop"
            )
          
          # 标准化城市级失业人口
          city_unemployed_avg$unemployed_pop_mean_scaled <- scale(city_unemployed_avg$unemployed_pop_mean)[,1]
          
          # 合并到city_socioecon_avg
          city_socioecon_avg <- city_socioecon_avg %>%
            left_join(city_unemployed_avg %>% select(city, unemployed_pop_mean_scaled), 
                      by = "city")
          
          # 【修复】检查缺失值并填充
          n_missing_unemployed_city <- sum(is.na(city_socioecon_avg$unemployed_pop_mean_scaled))
          if (n_missing_unemployed_city > 0) {
            cat("        ⚠", n_missing_unemployed_city, "个城市缺失失业人口数据\n")
            # 使用所有城市的中位数填充（而非0，避免偏差）
            median_unemployed <- median(city_socioecon_avg$unemployed_pop_mean_scaled, na.rm = TRUE)
            if (is.na(median_unemployed)) {
              median_unemployed <- 0
            }
            city_socioecon_avg$unemployed_pop_mean_scaled[is.na(city_socioecon_avg$unemployed_pop_mean_scaled)] <- median_unemployed
            cat("        → 已用中位数", round(median_unemployed, 3), "填充\n")
          }
          
          cat("        ✓ 城市级失业人口已标准化\n")
        }
        
        # 匹配城市顺序
        cat("      【调试】匹配城市协变量...\n")
        cat("        - coef_matrix城市数:", length(city_names_in_meta), "\n")
        cat("        - city_socioecon_avg城市数:", ifelse(is.null(city_socioecon_avg), 0, nrow(city_socioecon_avg)), "\n")
        
        # 检查哪些城市在city_socioecon_avg中缺失
        missing_cities <- if (!is.null(city_socioecon_avg)) {
          city_names_in_meta[!city_names_in_meta %in% city_socioecon_avg$city]
        } else {
          city_names_in_meta
        }
        if (!is.null(city_socioecon_avg) && length(missing_cities) > 0) {
          cat("        ⚠ 以下城市在社会经济数据中缺失:", paste(missing_cities, collapse = ", "), "\n")
          cat("        【调试】前10个city_socioecon_avg中的城市名:\n")
          cat("          ", paste(head(city_socioecon_avg$city, 10), collapse = ", "), "\n")
          cat("        【调试】前10个coef_matrix中的城市名:\n")
          cat("          ", paste(head(city_names_in_meta, 10), collapse = ", "), "\n")
        }
        
        if (!is.null(city_socioecon_avg)) {
          city_covariates <- city_socioecon_avg %>%
            filter(city %in% city_names_in_meta) %>%
            arrange(match(city, city_names_in_meta)) %>%
            select(-city) %>%
            as.data.frame()
        }
        
        cat("        - 匹配后city_covariates行数:", nrow(city_covariates), "\n")
        
        # 【修复】检查行数是否匹配；若有城市在社会经济数据中缺失，则仅保留有协变量的城市做 meta
        keep_rows <- city_names_in_meta %in% city_socioecon_avg$city
        if (nrow(city_covariates) != nrow(coef_matrix)) {
          n_drop <- sum(!keep_rows)
          if (sum(keep_rows) < 3) {
            cat("        ✗ 错误: 有协变量的城市不足3个，无法进行 meta-regression\n")
            city_covariates <- NULL
          } else {
            cat("        → 排除", n_drop, "个无协变量城市（如:", paste(missing_cities, collapse = ", "), "），使用", sum(keep_rows), "个城市进行 meta-regression\n")
            coef_matrix <- coef_matrix[keep_rows, , drop = FALSE]
            vcov_list <- vcov_list[keep_rows]
            model_results <- model_results[keep_rows]
            rownames(city_covariates) <- rownames(coef_matrix)
            cat("      ✓ 城市级协变量已准备（", ncol(city_covariates), "个变量，", nrow(city_covariates), "个城市）\n")
            cat("        变量:", paste(names(city_covariates), collapse = ", "), "\n")
          }
        } else {
          # 确保行顺序与coef_matrix一致
          rownames(city_covariates) <- rownames(coef_matrix)
          
          cat("      ✓ 城市级协变量已准备（", ncol(city_covariates), "个变量）\n")
          cat("        变量:", paste(names(city_covariates), collapse = ", "), "\n")
        }
        
        # 【V4新增】VIF检查（避免meta-predictors共线性）
        if (!is.null(city_covariates) && ncol(city_covariates) > 1 && nrow(city_covariates) > 10) {
          cat("      【V4】检查meta-predictors的VIF...\n")
          
          tryCatch({
            # 用简单线性模型计算VIF
            library(car)
            
            # 为每个变量计算VIF
            vif_values <- numeric(ncol(city_covariates))
            names(vif_values) <- names(city_covariates)
            
            for (i in seq_along(vif_values)) {
              var_name <- names(vif_values)[i]
              other_vars <- setdiff(names(city_covariates), var_name)
              
              if (length(other_vars) > 0) {
                formula_vif <- as.formula(paste(var_name, "~", paste(other_vars, collapse = " + ")))
                lm_vif <- lm(formula_vif, data = city_covariates)
                r_squared <- summary(lm_vif)$r.squared
                vif_values[var_name] <- 1 / (1 - r_squared)
              } else {
                vif_values[var_name] <- 1
              }
            }
            
            cat("        VIF值:\n")
            for (var_name in names(vif_values)) {
              vif_val <- vif_values[var_name]
              status <- if (vif_val > 10) "⚠ 严重共线性" else if (vif_val > 5) "⚠ 中度共线性" else "✓"
              cat(sprintf("          %s: %.2f %s\n", var_name, vif_val, status))
            }
            
            # 如果有严重共线性：先删交通距离/FAR（均值或Gini模式），重算VIF；若 max(VIF)<15 则不再删别的
            VIF_TRY_REMOVE_FIRST <- c("Distance_to_Transit_mean", "FAR_mean",
                                      "Distance_to_Transit_gini_mean", "FAR_gini_mean")
            VIF_IMPROVED_THRESHOLD <- 15   # 删完这两者后若 max(VIF)<15 视为改善，不再删其他变量
            if (FALSE && any(vif_values > 10)) {
              cat("        【V4】检测到严重共线性（VIF>10），先移除 Distance_to_Transit_mean、FAR_mean 看是否改善\n")
              
              to_remove_first <- intersect(names(city_covariates), VIF_TRY_REMOVE_FIRST)
              if (length(to_remove_first) > 0) {
                city_covariates <- city_covariates[, setdiff(names(city_covariates), to_remove_first), drop = FALSE]
                cat("          已移除:", paste(to_remove_first, collapse = ", "), "\n")
                # 重新计算VIF
                if (ncol(city_covariates) > 1) {
                  vif_values <- numeric(ncol(city_covariates))
                  names(vif_values) <- names(city_covariates)
                  for (i in 1:ncol(city_covariates)) {
                    var_name <- names(city_covariates)[i]
                    other_vars <- setdiff(names(city_covariates), var_name)
                    formula_vif <- as.formula(paste(var_name, "~", paste(other_vars, collapse = " + ")))
                    lm_vif <- lm(formula_vif, data = city_covariates)
                    r_squared <- summary(lm_vif)$r.squared
                    vif_values[var_name] <- 1 / (1 - r_squared)
                  }
                }
                if (max(vif_values) < VIF_IMPROVED_THRESHOLD) {
                  cat("        ✓ 移除上述变量后已改善（max VIF = ", round(max(vif_values), 2), " < ", VIF_IMPROVED_THRESHOLD, "），不再移除其他变量\n", sep = "")
                }
              }
              
              # 若删完两者后仍超过阈值，再按“删VIF最高”继续
              while (max(vif_values) >= VIF_IMPROVED_THRESHOLD && length(vif_values) > 2) {
                max_vif_var <- names(which.max(vif_values))
                cat("          max VIF 仍 ≥ ", VIF_IMPROVED_THRESHOLD, "，继续移除: ", max_vif_var, " (VIF = ", round(vif_values[max_vif_var], 2), ")\n", sep = "")
                city_covariates <- city_covariates[, setdiff(names(city_covariates), max_vif_var), drop = FALSE]
                if (ncol(city_covariates) > 1) {
                  vif_values <- numeric(ncol(city_covariates))
                  names(vif_values) <- names(city_covariates)
                  for (i in 1:ncol(city_covariates)) {
                    var_name <- names(city_covariates)[i]
                    other_vars <- setdiff(names(city_covariates), var_name)
                    formula_vif <- as.formula(paste(var_name, "~", paste(other_vars, collapse = " + ")))
                    lm_vif <- lm(formula_vif, data = city_covariates)
                    r_squared <- summary(lm_vif)$r.squared
                    vif_values[var_name] <- 1 / (1 - r_squared)
                  }
                } else {
                  break
                }
              }
              
              cat("        ✓ 共线性处理完成，保留", ncol(city_covariates), "个变量\n")
            }
            
          }, error = function(e) {
            cat("      ⚠ VIF检查失败:", conditionMessage(e), "\n")
          })
        }
        
        # 检查缺失值
        n_missing_city <- sum(!complete.cases(city_covariates))
        if (n_missing_city > 0) {
          cat("      ⚠ 警告:", n_missing_city, "个城市有缺失值，将在meta-regression中移除\n")
        }
      }
      
      # 【V4】使用mvmeta进行随机效应meta-regression
      cat("    【V4】拟合随机效应meta-regression...\n")
      
      # 【V6改进】强力数值稳定性处理
      cat("      【V6】检查并修复协方差矩阵稳定性...\n")
      n_fixed <- 0
      n_bad <- 0
      
      for (i in seq_along(vcov_list)) {
        # 检查NA/Inf
        if (any(is.na(vcov_list[[i]])) || any(!is.finite(vcov_list[[i]]))) {
          n_bad <- n_bad + 1
          # 用单位矩阵的一个很大的值替换（相当于给这个城市很大的不确定性）
          vcov_list[[i]] <- diag(1000, nrow(vcov_list[[i]]))
          next
        }
        
        # 检查矩阵条件数
        eig_vals <- tryCatch({
          eigen(vcov_list[[i]], only.values = TRUE)$values
        }, error = function(e) {
          rep(0, nrow(vcov_list[[i]]))  # 如果eigen失败，返回0
        })
        
        # 如果有负特征值或条件数过大，添加强正则化
        if (any(eig_vals <= 1e-10) || max(abs(eig_vals)) / (min(abs(eig_vals[abs(eig_vals) > 1e-10])) + 1e-10) > 1e8) {
          n_fixed <- n_fixed + 1
          # 强力正则化：添加较大的值到对角线
          reg_strength <- max(abs(eig_vals)) * 0.01  # 最大特征值的1%
          vcov_list[[i]] <- vcov_list[[i]] + diag(max(reg_strength, 0.001), nrow(vcov_list[[i]]))
        }
      }
      
      if (n_fixed > 0 || n_bad > 0) {
        cat("        → 修复了", n_fixed, "个奇异矩阵，", n_bad, "个损坏矩阵\n")
      } else {
        cat("        ✓ 所有协方差矩阵状态良好\n")
      }
      
      # 尝试拟合meta-regression（带错误处理）
      mv_model <- NULL
      mv_model_error <- NULL
      
      if (!is.null(city_covariates) && nrow(city_covariates) > 0 && ncol(city_covariates) > 0) {
        # 包含城市级协变量的meta-regression（动态构建公式）
        covariate_formula_parts <- paste(names(city_covariates), collapse = " + ")
        meta_formula <- as.formula(paste("coef_matrix ~", covariate_formula_parts))
        
        mv_model <- tryCatch({
          mvmeta(meta_formula, data = city_covariates, S = vcov_list, method = "reml", 
                 control = list(maxiter = 1000, reltol = 1e-6))
        }, error = function(e) {
          mv_model_error <<- as.character(e$message)
          NULL
        })
        
        # 若失败（如 leading minor not positive definite），对 vcov 做轻度正则化后重试一次
        if (is.null(mv_model) && length(vcov_list) > 0) {
          cat("    → 尝试对协方差矩阵正则化后重试带协变量模型...\n")
          max_eigs <- sapply(vcov_list, function(V) {
            tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
          })
          reg_str <- max(0.001, max(max_eigs, na.rm = TRUE) * 0.05)
          vcov_list_reg <- lapply(vcov_list, function(V) V + diag(reg_str, nrow(V)))
          mv_model <- tryCatch({
            mvmeta(meta_formula, data = city_covariates, S = vcov_list_reg, method = "reml",
                   control = list(maxiter = 1000, reltol = 1e-6))
          }, error = function(e) NULL)
          if (!is.null(mv_model)) {
            cat("    ✓ 随机效应meta-regression完成（正则化后成功，", ncol(city_covariates), "个meta-predictors）\n")
            cat("      Meta-predictors:", paste(names(city_covariates), collapse = ", "), "\n")
          }
        } else if (!is.null(mv_model)) {
          cat("    ✓ 随机效应meta-regression完成（", ncol(city_covariates), "个meta-predictors）\n")
          cat("      Meta-predictors:", paste(names(city_covariates), collapse = ", "), "\n")
        }
        
        if (is.null(mv_model)) {
          cat("    ⚠ Meta-regression失败:", mv_model_error, "\n")
          cat("      → 回退到无协变量的随机效应模型\n")
          city_covariates <- NULL  # 回退
        }
      }
      
      # 如果有协变量的模型失败，或者本来就没有协变量，使用简单模型
      if (is.null(mv_model)) {
        mv_model <- tryCatch({
          mvmeta(coef_matrix, vcov_list, method = "reml", 
                 control = list(maxiter = 1000, reltol = 1e-6))
        }, error = function(e) {
          mv_model_error <<- as.character(e$message)
          NULL
        })
        
        if (!is.null(mv_model)) {
          cat("    ✓ 随机效应meta-regression完成（无meta-predictors）\n")
          write_meta_predictor_status(
            pooled_output_dir,
            c(
              "Meta-predictors were not run in this pooled model.",
              "Reason: city-level socioeconomic data were unavailable for the current run.",
              "If city_level_covariates* files are present in this folder, they are from an earlier run rather than the current one."
            )
          )
        } else {
          cat("    ✗ Meta-regression完全失败:", mv_model_error, "\n")
          cat("      → 跳过该模型的后续分析\n")
        }
      }
      
      # 如果模型完全失败，跳过后续分析
      if (is.null(mv_model)) {
        next
      }

      mv_model_predictor <- if (!is.null(city_covariates)) mv_model else NULL
      mv_model_for_pooled <- mv_model

      if (!is.null(city_covariates)) {
        mv_model_overall <- tryCatch({
          mvmeta(coef_matrix, vcov_list, method = "reml",
                 control = list(maxiter = 1000, reltol = 1e-6))
        }, error = function(e) {
          cat("    ⚠ Overall pooled intercept-only meta failed; pooled RR will fall back to current meta-predictor model: ",
              conditionMessage(e), "\n", sep = "")
          NULL
        })

        if (!is.null(mv_model_overall)) {
          mv_model_for_pooled <- mv_model_overall
          cat("    ✓ Overall pooled RR will use intercept-only random-effects meta (independent of MEAN/GINI mode)\n")
        }
      }
      
      # ========== 【V4输出】城市级meta-predictors系数 ==========
      if (!is.null(city_covariates)) {
        cat("\n    === 【V4】Meta-predictors贡献（解释城市间异质性）===\n")
        
        mv_summary_temp <- summary(mv_model)
        
        # 提取协变量系数（跳过截距和DLNM系数）
        if ("coefficients" %in% names(mv_summary_temp)) {
          coef_table_temp <- mv_summary_temp$coefficients
          
          # 查找社会经济变量的行（包含_mean的）
          socioecon_rows <- grep("_mean", rownames(coef_table_temp))
          
          if (length(socioecon_rows) > 0) {
            for (i in socioecon_rows) {
              var_name <- rownames(coef_table_temp)[i]
              coef_val <- coef_table_temp[i, "Estimate"]
              se_val <- coef_table_temp[i, "Std. Error"]
              
              # 安全提取p值
              if ("Pr(>|z|)" %in% colnames(coef_table_temp)) {
                p_val <- coef_table_temp[i, "Pr(>|z|)"]
              } else if ("Pr(>|t|)" %in% colnames(coef_table_temp)) {
                p_val <- coef_table_temp[i, "Pr(>|t|)"]
              } else {
                p_val <- NA
              }
              
              sig <- if (!is.na(p_val)) {
                ifelse(p_val < 0.001, "***",
                       ifelse(p_val < 0.01, "**",
                              ifelse(p_val < 0.05, "*", "ns")))
              } else {
                "NA"
              }
              
              cat(sprintf("      %s: coef=%.4f (SE=%.4f, p=%s) %s\n",
                          var_name, coef_val, se_val, 
                          ifelse(is.na(p_val), "NA", format.pval(p_val, digits=3)), sig))
            }
          }
        }
      }
      
      # ========== 【新增】输出Meta-regression模型评价指标 ==========
      cat("\n    === Meta-regression模型评价 ===\n")
      
      # 1. 基本统计量
      mv_summary <- summary(mv_model)
      
      # 2. AIC/BIC
      aic_value <- AIC(mv_model)
      bic_value <- BIC(mv_model)
      cat("      - AIC =", round(aic_value, 2), "\n")
      cat("      - BIC =", round(bic_value, 2), "\n")
      
      # 3. 异质性检验（Heterogeneity test）
      # Cochran's Q统计量和I²
      if (!is.null(mv_summary$qstat)) {
        q_stat <- mv_summary$qstat$Q
        q_df <- mv_summary$qstat$df
        q_pval <- mv_summary$qstat$pvalue
        
        # 【修复】检查Q统计量是否为向量（多个系数）
        if (length(q_stat) > 1) {
          # 使用第一个Q统计量（整体异质性）
          q_stat_overall <- q_stat[1]
          q_df_overall <- q_df[1]
          q_pval_overall <- q_pval[1]
          
          cat("      - Cochran's Q =", round(q_stat_overall, 2), 
              "(df =", q_df_overall, ", p =", format.pval(q_pval_overall, digits = 3), ")\n")
          
          # 计算I²（异质性程度）
          if (q_stat_overall > q_df_overall) {
            i_squared <- ((q_stat_overall - q_df_overall) / q_stat_overall) * 100
          } else {
            i_squared <- 0
          }
        } else {
          # 单个Q统计量
          cat("      - Cochran's Q =", round(q_stat, 2), 
              "(df =", q_df, ", p =", format.pval(q_pval, digits = 3), ")\n")
          
          # 计算I²
          if (q_stat > q_df) {
            i_squared <- ((q_stat - q_df) / q_stat) * 100
          } else {
            i_squared <- 0
          }
        }
        
        cat("      - I² =", round(i_squared, 1), "% ")
        
        # 解释I²
        if (i_squared < 25) {
          cat("(低异质性)\n")
        } else if (i_squared < 50) {
          cat("(中等异质性)\n")
        } else if (i_squared < 75) {
          cat("(高异质性)\n")
        } else {
          cat("(极高异质性)\n")
        }
      } else {
        # 如果没有qstat，设置默认值
        i_squared <- NA
      }
      
      # 4. 整体效应显著性（使用Wald检验）
      # 【修复】安全地提取p值
      coef_table_eval <- mv_summary$coefficients
      
      # 尝试多种p值列名
      if ("Pr(>|z|)" %in% colnames(coef_table_eval)) {
        coef_pvals <- coef_table_eval[, "Pr(>|z|)"]
      } else if ("Pr(>|t|)" %in% colnames(coef_table_eval)) {
        coef_pvals <- coef_table_eval[, "Pr(>|t|)"]
      } else if ("pvalue" %in% colnames(coef_table_eval)) {
        coef_pvals <- coef_table_eval[, "pvalue"]
      } else if (ncol(coef_table_eval) >= 4) {
        coef_pvals <- coef_table_eval[, 4]
      } else {
        coef_pvals <- rep(NA, nrow(coef_table_eval))
      }
      
      n_sig_coefs <- sum(coef_pvals < 0.05, na.rm = TRUE)
      cat("      - 显著系数数量:", n_sig_coefs, "/", length(coef_pvals), "\n")
      
      # 5. 整体效应强度（从最小到最大CEHWI的RR变化）
      # 这个需要在生成pooled RR曲线后计算
      
      cat("\n")
      
      # 汇总预测：生成pooled RR曲线
      cat("    生成pooled RR曲线...\n")
      
      pooled_cp <- NULL
      pooled_df <- NULL
      cb_template <- NULL
      cehwi_seq <- NULL
      
      # 【V6改进】分步错误处理，确保基本功能不受影响
      
      # Step 1: 准备cross-basis模板（必须成功）
      # 【与分区一致】暴露范围用城市 cehwi_range/cehwi_data，避免负值导致 cehwi_max 异常（仅跑选项4时也生效）
      tryCatch({
        cb_template <- model_results[[1]]$cb
        rangs <- lapply(model_results, function(x) x$cehwi_range)
        rangs <- rangs[!sapply(rangs, is.null)]
        if (length(rangs) > 0) {
          cehwi_range <- range(unlist(rangs), na.rm = TRUE)
          cehwi_max <- max(cehwi_range[2], 1)
        } else {
          cehwi_max <- 10
        }
        if (is.na(cehwi_max) || cehwi_max <= 0) {
          all_exp <- unlist(lapply(model_results, function(x) { d <- x$cehwi_data; if (!is.null(d)) d[d > 0] else NULL }))
          cehwi_max <- if (length(all_exp) > 0) max(max(all_exp, na.rm = TRUE), 1) else 10
        }
        if (is.na(cehwi_max) || cehwi_max <= 0) cehwi_max <- 10
        cehwi_seq <- seq(0, cehwi_max, length.out = 500)
        cat("      ✓ Cross-basis模板准备完成\n")
      }, error = function(e) {
        cat("      ✗ Cross-basis模板准备失败:", conditionMessage(e), "\n")
      })
      
      # 如果模板准备失败，跳过RR曲线生成
      if (is.null(cb_template) || is.null(cehwi_seq)) {
        cat("      ⚠ 无法准备cross-basis模板，跳过RR曲线生成\n")
      } else {
      
      # Step 2: 生成基本的Pooled RR曲线（核心功能）
      tryCatch({
        
        # 【V5完整版】生成Overall和Conditional RR曲线
        if (!is.null(city_covariates)) {
          cat("    【V5完整版】生成Overall和Conditional RR曲线...\n")
          
          # 【V5诊断】检测极端meta-predictor系数
          city_covar_coefs_df <- as.data.frame(coef(mv_model))
          city_covar_vcov <- vcov(mv_model)
          city_covar_ses <- sqrt(diag(city_covar_vcov))
          
          max_coef <- max(abs(city_covar_coefs_df[,1]), na.rm = TRUE)
          max_se <- max(city_covar_ses, na.rm = TRUE)
          
          # 【V5改进】更严格的极端检测：系数>10或SE>20即认为不稳定
          extreme_detected <- (max_coef > 10 || max_se > 20)
          
          if (extreme_detected) {
            cat("      ⚠ 警告: 检测到极端meta-predictor系数!\n")
            cat("        - Max |coef|:", round(max_coef, 2), "\n")
            cat("        - Max SE:", round(max_se, 2), "\n")
            cat("        → 条件RR曲线可能不稳定，将跳过生成\n")
          }
          
          # 1. Overall Pooled RR曲线（在meta-predictors均值处）
          cat("      [1/2] Overall Pooled RR曲线（meta-predictors=0，即均值）...\n")
          
          # 构建均值处的newdata（所有meta-predictors=0，因为已标准化）
          predictor_names <- colnames(city_covariates)[!colnames(city_covariates) %in% c("city")]
          newdata_mean <- as.data.frame(matrix(0, nrow = 1, ncol = length(predictor_names)))
          colnames(newdata_mean) <- predictor_names
          
          # 使用predict.mvmeta()在均值处预测
          pred_mean <- NULL
          # 【贴地修复】mvmeta 多变量时 predict() 返回 fit 为矩阵(1×9)，crosspred 需要长度为 9 的向量，否则会错用/回收导致曲线异常
          coef_mean <- as.vector(coef(mv_model_for_pooled))
          vcov_mean <- vcov(mv_model_for_pooled)
          if (length(coef_mean) != ncol(coef_matrix)) {
            cat("      ⚠ 预测系数长度(", length(coef_mean), ")与 cross-basis 维度(", ncol(coef_matrix), ")不一致，跳过 Overall RR\n")
            pooled_cp <- NULL
          } else {
            pooled_cp <- crosspred(
              cb_template,
              coef = coef_mean,
              vcov = vcov_mean,
              model.link = "log",
              at = cehwi_seq,
              cen = REFERENCE_CEHWI,
              cumul = TRUE
            )
            rr_range <- diff(range(pooled_cp$allRRfit, na.rm = TRUE))
            if (!is.na(rr_range) && rr_range < 0.05) {
              cat("      【说明】Overall RR 曲线变化很小（范围 ", round(rr_range, 4), "），属常见情况：在 meta-predictors 均值处平均效应接近无效；若希望看到更大变化可查看 Conditional RR（按变量分层）\n")
            }
          }
          cat("      ✓ Overall Pooled RR曲线生成完成\n")
          
          # 2. Conditional RR曲线（为每个主要meta-predictor生成高/中/低水平的曲线）
          if (!extreme_detected) {
            cat("      [2/2] Conditional RR曲线（按meta-predictor分层）...\n")
          } else {
            cat("      [2/2] ⚠ 跳过Conditional RR曲线（系数极端不稳定）\n")
            conditional_rr_list <- NULL
          }
          
          if (!extreme_detected) {
            # 选择主要的meta-predictors（排除Intercept）
            main_predictors <- predictor_names[!grepl("Intercept", predictor_names)]
            
            # 为每个meta-predictor生成条件RR曲线
            conditional_rr_list <- list()
            
            for (pred_var in main_predictors) {
              cat("        - 生成", pred_var, "的条件RR曲线（高/中/低）...\n")
            
            # 定义高/中/低水平（-1SD, 0, +1SD）
            levels <- c(-1, 0, 1)
            level_names <- c("Low", "Mean", "High")
            
            pred_curves <- list()
            
            for (i in seq_along(levels)) {
              # 构建newdata：该predictor=指定水平，其他=0
              newdata_cond <- newdata_mean
              newdata_cond[[pred_var]] <- levels[i]
              
              # 预测
              pred_cond <- tryCatch({
                predict(mv_model_predictor, newdata = newdata_cond, vcov = TRUE)
              }, error = function(e) {
                NULL
              })
              
              if (!is.null(pred_cond)) {
                # 【贴地修复】与 Overall 一致：确保 crosspred 收到向量及正确 vcov
                coef_cond <- as.vector(pred_cond$fit)
                vcov_cond <- pred_cond$vcov
                if (length(dim(vcov_cond)) == 3) vcov_cond <- vcov_cond[1, , ]
                if (length(coef_cond) != ncol(coef_matrix)) next
                cp_cond <- crosspred(
                  cb_template,
                  coef = coef_cond,
                  vcov = vcov_cond,
                  model.link = "log",
                  at = cehwi_seq,
                  cen = REFERENCE_CEHWI,
                  cumul = TRUE
                )
                
                # 保存数据
                pred_curves[[level_names[i]]] <- data.frame(
                  cehwi = cehwi_seq,
                  rr = cp_cond$allRRfit,
                  rr_low = cp_cond$allRRlow,
                  rr_high = cp_cond$allRRhigh,
                  level = level_names[i],
                  predictor = pred_var
                )
              }
            }
            
            if (length(pred_curves) > 0) {
              conditional_rr_list[[pred_var]] <- do.call(rbind, pred_curves)
              # 若高/中/低三条线几乎重合，提示为正常现象（该 meta-predictor 修饰作用弱）
              rr_by_level <- sapply(pred_curves, function(d) max(d$rr, na.rm = TRUE))
              if (length(rr_by_level) >= 2 && diff(range(rr_by_level)) < 0.05) {
                cat("          （三条线接近时表示该变量对效应修饰较弱，属正常）\n")
              }
              cat("          ✓", pred_var, "完成\n")
            }
          }
          
            # 保存conditional RR曲线数据
            if (length(conditional_rr_list) > 0) {
              conditional_rr_df <- do.call(rbind, conditional_rr_list)
              safe_write_csv(
                conditional_rr_df,
                file.path(pooled_mode_output_dir, "conditional_RR_curves.csv"),
                label = "conditional RR curves"
              )
              cat("      ✓ Conditional RR曲线数据已保存\n")
            }
          } # end if (!extreme_detected)
          
        } else {
          # 无meta-predictors时，使用标准方法
          pooled_cp <- crosspred(
            cb_template,
            coef = coef(mv_model_for_pooled),
            vcov = vcov(mv_model_for_pooled),
            model.link = "log",
            at = cehwi_seq,
            cen = REFERENCE_CEHWI,
            cumul = TRUE
          )
          cat("    ✓ Pooled RR曲线生成完成\n")
          conditional_rr_list <- NULL
        }
        
        cat("    ✓ Pooled RR曲线生成完成\n")
      }, error = function(e) {
        cat("    ✗ Pooled RR曲线生成失败:", conditionMessage(e), "\n")
        pooled_cp <<- NULL
      })
      
      # Step 3: 如果基本RR曲线成功，创建pooled_df
      if (!is.null(pooled_cp)) {
        # 【V6新增】计算95% Prediction Interval（考虑城市间异质性）
        # PI = pooled ± 1.96 * sqrt(τ² + SE²)
        # τ² 是城市间方差，从mvmeta模型提取
        pooled_pi_low <- NULL
        pooled_pi_high <- NULL
        
        if (!is.null(mv_model_for_pooled) && !is.null(mv_model_for_pooled$Psi)) {
          # 提取τ²（城市间方差）
          tau_sq <- tryCatch({
            psi_matrix <- mv_model_for_pooled$Psi
            if (is.matrix(psi_matrix)) {
              # 对于每个DLNM系数，提取对应的τ²
              diag(psi_matrix)
            } else {
              rep(0, length(pooled_cp$allRRfit))
            }
          }, error = function(e) {
            rep(0, length(pooled_cp$allRRfit))
          })
          
          # 计算PI（在log scale上）
          # SE from pooled estimate
          pooled_se <- (log(pooled_cp$allRRhigh) - log(pooled_cp$allRRlow)) / (2 * 1.96)
          
          # 平均τ²（因为可能是矩阵）
          tau_sq_avg <- ifelse(length(tau_sq) > 1, mean(tau_sq, na.rm = TRUE), tau_sq[1])
          
          # PI = exp(log(RR) ± 1.96 * sqrt(τ² + SE²))
          if (!is.na(tau_sq_avg) && tau_sq_avg > 0) {
            pi_factor <- 1.96 * sqrt(tau_sq_avg + pooled_se^2) / pooled_se
            pooled_pi_low <- pooled_cp$allRRfit^(pi_factor * sign(log(pooled_cp$allRRlow) - log(pooled_cp$allRRfit)))
            pooled_pi_high <- pooled_cp$allRRfit^(pi_factor * sign(log(pooled_cp$allRRhigh) - log(pooled_cp$allRRfit)))
            cat("    ✓ 95% Prediction Interval已计算（τ² =", round(tau_sq_avg, 4), "）\n")
          }
        }
        
        # 保存pooled结果（包含CI和PI）
        pooled_df <- data.frame(
          cehwi = cehwi_seq,
          rr = pooled_cp$allRRfit,
          rr_low = pooled_cp$allRRlow,
          rr_high = pooled_cp$allRRhigh,
          pi_low = if (!is.null(pooled_pi_low)) pooled_pi_low else NA,
          pi_high = if (!is.null(pooled_pi_high)) pooled_pi_high else NA
        )
        cat("      ✓ Pooled数据框已创建 (", nrow(pooled_df), "行)\n")
      } else {
        cat("      ⚠ 基本RR曲线生成失败，跳过后续步骤\n")
      }
      
      }  # 结束 if (is.null(cb_template) || is.null(cehwi_seq)) 的 else 分支
      
      # 【新增】计算Pooled效应百分比
      pooled_effect_pct <- NA
      pooled_rr_at_min <- NA
      pooled_rr_at_max <- NA
      
      if (!is.null(pooled_df) && nrow(pooled_df) >= 2) {
        min_cehwi_idx <- which.min(pooled_df$cehwi)
        max_cehwi_idx <- which.max(pooled_df$cehwi)
        pooled_rr_at_min <- pooled_df$rr[min_cehwi_idx]
        pooled_rr_at_max <- pooled_df$rr[max_cehwi_idx]
        
        if (is.finite(pooled_rr_at_min) && is.finite(pooled_rr_at_max) && pooled_rr_at_min > 0) {
          pooled_effect_pct <- (pooled_rr_at_max - pooled_rr_at_min) / pooled_rr_at_min * 100
          
          cat("    - Pooled效应变化: ", round(pooled_effect_pct, 1), "% ", 
              ifelse(pooled_effect_pct > 0, "↑", "↓"),
              " (RR从", round(pooled_rr_at_min, 3), "到", round(pooled_rr_at_max, 3), ")\n", sep = "")
          
          if (abs(pooled_effect_pct) > 1000) {
            cat("      ⚠ Pooled效应变化异常大，可能数值不稳定\n")
          }
        }
      }
      
      pooled_output_dir <- file.path(OUTPUT_DIR, paste0("POOLED_META_", indicator, "_", model_type))
      dir.create(pooled_output_dir, showWarnings = FALSE, recursive = TRUE)
      pooled_mode_output_dir <- meta_model_output_dir(pooled_output_dir)
      
      if (!is.null(pooled_df)) {
        safe_write_csv(pooled_df, file.path(pooled_output_dir, paste0("pooled_RR_curve.csv")), label = "national pooled RR curve")
        save_pooled_lag_response(
          model_results,
          pooled_output_dir,
          indicator = indicator,
          model_type = model_type,
          group_label = "National",
          reliability = "STAGE2_POOLED"
        )
      }
      
      # 【新增】保存pooled系数（Meta-regression的结果）
      pooled_coef_summary <- summary(mv_model)
      
      # 【修复】安全地提取系数，检查列名是否存在
      coef_table <- pooled_coef_summary$coefficients
      coef_names_vec <- rownames(coef_table)
      col_names_vec <- colnames(coef_table)
      
      # 构建基础数据框
      pooled_coefs_df <- data.frame(
        indicator = indicator,
        model_type = model_type,
        n_cities = length(model_results),
        coef_name = coef_names_vec,
        stringsAsFactors = FALSE
      )
      
      # 安全地添加每一列（检查是否存在）
      if ("Estimate" %in% col_names_vec) {
        pooled_coefs_df$coefficient <- coef_table[, "Estimate"]
      } else if (ncol(coef_table) >= 1) {
        pooled_coefs_df$coefficient <- coef_table[, 1]
      }
      
      if ("Std. Error" %in% col_names_vec) {
        pooled_coefs_df$se <- coef_table[, "Std. Error"]
      } else if ("Std.Error" %in% col_names_vec) {
        pooled_coefs_df$se <- coef_table[, "Std.Error"]
      } else if (ncol(coef_table) >= 2) {
        pooled_coefs_df$se <- coef_table[, 2]
      }
      
      if ("z" %in% col_names_vec) {
        pooled_coefs_df$z_value <- coef_table[, "z"]
      } else if ("t" %in% col_names_vec) {
        pooled_coefs_df$z_value <- coef_table[, "t"]
      } else if (ncol(coef_table) >= 3) {
        pooled_coefs_df$z_value <- coef_table[, 3]
      }
      
      if ("Pr(>|z|)" %in% col_names_vec) {
        pooled_coefs_df$p_value <- coef_table[, "Pr(>|z|)"]
      } else if ("Pr(>|t|)" %in% col_names_vec) {
        pooled_coefs_df$p_value <- coef_table[, "Pr(>|t|)"]
      } else if (ncol(coef_table) >= 4) {
        pooled_coefs_df$p_value <- coef_table[, 4]
      }
      
      if ("ci.lb" %in% col_names_vec) {
        pooled_coefs_df$ci_low <- coef_table[, "ci.lb"]
      } else if (ncol(coef_table) >= 5) {
        pooled_coefs_df$ci_low <- coef_table[, 5]
      }
      
      if ("ci.ub" %in% col_names_vec) {
        pooled_coefs_df$ci_high <- coef_table[, "ci.ub"]
      } else if (ncol(coef_table) >= 6) {
        pooled_coefs_df$ci_high <- coef_table[, 6]
      }
      
      write_csv(pooled_coefs_df, file.path(pooled_output_dir, paste0("pooled_coefficients.csv")))
      cat("    ✓ Pooled系数已保存\n")
      
      # 【V5新增】单独保存和可视化城市级社会经济协变量系数
      if (!is.null(city_covariates)) {
        cat("    【V5】保存城市级协变量系数...\n")
        
        # 提取社会经济协变量行
        socioecon_covariate_rows <- grep("_mean", coef_names_vec)
        
        if (length(socioecon_covariate_rows) > 0) {
          pooled_predictor_output_dir <- meta_predictor_output_dir(pooled_output_dir)
          # 构建城市级协变量系数数据框
          city_covar_coefs_df <- pooled_coefs_df[socioecon_covariate_rows, ]
          
          # 添加变量标签
          city_covar_coefs_df <- city_covar_coefs_df %>%
            mutate(
              variable_label = case_when(
                grepl("BD|Building_Density", coef_name) ~ "Building Density (City Avg)",
                grepl("FAR_mean", coef_name) ~ "Floor Area Ratio (City Avg)",
                grepl("NDVI_mean", coef_name) ~ "NDVI (City Avg)",
                grepl("Pop_mean", coef_name) ~ "Population (City Avg)",
                grepl("GDP_mean", coef_name) ~ "GDP (City Avg)",
                grepl("unemployed_pop_mean", coef_name) ~ "Unemployed Population (City Avg)",  # 【新增】
                grepl("WS_mean", coef_name) ~ "Wind Speed (City Avg)",
                grepl("Walk_mean", coef_name) ~ "Walkability (City Avg)",
                TRUE ~ coef_name
              ),
              significant = ifelse(!is.na(p_value) & p_value < 0.05, "sig", "ns")
            )
          
          # 保存CSV
          write_csv(city_covar_coefs_df, 
                    file.path(pooled_predictor_output_dir, paste0("city_level_covariates.csv")))
          cat("      ✓ 城市级协变量系数已保存（完整版：", nrow(city_covar_coefs_df), "个系数）\n")
          pooled_meta_output_success <- save_meta_predictor_outputs(
            meta_coef_df = city_covar_coefs_df,
            output_dir = pooled_predictor_output_dir,
            full_title = paste0("Meta-Predictors (", meta_predictor_mode_label(), ", Full Detail) - ", toupper(indicator), " - ", toupper(model_type)),
            full_subtitle = paste0(
              "Predictor mode: ", meta_predictor_mode_label(), " | Faceted by retained city-level covariate | ",
              nrow(city_covar_coefs_df), " coefficient rows | Blue = positive | Red = negative"
            ),
            summary_title = paste0("Meta-Predictors (", meta_predictor_mode_label(), "): How City Characteristics Modify Heatwave Effects"),
            summary_subtitle = paste0(
              "Model: ", toupper(indicator), " - ", toupper(model_type),
              " | Random-effects meta-regression | Summary keeps one row per retained covariate"
            )
          )
          if (pooled_meta_output_success) {
            write_meta_predictor_status(
              pooled_output_dir,
              c(
                "Meta-predictors were generated successfully for this pooled model.",
                paste0("Detailed files are in: ", basename(pooled_predictor_output_dir)),
                paste0("Predictor mode: ", meta_predictor_mode_label()),
                paste0("Predictors retained: ", paste(names(city_covariates), collapse = ", "))
              )
            )
            cat("      ✓ pooled city_level_covariates 图表已重建为分面版，避免 full 图所有系数叠在一起\n")
          }
          
          # 【V5新增】绘制完整版森林图（所有72个系数）
          if (FALSE && nrow(city_covar_coefs_df) > 0) {
            # 【V5改进】按meta-predictor分组排序，而不是按系数大小排序
            city_covar_coefs_df <- city_covar_coefs_df %>%
              mutate(
                # 提取meta-predictor名称
                predictor_name = case_when(
                  grepl("NDVI", coef_name) ~ "1_NDVI",
                  grepl("total_20", coef_name) ~ "2_Population",
                  grepl("BD|Building_Density", coef_name) ~ "3_Building_Density",
                  grepl("FAR_mean", coef_name) ~ "4_Floor_Area_Ratio",
                  grepl("GDP", coef_name) ~ "5_GDP",
                  grepl("unemployed", coef_name) ~ "6_Unemployed_Pop",
                  grepl("WS", coef_name) ~ "7_Wind_Speed",
                  grepl("Walk", coef_name) ~ "8_Walkability",
                  grepl("Intercept", coef_name) ~ "0_Intercept",
                  TRUE ~ "9_Other"
                ),
                # Reduced overall spline term.
                cb_lag = str_extract(coef_name, "(cb_cehwiv[0-9]+(\\.l[0-9]+)?|overall_spline[0-9]+|basis[0-9]+|b[0-9]+)"),
                cb_lag = ifelse(is.na(cb_lag), "overall_reduced", cb_lag)
              ) %>%
              arrange(predictor_name, cb_lag)  # 先按predictor分组，再按cb_lag排序
            
            # 创建分组标签（用于Y轴）
            city_covar_coefs_df <- city_covar_coefs_df %>%
              mutate(
                display_label = paste0(predictor_name, " | ", cb_lag),
                y_order = row_number()  # 按分组排序后的顺序
              )
            
            # 动态计算图片高度（完整版需要更高，但间距更紧凑）
            plot_height_full <- max(14, 5 + nrow(city_covar_coefs_df) * 0.25)
            
            p_city_covar_full <- ggplot(city_covar_coefs_df, 
                                        aes(x = coefficient, y = reorder(display_label, y_order))) +
              geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1) +
              geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, 
                                 alpha = ifelse(significant == "sig", 1, 0.4)), 
                             height = 0.2, linewidth = 0.9, color = "gray20") +
              geom_point(aes(color = ifelse(coefficient > 0, "#D62728", "#1F77B4"),
                             alpha = ifelse(significant == "sig", 1, 0.4)), 
                         size = 3, shape = 19) +
              scale_color_identity() +
              scale_alpha_identity() +
              labs(
                title = paste0("Meta-Predictors (Full Detail) - ", toupper(indicator), " - ", toupper(model_type)),
                subtitle = paste0("All ", nrow(city_covar_coefs_df), " reduced-basis × meta-predictor coefficients | ",
                                 "Each social variable × overall reduced spline coefficients\n",
                                 "Red = Positive Effect | Blue = Negative Effect | Solid = p<0.05 | Faded = p≥0.05"),
                x = "Meta-Regression Coefficient",
                y = ""
              ) +
              theme_minimal(base_size = 12) +
              theme(
                plot.title = element_text(face = "bold", size = 15, hjust = 0),
                plot.subtitle = element_text(size = 10, color = "gray30", hjust = 0, lineheight = 1.2),
                axis.title.x = element_text(face = "bold", size = 12, margin = margin(t = 8)),
                axis.text.y = element_text(size = 9, color = "black", hjust = 1),
                axis.text.x = element_text(size = 11),
                panel.grid.major.y = element_blank(),  # 移除Y轴网格线
                panel.grid.major.x = element_blank(),  # 移除X轴网格线
                panel.grid.minor = element_blank(),
                axis.line.x = element_line(color = "black", linewidth = 1),
                axis.line.y = element_line(color = "black", linewidth = 1),
                panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2),
                plot.margin = margin(12, 12, 12, 12)
              )
            
            ggsave(file.path(pooled_output_dir, "city_level_covariates_forest_FULL.png"),
                   p_city_covar_full, width = 16, height = plot_height_full, dpi = 300)
            cat("      ✓ 完整版森林图已保存（", nrow(city_covar_coefs_df), "个系数，", 
                round(plot_height_full, 1), "英寸高）\n")
          }
          
          # 【V5改进】汇总同一社会经济变量在不同cross-basis系数上的效应
          # 提取变量类型（不含cross-basis和lag信息）
          # 覆盖全部 retained meta-predictors，避免 Crime/Unemployment/Urbanization_Rate/Street_Intersection_Density 落入 Other
          city_covar_coefs_df <- city_covar_coefs_df %>%
            mutate(
              # 提取纯变量名（去掉cb_cehwiv1.l1.等前缀）
              variable_only = case_when(
                grepl("Intercept", coef_name) ~ "Intercept",
                grepl("NDVI", coef_name) & grepl("_mean", coef_name) ~ "NDVI",
                grepl("total_20_55_mean|total_20_mean", coef_name) ~ "Population (20-55)",
                grepl("GDP_mean", coef_name) ~ "GDP",
                grepl("Crime_mean", coef_name) ~ "Crime",
                grepl("Unemployment_mean", coef_name) ~ "Unemployment",
                grepl("BD|Building_Density", coef_name) ~ "Building Density",
                grepl("FAR_mean", coef_name) ~ "Floor Area Ratio",
                grepl("Urbanization_Rate_mean", coef_name) ~ "Urbanization Rate",
                grepl("Street_Intersection_Density_mean", coef_name) ~ "Street Intersection Density",
                grepl("Distance_to_Transit_mean", coef_name) ~ "Distance to Transit",
                grepl("Walkability_Index_mean|Walk_mean", coef_name) ~ "Walkability Index",
                grepl("unemployed_pop_mean", coef_name) ~ "Unemployed Population",
                grepl("WS", coef_name) & grepl("_mean", coef_name) ~ "Wind Speed",
                TRUE ~ "Other"
              )
            )
          
          # 汇总：对每个社会经济变量，计算加权平均系数
          city_covar_summary <- NULL
          
          # 汇总版 CSV 与森林图已由 save_meta_predictor_outputs() 统一生成
          
          # 绘制Forest Plot（汇总版 - 短版）
          if (FALSE && nrow(city_covar_summary) > 0) {
            # 动态计算图片高度（短版，只有6-8个变量左右）
            plot_height <- max(6, 4 + nrow(city_covar_summary) * 0.8)
            
            p_city_covar <- ggplot(city_covar_summary, 
                                   aes(x = coefficient, y = reorder(variable_only, coefficient))) +
              geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1.5) +
              geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, 
                                 alpha = ifelse(significant == "sig", 1, 0.5)), 
                             height = 0.4, linewidth = 2, color = "gray20") +
              geom_point(aes(color = ifelse(coefficient > 0, "#D62728", "#1F77B4"),
                             alpha = ifelse(significant == "sig", 1, 0.5)), 
                         size = 12, shape = 19) +
              scale_color_identity() +
              scale_alpha_identity() +
              labs(
                title = "Meta-Predictors: How City Characteristics Modify Heatwave Effects",
                subtitle = paste0("Model: ", toupper(indicator), " - ", toupper(model_type), 
                                 " | Random-Effects Meta-Regression | ", nrow(city_covar_summary), " Variables\n",
                                 "Weighted average across all DLNM coefficients | Red = Amplify Effect | Blue = Reduce Effect | Solid = p<0.05"),
                x = "Meta-Regression Coefficient (Standardized)",
                y = ""
              ) +
              theme_minimal(base_size = 20) +
              theme(
                plot.title = element_text(face = "bold", size = 22, hjust = 0),
                plot.subtitle = element_text(size = 14, color = "gray30", hjust = 0, lineheight = 1.3),
                axis.title.x = element_text(face = "bold", size = 18, margin = margin(t = 15)),
                axis.text.y = element_text(size = 18, face = "bold", color = "black", hjust = 1),
                axis.text.x = element_text(size = 16),
                panel.grid.major.y = element_blank(),  # 移除Y轴网格线
                panel.grid.major.x = element_blank(),  # 移除X轴网格线
                panel.grid.minor = element_blank(),
                axis.line.x = element_line(color = "black", linewidth = 1.5),
                axis.line.y = element_line(color = "black", linewidth = 1.5),
                panel.border = element_rect(color = "black", fill = NA, linewidth = 1.5),
                plot.margin = margin(20, 20, 20, 20)
              )
            
            ggsave(file.path(pooled_output_dir, "city_level_covariates_forest.png"),
                   p_city_covar, width = 15, height = plot_height, dpi = 300)
            cat("      ✓ 汇总版森林图已保存（", nrow(city_covar_summary), "个变量，", 
                round(plot_height, 1), "英寸高）\n")
          }
        }
      }
      
      # 【新增】保存模型评价摘要
      model_eval_df <- data.frame(
        indicator = indicator,
        model_type = model_type,
        n_cities = length(model_results),
        n_coefs = length(coef_pvals),
        n_sig_coefs = n_sig_coefs,
        AIC = aic_value,
        BIC = bic_value,
        effect_pct = ifelse(is.na(pooled_effect_pct), NA, pooled_effect_pct),
        rr_at_min = ifelse(is.na(pooled_rr_at_min), NA, pooled_rr_at_min),
        rr_at_max = ifelse(is.na(pooled_rr_at_max), NA, pooled_rr_at_max),
        stringsAsFactors = FALSE
      )
      
      # 【修复】添加异质性指标（安全处理向量）
      if (!is.null(mv_summary$qstat)) {
        # 提取Q统计量（如果是向量，使用第一个）
        q_stat_save <- if (length(mv_summary$qstat$Q) > 1) {
          mv_summary$qstat$Q[1]
        } else {
          mv_summary$qstat$Q
        }
        
        q_df_save <- if (length(mv_summary$qstat$df) > 1) {
          mv_summary$qstat$df[1]
        } else {
          mv_summary$qstat$df
        }
        
        q_pval_save <- if (length(mv_summary$qstat$pvalue) > 1) {
          mv_summary$qstat$pvalue[1]
        } else {
          mv_summary$qstat$pvalue
        }
        
        model_eval_df$Q_statistic <- q_stat_save
        model_eval_df$Q_df <- q_df_save
        model_eval_df$Q_pvalue <- q_pval_save
        
        # 使用前面计算的i_squared（已经是标量）
        model_eval_df$I_squared <- ifelse(exists("i_squared") && !is.na(i_squared), i_squared, NA)
      } else {
        model_eval_df$Q_statistic <- NA
        model_eval_df$Q_df <- NA
        model_eval_df$Q_pvalue <- NA
        model_eval_df$I_squared <- NA
      }
      
      write_csv(model_eval_df, file.path(pooled_output_dir, paste0("model_evaluation.csv")))
      cat("    ✓ 模型评价已保存\n")
      
      # 【V6修复】检查pooled_df是否存在
      if (is.null(pooled_df) || nrow(pooled_df) == 0) {
        cat("    ⚠ 无法生成Pooled RR曲线（数据缺失）\n")
        next  # 跳到下一个模型
      }
      
      # 绘制pooled RR曲线
      model_label <- stage1_model_display_label(model_type)
      
      # 【颜色方案】根据model_type选择颜色
      line_colors <- c(
        "composite" = "#D53E4F",  # 红色
        "day" = "#FF8C00",         # 橙色
        "night" = "#9B59B6"        # 紫色
      )
      line_color <- stage1_model_color(model_type)
      if (is.null(line_color) || is.na(line_color)) line_color <- "#D53E4F"
      
      # 【优化】检查pooled RR范围，决定是否使用log scale
      pooled_rr_max <- max(pooled_df$rr_high, na.rm = TRUE)
      pooled_rr_min <- min(pooled_df$rr_low, na.rm = TRUE)
      pooled_rr_curve_max <- max(pooled_df$rr, na.rm = TRUE)
      pooled_rr_curve_min <- min(pooled_df$rr, na.rm = TRUE)
      use_log_pooled <- (pooled_rr_max > 10 || pooled_rr_min < 0.1)
      
      # 【优化】只对真正极端的值截断（>100或<0.01）
      pooled_has_extreme_high <- pooled_rr_max > 100
      pooled_has_extreme_low <- pooled_rr_min < 0.01
      
      if (pooled_has_extreme_high || pooled_has_extreme_low) {
        cat("      ⚠ Pooled CI有极端值，将适度截断\n")
        pooled_df_plot <- pooled_df %>%
          mutate(
            rr_low_clipped = ifelse(rr_low < 0.01, 0.01, rr_low),
            rr_high_clipped = ifelse(rr_high > 100, 100, rr_high)
          )
      } else {
        # 正常情况：完全保留
        pooled_df_plot <- pooled_df %>%
          mutate(
            rr_low_clipped = rr_low,
            rr_high_clipped = rr_high
          )
      }
      
      # 构建统计信息文本（放在subtitle）
      pooled_effect_text <- if (is.na(pooled_effect_pct)) {
        "N/A"
      } else if (abs(pooled_effect_pct) > 1000) {
        "Unstable"
      } else {
        paste0(round(pooled_effect_pct, 1), "%", ifelse(pooled_effect_pct > 0, "↑", "↓"))
      }
      
      # 【修复】异质性文本（使用前面计算的i_squared）
      heterogeneity_text <- if (exists("i_squared") && !is.na(i_squared)) {
        paste0("I² = ", round(i_squared, 1), "%")
      } else {
        "I² = N/A"
      }
      
      # 显著系数文本
      sig_coefs_text <- paste0(n_sig_coefs, "/", length(coef_pvals), " sig. coefs")
      
      # 【V6改进】根据异质性添加解释文本
      heterogeneity_interp <- if (exists("i_squared") && !is.na(i_squared)) {
        if (i_squared > 90) {
          " (Very High - Pooled estimate has limited generalizability)"
        } else if (i_squared > 75) {
          " (High - Substantial variation across cities)"
        } else if (i_squared > 50) {
          " (Moderate)"
        } else {
          " (Low)"
        }
      } else {
        ""
      }
      
      # 构建完整的subtitle（两行）
      pooled_subtitle_text <- paste0(
        "Meta-regression: ", toupper(indicator), " vs PA (scaled by ", SCALE_FACTOR, ") | Lag 0-", MAX_LAG, " days | Reference: CEHWI = 0\n",
        "N cities = ", length(model_results), 
        "  |  Pooled Effect: ", pooled_effect_text,
        "  |  ", heterogeneity_text, heterogeneity_interp,
        "  |  AIC = ", round(aic_value, 1),
        "  |  ", sig_coefs_text
      )
      
      # 【V5.1】计算暴露数据的98th分位数，用于截断x轴
      all_cehwi_for_98th <- unlist(lapply(model_results, function(x) {
        if (!is.null(x$cehwi_data)) x$cehwi_data[x$cehwi_data > 0] else NULL
      }))
      all_cehwi_for_98th <- all_cehwi_for_98th[!is.na(all_cehwi_for_98th)]
      
      if (length(all_cehwi_for_98th) > 10) {
        cehwi_98th <- quantile(all_cehwi_for_98th, 0.98, na.rm = TRUE)
        cat("    【V5.1】RR_with_distribution组合图X轴将截断至98th分位数:", round(cehwi_98th, 2), "\n")
      } else {
        # Fallback: 使用pooled_df_plot的98th分位数
        cehwi_98th <- quantile(pooled_df_plot$cehwi, 0.98, na.rm = TRUE)
        cat("    【V5.1】RR_with_distribution组合图X轴将截断至98th分位数（从预测值）:", round(cehwi_98th, 2), "\n")
      }
      
      # 【V6新增】检查是否有有效的PI数据
      has_valid_pi <- !all(is.na(pooled_df_plot$pi_low)) && !all(is.na(pooled_df_plot$pi_high))
      
      # 如果有PI，也需要截断
      if (has_valid_pi) {
        pooled_df_plot <- pooled_df_plot %>%
          mutate(
            pi_low_clipped = ifelse(is.na(pi_low), NA, ifelse(pi_low < 0.01, 0.01, pi_low)),
            pi_high_clipped = ifelse(is.na(pi_high), NA, ifelse(pi_high > 100, 100, pi_high))
          )
        # 更新caption
        pi_caption <- "95% CI (dark gray) shows uncertainty of pooled estimate; 95% PI (light gray) shows expected range for individual cities | X-axis truncated at 98th percentile"
      } else {
        pooled_df_plot$pi_low_clipped <- NA
        pooled_df_plot$pi_high_clipped <- NA
        pi_caption <- "X-axis truncated at 98th percentile"
      }
      
      p_pooled <- ggplot(pooled_df_plot, aes(x = cehwi, y = rr)) +
        geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8)
      
      # 【V6新增】先绘制PI（如果存在，作为最外层、最浅的区域）
      if (has_valid_pi) {
        p_pooled <- p_pooled +
          geom_ribbon(aes(ymin = pi_low_clipped, ymax = pi_high_clipped), 
                      fill = "gray90", alpha = 0.3)  # 非常浅的灰色PI
      }
      
      # 然后绘制CI（中间层）
      p_pooled <- p_pooled +
        geom_ribbon(aes(ymin = rr_low_clipped, ymax = rr_high_clipped), 
                    fill = "gray70", alpha = 0.5) +  # 深一点的灰色CI
        geom_line(color = line_color, linewidth = 2)  # 最上层的点估计线
      
      # 构建caption
      full_caption <- build_rr_caption(
        indicator = indicator,
        has_pi = has_valid_pi,
        pi_caption = if (has_valid_pi) pi_caption else NULL,
        ci_clipped = pooled_has_extreme_high || pooled_has_extreme_low,
        reliability = "STABLE",
        include_percentiles = length(af_list) > 0,
        x_truncated = TRUE
      )
      
      # 【V6新增】添加百分位数虚线（如果计算了AF，则有百分位数数据）
      if (length(af_list) > 0) {
        # 从最后一个af_summary提取百分位数
        last_af <- af_list[[length(af_list)]]
        if (!is.null(last_af) && nrow(last_af) > 0) {
          p25_val <- last_af$cehwi_p75[1]  # 实际上是25-75-90的75
          p75_val <- last_af$cehwi_p90[1]  # 90th
          p90_val <- last_af$cehwi_p95[1]  # 95th
          
          # 或者重新计算（更准确）
          all_cehwi_for_percentile <- unlist(lapply(model_results, function(x) {
            if (!is.null(x$cehwi_data)) x$cehwi_data[x$cehwi_data > 0] else NULL
          }))
          
          if (length(all_cehwi_for_percentile) > 10) {
            p25_val <- quantile(all_cehwi_for_percentile, 0.25, na.rm = TRUE)
            p75_val <- quantile(all_cehwi_for_percentile, 0.75, na.rm = TRUE)
            p90_val <- quantile(all_cehwi_for_percentile, 0.90, na.rm = TRUE)
            
            # 只添加在X轴范围内的虚线
            x_range <- range(pooled_df_plot$cehwi, na.rm = TRUE)
            percentiles_to_plot <- c()
            labels_to_plot <- c()
            
            if (p25_val >= x_range[1] && p25_val <= x_range[2]) {
              percentiles_to_plot <- c(percentiles_to_plot, p25_val)
              labels_to_plot <- c(labels_to_plot, "25th")
            }
            if (p75_val >= x_range[1] && p75_val <= x_range[2]) {
              percentiles_to_plot <- c(percentiles_to_plot, p75_val)
              labels_to_plot <- c(labels_to_plot, "75th")
            }
            if (p90_val >= x_range[1] && p90_val <= x_range[2]) {
              percentiles_to_plot <- c(percentiles_to_plot, p90_val)
              labels_to_plot <- c(labels_to_plot, "90th")
            }
            
            if (length(percentiles_to_plot) > 0) {
              p_pooled <- p_pooled +
                geom_vline(xintercept = percentiles_to_plot, 
                           linetype = "dotted", color = "gray40", linewidth = 1) +
                annotate("text", 
                         x = percentiles_to_plot, 
                         y = max(pooled_df_plot$rr, na.rm = TRUE) * 1.05,
                         label = labels_to_plot, 
                         size = 3.5, angle = 90, vjust = -0.5, color = "gray40")
            }
          }
        }
      }
      
      p_pooled <- p_pooled +
        labs(
          title = paste0("Pooled RR Curve: ", model_label, " (", length(model_results), " cities)"),
          subtitle = pooled_subtitle_text,
          x = paste0(toupper(indicator)),
          y = "Pooled Relative Risk (RR)",
          caption = full_caption
        ) +
        rr_plot_theme(14) +
        # 【V5.1】截断x轴到98th分位数
        coord_cartesian(xlim = c(0, cehwi_98th))
      
      # 【贴地修复】智能Y轴刻度：CI 爆炸时改用 RR 定上限
      pooled_ci_min <- min(pooled_df_plot$rr_low_clipped, na.rm = TRUE)
      pooled_ci_max <- max(pooled_df_plot$rr_high_clipped, na.rm = TRUE)
      pooled_rr_min <- min(pooled_df_plot$rr, na.rm = TRUE)
      pooled_rr_max <- max(pooled_df_plot$rr, na.rm = TRUE)
      pooled_rr_range <- pooled_rr_max - pooled_rr_min
      pooled_ci_range <- pooled_ci_max - pooled_ci_min
      
      # 判断 CI 是否爆炸：CI 上限 > RR 最大值 * 10 或 CI 范围 > RR 范围 * 15
      ci_exploded <- (pooled_ci_max > pooled_rr_max * 10 || pooled_ci_range > pooled_rr_range * 15)
      
      if (ci_exploded) {
        cat("    ⚠ 检测到 CI 爆炸（CI 上限 ", round(pooled_ci_max, 1), " >> RR 最大值 ", round(pooled_rr_max, 1), "），Y 轴改用 RR * 2.5\n")
        pooled_y_lim_max <- pooled_rr_max * 2.5
        pooled_y_lim_min <- max(0.005, pooled_rr_min * 0.5)
        use_log_pooled <- (pooled_y_lim_max / pooled_y_lim_min > 10)
      } else {
        # CI 正常，用 CI 定 Y 轴
        pooled_y_lim_min <- max(0.005, pooled_ci_min * 0.7)
        pooled_y_lim_max <- min(200, pooled_ci_max * 1.3)
      }
      
      if (use_log_pooled) {
        # 对数Y轴
        pooled_possible_breaks <- c(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
        pooled_y_breaks <- pooled_possible_breaks[pooled_possible_breaks >= pooled_y_lim_min * 0.8 & 
                                     pooled_possible_breaks <= pooled_y_lim_max * 1.2]
        if (!1 %in% pooled_y_breaks) pooled_y_breaks <- sort(c(pooled_y_breaks, 1))
        
        p_pooled <- p_pooled + 
          scale_y_log10(
            breaks = pooled_y_breaks,
            labels = as.character(pooled_y_breaks),
            limits = c(pooled_y_lim_min, pooled_y_lim_max)
          ) +
          labs(y = "Pooled Relative Risk (RR, log scale)")
        
        if (ci_exploded) {
          cat("    ⚠ 使用对数Y轴（CI 爆炸已改用 RR 定界） (Y轴:", round(pooled_y_lim_min, 2), "-", round(pooled_y_lim_max, 2), ")\n")
        } else {
          cat("    ⚠ Pooled RR范围较大，使用对数Y轴 (CI范围:", round(pooled_ci_min, 2), "-", round(pooled_ci_max, 2), ")\n")
        }
      } else {
        # 线性Y轴
        pooled_y_min <- max(0, pooled_ci_min - (pooled_ci_max - pooled_ci_min) * 0.1)
        pooled_y_max <- pooled_ci_max + (pooled_ci_max - pooled_ci_min) * 0.1
        pooled_y_range <- pooled_y_max - pooled_y_min
        
        if (pooled_y_range < 1) {
          pooled_y_breaks <- seq(floor(pooled_y_min * 10) / 10, ceiling(pooled_y_max * 10) / 10, by = 0.2)
        } else if (pooled_y_range < 3) {
          pooled_y_breaks <- seq(floor(pooled_y_min * 2) / 2, ceiling(pooled_y_max * 2) / 2, by = 0.5)
        } else {
          pooled_y_breaks <- seq(floor(pooled_y_min), ceiling(pooled_y_max), by = 1)
        }
        
        if (!1 %in% pooled_y_breaks && 1 >= pooled_y_min && 1 <= pooled_y_max) {
          pooled_y_breaks <- sort(c(pooled_y_breaks, 1))
        }
        
        p_pooled <- p_pooled + 
          scale_y_continuous(
            breaks = pooled_y_breaks,
            limits = c(pooled_y_min, pooled_y_max)
          )
      }
      
      ggsave(file.path(pooled_output_dir, "pooled_RR_curve.png"),
             p_pooled, width = 12, height = 8, dpi = 300)
      
      cat("    ✓ Pooled RR曲线图已保存\n")
      
      # 【V6新增】生成暴露分布直方图（与RR曲线X轴对齐）
      cat("    【V6】生成暴露分布直方图...\n")
      
      tryCatch({
        # 【V6修复】收集所有城市的CEHWI数据
        all_cehwi_hist <- unlist(lapply(model_results, function(x) {
          if (!is.null(x$cehwi_data)) x$cehwi_data[x$cehwi_data > 0] else NULL
        }))
        all_cehwi_hist <- all_cehwi_hist[!is.na(all_cehwi_hist)]
        
        # 【V6修复】Fallback: 如果RDS里没有cehwi_data（旧版本），从原始数据读取
        if (length(all_cehwi_hist) <= 10) {
          cat("      【V6修复】RDS中无暴露数据，从原始文件重新读取...\n")
          for (city_result in model_results) {
            city_name <- city_result$city
            city_temp_file <- list.files(
              path = file.path(DATA_DIR, "by_city"),
              pattern = paste0("^", city_name, "_.*\\.csv$"),
              full.names = TRUE
            )
            if (length(city_temp_file) > 0) {
              temp_data <- tryCatch({
                read_csv(city_temp_file[1], show_col_types = FALSE) %>%
                  filter(date >= START_DATE, date <= END_DATE)
              }, error = function(e) NULL)
              
              if (!is.null(temp_data)) {
                # 提取对应indicator的列
                cehwi_col <- paste0(tolower(indicator), "_", tolower(model_type))
                if (cehwi_col %in% names(temp_data)) {
                  cehwi_vals <- temp_data[[cehwi_col]][temp_data[[cehwi_col]] > 0]
                  all_cehwi_hist <- c(all_cehwi_hist, cehwi_vals)
                }
              }
            }
          }
          all_cehwi_hist <- all_cehwi_hist[!is.na(all_cehwi_hist)]
          cat("        → 成功读取", length(all_cehwi_hist), "个暴露数据点\n")
        }
        
        if (length(all_cehwi_hist) > 10) {
          # 计算百分位数
          p25_hist <- quantile(all_cehwi_hist, 0.25, na.rm = TRUE)
          p75_hist <- quantile(all_cehwi_hist, 0.75, na.rm = TRUE)
          p90_hist <- quantile(all_cehwi_hist, 0.90, na.rm = TRUE)
          
          # 创建直方图数据框
          hist_df <- data.frame(cehwi = all_cehwi_hist)
          
          # 绘制直方图
          p_hist <- ggplot(hist_df, aes(x = cehwi)) +
            geom_histogram(bins = 60, fill = "steelblue", color = "white", alpha = 0.7) +
            geom_vline(xintercept = c(p25_hist, p75_hist, p90_hist), 
                       linetype = "dotted", color = "gray40", linewidth = 1) +
            annotate("text", x = p25_hist, y = Inf, 
                     label = "25th", angle = 90, vjust = 1.2, hjust = 1.1, size = 3.5, color = "gray40") +
            annotate("text", x = p75_hist, y = Inf, 
                     label = "75th", angle = 90, vjust = 1.2, hjust = 1.1, size = 3.5, color = "gray40") +
            annotate("text", x = p90_hist, y = Inf, 
                     label = "90th", angle = 90, vjust = 1.2, hjust = 1.1, size = 3.5, color = "gray40") +
            labs(
              title = paste0("Exposure Distribution: ", model_label),
              x = paste0(toupper(indicator)),
              y = "Frequency",
              caption = paste0("n = ", length(all_cehwi_hist), " heatwave-days from ", 
                              length(model_results), " cities")
            ) +
            theme_minimal(base_size = 14) +
            theme(
              plot.title = element_text(face = "bold", size = 16),
              axis.title = element_text(face = "bold", size = 12),
              plot.caption = element_text(size = 10, color = "gray40", hjust = 1),
              panel.grid = element_blank(),
              axis.line = element_line(color = "gray30", linewidth = 0.5),
              plot.margin = margin(5, 10, 5, 10)  # 与RR曲线保持一致的边距
            ) +
            coord_cartesian(xlim = c(0, cehwi_98th))  # 【V5.1】与RR曲线X轴一致，截断至98th
          
          # 保存直方图
          ggsave(file.path(pooled_output_dir, "exposure_distribution.png"),
                 p_hist, width = 12, height = 4, dpi = 300)
          
          cat("      ✓ 暴露分布直方图已保存\n")
          cat("      提示: 此直方图X轴与RR曲线对齐，可并排拼接\n")
          
          # 可选：生成组合图（RR曲线在上，直方图在下）
          library(patchwork)
          
          p_combined <- build_rr_distribution_combined_plot(
            rr_plot = p_pooled,
            hist_plot = p_hist,
            caption_text = full_caption,
            heights = c(2, 1),
            caption_size = 10
          )
          
          ggsave(file.path(pooled_output_dir, "pooled_RR_with_distribution.png"),
                 p_combined, width = 12, height = 10, dpi = 300)
          
          cat("      ✓ 组合图已保存（RR曲线+分布直方图）\n")
          
        } else {
          cat("      ⚠ 暴露数据不足，跳过直方图生成\n")
        }
        
      }, error = function(e) {
        cat("      ✗ 直方图生成失败:", conditionMessage(e), "\n")
      })
      
      # ========== 【V6.3】计算归因分数（Attributable Fraction, AF） - 已禁用 ==========
      # 【V6.3说明】基于conditional RR的AF计算不稳定（导致数值爆炸）
      # 正确的AF应该在第一阶段每个城市独立计算，然后在第二阶段pooled
      # 参考：参考文献的方法
      
      # 【V6.3暂时禁用】等待第一阶段实现城市级AF计算
      if (FALSE && !is.null(mv_model) && !is.null(conditional_rr_list) && length(conditional_rr_list) > 0) {
        cat("\n    【V6.2】计算meta-predictor的归因分数（AF）...\n")
        
        tryCatch({
          # 1. 收集所有城市的CEHWI数据（用于计算加权AF）
          all_cehwi_data <- unlist(lapply(model_results, function(x) {
            if (!is.null(x$cehwi_data)) {
              x$cehwi_data[x$cehwi_data > 0]
            } else {
              NULL
            }
          }))
          all_cehwi_data <- all_cehwi_data[!is.na(all_cehwi_data)]
          
          # Fallback: 从CSV读取
          if (length(all_cehwi_data) <= 10) {
            cat("      【V6修复】RDS中无暴露数据，从原始文件重新读取...\n")
            for (city_result in model_results) {
              city_name <- city_result$city
              city_temp_file <- list.files(
                path = file.path(DATA_DIR, "by_city"),
                pattern = paste0("^", city_name, "_.*\\.csv$"),
                full.names = TRUE
              )
              if (length(city_temp_file) > 0) {
                temp_data <- tryCatch({
                  read_csv(city_temp_file[1], show_col_types = FALSE) %>%
                    filter(date >= START_DATE, date <= END_DATE)
                }, error = function(e) NULL)
                
                if (!is.null(temp_data)) {
                  cehwi_col <- paste0(tolower(indicator), "_", tolower(model_type))
                  if (cehwi_col %in% names(temp_data)) {
                    cehwi_vals <- temp_data[[cehwi_col]][temp_data[[cehwi_col]] > 0]
                    all_cehwi_data <- c(all_cehwi_data, cehwi_vals)
                  }
                }
              }
            }
            all_cehwi_data <- all_cehwi_data[!is.na(all_cehwi_data)]
            cat("        → 成功读取", length(all_cehwi_data), "个暴露数据点\n")
          }
          
          if (length(all_cehwi_data) > 10) {
            # 【V6.2新逻辑】计算每个meta-predictor的AF贡献
            # 参考：Paper 1 (Fig. 2), Paper 3 (Fig. 5)
            
            # 2. 创建暴露分布的直方图（用于加权AF计算）
            cehwi_breaks <- seq(min(all_cehwi_data, na.rm = TRUE), 
                               max(all_cehwi_data, na.rm = TRUE), 
                               length.out = 50)
            cehwi_hist <- hist(all_cehwi_data, breaks = cehwi_breaks, plot = FALSE)
            cehwi_midpoints <- cehwi_hist$mids
            cehwi_freqs <- cehwi_hist$counts / sum(cehwi_hist$counts)  # 归一化为概率
            
            cat("      暴露分布: n =", length(all_cehwi_data), 
                ", 范围 =", round(min(all_cehwi_data), 2), "-", round(max(all_cehwi_data), 2), "\n")
            
            # 3. 对每个meta-predictor计算AF
            # AF_predictor = Σ [p(x) * (RR_high(x) - RR_low(x)) / RR_high(x)]
            # 其中 RR_high 是该predictor在+1SD时的RR，RR_low 是-1SD时的RR
            
            af_by_predictor <- list()
            
            for (pred_var in names(conditional_rr_list)) {
              pred_data <- conditional_rr_list[[pred_var]]
              
              # 提取Low和High的RR曲线
              rr_low_curve <- pred_data %>% filter(level == "Low")
              rr_high_curve <- pred_data %>% filter(level == "High")
              
              if (nrow(rr_low_curve) > 0 && nrow(rr_high_curve) > 0) {
                # 对每个暴露水平，插值得到RR值
                rr_low_at_x <- approx(rr_low_curve$cehwi, rr_low_curve$rr, 
                                     xout = cehwi_midpoints, rule = 2)$y
                rr_high_at_x <- approx(rr_high_curve$cehwi, rr_high_curve$rr, 
                                      xout = cehwi_midpoints, rule = 2)$y
                
                rr_low_ci_low <- approx(rr_low_curve$cehwi, rr_low_curve$rr_low, 
                                       xout = cehwi_midpoints, rule = 2)$y
                rr_high_ci_high <- approx(rr_high_curve$cehwi, rr_high_curve$rr_high, 
                                         xout = cehwi_midpoints, rule = 2)$y
                
                # 计算加权AF
                # AF = Σ [p(x) * (RR_high(x) - RR_low(x)) / RR_high(x)]
                af_at_each_x <- (rr_high_at_x - rr_low_at_x) / rr_high_at_x
                af_weighted <- sum(cehwi_freqs * af_at_each_x, na.rm = TRUE)
                
                # 计算置信区间（简化：使用CI的边界）
                af_at_each_x_low <- (rr_high_ci_high - rr_low_ci_low) / rr_high_ci_high
                af_weighted_low <- sum(cehwi_freqs * af_at_each_x_low, na.rm = TRUE)
                
                # 另一个边界（保守估计）
                af_at_each_x_high <- (rr_high_at_x - rr_low_at_x) / rr_low_at_x
                af_weighted_high <- sum(cehwi_freqs * af_at_each_x_high, na.rm = TRUE)
                
                # 保存结果
                af_by_predictor[[pred_var]] <- data.frame(
                  predictor = pred_var,
                  af_pct = af_weighted * 100,
                  af_low = min(af_weighted_low, af_weighted_high) * 100,
                  af_high = max(af_weighted_low, af_weighted_high) * 100,
                  interpretation = ifelse(af_weighted > 0, "增加风险", "降低风险")
                )
                
                cat("      ", pred_var, ": AF =", round(af_weighted * 100, 2), "%\n")
              }
            }
            
            # 4. 合并所有predictor的AF结果
            if (length(af_by_predictor) > 0) {
              af_summary <- do.call(rbind, af_by_predictor) %>%
                mutate(
                  indicator = indicator,
                  model_type = model_type,
                  n_cities = length(model_results)
                )
              
              # 保存AF汇总
              write_csv(af_summary, file.path(pooled_output_dir, "attributable_fraction_by_predictor.csv"))
              
              cat("      ✓ Meta-predictor的AF已计算并保存\n")
              cat("      → 文件:", "attributable_fraction_by_predictor.csv\n")
              
              # 保存AF数据供后续森林图使用
              af_list[[length(af_list) + 1]] <- af_summary
            } else {
              cat("      ⚠ 无可用的conditional RR数据，跳过AF计算\n")
            }
            
          } else {
            cat("      ⚠ 暴露数据不足，跳过AF计算\n")
          }
          
        }, error = function(e) {
          cat("      ✗ AF计算失败:", conditionMessage(e), "\n")
        })
      }
      
      # 【V5新增】绘制Conditional RR曲线（如果有）
      if (!is.null(conditional_rr_list) && length(conditional_rr_list) > 0) {
        cat("    【V5】绘制Conditional RR曲线对比图...\n")
        
        conditional_rr_df <- do.call(rbind, conditional_rr_list)
        
        # 为每个meta-predictor生成一张图
        for (pred_var in unique(conditional_rr_df$predictor)) {
          pred_data <- conditional_rr_df %>% filter(predictor == pred_var)
          # Keep conditional RR plots readable: CIs are retained in CSV, but
          # old plotting path should not let exploded CIs determine the y-axis.
          pred_data$rr_low <- pred_data$rr
          pred_data$rr_high <- pred_data$rr
          y_limits <- conditional_rr_ylim(pred_data)
          
          # 清理predictor名称用于标题
          pred_label <- case_when(
            grepl("NDVI", pred_var) ~ "NDVI (Vegetation Index)",
            grepl("total_20", pred_var) ~ "Population (20-55 years)",
            grepl("BD|Building_Density", pred_var) ~ "Building Density",
            grepl("FAR", pred_var) ~ "Floor Area Ratio",
            grepl("GDP", pred_var) ~ "GDP",
            grepl("unemployed", pred_var) ~ "Unemployed Population",
            grepl("WS", pred_var) ~ "Wind Speed",
            grepl("Walk", pred_var) ~ "Walkability Index",
            TRUE ~ pred_var
          )
          
          # 定义颜色：Low=红，Mean=黄，High=蓝
          color_map <- c("Low" = "#D62728", "Mean" = "#FFA500", "High" = "#1F77B4")
          
          p_cond <- ggplot(pred_data, aes(x = cehwi, y = rr, color = level, fill = level)) +
            geom_hline(yintercept = 1, linetype = "dashed", color = "gray40", linewidth = 1) +
            geom_ribbon(aes(ymin = rr_low, ymax = rr_high), alpha = 0.2, color = NA) +
            geom_line(linewidth = 2) +
            scale_color_manual(values = color_map,
                              labels = c("Low" = "Low (-1 SD)", 
                                        "Mean" = "Mean (0 SD)", 
                                        "High" = "High (+1 SD)")) +
            scale_fill_manual(values = color_map,
                             labels = c("Low" = "Low (-1 SD)", 
                                       "Mean" = "Mean (0 SD)", 
                                       "High" = "High (+1 SD)")) +
            labs(
              title = paste0("Conditional RR Curves by ", pred_label),
              subtitle = paste0("Model: ", toupper(indicator), " - ", toupper(model_type), 
                               " | How ", pred_label, " modifies heatwave effects\n",
                               "Red = Low ", pred_label, " cities | Orange = Mean | Blue = High ", pred_label, " cities"),
              x = toupper(indicator),
              y = "Relative Risk (RR)",
              color = pred_label,
              fill = pred_label
            ) +
            theme_minimal(base_size = 16) +
            theme(
              plot.title = element_text(face = "bold", size = 18, hjust = 0),
              plot.subtitle = element_text(size = 12, color = "gray30", hjust = 0, lineheight = 1.25),
              axis.title = element_text(face = "bold", size = 14),
              axis.text = element_text(size = 13),
              legend.title = element_text(face = "bold", size = 14),
              legend.text = element_text(size = 13),
              legend.position = "right",
              panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
              panel.grid.minor = element_blank(),
              axis.line = element_line(color = "black", linewidth = 1),
              panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2)
            )
          if (!is.null(y_limits)) {
            p_cond <- p_cond + coord_cartesian(ylim = y_limits)
          }
          
          # 保存图片
          safe_pred_name <- str_replace_all(pred_var, "[^a-zA-Z0-9_]", "_")
          ggsave(file.path(pooled_mode_output_dir, paste0("conditional_RR_", safe_pred_name, ".png")),
                 p_cond, width = 14, height = 9, dpi = 300)
          
          cat("      ✓", pred_label, "条件RR曲线已保存\n")
        }
        
        cat("    ✓ 所有Conditional RR曲线图已保存\n")
      }
      
      # Root stores the overall pooled model; MEAN/GINI predictor models go to mode folders.
      save_meta_model_artifact(
        list(model = mv_model_for_pooled, crosspred = pooled_cp, pred_df = pooled_df,
             meta_predictor_mode = "OVERALL_INTERCEPT_ONLY"),
        file.path(pooled_output_dir, "meta_model.rds"),
        label = "overall pooled meta model"
      )
      if (!is.null(mv_model_predictor)) {
        save_meta_model_artifact(
          list(model = mv_model_predictor, crosspred = pooled_cp, pred_df = pooled_df,
               conditional_rr = conditional_rr_list,
               meta_predictor_mode = meta_predictor_mode_label()),
          file.path(pooled_mode_output_dir, "meta_model.rds"),
          label = "meta-predictor model"
        )
      }
      
      cat("    Meta model artifact handling complete (heavy RDS save = ", SAVE_META_MODEL_RDS, ")\n\n", sep = "")
      
    }, error = function(e) {
      cat("    ✗ Meta-regression失败:", conditionMessage(e), "\n\n")
    })
  }
  }  # 结束指标循环
  
  # ========== 【V5新增】汇总Weekend Effect ==========
  
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("【V5】汇总Weekend Effect（控制变量分析）\n")
  cat(rep("=", 100), "\n\n", sep = "")
  
  # 从所有城市的RDS结果中提取weekend effect
  # 注意：successful_cities已经在第二阶段开始时加载过了，这里直接使用
  all_weekend_effects <- list()
  
  if (length(successful_cities) > 0) {
    for (city_key in names(successful_cities)) {
      city_results <- successful_cities[[city_key]]
      
      for (model_type in names(city_results)) {
        result <- city_results[[model_type]]
        
        if (!is.null(result$socioecon_coefs)) {
          # 只提取weekend effect和grid heterogeneity
          weekend_data <- result$socioecon_coefs %>%
            mutate(
              city = result$city,
              indicator = result$indicator,
              model_type = model_type
            )
          
          key <- paste0(city_key, "_", model_type)
          all_weekend_effects[[key]] <- weekend_data
          
          cat("    ✓ 提取", result$city, "-", model_type, "weekend effect\n")
        }
      }
    }
  } else {
    cat("  ⚠ 未找到successful_cities数据，尝试从RDS文件加载...\n")
    # 作为备用方案，从RDS文件加载
    for (indicator in INDICATORS) {
      loaded_results <- load_stage1_results(OUTPUT_DIR, indicator)
      for (city_key in names(loaded_results)) {
        city_results <- loaded_results[[city_key]]
        for (model_type in names(city_results)) {
          result <- city_results[[model_type]]
          if (!is.null(result$socioecon_coefs)) {
            weekend_data <- result$socioecon_coefs %>%
              mutate(city = result$city, indicator = result$indicator, model_type = model_type)
            key <- paste0(city_key, "_", model_type)
            all_weekend_effects[[key]] <- weekend_data
          }
        }
      }
    }
  }
  
  if (length(all_weekend_effects) > 0) {
    cat("\n  ✓ 提取完成，共", length(all_weekend_effects), "个模型结果\n\n")
    
    # 合并所有数据
    weekend_df <- bind_rows(all_weekend_effects)
    
    # 保存完整数据
    weekend_output_dir <- file.path(OUTPUT_DIR, "WEEKEND_EFFECT_SUMMARY")
    dir.create(weekend_output_dir, showWarnings = FALSE, recursive = TRUE)
    write_csv(weekend_df, file.path(weekend_output_dir, "all_cities_weekend_grid_effects.csv"))
    
    cat("  ✓ 所有城市Weekend Effect已汇总: ", nrow(weekend_df), " 条记录\n")
    cat("    - 文件: WEEKEND_EFFECT_SUMMARY/all_cities_weekend_grid_effects.csv\n\n")
    
    # 按indicator和model_type汇总
    weekend_summary <- weekend_df %>%
      group_by(indicator, model_type, variable) %>%
      summarise(
        n_cities = n(),
        mean_coef = mean(coefficient, na.rm = TRUE),
        se_pooled = 1 / sqrt(sum(1 / (se^2 + 1e-10), na.rm = TRUE)),
        mean_rr = ifelse(all(is.na(rr)), NA, mean(rr, na.rm = TRUE)),
        n_significant = sum(significant %in% c("*", "**", "***"), na.rm = TRUE),
        .groups = "drop"
      )
    
    write_csv(weekend_summary, file.path(weekend_output_dir, "weekend_effect_summary.csv"))
    cat("  ✓ Weekend Effect汇总统计已保存\n\n")
    
    # 【V5新增】生成Weekend Effect森林图
    cat("  【V5】生成Weekend Effect森林图...\n")
    
    # 只提取Weekend Effect（不包括Grid Heterogeneity）
    weekend_only <- weekend_df %>%
      filter(variable == "Weekend_Effect")
    
    if (nrow(weekend_only) > 0) {
      # 为每个indicator和model_type生成森林图
      for (indic in unique(weekend_only$indicator)) {
        for (mtype in unique(weekend_only$model_type)) {
          subset_data <- weekend_only %>%
            filter(indicator == indic, model_type == mtype) %>%
            arrange(desc(coefficient))
          
          if (nrow(subset_data) > 0) {
            # 计算图片高度
            plot_height <- max(8, 4 + nrow(subset_data) * 0.3)
            
            # 绘制森林图
            p_weekend <- ggplot(subset_data, aes(x = rr, y = reorder(city, rr))) +
              geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 1.5) +
              geom_errorbarh(aes(xmin = rr_low, xmax = rr_high, 
                                 color = significant != "ns"), 
                             height = 0.4, linewidth = 1.2) +
              geom_point(aes(color = significant != "ns"), size = 4) +
              scale_color_manual(
                values = c("TRUE" = "#D62728", "FALSE" = "gray50"),
                labels = c("TRUE" = "Significant (p<0.05)", "FALSE" = "Non-significant"),
                name = ""
              ) +
              labs(
                title = paste0("Weekend Effect on Physical Activity"),
                subtitle = paste0(
                  "Indicator: ", toupper(indic), " | Model: ", toupper(mtype), "\n",
                  "RR > 1: Activity increases on weekends | ",
                  "Solid: p<0.05, Faded: ns"
                ),
                x = "Relative Risk (RR)",
                y = "City"
              ) +
              theme_minimal(base_size = 16) +
              theme(
                plot.title = element_text(face = "bold", size = 20, hjust = 0),
                plot.subtitle = element_text(size = 13, color = "gray30", hjust = 0, lineheight = 1.2),
                axis.title = element_text(face = "bold", size = 16),
                axis.text.y = element_text(size = 12),
                axis.text.x = element_text(size = 14),
                legend.position = "bottom",
                legend.text = element_text(size = 14),
                panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5),
                panel.grid.major.y = element_blank(),
                panel.grid.minor = element_blank(),
                axis.line = element_line(color = "black", linewidth = 1),
                panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2)
              )
            
            # 保存图片
            ggsave(
              file.path(weekend_output_dir, paste0("weekend_effect_forest_", indic, "_", mtype, ".png")),
              p_weekend, 
              width = 12, 
              height = plot_height, 
              dpi = 300
            )
            
            cat("    ✓", toupper(indic), "-", toupper(mtype), "森林图已保存 (", nrow(subset_data), "个城市)\n")
          }
        }
      }
      
      cat("  ✓ 所有Weekend Effect森林图已保存\n\n")
      
      # 【V5新增】生成汇总森林图（按indicator和model_type平均）
      cat("  【V5】生成Weekend Effect汇总森林图...\n")
      
      weekend_summary_plot <- weekend_summary %>%
        filter(variable == "Weekend_Effect") %>%
        mutate(
          rr_low = exp(mean_coef - 1.96 * se_pooled),
          rr_high = exp(mean_coef + 1.96 * se_pooled),
          model_label = paste0(toupper(indicator), " - ", toupper(model_type)),
          sig_label = ifelse(n_significant >= n_cities * 0.5, 
                            paste0("Significant (", n_significant, "/", n_cities, " cities)"),
                            paste0("Mostly NS (", n_significant, "/", n_cities, " cities)"))
        )
      
      if (nrow(weekend_summary_plot) > 0) {
        p_weekend_summary <- ggplot(weekend_summary_plot, 
                                     aes(x = mean_rr, y = reorder(model_label, mean_rr))) +
          geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 2) +
          geom_errorbarh(aes(xmin = rr_low, xmax = rr_high,
                             color = n_significant >= n_cities * 0.5), 
                         height = 0.5, linewidth = 2) +
          geom_point(aes(color = n_significant >= n_cities * 0.5, 
                        size = n_cities)) +
          scale_color_manual(
            values = c("TRUE" = "#D62728", "FALSE" = "gray50"),
            labels = c("TRUE" = "Majority Significant", "FALSE" = "Majority NS"),
            name = ""
          ) +
          scale_size_continuous(range = c(4, 10), name = "Number of Cities") +
          labs(
            title = "Weekend Effect on Physical Activity - Overall Summary",
            subtitle = paste0(
              "Pooled estimates across all cities | RR > 1: Activity increases on weekends\n",
              "Error bars: 95% CI | Point size: Number of cities"
            ),
            x = "Relative Risk (RR)",
            y = "Model"
          ) +
          theme_minimal(base_size = 18) +
          theme(
            plot.title = element_text(face = "bold", size = 22, hjust = 0),
            plot.subtitle = element_text(size = 14, color = "gray30", hjust = 0, lineheight = 1.2),
            axis.title = element_text(face = "bold", size = 18),
            axis.text = element_text(size = 16),
            legend.position = "bottom",
            legend.text = element_text(size = 14),
            panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5),
            panel.grid.major.y = element_blank(),
            panel.grid.minor = element_blank(),
            axis.line = element_line(color = "black", linewidth = 1.2),
            panel.border = element_rect(color = "black", fill = NA, linewidth = 1.5)
          )
        
        ggsave(
          file.path(weekend_output_dir, "weekend_effect_forest_SUMMARY.png"),
          p_weekend_summary, 
          width = 14, 
          height = 8, 
          dpi = 300
        )
        
        cat("  ✓ Weekend Effect汇总森林图已保存\n\n")
      }
    }
    
  } else {
    cat("  ⚠ 未找到Weekend Effect数据\n\n")
  }
  
  # ========== 【V6新增】AF（归因分数）森林图 ==========
  
  cat(rep("=", 100), "\n", sep = "")
  cat("【V6.2】生成归因分数（AF）森林图 - Meta-Predictor版本\n")
  cat(rep("=", 100), "\n", sep = "")
  
  if (length(af_list) > 0) {
    cat("\n  汇总所有模型的AF数据...\n")
    
    # 合并所有AF数据
    af_all <- bind_rows(af_list)
    meta_af_output_dir <- meta_model_output_dir(OUTPUT_DIR)
    meta_af_mode_label <- meta_predictor_mode_label()
    
    # 保存汇总的AF数据
    write_csv(af_all, file.path(meta_af_output_dir, paste0("AF_by_predictor_all_models_", meta_af_mode_label, ".csv")))
    cat("  ✓ AF汇总数据已保存:", nrow(af_all), "个predictor-model组合\n\n")
    
    # 【V6.2新逻辑】森林图Y轴是meta-predictor，X轴是AF%
    # 参考：Paper 1 (Fig. 2), Paper 3 (Fig. 5)
    
    # 1. 总体AF森林图（按meta-predictor）
    cat("  【1/3】生成Meta-Predictor AF森林图...\n")
    
    # 对每个predictor，跨所有模型计算平均AF
    af_by_pred_summary <- af_all %>%
      group_by(predictor) %>%
      summarise(
        af_mean = mean(af_pct, na.rm = TRUE),
        af_se = sd(af_pct, na.rm = TRUE) / sqrt(n()),
        af_low_pooled = mean(af_low, na.rm = TRUE),
        af_high_pooled = mean(af_high, na.rm = TRUE),
        n_models = n(),
        .groups = "drop"
      ) %>%
      mutate(
        # 清理predictor名称
        predictor_label = case_when(
          grepl("NDVI", predictor) ~ "NDVI (Vegetation Index)",
          grepl("total_20", predictor) ~ "Population (20-55 years)",
          grepl("BD|Building_Density", predictor) ~ "Building Density",
          grepl("FAR", predictor) ~ "Floor Area Ratio",
          grepl("GDP", predictor) ~ "GDP",
          grepl("unemployed", predictor) ~ "Unemployed Population",
          grepl("WS", predictor) ~ "Wind Speed",
          grepl("Walk", predictor) ~ "Walkability Index",
          TRUE ~ predictor
        )
      ) %>%
      mutate(
        p_value = 2 * pnorm(-abs(af_mean / pmax(af_se, 1e-10))),
        direction = forest_effect_direction(af_mean, neutral_zero = TRUE),
        star_label = forest_significance_stars(p_value)
      ) %>%
      arrange(desc(abs(af_mean)))  # 按AF绝对值排序
    
    # 保存汇总统计
    write_csv(af_by_pred_summary, file.path(meta_af_output_dir, paste0("AF_by_predictor_summary_", meta_af_mode_label, ".csv")))
    
    # 绘制森林图
    af_pred_star_nudge <- forest_star_offset(af_by_pred_summary$af_low_pooled, af_by_pred_summary$af_high_pooled, fallback = 0.1)
    p_af_forest <- ggplot(af_by_pred_summary, aes(x = af_mean, y = reorder(predictor_label, af_mean))) +
      geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1.2) +
      geom_errorbarh(aes(xmin = af_low_pooled, xmax = af_high_pooled, color = direction),
                     height = 0.4, linewidth = 2) +
      geom_point(aes(color = direction), size = 8, shape = 19) +
      geom_text(
        data = af_by_pred_summary %>% filter(nzchar(star_label)),
        aes(label = star_label),
        nudge_x = af_pred_star_nudge,
        nudge_y = 0.24,
        size = 5,
        color = "black",
        fontface = "bold",
        show.legend = FALSE
      ) +
      scale_color_manual(values = forest_direction_colors, guide = "none") +
      labs(
        title = "Attributable Fraction by Meta-Predictor",
        subtitle = paste0("AF from Low (-1 SD) to High (+1 SD) | Pooled across ", 
                        length(unique(af_all$indicator)), " indicators × ", 
                        length(unique(af_all$model_type)), " model types\n",
                        "Blue=Positive, Red=Negative | * p<0.05, ** p<0.01, *** p<0.001"),
        x = "Attributable Fraction (%)",
        y = "",
        caption = "AF > 0: Predictor increases heatwave-related PA risk | AF < 0: Predictor decreases risk\nErrorbars: 95% CI across models"
      ) +
      coord_cartesian(clip = "off") +
      theme_minimal(base_size = 18) +
      theme(
        plot.title = element_text(face = "bold", size = 22, hjust = 0),
        plot.subtitle = element_text(size = 15, color = "gray30"),
        axis.text.y = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 15),
        axis.title.x = element_text(size = 18, face = "bold"),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5),
        plot.caption = element_text(size = 11, hjust = 0, color = "gray50", lineheight = 1.2)
      )
    
    # 动态调整高度
    plot_height <- max(10, 5 + nrow(af_by_pred_summary) * 1.2)
    
    ggsave(
      file.path(meta_af_output_dir, paste0("AF_forest_plot_by_predictor_", meta_af_mode_label, ".png")),
      p_af_forest,
      width = 16,
      height = plot_height,
      dpi = 300
    )
    
    cat("  ✓ Meta-Predictor AF森林图已保存（", plot_height, "英寸高）\n\n")
    
    # 【V6.3已废弃】以下代码用于生成按暴露百分位数的AF对比图
    # 由于AF计算方式改变（第一阶段计算），这些图暂时禁用
    
    # 2. 【V6.3已禁用】按模型分组的AF对比图
    if (FALSE) {  # 暂时禁用
      cat("  【2/3】生成按模型分组的AF森林图（可选）...\n")
      
      af_comparison <- af_all %>%
        select(indicator, model_type, n_cities, 
               cehwi_p75, af_at_p75, af_at_p75_low, af_at_p75_high,
               cehwi_p90, af_at_p90, af_at_p90_low, af_at_p90_high,
               cehwi_p95, af_at_p95, af_at_p95_low, af_at_p95_high) %>%
      pivot_longer(
        cols = c(af_at_p75, af_at_p90, af_at_p95),
        names_to = "percentile",
        values_to = "af"
      ) %>%
      mutate(
        percentile = case_when(
          percentile == "af_at_p75" ~ "75th percentile",
          percentile == "af_at_p90" ~ "90th percentile",
          percentile == "af_at_p95" ~ "95th percentile"
        ),
        percentile = factor(percentile, levels = c("75th percentile", "90th percentile", "95th percentile")),
        model_label = paste0(toupper(indicator), " - ", str_to_title(model_type)),
        af_pct = af * 100
      )
    
    # 添加CI（需要手动匹配）
    af_comparison <- af_comparison %>%
      left_join(
        af_all %>%
          select(indicator, model_type, 
                 af_at_p75_low, af_at_p75_high,
                 af_at_p90_low, af_at_p90_high,
                 af_at_p95_low, af_at_p95_high) %>%
          pivot_longer(
            cols = -c(indicator, model_type),
            names_to = "var",
            values_to = "val"
          ) %>%
          mutate(
            percentile = case_when(
              str_detect(var, "p75") ~ "75th percentile",
              str_detect(var, "p90") ~ "90th percentile",
              str_detect(var, "p95") ~ "95th percentile"
            ),
            bound = ifelse(str_detect(var, "_low"), "low", "high")
          ) %>%
          select(-var) %>%
          pivot_wider(names_from = bound, values_from = val) %>%
          mutate(
            low_pct = low * 100,
            high_pct = high * 100
          )
        ,
        by = c("indicator", "model_type", "percentile")
      )
    
    p_af_comparison <- ggplot(af_comparison, aes(x = percentile, y = af_pct, color = model_label, group = model_label)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      geom_line(linewidth = 1.5) +
      geom_pointrange(aes(ymin = low_pct, ymax = high_pct), size = 0.8, linewidth = 1.2) +
      facet_wrap(~ indicator, scales = "free_y", ncol = 2) +
      labs(
        title = "Attributable Fraction by Exposure Intensity",
        subtitle = "Comparison across 75th, 90th, and 95th percentiles | 95% CI shown",
        x = "",
        y = "Attributable Fraction (%)",
        color = "Model Type"
      ) +
      theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(face = "bold", size = 16),
        strip.text = element_text(face = "bold", size = 14),
        axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "bottom",
        panel.grid.minor = element_blank()
      ) +
      scale_color_brewer(palette = "Set1")
    
    ggsave(
      file.path(national_af_output_dir(OUTPUT_DIR), paste0("AF_comparison_by_percentile_", meta_predictor_mode_label(), ".png")),
      p_af_comparison,
      width = 14,
      height = 10,
      dpi = 300
    )
    
    cat("  ✓ AF对比图已保存\n\n")
    
    # 3. AF vs RR 散点图（显示两者关系）
    cat("  【3/3】生成AF vs RR关系图...\n")
    
    af_rr_data <- af_all %>%
      mutate(
        model_label = paste0(toupper(indicator), "\n", str_to_title(model_type)),
        af_pct_p90 = af_at_p90 * 100
      )
    
    p_af_vs_rr <- ggplot(af_rr_data, aes(x = rr_at_p90, y = af_pct_p90)) +
      geom_point(aes(color = indicator, shape = model_type), size = 5, alpha = 0.8) +
      geom_text(aes(label = model_label), size = 3, nudge_y = 0.3, check_overlap = TRUE) +
      labs(
        title = "Relationship between RR and AF at 90th Percentile",
        subtitle = "AF = (RR - 1) / RR",
        x = "Relative Risk (RR)",
        y = "Attributable Fraction (%)",
        color = "Indicator",
        shape = "Model Type"
      ) +
      theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(face = "bold", size = 16),
        legend.position = "right",
        panel.grid.minor = element_blank()
      ) +
      scale_color_brewer(palette = "Dark2")
    
    ggsave(
      file.path(national_af_output_dir(OUTPUT_DIR), paste0("AF_vs_RR_scatter_", meta_predictor_mode_label(), ".png")),
      p_af_vs_rr,
      width = 12,
      height = 8,
      dpi = 300
    )
    
    cat("  ✓ AF vs RR关系图已保存\n\n")
    
      cat("  【汇总】AF森林图生成完成！\n")
      cat("    - 总体AF森林图: AF_forest_plot_all_models_p90.png\n")
      cat("    - AF对比图: AF_comparison_by_percentile.png\n")
      cat("    - AF vs RR关系图: AF_vs_RR_scatter.png\n")
      cat("    - AF数据: AF_all_models_summary.csv\n\n")
    }  # 结束 if (FALSE)
    
    # 【V6.3】暂时跳过旧的AF森林图，等待实现第一阶段城市级AF计算
    cat("  【V6.3】注意：AF计算已改为第一阶段方法，旧的AF森林图暂时禁用\n")
    cat("  → 等待实现城市级AF计算和meta-analysis\n\n")
    
  } else {
    cat("  ⚠ 没有AF数据，跳过森林图生成\n\n")
  }
  
  # ========== 【V7新增】城市级AF Meta-Analysis ==========
  
  cat(rep("=", 100), "\n", sep = "")
  cat("【V7】归因分数（AF）Meta-Analysis - 城市级\n")
  cat(rep("=", 100), "\n\n")
  
  all_city_af <- bind_rows(
    extract_city_af_records(successful_cities_cehwi, "cehwi"),
    extract_city_af_records(successful_cities_exceeded, "exceeded_quantity")
  )
  national_af_dir <- national_af_output_dir(OUTPUT_DIR)
  national_af_mode_label <- meta_predictor_mode_label()
  national_af_mode_suffix <- analysis_output_suffix(national_af_mode_label)
  
  if (nrow(all_city_af) > 0) {
    safe_write_csv(all_city_af, file.path(national_af_dir, paste0("ALL_CITIES_AF", national_af_mode_suffix, ".csv")), label = "national all city AF")
    cat("  ✓ 所有城市AF已保存: ", nrow(all_city_af), "条记录\n\n")
    
    pooled_af_results <- list()
    city_af_forest_saved <- 0
    
    cat("  【V7】生成城市级AF Meta-Analysis 与森林图...\n")
    
    for (ind in c("cehwi", "exceeded_quantity")) {
      for (mtype in STAGE1_MODEL_TYPES) {
        for (pct in c("overall", "p25", "p50", "p75", "p90", "p95")) {
          af_subset <- all_city_af %>%
            filter(indicator == ind, model_type == mtype, percentile == pct)
          
          if (nrow(af_subset) < 3) next
          
          af_meta <- fit_univariate_meta_af(af_subset)
          if (is.null(af_meta)) next
          af_meta_row <- af_meta$summary[1, , drop = FALSE]
          pooled_af_results[[length(pooled_af_results) + 1]] <- data.frame(
            indicator = ind,
            model_type = mtype,
            percentile = pct,
            n_cities = af_meta_row$n_cities,
            pooled_af = af_meta_row$pooled_af,
            pooled_se = af_meta_row$pooled_se,
            pooled_low = af_meta_row$pooled_low,
            pooled_high = af_meta_row$pooled_high,
            method = af_meta_row$method,
            q = af_meta_row$q_statistic,
            q_df = af_meta_row$q_df,
            q_p = af_meta_row$q_pvalue,
            stringsAsFactors = FALSE
          )
          
          if (pct == "overall") {
            af_subset <- af_subset %>%
              arrange(desc(af_pct)) %>%
              mutate(
                city = factor(city, levels = unique(city)),
                p_value = 2 * pnorm(-abs(af_pct / pmax(af_se, 1e-10))),
                direction = forest_effect_direction(af_pct, neutral_zero = TRUE),
                star_label = forest_significance_stars(p_value)
              )
            af_city_star_nudge <- forest_star_offset(af_subset$af_low, af_subset$af_high, fallback = 0.1)
            
            p_af_forest <- ggplot(af_subset, aes(x = af_pct, y = city)) +
              geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
              geom_errorbarh(
                aes(xmin = af_low, xmax = af_high, color = direction),
                height = 0.22,
                linewidth = 1
              ) +
              geom_point(aes(color = direction), size = 3.5, shape = 19) +
              geom_text(
                data = af_subset %>% filter(nzchar(star_label)),
                aes(label = star_label),
                nudge_x = af_city_star_nudge,
                nudge_y = 0.22,
                size = 3.2,
                color = "black",
                fontface = "bold",
                show.legend = FALSE
              ) +
              scale_color_manual(values = forest_direction_colors, guide = "none") +
              labs(
                title = paste0("Attributable Fraction (AF) - ", toupper(ind), " - ", toupper(mtype)),
                subtitle = paste0(
                  nrow(af_subset), " cities | Pooled AF = ",
                  round(af_meta_row$pooled_af, 2), "% (95% CI ",
                  round(af_meta_row$pooled_low, 2), ", ",
                  round(af_meta_row$pooled_high, 2), ")\n",
                  "Blue=Positive, Red=Negative | * p<0.05, ** p<0.01, *** p<0.001"
                ),
                x = "Attributable Fraction (%)",
                y = ""
              ) +
              coord_cartesian(clip = "off") +
              theme_minimal(base_size = 14) +
              theme(
                plot.title = element_text(face = "bold", size = 16),
                axis.text.y = element_text(size = 10),
                panel.grid.major.y = element_blank(),
                panel.grid.minor = element_blank()
              )
            
            af_file <- file.path(national_af_dir, paste0(toupper(ind), "_", toupper(mtype), "_AF_forest", national_af_mode_suffix, ".png"))
            ggsave(af_file, p_af_forest, width = 10, height = max(6, nrow(af_subset) * 0.3), dpi = 300)
            city_af_forest_saved <- city_af_forest_saved + 1
            cat("    ✓ ", toupper(ind), "-", toupper(mtype), "AF森林图已保存\n", sep = "")
          }
        }
      }
    }
    
    pooled_af_summary <- bind_rows(pooled_af_results) %>%
      add_pooled_af_significance(mode = national_af_mode_label)
    safe_write_csv(pooled_af_summary, file.path(national_af_dir, paste0("POOLED_AF_SUMMARY", national_af_mode_suffix, ".csv")), label = "national pooled AF summary")
    cat("  ✓ Pooled AF汇总已保存\n")
    cat("  ✓ 城市级AF森林图数量:", city_af_forest_saved, "\n\n")
  } else {
    cat("  ⚠ 未找到城市级AF数据（可能需要重新运行第一阶段）\n\n")
  }
  
  if (exists("pooled_af_summary") && exists("all_city_af") &&
      nrow(pooled_af_summary) > 0 && nrow(all_city_af) > 0) {
    save_pooled_af_comparison_plot(pooled_af_summary, national_af_dir, mode = national_af_mode_label)
    save_af_vs_rr_scatter_plot(all_city_af, national_af_dir, mode = national_af_mode_label)
    cat("  ✓ National AF summary plots saved from canonical AF CSV outputs\n")
  }
  if (FALSE) {
  
  # 收集所有城市的AF数据
  all_city_af <- data.frame()
  
  # 【V7.1修复】分别从cehwi和exceeded_quantity收集AF，避免重复
  for (ind in c("cehwi", "exceeded_quantity")) {
    # 选择对应指标的successful_cities
    cities_to_process <- if (ind == "cehwi") {
      successful_cities_cehwi
    } else {
      successful_cities_exceeded
    }
    
    for (city_key in names(cities_to_process)) {
      city_result <- cities_to_process[[city_key]]
      
      for (mtype in STAGE1_MODEL_TYPES) {
        if (mtype %in% names(city_result)) {
          model_result <- city_result[[mtype]]
          
          if (!is.null(model_result$af)) {
            # 【V7.1修复】从model_result$city中移除后缀，只保留纯城市名
            city_name_clean <- sub("_(Composite|Day|Night|DayOnly)$", "", model_result$city, ignore.case = TRUE)
            
            all_city_af <- bind_rows(all_city_af, data.frame(
              city = city_name_clean,
              indicator = ind,
              model_type = mtype,
              af_overall = model_result$af$af_overall,
              af_p75 = model_result$af$af_p75,
              af_p90 = model_result$af$af_p90,
              af_p95 = model_result$af$af_p95,
              n_exposure = model_result$af$n_exposure,
              stringsAsFactors = FALSE
            ))
          }
        }
      }
    }
  }
  
  if (nrow(all_city_af) > 0) {
    # 保存所有城市AF
    safe_write_csv(all_city_af, file.path(national_af_output_dir(OUTPUT_DIR), paste0("ALL_CITIES_AF_", meta_predictor_mode_label(), ".csv")), label = "legacy national all city AF")
    cat("  ✓ 所有城市AF已保存: ", nrow(all_city_af), "条记录\n\n")
    
    # 计算Pooled AF
    af_pooled <- all_city_af %>%
      group_by(indicator, model_type) %>%
      summarise(
        n_cities = n(),
        af_mean = mean(af_overall, na.rm = TRUE),
        af_sd = sd(af_overall, na.rm = TRUE),
        af_se = sd(af_overall, na.rm = TRUE) / sqrt(n()),
        af_low = af_mean - 1.96 * af_se,
        af_high = af_mean + 1.96 * af_se,
        af_p75_mean = mean(af_p75, na.rm = TRUE),
        af_p90_mean = mean(af_p90, na.rm = TRUE),
        af_p95_mean = mean(af_p95, na.rm = TRUE),
        .groups = "drop"
      )
    
    safe_write_csv(af_pooled, file.path(national_af_output_dir(OUTPUT_DIR), paste0("POOLED_AF_SUMMARY_", meta_predictor_mode_label(), ".csv")), label = "legacy national pooled AF summary")
    cat("  ✓ Pooled AF汇总已保存\n\n")
    
    # 生成AF森林图（按indicator和model_type）
    cat("  【V7】生成城市级AF森林图...\n")
    
    for (ind in c("cehwi", "exceeded_quantity")) {
      for (mtype in STAGE1_MODEL_TYPES) {
        
        af_subset <- all_city_af %>%
          filter(indicator == ind, model_type == mtype)
        
        if (nrow(af_subset) >= 3) {
          
          # 排序（按AF大小）
          af_subset <- af_subset %>%
            arrange(desc(af_overall)) %>%
            mutate(
              city = factor(city, levels = unique(city)),  # 修复重复factor level
              af_se = calc_se_from_ci(af_overall_low, af_overall_high),
              af_low = af_overall - 1.96 * af_se,
              af_high = af_overall + 1.96 * af_se
            )
          
          # 绘制森林图
          p_af_forest <- ggplot(af_subset, aes(x = af_overall, y = city)) +
            geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
            geom_pointrange(aes(xmin = af_low, xmax = af_high, 
                               color = ifelse(af_overall > 0, "Positive", "Negative")),
                           size = 0.6, linewidth = 1) +
            scale_color_manual(values = c("Positive" = "#D62728", "Negative" = "#1F77B4"),
                              name = "AF Direction") +
            labs(
              title = paste0("Attributable Fraction (AF) - ", toupper(ind), " - ", toupper(mtype)),
              subtitle = paste0(nrow(af_subset), " cities | Mean AF = ", 
                               round(mean(af_subset$af_overall, na.rm = TRUE), 2), "%"),
              x = "Attributable Fraction (%)",
              y = ""
            ) +
            theme_minimal(base_size = 14) +
            theme(
              plot.title = element_text(face = "bold", size = 16),
              axis.text.y = element_text(size = 10),
              legend.position = "bottom",
              panel.grid.major.y = element_blank(),
              panel.grid.minor = element_blank()
            )
          
          # 保存
          af_file <- file.path(national_af_output_dir(OUTPUT_DIR), paste0(toupper(ind), "_", toupper(mtype), "_AF_forest_", meta_predictor_mode_label(), ".png"))
          ggsave(af_file, p_af_forest, width = 10, height = max(6, nrow(af_subset) * 0.3), dpi = 300)
          
          cat("    ✓ ", toupper(ind), "-", toupper(mtype), "AF森林图已保存\n")
        }
      }
    }
    
    cat("\n  【V7】所有AF森林图生成完成！\n\n")
    
  } else {
    cat("  ⚠ 未找到城市级AF数据（可能需要重新运行第一阶段）\n\n")
  }
  }
  
  # ========== 【V5新增】按分区的Meta-Regression ==========
  
  cat(rep("=", 100), "\n", sep = "")
  cat("【V6】按分区进行Meta-Regression（Climate Zone & City Cluster & Geographic Region）\n")
  cat(rep("=", 100), "\n\n")
  
  cat("  ✓ 【V6完整实现】为每个分区运行完整的meta-regression分析\n")
  cat("  包括: Pooled RR曲线、Meta-predictor森林图、AF计算、直方图\n\n")
  
  city_zone_cluster <- tibble(city = CITY_LIST) %>%
    mutate(
      climate_zone = case_when(
        city %in% CLIMATE_ZONE_MAPPING$Temperate ~ "Temperate",
        city %in% CLIMATE_ZONE_MAPPING$Cold ~ "Cold",
        city %in% CLIMATE_ZONE_MAPPING$Arid ~ "Arid",
        city %in% CLIMATE_ZONE_MAPPING$Tropical ~ "Tropical",
        TRUE ~ NA_character_
      ),
      city_cluster = case_when(
        city %in% CITY_CLUSTER_MAPPING$Texas_Southern_Hot ~ "Texas_Southern_Hot",
        city %in% CITY_CLUSTER_MAPPING$East_Coast_Moderate ~ "East_Coast_Moderate",
        city %in% CITY_CLUSTER_MAPPING$Northern_Cold ~ "Northern_Cold",
        city %in% CITY_CLUSTER_MAPPING$West_Coast_Mediterranean ~ "West_Coast_Mediterranean",
        city %in% CITY_CLUSTER_MAPPING$Arid_Desert ~ "Arid_Desert",
        city %in% CITY_CLUSTER_MAPPING$Tropical ~ "Tropical",
        city %in% CITY_CLUSTER_MAPPING$Transition_Special ~ "Transition_Special",
        TRUE ~ NA_character_
      ),
      geographic_region = case_when(
        city %in% GEOGRAPHIC_REGION_MAPPING$Northeast ~ "Northeast",
        city %in% GEOGRAPHIC_REGION_MAPPING$Southeast ~ "Southeast",
        city %in% GEOGRAPHIC_REGION_MAPPING$South ~ "South",
        city %in% GEOGRAPHIC_REGION_MAPPING$Midwest ~ "Midwest",
        city %in% GEOGRAPHIC_REGION_MAPPING$Southwest ~ "Southwest",
        city %in% GEOGRAPHIC_REGION_MAPPING$West ~ "West",
        TRUE ~ NA_character_
      )
    )
  
  partition_socioecon_avg <- NULL
  if (!is.null(df_socioecon_global)) {
    partition_socioecon_avg <- build_city_covariates_table(df_socioecon_global, SOCIOECON_VARS_USED)
    if (!is.null(partition_socioecon_avg) && nrow(partition_socioecon_avg) > 0) {
      cat("  ✓ 分区用城市社会经济均值已准备（", nrow(partition_socioecon_avg), " 城市）\n\n")
    }
  } else {
    cat("  ⚠ 社会经济meta-predictors未加载，分区分析将只输出 pooled RR / AF，不做 city-level covariates\n\n")
  }
  
  partition_meta_run_summary <- list()
    
    # 按Climate Zone分区
    cat("  [1/5] 按Climate Zone分区分析...\n\n")
    
    zone_summary_list <- list()
    
    for (zone in unique(city_zone_cluster$climate_zone)) {
      if (is.na(zone)) next
      
      zone_cities <- city_zone_cluster %>%
        filter(climate_zone == zone) %>%
        pull(city)
      
      cat("    ══ Climate Zone:", zone, "（", length(zone_cities), "个城市）══\n")
      
      # 对该Zone的城市进行meta-regression（简化版：只输出汇总）
      # 为节省时间，只针对CEHWI指标的composite模型
      zone_output_dir <- file.path(OUTPUT_DIR, paste0("ZONE_", zone))
      dir.create(zone_output_dir, showWarnings = FALSE, recursive = TRUE)
      zone_af_model_results <- list()
      
      # 提取该Zone城市的第一阶段结果
      zone_results <- list()
      for (city_name in zone_cities) {
        city_key <- paste0(city_name, "_cehwi")
        if (city_key %in% names(successful_cities)) {
          city_result <- successful_cities[[city_key]]
          base_model_for_partition <- stage1_model_name("composite", "all", STAGE1_ACTIVITY_MODE)
          if (base_model_for_partition %in% names(city_result)) {
            zone_results[[city_key]] <- city_result[[base_model_for_partition]]
          }
        }
      }
      
      if (length(zone_results) >= 3) {
        cat("      - 成功纳入", length(zone_results), "个城市\n")
        cat("      ✓ 目录已创建:", zone_output_dir, "\n")
        
        # 保存分区信息
        zone_info <- data.frame(
          zone = zone,
          n_cities_total = length(zone_cities),
          n_cities_included = length(zone_results),
          cities = paste(names(zone_results), collapse = ", ")
        )
        write_csv(zone_info, file.path(zone_output_dir, "zone_info.csv"))
        
        zone_summary_list[[zone]] <- zone_info
        zone_af_model_results <- list()
        
        # 【V6完整实现】运行完整的meta-regression（复制主循环逻辑）
        cat("      → 【V6】开始完整meta-regression分析...\n")
        
        # 遍历所有indicator和model_type组合
        for (ind in c("cehwi", "exceeded_quantity")) {
          for (mtype in STAGE1_MODEL_TYPES) {
            # 提取该zone该模型的结果
            partition_results <- list()
            for (city_name in zone_cities) {
              city_key <- paste0(city_name, "_", ind)
              if (city_key %in% names(successful_cities)) {
                city_result <- successful_cities[[city_key]]
                if (mtype %in% names(city_result)) {
                  partition_results[[city_name]] <- city_result[[mtype]]
                }
              }
            }
            partition_results <- filter_current_stage1_results(
              partition_results,
              paste0("Zone ", zone, " ", toupper(ind), " ", toupper(mtype))
            )
            
            if (length(partition_results) < 3) {
              cat("        [", toupper(ind), "-", toupper(mtype), "] 城市数不足，跳过\n")
              partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
                partition_family = "Zone",
                partition_name = zone,
                indicator = ind,
                model_type = mtype,
                n_cities = length(partition_results),
                n_cities_total = length(zone_cities),
                model_status = "SKIPPED_LT3",
                output_dir = file.path(zone_output_dir, paste0(ind, "_", mtype)),
                status_note = "Skipped before fitting because fewer than 3 cities had usable first-stage results."
              )
              next
            }
            
            cat("        [", toupper(ind), "-", toupper(mtype), "] 纳入", length(partition_results), "个城市\n")
            
            # 【V6.3.2】创建输出目录（将在拟合后根据稳定性重命名）
            if (is.null(zone_af_model_results[[ind]])) zone_af_model_results[[ind]] <- list()
            zone_af_model_results[[ind]][[mtype]] <- partition_results
            
            partition_meta_dir_base <- file.path(zone_output_dir, paste0(ind, "_", mtype))
            partition_meta_dir <- partition_meta_dir_base
            dir.create(partition_meta_dir, showWarnings = FALSE, recursive = TRUE)
            
            # 【V6.3修复】运行meta-regression（添加详细调试）
            summary_result <- list(
              model_status = "FAILED",
              stability_flag = NA_character_,
              meta_predictors_output = FALSE,
              af_output = FALSE,
              output_dir = partition_meta_dir,
              status_note = NA_character_
            )
            tryCatch({
              cat("          【调试】提取系数和协方差矩阵...\n")
              
              # 1. 提取系数和协方差
              coef_list <- lapply(partition_results, function(x) x$coef)
              vcov_list <- lapply(partition_results, function(x) x$vcov)
              
              # 检查是否有空值
              if (any(sapply(coef_list, is.null))) {
                stop("部分城市的coef为空")
              }
              if (any(sapply(vcov_list, is.null))) {
                stop("部分城市的vcov为空")
              }
              
              coef_matrix <- do.call(rbind, coef_list)
              cat("          【调试】coef_matrix: ", nrow(coef_matrix), "行×", ncol(coef_matrix), "列\n")
              
              # 2. 【V6.3.2】拟合简单mvmeta（多重策略）
              cat("          【调试】拟合mvmeta（多重策略）...\n")
              
              mv_model_simple <- NULL
              model_reliability <- "STABLE"
              
              # 策略1: 标准REML
              mv_model_simple <- tryCatch({
                mvmeta(coef_matrix, S = vcov_list, method = "reml")
              }, error = function(e) {
                cat("          ⚠ REML失败: ", substr(conditionMessage(e), 1, 40), "\n")
                NULL
              })
              
              # 策略2: 强力regularization
              if (is.null(mv_model_simple)) {
                cat("          → 尝试regularization...\n")
                max_eigs <- sapply(vcov_list, function(V) {
                  tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
                })
                reg_str <- max(0.01, max(max_eigs, na.rm = TRUE) * 0.1)
                vcov_reg <- lapply(vcov_list, function(V) V + diag(reg_str, nrow(V)))
                
                mv_model_simple <- tryCatch({
                  mvmeta(coef_matrix, S = vcov_reg, method = "reml")
                }, error = function(e) NULL)
                
                if (!is.null(mv_model_simple)) {
                  model_reliability <- "UNSTABLE_REG"
                  cat("          ⚠ Regularization成功（⚠不稳定）\n")
                }
              }
              
              # 策略3: Fixed effects
              if (is.null(mv_model_simple)) {
                cat("          → 尝试fixed effects...\n")
                mv_model_simple <- tryCatch({
                  mvmeta(coef_matrix, method = "fixed")
                }, error = function(e) NULL)
                
                if (!is.null(mv_model_simple)) {
                  model_reliability <- "HIGHLY_UNSTABLE_FIXED"
                  cat("          ⚠⚠ Fixed effects成功（⚠⚠极不稳定）\n")
                }
              }
              
              if (is.null(mv_model_simple)) {
                stop("所有策略失败")
              }
              
              cat("          【调试】mvmeta拟合成功 [", model_reliability, "]\n")
              
              # 3. 提取pooled系数
              pooled_coef <- coef(mv_model_simple)
              pooled_vcov <- vcov(mv_model_simple)
              cat("          【调试】pooled_coef长度: ", length(pooled_coef), "\n")
              
              # 4. 使用第一个城市的cb作为模板
              cb_template <- partition_results[[1]]$cb
              if (is.null(cb_template)) {
                stop("cb_template为空")
              }
              cat("          【调试】cb_template准备完成\n")
              
              # 5. 生成crosspred（暴露范围用城市实际 cehwi_range，避免 coef_matrix[,1] 为负导致异常）
              cat("          【调试】生成crosspred...\n")
              part_ranges <- lapply(partition_results, function(x) x$cehwi_range)
              part_ranges <- part_ranges[!sapply(part_ranges, is.null)]
              if (length(part_ranges) > 0) {
                cehwi_range_p <- range(unlist(part_ranges), na.rm = TRUE)
                cehwi_max <- max(cehwi_range_p[2], 1)
              } else {
                all_exp <- unlist(lapply(partition_results, function(x) { d <- x$cehwi_data; if (!is.null(d)) d[d > 0] else NULL }))
                cehwi_max <- if (length(all_exp) > 0) max(max(all_exp, na.rm = TRUE), 1) else 10
              }
              if (is.na(cehwi_max) || cehwi_max <= 0) cehwi_max <- 10
              
              pooled_cp <- crosspred(
                cb_template,
                coef = pooled_coef,
                vcov = pooled_vcov,
                model.link = "log",
                at = seq(0, cehwi_max, length.out = 500),
                cen = 0,
                cumul = TRUE
              )
              cat("          【调试】crosspred成功\n")
              
              # 6. 创建DataFrame
              pooled_df <- data.frame(
                cehwi = pooled_cp$predvar,
                rr = pooled_cp$allRRfit,
                rr_low = pooled_cp$allRRlow,
                rr_high = pooled_cp$allRRhigh
              )
              cat("          【调试】pooled_df: ", nrow(pooled_df), "行\n")
              
              # 7. 【V6.3.2】绘制图形（添加稳定性警告）
              cat("          【调试】绘制RR曲线...\n")
              
              # 根据稳定性生成标题
              title_suffix <- ifelse(model_reliability == "STABLE", "",
                              ifelse(model_reliability == "UNSTABLE_REG", " ⚠ UNSTABLE",
                              " ⚠⚠ HIGHLY UNSTABLE"))
              
              subtitle_text <- build_partition_rr_subtitle(
                partition_family = "Zone",
                partition_name = zone,
                indicator = ind,
                model_type = mtype,
                n_cities = length(partition_results),
                pooled_df = pooled_df,
                meta_model = mv_model_simple,
                reliability = model_reliability
              )
              
              # 【颜色修复】根据 model_type 选择颜色（与主分析保持一致）
              line_color <- stage1_model_color(mtype)
              if (is.null(line_color) || is.na(line_color)) line_color <- "#D53E4F"
              
              # 【贴地修复】智能Y轴：CI 爆炸时改用 RR 定上限
              part_rr_min <- min(pooled_df$rr, na.rm = TRUE)
              part_rr_max <- max(pooled_df$rr, na.rm = TRUE)
              part_ci_min <- min(pooled_df$rr_low, na.rm = TRUE)
              part_ci_max <- max(pooled_df$rr_high, na.rm = TRUE)
              part_rr_range <- part_rr_max - part_rr_min
              part_ci_range <- part_ci_max - part_ci_min
              
              # 判断 CI 是否爆炸：CI 上限 > RR 最大值 * 10 或 CI 范围 > RR 范围 * 15
              part_ci_exploded <- (part_ci_max > part_rr_max * 10 || part_ci_range > part_rr_range * 15)
              
              # 【组合图修复】创建裁剪版本的 pooled_df_plot（与主分析一致）
              if (part_ci_exploded) {
                cat("          ⚠ CI 爆炸（max=", round(part_ci_max, 1), "），裁剪 CI 用于绘图\n")
                part_y_max <- part_rr_max * 2.5
                part_y_min <- max(0, part_rr_min * 0.5)
                pooled_df_plot <- pooled_df %>%
                  mutate(
                    rr_low_clipped = pmax(rr_low, part_y_min),
                    rr_high_clipped = pmin(rr_high, part_y_max)
                  )
              } else {
                pooled_df_plot <- pooled_df %>%
                  mutate(
                    rr_low_clipped = rr_low,
                    rr_high_clipped = rr_high
                  )
              }
              
              part_caption <- build_rr_caption(
                indicator = ind,
                has_pi = FALSE,
                ci_clipped = part_ci_exploded,
                reliability = model_reliability,
                include_percentiles = FALSE,
                x_truncated = TRUE
              )
              p_pooled_rr <- ggplot(pooled_df_plot, aes(x = cehwi, y = rr)) +
                geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8) +
                geom_ribbon(aes(ymin = rr_low_clipped, ymax = rr_high_clipped), fill = "gray80", alpha = 0.5) +
                geom_line(color = line_color, linewidth = 2) +
                labs(
                  title = paste0("Zone: ", zone, " - ", toupper(ind), " - ", toupper(mtype), title_suffix,
                                 if(part_ci_exploded) " [Y-axis: RR-based]" else ""),
                  subtitle = subtitle_text,
                  x = toupper(ind),
                  y = "Relative Risk (RR)",
                  caption = part_caption
                ) +
                rr_plot_theme(14)
              
              # 【注释】Y轴不需要额外限制，数据已裁剪
              
              # 8. 保存图形
              rr_file <- file.path(partition_meta_dir, "pooled_RR_curve.png")
              ggsave(rr_file, p_pooled_rr, width = 12, height = 8, dpi = 300)
              cat("          【调试】RR曲线已保存: ", basename(rr_file), "\n")
              
              # 9. 【V6.3.2】保存CSV（添加reliability标记）
              coef_df <- as.data.frame(coef_matrix)
              coef_df$reliability <- model_reliability
              coef_file <- file.path(partition_meta_dir, "pooled_coefs.csv")
              safe_write_csv(coef_df, coef_file, label = "partition pooled coefficients")
              cat("          【调试】系数矩阵已保存: ", basename(coef_file), "\n")
              
              pooled_df$reliability <- model_reliability
              pooled_df_file <- file.path(partition_meta_dir, "pooled_RR_data.csv")
              safe_write_csv(pooled_df, pooled_df_file, label = "partition pooled RR data")
              save_pooled_lag_response(
                partition_results,
                partition_meta_dir,
                indicator = ind,
                model_type = mtype,
                group_label = paste0("Partition: ", basename(dirname(partition_meta_dir))),
                reliability = model_reliability
              )
              cat("          【调试】RR数据已保存: ", basename(pooled_df_file), "\n")
              
              # 【V6.3.2】如果不稳定，重命名文件夹添加标记
              if (model_reliability != "STABLE") {
                new_dir_name <- paste0(partition_meta_dir_base, "_", model_reliability)
                if (dir.exists(partition_meta_dir) && partition_meta_dir != new_dir_name) {
                  file.rename(partition_meta_dir, new_dir_name)
                  partition_meta_dir <- new_dir_name
                  cat("          【V6.3.2】文件夹已重命名添加⚠标记\n")
                }
              }
              
              cat("          ✓ Pooled RR曲线已保存 (", nrow(pooled_df), " points) [", model_reliability, "]\n")
              cat("          ✓ 系数矩阵已保存 (", nrow(coef_matrix), " cities)\n")
              
              # 【V6.3新增】生成暴露分布直方图和组合图
              cat("          【V6.3】生成暴露分布直方图...\n")
              
              # 提取所有城市的暴露数据
              all_cehwi_data <- unlist(lapply(partition_results, function(x) {
                if (!is.null(x$cehwi_data)) {
                  return(x$cehwi_data[x$cehwi_data > 0])
                } else {
                  return(NULL)
                }
              }))
              
              if (!is.null(all_cehwi_data) && length(all_cehwi_data) > 10) {
                # 计算分位数
                q25 <- quantile(all_cehwi_data, 0.25, na.rm = TRUE)
                q75 <- quantile(all_cehwi_data, 0.75, na.rm = TRUE)
                q90 <- quantile(all_cehwi_data, 0.90, na.rm = TRUE)
                # 【V5.1】计算98th分位数用于x轴截断
                q95 <- quantile(all_cehwi_data, 0.95, na.rm = TRUE)
                q98 <- quantile(all_cehwi_data, 0.98, na.rm = TRUE)
                cat("          【V5.1】RR_with_distribution组合图X轴将截断至98th分位数:", round(q98, 2), "\n")
                
                # 1. 单独的直方图
                p_hist <- ggplot(data.frame(cehwi = all_cehwi_data), aes(x = cehwi)) +
                  geom_histogram(bins = 30, fill = "#3B9AB2", alpha = 0.7, color = "white") +
                  geom_vline(xintercept = q25, linetype = "dotted", color = "#E69F00", linewidth = 1) +
                  geom_vline(xintercept = q75, linetype = "dotted", color = "#D55E00", linewidth = 1) +
                  geom_vline(xintercept = q90, linetype = "dashed", color = "#CC0000", linewidth = 1.2) +
                  labs(
                    title = paste0("Exposure Distribution: ", zone),
                    subtitle = paste0(toupper(ind), " - ", toupper(mtype), " (", length(partition_results), " cities)"),
                    x = toupper(ind),
                    y = "Frequency",
                    caption = paste0("25th: ", round(q25, 2), " | 75th: ", round(q75, 2), " | 90th: ", round(q90, 2))
                  ) +
                  rr_hist_theme(12) +
                  # 【V5.1】与RR曲线X轴一致，截断至98th
                  coord_cartesian(xlim = c(0, q98))
                
                hist_file <- file.path(partition_meta_dir, "exposure_distribution.png")
                ggsave(hist_file, p_hist, width = 12, height = 4, dpi = 300)
                
                # 2. RR曲线添加分位线（并添加98th截断）
                # 【组合图修复】只设置 xlim，ylim 由裁剪后的数据决定
                part_rr_caption <- build_rr_caption(
                  indicator = ind,
                  has_pi = FALSE,
                  ci_clipped = part_ci_exploded,
                  reliability = model_reliability,
                  include_percentiles = TRUE,
                  x_truncated = TRUE
                )
                p_pooled_rr_with_lines <- p_pooled_rr +
                  coord_cartesian(xlim = c(0, q98)) +
                  geom_vline(xintercept = q25, linetype = "dotted", color = "#E69F00", linewidth = 0.8) +
                  geom_vline(xintercept = q75, linetype = "dotted", color = "#D55E00", linewidth = 0.8) +
                  geom_vline(xintercept = q90, linetype = "dashed", color = "#CC0000", linewidth = 1) +
                  labs(caption = part_rr_caption)
                
                rr_with_lines_file <- file.path(partition_meta_dir, "pooled_RR_curve_with_percentiles.png")
                ggsave(rr_with_lines_file, p_pooled_rr_with_lines, width = 12, height = 8, dpi = 300)
                
                # 3. 组合图（RR曲线 + 直方图对齐）
                p_combined <- build_rr_distribution_combined_plot(
                  rr_plot = p_pooled_rr_with_lines,
                  hist_plot = p_hist,
                  caption_text = part_rr_caption,
                  heights = c(2, 1),
                  caption_size = 10
                )
                
                combined_file <- file.path(partition_meta_dir, "pooled_RR_with_distribution.png")
                ggsave(combined_file, p_combined, width = 12, height = 10, dpi = 300)
                
                cat("          ✓ 暴露分布直方图已生成（", length(all_cehwi_data), "个观测）\n")
                cat("          ✓ 带分位线RR曲线已生成\n")
                cat("          ✓ 组合图已生成\n")
              } else {
                cat("          ⚠ 暴露数据不足，跳过直方图生成\n")
              }
              
              # 【V7新增】生成AF森林图
              cat("          【V7】生成AF森林图...\n")
              
              plot_partition_af_results(
                partition_results = partition_results,
                partition_meta_dir = partition_meta_dir,
                partition_label = paste0("Zone: ", zone),
                indicator = ind,
                model_type = mtype,
                percentile = "overall"
              )
              summary_result$af_output <- partition_has_af_outputs(partition_meta_dir)
              
              meta_predictor_result <- run_partition_meta_predictors(
                partition_socioecon_avg = partition_socioecon_avg,
                coef_matrix = coef_matrix,
                vcov_list = vcov_list,
                partition_meta_dir = partition_meta_dir,
                partition_label = paste0("Zone: ", zone),
                indicator = ind,
                model_type = mtype,
                partition_results = partition_results
              )
              
              summary_result$model_status <- "SUCCESS"
              summary_result$stability_flag <- model_reliability
              summary_result$meta_predictors_output <- partition_has_meta_predictors(partition_meta_dir)
              summary_result$output_dir <- partition_meta_dir
              note_parts <- c(
                if (!summary_result$meta_predictors_output) meta_predictor_result$note else NULL,
                if (!summary_result$af_output) "AF outputs missing" else NULL
              )
              summary_result$status_note <- if (length(note_parts) > 0) paste(note_parts, collapse = " | ") else NA_character_
              
            }, error = function(e) {
              cat("          ✗ Meta-regression完全失败!\n")
              cat("          错误类型: ", class(e)[1], "\n")
              cat("          错误信息: ", conditionMessage(e), "\n")
              cat("          请检查日志以获取详细信息\n")
              summary_result <<- modifyList(summary_result, list(
                model_status = "FAILED",
                output_dir = partition_meta_dir,
                meta_predictors_output = partition_has_meta_predictors(partition_meta_dir),
                af_output = partition_has_af_outputs(partition_meta_dir),
                status_note = conditionMessage(e)
              ))
            })
            
            partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
              partition_family = "Zone",
              partition_name = zone,
              indicator = ind,
              model_type = mtype,
              n_cities = length(partition_results),
              n_cities_total = length(zone_cities),
              model_status = summary_result$model_status,
              stability_flag = summary_result$stability_flag,
              meta_predictors_output = summary_result$meta_predictors_output,
              af_output = summary_result$af_output,
              output_dir = summary_result$output_dir,
              status_note = summary_result$status_note
            )
          }
        }
        
        cat("      ✓ 【V6】Zone meta-regression完成\n")
      } else {
        cat("      ⚠ 城市数不足3个，跳过\n")
        for (ind in c("cehwi", "exceeded_quantity")) {
          for (mtype in STAGE1_MODEL_TYPES) {
            partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
              partition_family = "Zone",
              partition_name = zone,
              indicator = ind,
              model_type = mtype,
              n_cities = count_partition_model_cities(zone_cities, successful_cities, ind, mtype),
              n_cities_total = length(zone_cities),
              model_status = "SKIPPED_PARTITION_LT3",
              output_dir = zone_output_dir,
              status_note = "Entire partition skipped because fewer than 3 cities were available for the base partition result set."
            )
          }
        }
      }
      if (length(zone_af_model_results) > 0) {
        save_partition_af_percentile_outputs(
          partition_model_results = zone_af_model_results,
          partition_output_dir = zone_output_dir,
          partition_label = paste0("Zone: ", zone),
          partition_family = "Zone",
          partition_name = zone,
          mode = meta_predictor_mode_label()
        )
      }
      cat("\n")
    }
    
    # 按City Cluster分区
    cat("  [2/5] 按City Cluster分区分析...\n\n")
    
    cluster_summary_list <- list()
    
    for (cluster in unique(city_zone_cluster$city_cluster)) {
      if (is.na(cluster)) next
      
      cluster_cities <- city_zone_cluster %>%
        filter(city_cluster == cluster) %>%
        pull(city)
      
      cat("    ══ City Cluster:", cluster, "（", length(cluster_cities), "个城市）══\n")
      
      # 对该Cluster的城市进行meta-regression
      cluster_output_dir <- file.path(OUTPUT_DIR, paste0("CLUSTER_", cluster))
      dir.create(cluster_output_dir, showWarnings = FALSE, recursive = TRUE)
      cluster_af_model_results <- list()
      
      # 提取该Cluster城市的第一阶段结果
      cluster_results <- list()
      for (city_name in cluster_cities) {
        city_key <- paste0(city_name, "_cehwi")
        if (city_key %in% names(successful_cities)) {
          city_result <- successful_cities[[city_key]]
          base_model_for_partition <- stage1_model_name("composite", "all", STAGE1_ACTIVITY_MODE)
          if (base_model_for_partition %in% names(city_result)) {
            cluster_results[[city_key]] <- city_result[[base_model_for_partition]]
          }
        }
      }
      
      if (length(cluster_results) >= 3) {
        cat("      - 成功纳入", length(cluster_results), "个城市\n")
        cat("      ✓ 目录已创建:", cluster_output_dir, "\n")
        
        # 保存分区信息
        cluster_info <- data.frame(
          cluster = cluster,
          n_cities_total = length(cluster_cities),
          n_cities_included = length(cluster_results),
          cities = paste(names(cluster_results), collapse = ", ")
        )
        write_csv(cluster_info, file.path(cluster_output_dir, "cluster_info.csv"))
        
        cluster_summary_list[[cluster]] <- cluster_info
        
        # 【V6完整实现】运行完整的meta-regression（复制主循环逻辑）
        cat("      → 【V6】开始完整meta-regression分析...\n")
        
        # 遍历所有indicator和model_type组合
        for (ind in c("cehwi", "exceeded_quantity")) {
          for (mtype in STAGE1_MODEL_TYPES) {
            # 提取该cluster该模型的结果
            partition_results <- list()
            for (city_name in cluster_cities) {
              city_key <- paste0(city_name, "_", ind)
              if (city_key %in% names(successful_cities)) {
                city_result <- successful_cities[[city_key]]
                if (mtype %in% names(city_result)) {
                  partition_results[[city_name]] <- city_result[[mtype]]
                }
              }
            }
            partition_results <- filter_current_stage1_results(
              partition_results,
              paste0("Cluster ", cluster, " ", toupper(ind), " ", toupper(mtype))
            )
            
            if (length(partition_results) < 3) {
              cat("        [", toupper(ind), "-", toupper(mtype), "] 城市数不足，跳过\n")
              partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
                partition_family = "Cluster",
                partition_name = cluster,
                indicator = ind,
                model_type = mtype,
                n_cities = length(partition_results),
                n_cities_total = length(cluster_cities),
                model_status = "SKIPPED_LT3",
                output_dir = file.path(cluster_output_dir, paste0(ind, "_", mtype)),
                status_note = "Skipped before fitting because fewer than 3 cities had usable first-stage results."
              )
              next
            }
            
            cat("        [", toupper(ind), "-", toupper(mtype), "] 纳入", length(partition_results), "个城市\n")
            
            # 【V6.3.2】创建输出目录（将在拟合后根据稳定性重命名）
            if (is.null(cluster_af_model_results[[ind]])) cluster_af_model_results[[ind]] <- list()
            cluster_af_model_results[[ind]][[mtype]] <- partition_results
            
            partition_meta_dir_base <- file.path(cluster_output_dir, paste0(ind, "_", mtype))
            partition_meta_dir <- partition_meta_dir_base
            dir.create(partition_meta_dir, showWarnings = FALSE, recursive = TRUE)
            
            # 【V6.3修复】运行meta-regression（添加详细调试）
            summary_result <- list(
              model_status = "FAILED",
              stability_flag = NA_character_,
              meta_predictors_output = FALSE,
              af_output = FALSE,
              output_dir = partition_meta_dir,
              status_note = NA_character_
            )
            tryCatch({
              cat("          【调试】提取系数和协方差矩阵...\n")
              
              # 1. 提取系数和协方差
              coef_list <- lapply(partition_results, function(x) x$coef)
              vcov_list <- lapply(partition_results, function(x) x$vcov)
              
              # 检查是否有空值
              if (any(sapply(coef_list, is.null))) {
                stop("部分城市的coef为空")
              }
              if (any(sapply(vcov_list, is.null))) {
                stop("部分城市的vcov为空")
              }
              
              coef_matrix <- do.call(rbind, coef_list)
              cat("          【调试】coef_matrix: ", nrow(coef_matrix), "行×", ncol(coef_matrix), "列\n")
              
              # 2. 【V6.3.2】拟合简单mvmeta（多重策略）
              cat("          【调试】拟合mvmeta（多重策略）...\n")
              
              mv_model_simple <- NULL
              model_reliability <- "STABLE"
              
              # 策略1: 标准REML
              mv_model_simple <- tryCatch({
                mvmeta(coef_matrix, S = vcov_list, method = "reml")
              }, error = function(e) {
                cat("          ⚠ REML失败: ", substr(conditionMessage(e), 1, 40), "\n")
                NULL
              })
              
              # 策略2: 强力regularization
              if (is.null(mv_model_simple)) {
                cat("          → 尝试regularization...\n")
                max_eigs <- sapply(vcov_list, function(V) {
                  tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
                })
                reg_str <- max(0.01, max(max_eigs, na.rm = TRUE) * 0.1)
                vcov_reg <- lapply(vcov_list, function(V) V + diag(reg_str, nrow(V)))
                
                mv_model_simple <- tryCatch({
                  mvmeta(coef_matrix, S = vcov_reg, method = "reml")
                }, error = function(e) NULL)
                
                if (!is.null(mv_model_simple)) {
                  model_reliability <- "UNSTABLE_REG"
                  cat("          ⚠ Regularization成功（⚠不稳定）\n")
                }
              }
              
              # 策略3: Fixed effects
              if (is.null(mv_model_simple)) {
                cat("          → 尝试fixed effects...\n")
                mv_model_simple <- tryCatch({
                  mvmeta(coef_matrix, method = "fixed")
                }, error = function(e) NULL)
                
                if (!is.null(mv_model_simple)) {
                  model_reliability <- "HIGHLY_UNSTABLE_FIXED"
                  cat("          ⚠⚠ Fixed effects成功（⚠⚠极不稳定）\n")
                }
              }
              
              if (is.null(mv_model_simple)) {
                stop("所有策略失败")
              }
              
              cat("          【调试】mvmeta拟合成功 [", model_reliability, "]\n")
              
              # 3. 提取pooled系数
              pooled_coef <- coef(mv_model_simple)
              pooled_vcov <- vcov(mv_model_simple)
              cat("          【调试】pooled_coef长度: ", length(pooled_coef), "\n")
              
              # 4. 使用第一个城市的cb作为模板
              cb_template <- partition_results[[1]]$cb
              if (is.null(cb_template)) {
                stop("cb_template为空")
              }
              cat("          【调试】cb_template准备完成\n")
              
              # 5. 生成crosspred（暴露范围用城市实际 cehwi_range）
              cat("          【调试】生成crosspred...\n")
              part_ranges <- lapply(partition_results, function(x) x$cehwi_range)
              part_ranges <- part_ranges[!sapply(part_ranges, is.null)]
              if (length(part_ranges) > 0) {
                cehwi_range_p <- range(unlist(part_ranges), na.rm = TRUE)
                cehwi_max <- max(cehwi_range_p[2], 1)
              } else {
                all_exp <- unlist(lapply(partition_results, function(x) { d <- x$cehwi_data; if (!is.null(d)) d[d > 0] else NULL }))
                cehwi_max <- if (length(all_exp) > 0) max(max(all_exp, na.rm = TRUE), 1) else 10
              }
              if (is.na(cehwi_max) || cehwi_max <= 0) cehwi_max <- 10
              
              pooled_cp <- crosspred(
                cb_template,
                coef = pooled_coef,
                vcov = pooled_vcov,
                model.link = "log",
                at = seq(0, cehwi_max, length.out = 500),
                cen = 0,
                cumul = TRUE
              )
              cat("          【调试】crosspred成功\n")
              
              # 6. 创建DataFrame
              pooled_df <- data.frame(
                cehwi = pooled_cp$predvar,
                rr = pooled_cp$allRRfit,
                rr_low = pooled_cp$allRRlow,
                rr_high = pooled_cp$allRRhigh
              )
              cat("          【调试】pooled_df: ", nrow(pooled_df), "行\n")
              
              # 7. 【V6.3.2】绘制图形（添加稳定性警告）
              cat("          【调试】绘制RR曲线...\n")
              
              # 根据稳定性生成标题
              title_suffix <- ifelse(model_reliability == "STABLE", "",
                              ifelse(model_reliability == "UNSTABLE_REG", " ⚠ UNSTABLE",
                              " ⚠⚠ HIGHLY UNSTABLE"))
              
              subtitle_text <- build_partition_rr_subtitle(
                partition_family = "Cluster",
                partition_name = cluster,
                indicator = ind,
                model_type = mtype,
                n_cities = length(partition_results),
                pooled_df = pooled_df,
                meta_model = mv_model_simple,
                reliability = model_reliability
              )
              
              # 【颜色修复】根据 model_type 选择颜色（与主分析保持一致）
              line_color <- stage1_model_color(mtype)
              if (is.null(line_color) || is.na(line_color)) line_color <- "#D53E4F"
              
              # 【贴地修复】智能Y轴：CI 爆炸时改用 RR 定上限
              part_rr_min <- min(pooled_df$rr, na.rm = TRUE)
              part_rr_max <- max(pooled_df$rr, na.rm = TRUE)
              part_ci_min <- min(pooled_df$rr_low, na.rm = TRUE)
              part_ci_max <- max(pooled_df$rr_high, na.rm = TRUE)
              part_rr_range <- part_rr_max - part_rr_min
              part_ci_range <- part_ci_max - part_ci_min
              
              # 判断 CI 是否爆炸：CI 上限 > RR 最大值 * 10 或 CI 范围 > RR 范围 * 15
              part_ci_exploded <- (part_ci_max > part_rr_max * 10 || part_ci_range > part_rr_range * 15)
              
              # 【组合图修复】创建裁剪版本的 pooled_df_plot（与主分析一致）
              if (part_ci_exploded) {
                cat("          ⚠ CI 爆炸（max=", round(part_ci_max, 1), "），裁剪 CI 用于绘图\n")
                part_y_max <- part_rr_max * 2.5
                part_y_min <- max(0, part_rr_min * 0.5)
                pooled_df_plot <- pooled_df %>%
                  mutate(
                    rr_low_clipped = pmax(rr_low, part_y_min),
                    rr_high_clipped = pmin(rr_high, part_y_max)
                  )
              } else {
                pooled_df_plot <- pooled_df %>%
                  mutate(
                    rr_low_clipped = rr_low,
                    rr_high_clipped = rr_high
                  )
              }
              
              part_caption <- build_rr_caption(
                indicator = ind,
                has_pi = FALSE,
                ci_clipped = part_ci_exploded,
                reliability = model_reliability,
                include_percentiles = FALSE,
                x_truncated = TRUE
              )
              p_pooled_rr <- ggplot(pooled_df_plot, aes(x = cehwi, y = rr)) +
                geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8) +
                geom_ribbon(aes(ymin = rr_low_clipped, ymax = rr_high_clipped), fill = "gray80", alpha = 0.5) +
                geom_line(color = line_color, linewidth = 2) +
                labs(
                  title = paste0("Cluster: ", cluster, " - ", toupper(ind), " - ", toupper(mtype), title_suffix,
                                 if(part_ci_exploded) " [Y-axis: RR-based]" else ""),
                  subtitle = subtitle_text,
                  x = toupper(ind),
                  y = "Relative Risk (RR)",
                  caption = part_caption
                ) +
                rr_plot_theme(14)
              
              # 【注释】Y轴不需要额外限制，数据已裁剪
              
              # 8. 保存图形
              rr_file <- file.path(partition_meta_dir, "pooled_RR_curve.png")
              ggsave(rr_file, p_pooled_rr, width = 12, height = 8, dpi = 300)
              cat("          【调试】RR曲线已保存: ", basename(rr_file), "\n")
              
              # 9. 【V6.3.2】保存CSV（添加reliability标记）
              coef_df <- as.data.frame(coef_matrix)
              coef_df$reliability <- model_reliability
              coef_file <- file.path(partition_meta_dir, "pooled_coefs.csv")
              safe_write_csv(coef_df, coef_file, label = "partition pooled coefficients")
              cat("          【调试】系数矩阵已保存: ", basename(coef_file), "\n")
              
              pooled_df$reliability <- model_reliability
              pooled_df_file <- file.path(partition_meta_dir, "pooled_RR_data.csv")
              safe_write_csv(pooled_df, pooled_df_file, label = "partition pooled RR data")
              save_pooled_lag_response(
                partition_results,
                partition_meta_dir,
                indicator = ind,
                model_type = mtype,
                group_label = paste0("Partition: ", basename(dirname(partition_meta_dir))),
                reliability = model_reliability
              )
              cat("          【调试】RR数据已保存: ", basename(pooled_df_file), "\n")
              
              # 【V6.3.2】如果不稳定，重命名文件夹添加标记
              if (model_reliability != "STABLE") {
                new_dir_name <- paste0(partition_meta_dir_base, "_", model_reliability)
                if (dir.exists(partition_meta_dir) && partition_meta_dir != new_dir_name) {
                  file.rename(partition_meta_dir, new_dir_name)
                  partition_meta_dir <- new_dir_name
                  cat("          【V6.3.2】文件夹已重命名添加⚠标记\n")
                }
              }
              
              cat("          ✓ Pooled RR曲线已保存 (", nrow(pooled_df), " points) [", model_reliability, "]\n")
              cat("          ✓ 系数矩阵已保存 (", nrow(coef_matrix), " cities)\n")
              
              # 【V6.3新增】生成暴露分布直方图和组合图
              cat("          【V6.3】生成暴露分布直方图...\n")
              
              # 提取所有城市的暴露数据
              all_cehwi_data <- unlist(lapply(partition_results, function(x) {
                if (!is.null(x$cehwi_data)) {
                  return(x$cehwi_data[x$cehwi_data > 0])
                } else {
                  return(NULL)
                }
              }))
              
              if (!is.null(all_cehwi_data) && length(all_cehwi_data) > 10) {
                # 计算分位数
                q25 <- quantile(all_cehwi_data, 0.25, na.rm = TRUE)
                q75 <- quantile(all_cehwi_data, 0.75, na.rm = TRUE)
                q90 <- quantile(all_cehwi_data, 0.90, na.rm = TRUE)
                # 【V5.1】计算98th分位数用于x轴截断
                q95 <- quantile(all_cehwi_data, 0.95, na.rm = TRUE)
                q98 <- quantile(all_cehwi_data, 0.98, na.rm = TRUE)
                cat("          【V5.1】RR_with_distribution组合图X轴将截断至98th分位数:", round(q98, 2), "\n")
                
                # 1. 单独的直方图
                p_hist <- ggplot(data.frame(cehwi = all_cehwi_data), aes(x = cehwi)) +
                  geom_histogram(bins = 30, fill = "#3B9AB2", alpha = 0.7, color = "white") +
                  geom_vline(xintercept = q25, linetype = "dotted", color = "#E69F00", linewidth = 1) +
                  geom_vline(xintercept = q75, linetype = "dotted", color = "#D55E00", linewidth = 1) +
                  geom_vline(xintercept = q90, linetype = "dashed", color = "#CC0000", linewidth = 1.2) +
                  labs(
                    title = paste0("Exposure Distribution: ", cluster),
                    subtitle = paste0(toupper(ind), " - ", toupper(mtype), " (", length(partition_results), " cities)"),
                    x = toupper(ind),
                    y = "Frequency",
                    caption = paste0("25th: ", round(q25, 2), " | 75th: ", round(q75, 2), " | 90th: ", round(q90, 2))
                  ) +
                  rr_hist_theme(12) +
                  # 【V5.1】与RR曲线X轴一致，截断至98th
                  coord_cartesian(xlim = c(0, q98))
                
                hist_file <- file.path(partition_meta_dir, "exposure_distribution.png")
                ggsave(hist_file, p_hist, width = 12, height = 4, dpi = 300)
                
                # 2. RR曲线添加分位线（并添加98th截断）
                # 【组合图修复】只设置 xlim，ylim 由裁剪后的数据决定
                part_rr_caption <- build_rr_caption(
                  indicator = ind,
                  has_pi = FALSE,
                  ci_clipped = part_ci_exploded,
                  reliability = model_reliability,
                  include_percentiles = TRUE,
                  x_truncated = TRUE
                )
                p_pooled_rr_with_lines <- p_pooled_rr +
                  coord_cartesian(xlim = c(0, q98)) +
                  geom_vline(xintercept = q25, linetype = "dotted", color = "#E69F00", linewidth = 0.8) +
                  geom_vline(xintercept = q75, linetype = "dotted", color = "#D55E00", linewidth = 0.8) +
                  geom_vline(xintercept = q90, linetype = "dashed", color = "#CC0000", linewidth = 1) +
                  labs(caption = part_rr_caption)
                
                rr_with_lines_file <- file.path(partition_meta_dir, "pooled_RR_curve_with_percentiles.png")
                ggsave(rr_with_lines_file, p_pooled_rr_with_lines, width = 12, height = 8, dpi = 300)
                
                # 3. 组合图（RR曲线 + 直方图对齐）
                p_combined <- build_rr_distribution_combined_plot(
                  rr_plot = p_pooled_rr_with_lines,
                  hist_plot = p_hist,
                  caption_text = part_rr_caption,
                  heights = c(2, 1),
                  caption_size = 10
                )
                
                combined_file <- file.path(partition_meta_dir, "pooled_RR_with_distribution.png")
                ggsave(combined_file, p_combined, width = 12, height = 10, dpi = 300)
                
                cat("          ✓ 暴露分布直方图已生成（", length(all_cehwi_data), "个观测）\n")
                cat("          ✓ 带分位线RR曲线已生成\n")
                cat("          ✓ 组合图已生成\n")
              } else {
                cat("          ⚠ 暴露数据不足，跳过直方图生成\n")
              }
              
              # 【V7新增】生成AF森林图
              cat("          【V7】生成AF森林图...\n")
              
              plot_partition_af_results(
                partition_results = partition_results,
                partition_meta_dir = partition_meta_dir,
                partition_label = paste0("Cluster: ", cluster),
                indicator = ind,
                model_type = mtype,
                percentile = "overall"
              )
              summary_result$af_output <- partition_has_af_outputs(partition_meta_dir)
              
              # 【分区】Meta-predictors 输出（与 Zone 相同逻辑）
              meta_predictor_result <- run_partition_meta_predictors(
                partition_socioecon_avg = partition_socioecon_avg,
                coef_matrix = coef_matrix,
                vcov_list = vcov_list,
                partition_meta_dir = partition_meta_dir,
                partition_label = paste0("Cluster: ", cluster),
                indicator = ind,
                model_type = mtype,
                partition_results = partition_results
              )
              if (FALSE && !is.null(partition_socioecon_avg) && nrow(partition_socioecon_avg) > 0) {
                city_names_p <- rownames(coef_matrix)
                city_covariates_p <- partition_socioecon_avg %>%
                  filter(city %in% city_names_p) %>%
                  arrange(match(city, city_names_p)) %>%
                  select(-city) %>%
                  as.data.frame()
                keep_p <- city_names_p %in% partition_socioecon_avg$city
                if (sum(keep_p) < 3) {
                  cat("          ⚠ Meta-predictors 跳过：城市数不足 (", sum(keep_p), " < 3)\n")
                } else if (nrow(city_covariates_p) != sum(keep_p)) {
                  cat("          ⚠ Meta-predictors 跳过：协变量数据不完整 (", nrow(city_covariates_p), " != ", sum(keep_p), ")\n")
                }
                if (sum(keep_p) >= 3 && nrow(city_covariates_p) == sum(keep_p)) {
                  if (nrow(city_covariates_p) != nrow(coef_matrix)) {
                    coef_matrix_p <- coef_matrix[keep_p, , drop = FALSE]
                    vcov_list_p <- vcov_list[keep_p]
                  } else {
                    coef_matrix_p <- coef_matrix
                    vcov_list_p <- vcov_list
                  }
                  rownames(city_covariates_p) <- rownames(coef_matrix_p)
                  meta_formula_p <- as.formula(paste("coef_matrix_p ~", paste(names(city_covariates_p), collapse = " + ")))
                  
                  # 【修复】多重策略拟合 meta-predictors（与主分析一致）
                  mv_cov_p <- NULL
                  # 策略1: 标准REML
                  mv_cov_p <- tryCatch({
                    eval(bquote(
                      mvmeta(.(meta_formula_p), data = city_covariates_p, S = vcov_list_p, method = "reml")
                    ))
                  }, error = function(e) {
                    cat("          ⚠ Meta-predictors REML失败: ", substr(conditionMessage(e), 1, 60), "\n")
                    NULL
                  })
                  
                  # 策略2: Regularization
                  if (is.null(mv_cov_p)) {
                    cat("          → Meta-predictors 尝试 regularization...\n")
                    max_eigs <- sapply(vcov_list_p, function(V) {
                      tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
                    })
                    reg_str <- max(0.01, max(max_eigs, na.rm = TRUE) * 0.1)
                    vcov_reg_p <- lapply(vcov_list_p, function(V) V + diag(reg_str, nrow(V)))
                    mv_cov_p <- tryCatch({
                      eval(bquote(
                        mvmeta(.(meta_formula_p), data = city_covariates_p, S = vcov_reg_p, method = "reml")
                      ))
                    }, error = function(e) NULL)
                    if (!is.null(mv_cov_p)) {
                      cat("          ⚠ Meta-predictors regularization 成功（结果仅供参考）\n")
                    }
                  }
                  
                  # 策略3: Fixed effects
                  if (is.null(mv_cov_p)) {
                    cat("          → Meta-predictors 尝试 fixed effects...\n")
                    mv_cov_p <- tryCatch({
                      eval(bquote(
                        mvmeta(.(meta_formula_p), data = city_covariates_p, method = "fixed")
                      ))
                    }, error = function(e) {
                      cat("          ✗ Meta-predictors 所有策略失败: ", substr(conditionMessage(e), 1, 60), "\n")
                      NULL
                    })
                    if (!is.null(mv_cov_p)) {
                      cat("          ⚠⚠ Meta-predictors fixed effects 成功（不稳定，慎用）\n")
                    }
                  }
                  
                  # 【Bug修复】防御性检查：mv_cov_p 非 NULL 且可正常处理
                  meta_output_success <- FALSE
                  if (!is.null(mv_cov_p)) {
                    p_coef_summary <- tryCatch({
                      summary(mv_cov_p)
                    }, error = function(e) {
                      cat("          ✗ Meta-predictors summary() 失败: ", substr(conditionMessage(e), 1, 50), "\n")
                      NULL
                    })
                    
                    if (!is.null(p_coef_summary)) {
                      meta_output_success <- TRUE
                      p_coef_table <- p_coef_summary$coefficients
                      p_coef_names <- rownames(p_coef_table)
                      p_cols <- colnames(p_coef_table)
                      p_coefs_df <- data.frame(indicator = ind, model_type = mtype, n_cities = nrow(coef_matrix_p), coef_name = p_coef_names, stringsAsFactors = FALSE)
                      if ("Estimate" %in% p_cols) p_coefs_df$coefficient <- p_coef_table[, "Estimate"] else p_coefs_df$coefficient <- p_coef_table[, 1]
                      if ("Std. Error" %in% p_cols) p_coefs_df$se <- p_coef_table[, "Std. Error"] else if (ncol(p_coef_table) >= 2) p_coefs_df$se <- p_coef_table[, 2]
                      if ("z" %in% p_cols) p_coefs_df$z_value <- p_coef_table[, "z"] else if (ncol(p_coef_table) >= 3) p_coefs_df$z_value <- p_coef_table[, 3]
                      if ("Pr(>|z|)" %in% p_cols) p_coefs_df$p_value <- p_coef_table[, "Pr(>|z|)"] else if (ncol(p_coef_table) >= 4) p_coefs_df$p_value <- p_coef_table[, 4]
                      if ("ci.lb" %in% p_cols) p_coefs_df$ci_low <- p_coef_table[, "ci.lb"] else if (ncol(p_coef_table) >= 5) p_coefs_df$ci_low <- p_coef_table[, 5]
                      if ("ci.ub" %in% p_cols) p_coefs_df$ci_high <- p_coef_table[, "ci.ub"] else if (ncol(p_coef_table) >= 6) p_coefs_df$ci_high <- p_coef_table[, 6]
                      p_socioecon_rows <- grep("_mean", p_coef_names)
                    if (length(p_socioecon_rows) > 0) {
                      p_city_covar_df <- p_coefs_df[p_socioecon_rows, ]
                      p_city_covar_df <- p_city_covar_df %>%
                        mutate(
                          variable_label = case_when(
                            grepl("BD|Building_Density", coef_name) ~ "Building Density (City Avg)",
                            grepl("FAR_mean", coef_name) ~ "Floor Area Ratio (City Avg)",
                            grepl("NDVI", coef_name) & grepl("_mean", coef_name) ~ "NDVI (City Avg)",
                            grepl("total_20|Pop_mean", coef_name) ~ "Population (City Avg)",
                            grepl("GDP_mean", coef_name) ~ "GDP (City Avg)",
                            grepl("unemployed", coef_name) ~ "Unemployed Population (City Avg)",
                            grepl("Crime_mean", coef_name) ~ "Crime (City Avg)",
                            grepl("Unemployment_mean", coef_name) ~ "Unemployment (City Avg)",
                            grepl("Urbanization", coef_name) ~ "Urbanization (City Avg)",
                            grepl("Street_Intersection|Intersection", coef_name) ~ "Street Intersection (City Avg)",
                            grepl("Walkability|Walk", coef_name) & grepl("_mean", coef_name) ~ "Walkability (City Avg)",
                            TRUE ~ coef_name
                          ),
                          significant = ifelse(!is.na(p_value) & p_value < 0.05, "sig", "ns")
                        )
                      write_csv(p_city_covar_df, file.path(partition_meta_dir, "city_level_covariates.csv"))
                      p_city_covar_df <- p_city_covar_df %>%
                        mutate(
                          predictor_name = case_when(
                            grepl("NDVI", coef_name) ~ "1_NDVI",
                            grepl("total_20", coef_name) ~ "2_Population",
                            grepl("BD|Building_Density", coef_name) ~ "3_Building_Density",
                            grepl("FAR_mean", coef_name) ~ "4_Floor_Area_Ratio",
                            grepl("GDP", coef_name) ~ "5_GDP",
                            grepl("unemployed", coef_name) ~ "6_Unemployed_Pop",
                            grepl("Crime", coef_name) ~ "7_Crime",
                            grepl("Unemployment", coef_name) ~ "8_Unemployment",
                            grepl("Walk", coef_name) ~ "9_Walkability",
                            TRUE ~ "0_Other"
                          ),
                          cb_lag = str_extract(coef_name, "cb_cehwiv[1-3]\\.l[1-3]")
                        ) %>%
                        arrange(predictor_name, cb_lag) %>%
                        mutate(display_label = paste0(predictor_name, " | ", cb_lag), y_order = row_number())
                      plot_h_full <- max(14, 5 + nrow(p_city_covar_df) * 0.25)
                      p_full <- ggplot(p_city_covar_df, aes(x = coefficient, y = reorder(display_label, y_order))) +
                        geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1) +
                        geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, alpha = ifelse(significant == "sig", 1, 0.4)), height = 0.2, linewidth = 0.9, color = "gray20") +
                        geom_point(aes(color = ifelse(coefficient > 0, "#D62728", "#1F77B4"), alpha = ifelse(significant == "sig", 1, 0.4)), size = 3, shape = 19) +
                        scale_color_identity() + scale_alpha_identity() +
                        labs(title = paste0("Meta-Predictors (Full Detail) - Cluster: ", cluster, " | ", toupper(ind), " - ", toupper(mtype)),
                             subtitle = paste0("All ", nrow(p_city_covar_df), " coefficients | Red = Positive | Blue = Negative | Solid = p<0.05"),
                             x = "Meta-Regression Coefficient", y = "") +
                        theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold", size = 15), panel.grid.major.y = element_blank(), panel.grid.major.x = element_blank())
                      ggsave(file.path(partition_meta_dir, "city_level_covariates_forest_FULL.png"), p_full, width = 16, height = plot_h_full, dpi = 300)
                      p_city_covar_df <- p_city_covar_df %>%
                        mutate(variable_only = case_when(
                          grepl("BD|Building_Density", coef_name) ~ "Building Density",
                          grepl("FAR_mean", coef_name) ~ "Floor Area Ratio",
                          grepl("NDVI", coef_name) & grepl("_mean", coef_name) ~ "NDVI",
                          grepl("total_20", coef_name) ~ "Population (20-55)",
                          grepl("GDP_mean", coef_name) ~ "GDP",
                          grepl("unemployed", coef_name) ~ "Unemployed Population",
                          grepl("Crime_mean", coef_name) ~ "Crime",
                          grepl("Unemployment_mean", coef_name) ~ "Unemployment",
                          grepl("Walk", coef_name) & grepl("_mean", coef_name) ~ "Walkability Index",
                          TRUE ~ "Other"
                        ))
                      p_covar_summary <- p_city_covar_df %>%
                        filter(variable_only != "Other") %>%
                        group_by(variable_only) %>%
                        summarise(
                          coefficient = sum(coefficient / (se^2 + 1e-10), na.rm = TRUE) / sum(1 / (se^2 + 1e-10), na.rm = TRUE),
                          se = sqrt(1 / sum(1 / (se^2 + 1e-10), na.rm = TRUE)),
                          z_value = coefficient / se, p_value = 2 * pnorm(-abs(coefficient / se)),
                          ci_low = coefficient - 1.96 * se, ci_high = coefficient + 1.96 * se,
                          significant = ifelse(p_value < 0.05, "sig", "ns"), n_coefs = n(), .groups = "drop"
                        )
                      write_csv(p_covar_summary, file.path(partition_meta_dir, "city_level_covariates_SUMMARY.csv"))
                      plot_h <- max(6, 4 + nrow(p_covar_summary) * 0.8)
                      p_summary <- ggplot(p_covar_summary, aes(x = coefficient, y = reorder(variable_only, coefficient))) +
                        geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1.5) +
                        geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, alpha = ifelse(significant == "sig", 1, 0.5)), height = 0.4, linewidth = 2, color = "gray20") +
                        geom_point(aes(color = ifelse(coefficient > 0, "#D62728", "#1F77B4"), alpha = ifelse(significant == "sig", 1, 0.5)), size = 12, shape = 19) +
                        scale_color_identity() + scale_alpha_identity() +
                        labs(title = "Meta-Predictors: How City Characteristics Modify Heatwave Effects",
                             subtitle = paste0("Cluster: ", cluster, " | ", toupper(ind), " - ", toupper(mtype), " | ", nrow(p_covar_summary), " Variables"),
                             x = "Meta-Regression Coefficient (Standardized)", y = "") +
                        theme_minimal(base_size = 20) + theme(plot.title = element_text(face = "bold", size = 22), axis.text.y = element_text(size = 18, face = "bold"), panel.grid.major.y = element_blank())
                      ggsave(file.path(partition_meta_dir, "city_level_covariates_forest.png"), p_summary, width = 15, height = plot_h, dpi = 300)
                      cat("          ✓ 分区 Meta-predictors 已保存（city_level_covariates*.csv + 森林图）\n")
                    }
                    } else {
                      cat("          ⚠ 跳过 Meta-predictors 输出（mvmeta 对象不完整或 summary 失败）\n")
                    }
                  }
                }
              } else {
                cat("          ⚠ Meta-predictors 跳过：无城市社会经济数据\n")
                writeLines(
                  c(
                    "Meta-predictors were skipped for this partition model.",
                    "Reason: city-level socioeconomic data were unavailable in the current run.",
                    "No city_level_covariates*.csv or forest plots were generated for this partition."
                  ),
                  file.path(partition_meta_dir, "meta_predictors_status.txt")
                )
              }
              
              summary_result$model_status <- "SUCCESS"
              summary_result$stability_flag <- model_reliability
              summary_result$meta_predictors_output <- partition_has_meta_predictors(partition_meta_dir)
              summary_result$output_dir <- partition_meta_dir
              note_parts <- c(
                if (!summary_result$meta_predictors_output) meta_predictor_result$note else NULL,
                if (!summary_result$af_output) "AF outputs missing" else NULL
              )
              summary_result$status_note <- if (length(note_parts) > 0) paste(note_parts, collapse = " | ") else NA_character_
              
            }, error = function(e) {
              cat("          ✗ Meta-regression完全失败!\n")
              cat("          错误类型: ", class(e)[1], "\n")
              cat("          错误信息: ", conditionMessage(e), "\n")
              cat("          请检查日志以获取详细信息\n")
              summary_result <<- modifyList(summary_result, list(
                model_status = "FAILED",
                output_dir = partition_meta_dir,
                meta_predictors_output = partition_has_meta_predictors(partition_meta_dir),
                af_output = partition_has_af_outputs(partition_meta_dir),
                status_note = conditionMessage(e)
              ))
            })
            
            partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
              partition_family = "Cluster",
              partition_name = cluster,
              indicator = ind,
              model_type = mtype,
              n_cities = length(partition_results),
              n_cities_total = length(cluster_cities),
              model_status = summary_result$model_status,
              stability_flag = summary_result$stability_flag,
              meta_predictors_output = summary_result$meta_predictors_output,
              af_output = summary_result$af_output,
              output_dir = summary_result$output_dir,
              status_note = summary_result$status_note
            )
          }
        }
        
        cat("      ✓ 【V6】Cluster meta-regression完成\n")
      } else {
        cat("      ⚠ 城市数不足3个，跳过\n")
        for (ind in c("cehwi", "exceeded_quantity")) {
          for (mtype in STAGE1_MODEL_TYPES) {
            partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
              partition_family = "Cluster",
              partition_name = cluster,
              indicator = ind,
              model_type = mtype,
              n_cities = count_partition_model_cities(cluster_cities, successful_cities, ind, mtype),
              n_cities_total = length(cluster_cities),
              model_status = "SKIPPED_PARTITION_LT3",
              output_dir = cluster_output_dir,
              status_note = "Entire partition skipped because fewer than 3 cities were available for the base partition result set."
            )
          }
        }
      }
      if (length(cluster_af_model_results) > 0) {
        save_partition_af_percentile_outputs(
          partition_model_results = cluster_af_model_results,
          partition_output_dir = cluster_output_dir,
          partition_label = paste0("Cluster: ", cluster),
          partition_family = "Cluster",
          partition_name = cluster,
          mode = meta_predictor_mode_label()
        )
      }
      cat("\n")
    }
    
    # 【V5新增】按Geographic Region分区
    cat("  [3/5] 按Geographic Region分区分析...\n\n")
    
    # geographic_region 已在 city_zone_cluster 中基于映射生成
    
    region_summary_list <- list()
    
    for (region in unique(city_zone_cluster$geographic_region)) {
      if (is.na(region) || region == "Unknown") next
      
      region_cities <- city_zone_cluster %>%
        filter(geographic_region == region) %>%
        pull(city)
      
      cat("    ══ Geographic Region:", region, "（", length(region_cities), "个城市）══\n")
      
      # 对该Region的城市进行meta-regression
      region_output_dir <- file.path(OUTPUT_DIR, paste0("REGION_", region))
      dir.create(region_output_dir, showWarnings = FALSE, recursive = TRUE)
      region_af_model_results <- list()
      
      # 提取该Region城市的第一阶段结果
      region_results <- list()
      for (city_name in region_cities) {
        city_key <- paste0(city_name, "_cehwi")
        if (city_key %in% names(successful_cities)) {
          city_result <- successful_cities[[city_key]]
          base_model_for_partition <- stage1_model_name("composite", "all", STAGE1_ACTIVITY_MODE)
          if (base_model_for_partition %in% names(city_result)) {
            region_results[[city_key]] <- city_result[[base_model_for_partition]]
          }
        }
      }
      
      if (length(region_results) >= 3) {
        cat("      - 成功纳入", length(region_results), "个城市\n")
        cat("      ✓ 目录已创建:", region_output_dir, "\n")
        
        # 保存分区信息
        region_info <- data.frame(
          region = region,
          n_cities_total = length(region_cities),
          n_cities_included = length(region_results),
          cities = paste(names(region_results), collapse = ", ")
        )
        write_csv(region_info, file.path(region_output_dir, "region_info.csv"))
        
        region_summary_list[[region]] <- region_info
        
        # 【V6完整实现】运行完整的meta-regression（与 Zone/Cluster 相同输出）
        cat("      → 【V6】开始完整meta-regression分析...\n")
        
        # 遍历所有indicator和model_type组合
        for (ind in c("cehwi", "exceeded_quantity")) {
          for (mtype in STAGE1_MODEL_TYPES) {
            # 提取该region该模型的结果
            partition_results <- list()
            for (city_name in region_cities) {
              city_key <- paste0(city_name, "_", ind)
              if (city_key %in% names(successful_cities)) {
                city_result <- successful_cities[[city_key]]
                if (mtype %in% names(city_result)) {
                  partition_results[[city_name]] <- city_result[[mtype]]
                }
              }
            }
            partition_results <- filter_current_stage1_results(
              partition_results,
              paste0("Region ", region, " ", toupper(ind), " ", toupper(mtype))
            )
            
            if (length(partition_results) < 3) {
              partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
                partition_family = "Region",
                partition_name = region,
                indicator = ind,
                model_type = mtype,
                n_cities = length(partition_results),
                n_cities_total = length(region_cities),
                model_status = "SKIPPED_LT3",
                output_dir = file.path(region_output_dir, paste0(toupper(ind), "_", toupper(mtype))),
                status_note = "Skipped before fitting because fewer than 3 cities had usable first-stage results."
              )
              next
            }
            
            cat("        → ", toupper(ind), "-", toupper(mtype), ": ", 
                length(partition_results), "个城市...\n")
            
            # 【V6完整实现】创建输出目录（与 Zone/Cluster 一致）
            if (is.null(region_af_model_results[[ind]])) region_af_model_results[[ind]] <- list()
            region_af_model_results[[ind]][[mtype]] <- partition_results
            
            partition_meta_dir_base <- file.path(region_output_dir, paste0(toupper(ind), "_", toupper(mtype)))
            partition_meta_dir <- partition_meta_dir_base
            dir.create(partition_meta_dir, showWarnings = FALSE, recursive = TRUE)
            
            # 【V6完整实现】运行完整 meta-regression（与 Climate Zone / City Cluster 相同流程）
            summary_result <- list(
              model_status = "FAILED",
              stability_flag = NA_character_,
              meta_predictors_output = FALSE,
              af_output = FALSE,
              output_dir = partition_meta_dir,
              status_note = NA_character_
            )
            tryCatch({
              cat("          【调试】提取系数和协方差矩阵...\n")
              
              coef_list <- lapply(partition_results, function(x) x$coef)
              vcov_list <- lapply(partition_results, function(x) x$vcov)
              
              if (any(sapply(coef_list, is.null))) {
                stop("部分城市的coef为空")
              }
              if (any(sapply(vcov_list, is.null))) {
                stop("部分城市的vcov为空")
              }
              
              coef_matrix <- do.call(rbind, coef_list)
              cat("          【调试】coef_matrix: ", nrow(coef_matrix), "行×", ncol(coef_matrix), "列\n")
              
              mv_model_simple <- NULL
              model_reliability <- "STABLE"
              
              mv_model_simple <- tryCatch({
                mvmeta(coef_matrix, S = vcov_list, method = "reml")
              }, error = function(e) {
                cat("          ⚠ REML失败: ", substr(conditionMessage(e), 1, 40), "\n")
                NULL
              })
              
              if (is.null(mv_model_simple)) {
                cat("          → 尝试regularization...\n")
                max_eigs <- sapply(vcov_list, function(V) {
                  tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
                })
                reg_str <- max(0.01, max(max_eigs, na.rm = TRUE) * 0.1)
                vcov_reg <- lapply(vcov_list, function(V) V + diag(reg_str, nrow(V)))
                mv_model_simple <- tryCatch({
                  mvmeta(coef_matrix, S = vcov_reg, method = "reml")
                }, error = function(e) NULL)
                if (!is.null(mv_model_simple)) {
                  model_reliability <- "UNSTABLE_REG"
                  cat("          ⚠ Regularization成功（⚠不稳定）\n")
                }
              }
              
              if (is.null(mv_model_simple)) {
                cat("          → 尝试fixed effects...\n")
                mv_model_simple <- tryCatch({
                  mvmeta(coef_matrix, method = "fixed")
                }, error = function(e) NULL)
                if (!is.null(mv_model_simple)) {
                  model_reliability <- "HIGHLY_UNSTABLE_FIXED"
                  cat("          ⚠⚠ Fixed effects成功（⚠⚠极不稳定）\n")
                }
              }
              
              if (is.null(mv_model_simple)) {
                stop("所有策略失败")
              }
              
              cat("          【调试】mvmeta拟合成功 [", model_reliability, "]\n")
              
              pooled_coef <- coef(mv_model_simple)
              pooled_vcov <- vcov(mv_model_simple)
              cb_template <- partition_results[[1]]$cb
              if (is.null(cb_template)) stop("cb_template为空")
              
              part_ranges <- lapply(partition_results, function(x) x$cehwi_range)
              part_ranges <- part_ranges[!sapply(part_ranges, is.null)]
              if (length(part_ranges) > 0) {
                cehwi_range_p <- range(unlist(part_ranges), na.rm = TRUE)
                cehwi_max <- max(cehwi_range_p[2], 1)
              } else {
                all_exp <- unlist(lapply(partition_results, function(x) { d <- x$cehwi_data; if (!is.null(d)) d[d > 0] else NULL }))
                cehwi_max <- if (length(all_exp) > 0) max(max(all_exp, na.rm = TRUE), 1) else 10
              }
              if (is.na(cehwi_max) || cehwi_max <= 0) cehwi_max <- 10
              
              pooled_cp <- crosspred(
                cb_template,
                coef = pooled_coef,
                vcov = pooled_vcov,
                model.link = "log",
                at = seq(0, cehwi_max, length.out = 500),
                cen = 0,
                cumul = TRUE
              )
              
              pooled_df <- data.frame(
                cehwi = pooled_cp$predvar,
                rr = pooled_cp$allRRfit,
                rr_low = pooled_cp$allRRlow,
                rr_high = pooled_cp$allRRhigh
              )
              
              title_suffix <- ifelse(model_reliability == "STABLE", "",
                          ifelse(model_reliability == "UNSTABLE_REG", " ⚠ UNSTABLE", " ⚠⚠ HIGHLY UNSTABLE"))
              subtitle_text <- build_partition_rr_subtitle(
                partition_family = "Region",
                partition_name = region,
                indicator = ind,
                model_type = mtype,
                n_cities = length(partition_results),
                pooled_df = pooled_df,
                meta_model = mv_model_simple,
                reliability = model_reliability
              )
              
              # 【颜色修复】根据 model_type 选择颜色（与主分析保持一致）
              line_color <- stage1_model_color(mtype)
              if (is.null(line_color) || is.na(line_color)) line_color <- "#D53E4F"
              
              # 【贴地修复】智能Y轴：CI 爆炸时改用 RR 定上限
              part_rr_min <- min(pooled_df$rr, na.rm = TRUE)
              part_rr_max <- max(pooled_df$rr, na.rm = TRUE)
              part_ci_min <- min(pooled_df$rr_low, na.rm = TRUE)
              part_ci_max <- max(pooled_df$rr_high, na.rm = TRUE)
              part_rr_range <- part_rr_max - part_rr_min
              part_ci_range <- part_ci_max - part_ci_min
              
              # 判断 CI 是否爆炸：CI 上限 > RR 最大值 * 10 或 CI 范围 > RR 范围 * 15
              part_ci_exploded <- (part_ci_max > part_rr_max * 10 || part_ci_range > part_rr_range * 15)
              
              # 【组合图修复】创建裁剪版本的 pooled_df_plot（与主分析一致）
              if (part_ci_exploded) {
                cat("          ⚠ CI 爆炸（max=", round(part_ci_max, 1), "），裁剪 CI 用于绘图\n")
                part_y_max <- part_rr_max * 2.5
                part_y_min <- max(0, part_rr_min * 0.5)
                pooled_df_plot <- pooled_df %>%
                  mutate(
                    rr_low_clipped = pmax(rr_low, part_y_min),
                    rr_high_clipped = pmin(rr_high, part_y_max)
                  )
              } else {
                pooled_df_plot <- pooled_df %>%
                  mutate(
                    rr_low_clipped = rr_low,
                    rr_high_clipped = rr_high
                  )
              }
              
              part_caption <- build_rr_caption(
                indicator = ind,
                has_pi = FALSE,
                ci_clipped = part_ci_exploded,
                reliability = model_reliability,
                include_percentiles = FALSE,
                x_truncated = TRUE
              )
              p_pooled_rr <- ggplot(pooled_df_plot, aes(x = cehwi, y = rr)) +
                geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8) +
                geom_ribbon(aes(ymin = rr_low_clipped, ymax = rr_high_clipped), fill = "gray80", alpha = 0.5) +
                geom_line(color = line_color, linewidth = 2) +
                labs(
                  title = paste0("Region: ", region, " - ", toupper(ind), " - ", toupper(mtype), title_suffix,
                                 if(part_ci_exploded) " [Y-axis: RR-based]" else ""),
                  subtitle = subtitle_text,
                  x = toupper(ind),
                  y = "Relative Risk (RR)",
                  caption = part_caption
                ) +
                rr_plot_theme(14)
              
              # 【Region修复】Y轴不需要额外限制，数据已裁剪
              if (part_ci_exploded) {
                cat("          ⚠ CI 爆炸（max=", round(part_ci_max, 1), "），Y 轴改用 RR * 2.5\n")
              }
              
              ggsave(file.path(partition_meta_dir, "pooled_RR_curve.png"), p_pooled_rr, width = 12, height = 8, dpi = 300)
              coef_df <- as.data.frame(coef_matrix)
              coef_df$reliability <- model_reliability
              safe_write_csv(coef_df, file.path(partition_meta_dir, "pooled_coefs.csv"), label = "partition pooled coefficients")
              pooled_df$reliability <- model_reliability
              safe_write_csv(pooled_df, file.path(partition_meta_dir, "pooled_RR_data.csv"), label = "partition pooled RR data")
              save_pooled_lag_response(
                partition_results,
                partition_meta_dir,
                indicator = ind,
                model_type = mtype,
                group_label = paste0("Partition: ", basename(dirname(partition_meta_dir))),
                reliability = model_reliability
              )
              
              if (model_reliability != "STABLE") {
                new_dir_name <- paste0(partition_meta_dir_base, "_", model_reliability)
                if (dir.exists(partition_meta_dir) && partition_meta_dir != new_dir_name) {
                  file.rename(partition_meta_dir, new_dir_name)
                  partition_meta_dir <- new_dir_name
                }
              }
              
              cat("          ✓ Meta-regression成功 | Pooled RR曲线与系数已保存 [", model_reliability, "]\n")
              
              all_cehwi_data <- unlist(lapply(partition_results, function(x) {
                if (!is.null(x$cehwi_data)) return(x$cehwi_data[x$cehwi_data > 0])
                return(NULL)
              }))
              
              if (!is.null(all_cehwi_data) && length(all_cehwi_data) > 10) {
                q25 <- quantile(all_cehwi_data, 0.25, na.rm = TRUE)
                q75 <- quantile(all_cehwi_data, 0.75, na.rm = TRUE)
                q90 <- quantile(all_cehwi_data, 0.90, na.rm = TRUE)
                q98 <- quantile(all_cehwi_data, 0.98, na.rm = TRUE)
                p_hist <- ggplot(data.frame(cehwi = all_cehwi_data), aes(x = cehwi)) +
                  geom_histogram(bins = 30, fill = "#3B9AB2", alpha = 0.7, color = "white") +
                  geom_vline(xintercept = c(q25, q75, q90), linetype = c("dotted", "dotted", "dashed"),
                             color = c("#E69F00", "#D55E00", "#CC0000"), linewidth = c(1, 1, 1.2)) +
                  labs(title = paste0("Exposure Distribution: ", region),
                       subtitle = paste0(toupper(ind), " - ", toupper(mtype), " (", length(partition_results), " cities)"),
                       x = toupper(ind), y = "Frequency",
                       caption = paste0("25th: ", round(q25, 2), " | 75th: ", round(q75, 2), " | 90th: ", round(q90, 2))) +
                  rr_hist_theme(12) +
                  coord_cartesian(xlim = c(0, q98))
                ggsave(file.path(partition_meta_dir, "exposure_distribution.png"), p_hist, width = 12, height = 4, dpi = 300)
                
                # 【Region组合图修复】只设置 xlim，ylim 由裁剪后的数据决定
                part_rr_caption <- build_rr_caption(
                  indicator = ind,
                  has_pi = FALSE,
                  ci_clipped = part_ci_exploded,
                  reliability = model_reliability,
                  include_percentiles = TRUE,
                  x_truncated = TRUE
                )
                p_pooled_rr_with_lines <- p_pooled_rr +
                  coord_cartesian(xlim = c(0, q98)) +
                  geom_vline(xintercept = q25, linetype = "dotted", color = "#E69F00", linewidth = 0.8) +
                  geom_vline(xintercept = q75, linetype = "dotted", color = "#D55E00", linewidth = 0.8) +
                  geom_vline(xintercept = q90, linetype = "dashed", color = "#CC0000", linewidth = 1) +
                  labs(caption = part_rr_caption)
                
                ggsave(file.path(partition_meta_dir, "pooled_RR_curve_with_percentiles.png"), p_pooled_rr_with_lines, width = 12, height = 8, dpi = 300)
                p_combined <- build_rr_distribution_combined_plot(
                  rr_plot = p_pooled_rr_with_lines,
                  hist_plot = p_hist,
                  caption_text = part_rr_caption,
                  heights = c(2, 1),
                  caption_size = 10
                )
                ggsave(file.path(partition_meta_dir, "pooled_RR_with_distribution.png"), p_combined, width = 12, height = 10, dpi = 300)
                cat("          ✓ 暴露分布直方图与组合图已生成\n")
              } else {
                cat("          ⚠ 暴露数据不足，跳过直方图\n")
              }
              
              plot_partition_af_results(
                partition_results = partition_results,
                partition_meta_dir = partition_meta_dir,
                partition_label = paste0("Region: ", region),
                indicator = ind,
                model_type = mtype,
                percentile = "overall"
              )
              summary_result$af_output <- partition_has_af_outputs(partition_meta_dir)
              
              # 【分区】Meta-predictors 输出（与 Zone/Cluster 相同逻辑）
              meta_predictor_result <- run_partition_meta_predictors(
                partition_socioecon_avg = partition_socioecon_avg,
                coef_matrix = coef_matrix,
                vcov_list = vcov_list,
                partition_meta_dir = partition_meta_dir,
                partition_label = paste0("Region: ", region),
                indicator = ind,
                model_type = mtype,
                partition_results = partition_results
              )
              if (FALSE && !is.null(partition_socioecon_avg) && nrow(partition_socioecon_avg) > 0) {
                city_names_p <- rownames(coef_matrix)
                city_covariates_p <- partition_socioecon_avg %>%
                  filter(city %in% city_names_p) %>%
                  arrange(match(city, city_names_p)) %>%
                  select(-city) %>%
                  as.data.frame()
                keep_p <- city_names_p %in% partition_socioecon_avg$city
                if (sum(keep_p) < 3) {
                  cat("          ⚠ Meta-predictors 跳过：城市数不足 (", sum(keep_p), " < 3)\n")
                } else if (nrow(city_covariates_p) != sum(keep_p)) {
                  cat("          ⚠ Meta-predictors 跳过：协变量数据不完整 (", nrow(city_covariates_p), " != ", sum(keep_p), ")\n")
                }
                if (sum(keep_p) >= 3 && nrow(city_covariates_p) == sum(keep_p)) {
                  if (nrow(city_covariates_p) != nrow(coef_matrix)) {
                    coef_matrix_p <- coef_matrix[keep_p, , drop = FALSE]
                    vcov_list_p <- vcov_list[keep_p]
                  } else {
                    coef_matrix_p <- coef_matrix
                    vcov_list_p <- vcov_list
                  }
                  rownames(city_covariates_p) <- rownames(coef_matrix_p)
                  meta_formula_p <- as.formula(paste("coef_matrix_p ~", paste(names(city_covariates_p), collapse = " + ")))
                  
                  # 【修复】多重策略拟合 meta-predictors（与主分析一致）
                  mv_cov_p <- NULL
                  # 策略1: 标准REML
                  mv_cov_p <- tryCatch({
                    eval(bquote(
                      mvmeta(.(meta_formula_p), data = city_covariates_p, S = vcov_list_p, method = "reml")
                    ))
                  }, error = function(e) {
                    cat("          ⚠ Meta-predictors REML失败: ", substr(conditionMessage(e), 1, 60), "\n")
                    NULL
                  })
                  
                  # 策略2: Regularization
                  if (is.null(mv_cov_p)) {
                    cat("          → Meta-predictors 尝试 regularization...\n")
                    max_eigs <- sapply(vcov_list_p, function(V) {
                      tryCatch(max(abs(eigen(V, only.values = TRUE)$values)), error = function(e) 0)
                    })
                    reg_str <- max(0.01, max(max_eigs, na.rm = TRUE) * 0.1)
                    vcov_reg_p <- lapply(vcov_list_p, function(V) V + diag(reg_str, nrow(V)))
                    mv_cov_p <- tryCatch({
                      eval(bquote(
                        mvmeta(.(meta_formula_p), data = city_covariates_p, S = vcov_reg_p, method = "reml")
                      ))
                    }, error = function(e) NULL)
                    if (!is.null(mv_cov_p)) {
                      cat("          ⚠ Meta-predictors regularization 成功（结果仅供参考）\n")
                    }
                  }
                  
                  # 策略3: Fixed effects
                  if (is.null(mv_cov_p)) {
                    cat("          → Meta-predictors 尝试 fixed effects...\n")
                    mv_cov_p <- tryCatch({
                      eval(bquote(
                        mvmeta(.(meta_formula_p), data = city_covariates_p, method = "fixed")
                      ))
                    }, error = function(e) {
                      cat("          ✗ Meta-predictors 所有策略失败: ", substr(conditionMessage(e), 1, 60), "\n")
                      NULL
                    })
                    if (!is.null(mv_cov_p)) {
                      cat("          ⚠⚠ Meta-predictors fixed effects 成功（不稳定，慎用）\n")
                    }
                  }
                  
                  # 【Bug修复】防御性检查：mv_cov_p 非 NULL 且可正常处理
                  meta_output_success <- FALSE
                  if (!is.null(mv_cov_p)) {
                    p_coef_summary <- tryCatch({
                      summary(mv_cov_p)
                    }, error = function(e) {
                      cat("          ✗ Meta-predictors summary() 失败: ", substr(conditionMessage(e), 1, 50), "\n")
                      NULL
                    })
                    
                    if (!is.null(p_coef_summary)) {
                      meta_output_success <- TRUE
                      p_coef_table <- p_coef_summary$coefficients
                      p_coef_names <- rownames(p_coef_table)
                      p_cols <- colnames(p_coef_table)
                      p_coefs_df <- data.frame(indicator = ind, model_type = mtype, n_cities = nrow(coef_matrix_p), coef_name = p_coef_names, stringsAsFactors = FALSE)
                      if ("Estimate" %in% p_cols) p_coefs_df$coefficient <- p_coef_table[, "Estimate"] else p_coefs_df$coefficient <- p_coef_table[, 1]
                      if ("Std. Error" %in% p_cols) p_coefs_df$se <- p_coef_table[, "Std. Error"] else if (ncol(p_coef_table) >= 2) p_coefs_df$se <- p_coef_table[, 2]
                      if ("z" %in% p_cols) p_coefs_df$z_value <- p_coef_table[, "z"] else if (ncol(p_coef_table) >= 3) p_coefs_df$z_value <- p_coef_table[, 3]
                      if ("Pr(>|z|)" %in% p_cols) p_coefs_df$p_value <- p_coef_table[, "Pr(>|z|)"] else if (ncol(p_coef_table) >= 4) p_coefs_df$p_value <- p_coef_table[, 4]
                      if ("ci.lb" %in% p_cols) p_coefs_df$ci_low <- p_coef_table[, "ci.lb"] else if (ncol(p_coef_table) >= 5) p_coefs_df$ci_low <- p_coef_table[, 5]
                      if ("ci.ub" %in% p_cols) p_coefs_df$ci_high <- p_coef_table[, "ci.ub"] else if (ncol(p_coef_table) >= 6) p_coefs_df$ci_high <- p_coef_table[, 6]
                      p_socioecon_rows <- grep("_mean", p_coef_names)
                    if (length(p_socioecon_rows) > 0) {
                      p_city_covar_df <- p_coefs_df[p_socioecon_rows, ]
                      p_city_covar_df <- p_city_covar_df %>%
                        mutate(
                          variable_label = case_when(
                            grepl("BD|Building_Density", coef_name) ~ "Building Density (City Avg)",
                            grepl("FAR_mean", coef_name) ~ "Floor Area Ratio (City Avg)",
                            grepl("NDVI", coef_name) & grepl("_mean", coef_name) ~ "NDVI (City Avg)",
                            grepl("total_20|Pop_mean", coef_name) ~ "Population (City Avg)",
                            grepl("GDP_mean", coef_name) ~ "GDP (City Avg)",
                            grepl("unemployed", coef_name) ~ "Unemployed Population (City Avg)",
                            grepl("Crime_mean", coef_name) ~ "Crime (City Avg)",
                            grepl("Unemployment_mean", coef_name) ~ "Unemployment (City Avg)",
                            grepl("Urbanization", coef_name) ~ "Urbanization (City Avg)",
                            grepl("Street_Intersection|Intersection", coef_name) ~ "Street Intersection (City Avg)",
                            grepl("Walkability|Walk", coef_name) & grepl("_mean", coef_name) ~ "Walkability (City Avg)",
                            TRUE ~ coef_name
                          ),
                          significant = ifelse(!is.na(p_value) & p_value < 0.05, "sig", "ns")
                        )
                      write_csv(p_city_covar_df, file.path(partition_meta_dir, "city_level_covariates.csv"))
                      p_city_covar_df <- p_city_covar_df %>%
                        mutate(
                          predictor_name = case_when(
                            grepl("NDVI", coef_name) ~ "1_NDVI",
                            grepl("total_20", coef_name) ~ "2_Population",
                            grepl("BD|Building_Density", coef_name) ~ "3_Building_Density",
                            grepl("FAR_mean", coef_name) ~ "4_Floor_Area_Ratio",
                            grepl("GDP", coef_name) ~ "5_GDP",
                            grepl("unemployed", coef_name) ~ "6_Unemployed_Pop",
                            grepl("Crime", coef_name) ~ "7_Crime",
                            grepl("Unemployment", coef_name) ~ "8_Unemployment",
                            grepl("Walk", coef_name) ~ "9_Walkability",
                            TRUE ~ "0_Other"
                          ),
                          cb_lag = str_extract(coef_name, "cb_cehwiv[1-3]\\.l[1-3]")
                        ) %>%
                        arrange(predictor_name, cb_lag) %>%
                        mutate(display_label = paste0(predictor_name, " | ", cb_lag), y_order = row_number())
                      plot_h_full <- max(14, 5 + nrow(p_city_covar_df) * 0.25)
                      p_full <- ggplot(p_city_covar_df, aes(x = coefficient, y = reorder(display_label, y_order))) +
                        geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1) +
                        geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, alpha = ifelse(significant == "sig", 1, 0.4)), height = 0.2, linewidth = 0.9, color = "gray20") +
                        geom_point(aes(color = ifelse(coefficient > 0, "#D62728", "#1F77B4"), alpha = ifelse(significant == "sig", 1, 0.4)), size = 3, shape = 19) +
                        scale_color_identity() + scale_alpha_identity() +
                        labs(title = paste0("Meta-Predictors (Full Detail) - Region: ", region, " | ", toupper(ind), " - ", toupper(mtype)),
                             subtitle = paste0("All ", nrow(p_city_covar_df), " coefficients | Red = Positive | Blue = Negative | Solid = p<0.05"),
                             x = "Meta-Regression Coefficient", y = "") +
                        theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold", size = 15), panel.grid.major.y = element_blank(), panel.grid.major.x = element_blank())
                      ggsave(file.path(partition_meta_dir, "city_level_covariates_forest_FULL.png"), p_full, width = 16, height = plot_h_full, dpi = 300)
                      p_city_covar_df <- p_city_covar_df %>%
                        mutate(variable_only = case_when(
                          grepl("BD|Building_Density", coef_name) ~ "Building Density",
                          grepl("FAR_mean", coef_name) ~ "Floor Area Ratio",
                          grepl("NDVI", coef_name) & grepl("_mean", coef_name) ~ "NDVI",
                          grepl("total_20", coef_name) ~ "Population (20-55)",
                          grepl("GDP_mean", coef_name) ~ "GDP",
                          grepl("unemployed", coef_name) ~ "Unemployed Population",
                          grepl("Crime_mean", coef_name) ~ "Crime",
                          grepl("Unemployment_mean", coef_name) ~ "Unemployment",
                          grepl("Walk", coef_name) & grepl("_mean", coef_name) ~ "Walkability Index",
                          TRUE ~ "Other"
                        ))
                      p_covar_summary <- p_city_covar_df %>%
                        filter(variable_only != "Other") %>%
                        group_by(variable_only) %>%
                        summarise(
                          coefficient = sum(coefficient / (se^2 + 1e-10), na.rm = TRUE) / sum(1 / (se^2 + 1e-10), na.rm = TRUE),
                          se = sqrt(1 / sum(1 / (se^2 + 1e-10), na.rm = TRUE)),
                          z_value = coefficient / se, p_value = 2 * pnorm(-abs(coefficient / se)),
                          ci_low = coefficient - 1.96 * se, ci_high = coefficient + 1.96 * se,
                          significant = ifelse(p_value < 0.05, "sig", "ns"), n_coefs = n(), .groups = "drop"
                        )
                      write_csv(p_covar_summary, file.path(partition_meta_dir, "city_level_covariates_SUMMARY.csv"))
                      plot_h <- max(6, 4 + nrow(p_covar_summary) * 0.8)
                      p_summary <- ggplot(p_covar_summary, aes(x = coefficient, y = reorder(variable_only, coefficient))) +
                        geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 1.5) +
                        geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, alpha = ifelse(significant == "sig", 1, 0.5)), height = 0.4, linewidth = 2, color = "gray20") +
                        geom_point(aes(color = ifelse(coefficient > 0, "#D62728", "#1F77B4"), alpha = ifelse(significant == "sig", 1, 0.5)), size = 12, shape = 19) +
                        scale_color_identity() + scale_alpha_identity() +
                        labs(title = "Meta-Predictors: How City Characteristics Modify Heatwave Effects",
                             subtitle = paste0("Region: ", region, " | ", toupper(ind), " - ", toupper(mtype), " | ", nrow(p_covar_summary), " Variables"),
                             x = "Meta-Regression Coefficient (Standardized)", y = "") +
                        theme_minimal(base_size = 20) + theme(plot.title = element_text(face = "bold", size = 22), axis.text.y = element_text(size = 18, face = "bold"), panel.grid.major.y = element_blank())
                      ggsave(file.path(partition_meta_dir, "city_level_covariates_forest.png"), p_summary, width = 15, height = plot_h, dpi = 300)
                      cat("          ✓ 分区 Meta-predictors 已保存（city_level_covariates*.csv + 森林图）\n")
                    }
                    } else {
                      cat("          ⚠ 跳过 Meta-predictors 输出（mvmeta 对象不完整或 summary 失败）\n")
                    }
                  }
                }
              } else {
                cat("          ⚠ Meta-predictors 跳过：无城市社会经济数据\n")
                writeLines(
                  c(
                    "Meta-predictors were skipped for this partition model.",
                    "Reason: city-level socioeconomic data were unavailable in the current run.",
                    "No city_level_covariates*.csv or forest plots were generated for this partition."
                  ),
                  file.path(partition_meta_dir, "meta_predictors_status.txt")
                )
              }
              
              summary_result$model_status <- "SUCCESS"
              summary_result$stability_flag <- model_reliability
              summary_result$meta_predictors_output <- partition_has_meta_predictors(partition_meta_dir)
              summary_result$output_dir <- partition_meta_dir
              note_parts <- c(
                if (!summary_result$meta_predictors_output) meta_predictor_result$note else NULL,
                if (!summary_result$af_output) "AF outputs missing" else NULL
              )
              summary_result$status_note <- if (length(note_parts) > 0) paste(note_parts, collapse = " | ") else NA_character_
              
            }, error = function(e) {
              cat("          ✗ Meta-regression完全失败!\n")
              cat("          错误类型: ", class(e)[1], "\n")
              cat("          错误信息: ", conditionMessage(e), "\n")
              summary_result <<- modifyList(summary_result, list(
                model_status = "FAILED",
                output_dir = partition_meta_dir,
                meta_predictors_output = partition_has_meta_predictors(partition_meta_dir),
                af_output = partition_has_af_outputs(partition_meta_dir),
                status_note = conditionMessage(e)
              ))
            })
            
            partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
              partition_family = "Region",
              partition_name = region,
              indicator = ind,
              model_type = mtype,
              n_cities = length(partition_results),
              n_cities_total = length(region_cities),
              model_status = summary_result$model_status,
              stability_flag = summary_result$stability_flag,
              meta_predictors_output = summary_result$meta_predictors_output,
              af_output = summary_result$af_output,
              output_dir = summary_result$output_dir,
              status_note = summary_result$status_note
            )
          }
        }
        
        cat("      ✓ 【V6】Region meta-regression完成\n")
      } else {
        cat("      ⚠ 城市数不足3个，跳过\n")
        for (ind in c("cehwi", "exceeded_quantity")) {
          for (mtype in STAGE1_MODEL_TYPES) {
            partition_meta_run_summary[[length(partition_meta_run_summary) + 1]] <- create_partition_meta_summary_row(
              partition_family = "Region",
              partition_name = region,
              indicator = ind,
              model_type = mtype,
              n_cities = count_partition_model_cities(region_cities, successful_cities, ind, mtype),
              n_cities_total = length(region_cities),
              model_status = "SKIPPED_PARTITION_LT3",
              output_dir = region_output_dir,
              status_note = "Entire partition skipped because fewer than 3 cities were available for the base partition result set."
            )
          }
        }
      }
      if (length(region_af_model_results) > 0) {
        save_partition_af_percentile_outputs(
          partition_model_results = region_af_model_results,
          partition_output_dir = region_output_dir,
          partition_label = paste0("Region: ", region),
          partition_family = "Region",
          partition_name = region,
          mode = meta_predictor_mode_label()
        )
      }
      cat("\n")
    }
    
    dtw3_family_result <- run_additional_partition_meta_family(
      partition_mapping = DTW_CLUSTER_OPTIMIZED3_MAPPING,
      family_key = "DTW_Optimized3",
      family_display_label = "DTW Optimized k=3",
      output_prefix = "DTW3",
      section_label = "4/6",
      successful_cities = successful_cities,
      OUTPUT_DIR = OUTPUT_DIR,
      partition_socioecon_avg = partition_socioecon_avg,
      partition_meta_run_summary = partition_meta_run_summary
    )
    partition_meta_run_summary <- dtw3_family_result$partition_meta_run_summary
    dtw3_summary_list <- dtw3_family_result$family_summary_list
    
    dtw4_family_result <- run_additional_partition_meta_family(
      partition_mapping = DTW_CLUSTER_OPTIMIZED4_MAPPING,
      family_key = "DTW_Optimized4",
      family_display_label = "DTW Optimized k=4",
      output_prefix = "DTW4",
      section_label = "5/6",
      successful_cities = successful_cities,
      OUTPUT_DIR = OUTPUT_DIR,
      partition_socioecon_avg = partition_socioecon_avg,
      partition_meta_run_summary = partition_meta_run_summary
    )
    partition_meta_run_summary <- dtw4_family_result$partition_meta_run_summary
    dtw4_summary_list <- dtw4_family_result$family_summary_list
    
    dtw4lag12_family_result <- run_additional_partition_meta_family(
      partition_mapping = DTW_CLUSTER_LAG12_MAPPING,
      family_key = "DTW4lag12",
      family_display_label = "DTW4lag12",
      output_prefix = "DTW4lag12",
      section_label = "6/6",
      successful_cities = successful_cities,
      OUTPUT_DIR = OUTPUT_DIR,
      partition_socioecon_avg = partition_socioecon_avg,
      partition_meta_run_summary = partition_meta_run_summary
    )
    partition_meta_run_summary <- dtw4lag12_family_result$partition_meta_run_summary
    dtw4lag12_summary_list <- dtw4lag12_family_result$family_summary_list
    
    cat("  ✓ 分区框架已完成\n")
    if (length(partition_meta_run_summary) > 0) {
      partition_meta_summary_dir <- file.path(OUTPUT_DIR, "PARTITION_META_SUMMARY")
      dir.create(partition_meta_summary_dir, showWarnings = FALSE, recursive = TRUE)
      
      partition_meta_summary_df <- bind_rows(partition_meta_run_summary) %>%
        arrange(partition_family, partition_name, indicator, model_type)
      
      write_csv(
        partition_meta_summary_df,
        file.path(partition_meta_summary_dir, "partition_meta_run_summary.csv")
      )
      
      partition_scheme_comparison_df <- partition_meta_summary_df %>%
        group_by(partition_family) %>%
        summarise(
          n_partition_models = n(),
          n_success = sum(model_status == "SUCCESS", na.rm = TRUE),
          n_failed = sum(model_status == "FAILED", na.rm = TRUE),
          n_skipped = sum(grepl("^SKIPPED", model_status), na.rm = TRUE),
          success_rate = round(100 * n_success / n_partition_models, 1),
          failure_rate = round(100 * n_failed / n_partition_models, 1),
          skipped_rate = round(100 * n_skipped / n_partition_models, 1),
          n_unstable = sum(stability_flag == "UNSTABLE_REG", na.rm = TRUE),
          unstable_rate_among_success = round(
            100 * n_unstable / ifelse(n_success > 0, n_success, NA_real_), 1
          ),
          meta_predictor_success = sum(meta_predictors_output, na.rm = TRUE),
          meta_predictor_success_rate = round(100 * meta_predictor_success / n_partition_models, 1),
          af_success = sum(af_output, na.rm = TRUE),
          af_success_rate = round(100 * af_success / n_partition_models, 1),
          mean_n_cities = round(mean(n_cities, na.rm = TRUE), 1),
          median_n_cities = round(median(n_cities, na.rm = TRUE), 1),
          min_n_cities = min(n_cities, na.rm = TRUE),
          max_n_cities = max(n_cities, na.rm = TRUE),
          .groups = "drop"
        ) %>%
        arrange(desc(success_rate), desc(meta_predictor_success_rate), desc(mean_n_cities))
      
      write_csv(
        partition_scheme_comparison_df,
        file.path(partition_meta_summary_dir, "partition_scheme_comparison_summary.csv")
      )
      
      cat("    - 分区总表已保存: ",
          file.path(partition_meta_summary_dir, "partition_meta_run_summary.csv"), "\n")
    }
    cat("    - 按Climate Zone: ", n_distinct(city_zone_cluster$climate_zone), "个区 | ", 
        length(zone_summary_list), "个可分析\n")
    cat("    - 按City Cluster: ", n_distinct(city_zone_cluster$city_cluster), "个簇 | ", 
        length(cluster_summary_list), "个可分析\n")
    cat("    - 【V5新增】按Geographic Region: ", n_distinct(city_zone_cluster$geographic_region), "个区域 | ", 
        length(region_summary_list), "个可分析\n")
    cat("    - 【V5新增】按DTW Optimized k=3: ", length(DTW_CLUSTER_OPTIMIZED3_MAPPING), "个簇 | ",
        length(dtw3_summary_list), "个可分析\n")
    cat("    - 【V5新增】按DTW Optimized k=4: ", length(DTW_CLUSTER_OPTIMIZED4_MAPPING), "个簇 | ",
        length(dtw4_summary_list), "个可分析\n")
    cat("    - DTW4lag12: ", length(DTW_CLUSTER_LAG12_MAPPING), " clusters | ",
        length(dtw4lag12_summary_list), " analyzable\n")
    cat("    - 各分区目录和信息文件已保存（含六种分类的 RR 曲线、直方图、AF 森林图、meta-predictors 与 CSV）\n\n")
  safe_create_stage1_master_rr_panels(successful_cities, OUTPUT_DIR)
  create_stage2_master_rr_panels(OUTPUT_DIR)
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("✅ 第二阶段完成！\n")
  cat("输出目录:", OUTPUT_DIR, "\n")
  cat(rep("=", 100), "\n\n", sep = "")
} else {
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("⊘ 跳过第二阶段（用户选择）\n")
  cat(rep("=", 100), "\n\n", sep = "")
}

# ========== 最终总结 ==========

cat("\n", rep("=", 100), "\n", sep = "")

# 【V5.2】更新的总结信息
completed_parts <- c()
if (RUN_CITIES) completed_parts <- c(completed_parts, "单城市DLNM")
if (RUN_PARTITIONS) completed_parts <- c(completed_parts, "分区DLNM")
if (RUN_STAGE2) completed_parts <- c(completed_parts, "Meta-regression")
if (RUN_RERENDER_ONLY) completed_parts <- c(completed_parts, "仅重绘可视化")

if (length(completed_parts) > 0) {
  cat("✅ 分析完成！\n")
  cat("  已完成:", paste(completed_parts, collapse = " + "), "\n")
} else {
  cat("⊘ 未运行任何分析\n")
}

cat("输出目录:", OUTPUT_DIR, "\n")
cat("\n核心方法:\n")

if (RUN_CITIES || RUN_PARTITIONS) {
  cat("  【第一阶段】GAM + quasi-Poisson + DLNM cross-basis\n")
  if (RUN_CITIES) {
    cat("    ✓ 单城市分析: 75个城市独立建模\n")
    cat("      - 格子固定效应: fish_id_fac\n")
  }
  if (RUN_PARTITIONS) {
    cat("    ✓ 分区分析: Climate Zone/City Cluster/Geographic Region/DTW k=3/DTW k=4/DTW4lag12\n")
    cat("      - 格子随机效应: s(fish_id, bs='re') (内存优化)\n")
  }
  cat("    - 暴露: CEHWI, city-specific positive-exposure p50/p90 natural cubic spline\n")
  cat("    - 滞后: 0-", MAX_LAG, "天, natural cubic spline (3 df)\n")
  cat("    - 控制: mean temperature cross-basis + natural-spline precipitation/wind + continuous calendar time (7 df/year) + dow + fish_id effect（RH不入模）\n")
  cat("    - 二阶段输入: crossreduce后的overall cumulative reduced spline coefficients\n")
}

if (RUN_STAGE2) {
  cat("  【第二阶段】随机效应meta-regression (mvmeta, REML)\n")
  cat("    - 汇总各城市reduced coefficients和协方差\n")
  cat("    - meta-predictors可选城市均值或城市内1km网格Gini\n")
  cat("    - VIF检查（避免共线性）\n")
  cat("    - 生成pooled RR曲线 + Forest Plot\n")
}

cat(rep("=", 100), "\n\n", sep = "")

