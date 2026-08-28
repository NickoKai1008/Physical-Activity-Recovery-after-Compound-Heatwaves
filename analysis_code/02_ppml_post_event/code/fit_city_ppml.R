# ============================================================
# 🌡️ Event-lag PPML (Composite_heatwave)
# 每城市一个 trip 文件 + 多个 fish 热浪文件
# ============================================================

library(tidyverse)
library(lubridate)
library(fixest)
library(broom)
library(data.table)
library(janitor)

# ---------------------------
# PATHS
# ---------------------------
trip_root <- Sys.getenv("HEATPA_PPML_TRIP_ROOT", unset = file.path("external_data", "ppml", "daily_trip_counts_by_city"))
heat_root <- Sys.getenv("HEATPA_PPML_HEAT_ROOT", unset = file.path("external_data", "ppml", "daily_heatwave"))
output_dir <- Sys.getenv("HEATPA_PPML_OUTPUT_DIR", unset = file.path("analysis", "02_ppml_post_event", "output", "model"))
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

L <- 12
heat_var <- "composite_heatwave"
to_pct <- function(b) (exp(b) - 1) * 100

# ---------------------------
# 列出城市（以热浪文件夹为准）
# ---------------------------
city_list <- list.dirs(heat_root, full.names = FALSE, recursive = FALSE)
cat("检测到城市数量:", length(city_list), "\n")

# ---------------------------
# 单城市函数
# ---------------------------
run_city <- function(city_name){
  
  cat("\n🏙️ 处理城市:", city_name, "\n")
  
  trip_dir <- file.path(trip_root, city_name)
  heat_dir <- file.path(heat_root, city_name)
  
  trip_file <- file.path(trip_dir, "daily_trip_counts_by_fish.csv")
  
  if(!file.exists(trip_file)){
    cat("  ⚠️ 无 trip 文件，跳过\n")
    return(NULL)
  }
  
  # -----------------------
  # 1️⃣ 读取 trip 数据
  # -----------------------
  trip_df <- read_csv(trip_file, show_col_types = FALSE) %>%
    clean_names() %>%
    mutate(
      fish_id = as.character(id),
      date = as.Date(date)
    ) %>%
    select(fish_id, date, trip_count)
  
  # -----------------------
  # 2️⃣ 读取该城市所有热浪文件
  # -----------------------
  hw_files <- list.files(heat_dir,
                         pattern = "_temperature",
                         full.names = TRUE)
  
  if(length(hw_files) == 0){
    cat("  ⚠️ 无热浪文件\n")
    return(NULL)
  }
  
  hw_all <- map_dfr(hw_files, function(f){
    
    fid <- stringr::str_extract(basename(f), "\\d+")
    if(is.na(fid)) return(NULL)
    
    df_raw <- read_csv(f, show_col_types = FALSE)
    names(df_raw)[1] <- "date"
    
    df_raw %>%
      clean_names() %>%
      mutate(
        # 🔥 去掉时间
        date = as.Date(date),
        fish_id = as.character(fid)
      ) %>%
      filter(date >= as.Date("2010-01-01")) %>%
      select(fish_id, date, all_of(heat_var))
  })
  
  if(nrow(hw_all) == 0){
    cat("  ⚠️ 2010+ 无热浪数据\n")
    return(NULL)
  }
  
  # -----------------------
  # 3️⃣ 合并
  # -----------------------
  df <- hw_all %>%
    left_join(trip_df, by = c("fish_id","date")) %>%
    mutate(
      trip_count = replace_na(trip_count, 0L),
      is_hw = as.integer(replace_na(.data[[heat_var]], 0))
    ) %>%
    arrange(fish_id, date)
  
  if(sum(df$is_hw)==0){
    cat("  ⚠️ 2010+ 无热浪事件\n")
    return(NULL)
  }
  
  # -----------------------
  # 4️⃣ 识别事件结束日
  # -----------------------
  df <- df %>%
    group_by(fish_id) %>%
    arrange(date) %>%
    mutate(run_id = rleid(is_hw)) %>%
    ungroup()
  
  ev <- df %>%
    filter(is_hw == 1) %>%
    group_by(fish_id, run_id) %>%
    summarise(event_end = max(date), .groups="drop")
  
  if(nrow(ev)==0){
    cat("  ⚠️ 无事件\n")
    return(NULL)
  }
  
  # -----------------------
  # 5️⃣ 构造滞后 1..L
  # -----------------------
  post_tbl <- ev %>%
    mutate(dummy=1) %>%
    uncount(weights=L, .id="lag") %>%
    mutate(date = event_end + days(lag)) %>%
    select(fish_id, date, lag) %>%
    group_by(fish_id,date) %>%
    summarise(lag=min(lag), .groups="drop")
  
  df_model <- df %>%
    left_join(post_tbl, by=c("fish_id","date")) %>%
    mutate(
      lag = replace_na(lag,0),
      lag = ifelse(is_hw==1,0,lag)
    )
  
  for(k in 1:L){
    df_model[[paste0("post",k)]] <- as.integer(df_model$lag==k)
  }
  
  post_vars <- paste0("post",1:L)
  
  # -----------------------
  # 6️⃣ PPML
  # -----------------------
  frm <- as.formula(
    paste0("trip_count ~ is_hw + ",
           paste(post_vars,collapse=" + "),
           " | fish_id")
  )
  
  m <- tryCatch(
    fepois(frm,
           data=df_model,
           cluster=~fish_id,
           ssc=ssc(adj=FALSE,cluster.adj=FALSE)),
    error=function(e) NULL
  )
  
  if(is.null(m)){
    cat("  ❌ 模型失败\n")
    return(NULL)
  }
  
  td <- tidy(m,conf.int=TRUE) %>%
    filter(str_detect(term,"^post\\d+$")) %>%
    mutate(
      city = city_name,
      lag = as.numeric(str_extract(term,"\\d+$")),
      pct = to_pct(estimate),
      pct_low = to_pct(conf.low),
      pct_high = to_pct(conf.high),
      n_obs = nobs(m)
    ) %>%
    select(city,lag,estimate,std.error,p.value,
           conf.low,conf.high,pct,pct_low,pct_high,n_obs)
  
  cat("  ✅ 有效lag行数:",nrow(td),"\n")
  return(td)
}

# ---------------------------
# 7️⃣ 跑全部城市
# ---------------------------
results <- map(city_list, run_city)
results_all <- bind_rows(compact(results))

out_file <- file.path(output_dir,"all_cities_eventlag_results.csv")
write_csv(results_all,out_file)

cat("\n🎯 全部完成，输出文件：",out_file,"\n")
