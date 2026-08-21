# =====================================================================
# Shared helpers for the parameter-sensitivity sweeps.
#
# Spatial statistics go through spdep / spatstat with the SAME calls as
# 3.integration/ecology_code/, so the published parameter setting
# reproduces numerically rather than approximately. 99_validate_published.R
# checks that, and the robustness claim depends on it.
# =====================================================================

suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(purrr); library(tibble) })

# ---- SLURM array sharding -------------------------------------------
shard <- function(x, task_id = NULL, n_tasks = NULL) {
  if (is.null(task_id)) task_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID", "1"))
  if (is.null(n_tasks)) n_tasks <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
  if (is.na(task_id) || task_id < 1) task_id <- 1L
  if (is.na(n_tasks) || n_tasks < 1) n_tasks <- 1L
  keep <- ((seq_along(x) - 1L) %% n_tasks) + 1L == task_id
  structure(x[keep], task_id = task_id, n_tasks = n_tasks)
}

setup_parallel <- function(workers = NULL) {
  suppressPackageStartupMessages({ library(future); library(furrr) })
  if (is.null(workers)) {
    workers <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", ""))
    if (is.na(workers) || workers < 1) workers <- max(1L, future::availableCores() - 1L)
  }
  options(future.globals.maxSize = 8 * 1024^3)
  if (workers > 1) future::plan(future::multisession, workers = workers)
  else             future::plan(future::sequential)
  message(sprintf("[parallel] workers = %d", workers)); workers
}

# ---- stratification --------------------------------------------------
# stratum 2 = all tissue, 1 = DNA-content-abnormal, 0 = normal.
# Published pipeline stratifies on attn_class_extended (perilesional).
#
# CAREFUL: attn_class_extended is stored as a FACTOR with levels "0","1".
# as.numeric() on it returns LEVEL INDICES (1,2), not the values, which
# silently inverts the stratum. Always go through as.character() first.
as_attn_numeric <- function(v) {
  if (is.factor(v))    return(suppressWarnings(as.numeric(as.character(v))))
  if (is.character(v)) return(suppressWarnings(as.numeric(v)))
  as.numeric(v)
}

filter_stratum <- function(df, stratum, attn_col, attn_threshold = 0.5) {
  if (stratum == 2) return(df)
  v <- df[[attn_col]]
  if (is.null(v)) stop("attention column not found: ", attn_col)
  vn <- as_attn_numeric(v)
  uv <- unique(vn[!is.na(vn)])
  is_binary_class <- length(uv) > 0 && all(uv %in% c(0, 1))
  hi <- if (is_binary_class) vn == 1 else vn >= attn_threshold
  hi[is.na(hi)] <- FALSE
  if (stratum == 1) df[which(hi), , drop = FALSE] else df[which(!hi), , drop = FALSE]
}

# ---- spdep wrappers (tile level, PIXEL coordinates) ------------------
make_listw <- function(X, Y, d2, style = "B") {
  suppressPackageStartupMessages(library(spdep))
  if (length(X) < 3) return(NULL)
  nb <- tryCatch(spdep::dnearneigh(cbind(X, Y), d1 = 0, d2 = d2), error = function(e) NULL)
  if (is.null(nb)) return(NULL)
  lw <- tryCatch(spdep::nb2listw(nb, style = style, zero.policy = TRUE), error = function(e) NULL)
  if (is.null(lw)) return(NULL)
  attr(lw, "n_nb_median") <- stats::median(spdep::card(nb))
  attr(lw, "n_nb_mean")   <- mean(spdep::card(nb))
  lw
}

# Getis-Ord Gi*: published call uses GeoDa = TRUE.
local_gi <- function(X, Y, x, d2, style = "B", geoda = TRUE) {
  lw <- make_listw(X, Y, d2, style); if (is.null(lw)) return(NULL)
  z <- tryCatch(suppressWarnings(as.numeric(
         spdep::localG(x, lw, zero.policy = TRUE, GeoDa = geoda))),
       error = function(e) tryCatch(suppressWarnings(as.numeric(
         spdep::localG(x, lw, zero.policy = TRUE))), error = function(e2) NULL))
  if (is.null(z)) return(NULL)
  z[!is.finite(z)] <- NA_real_
  list(z = z, n_nb_median = attr(lw, "n_nb_median"))
}

local_moran <- function(X, Y, x, d2, style = "B") {
  lw <- make_listw(X, Y, d2, style); if (is.null(lw)) return(NULL)
  r <- tryCatch(suppressWarnings(spdep::localmoran(x, lw, zero.policy = TRUE)),
                error = function(e) NULL)
  if (is.null(r)) return(NULL)
  Ii <- as.numeric(r[, 1]); Ii[!is.finite(Ii)] <- NA_real_
  list(Ii = Ii, n_nb_median = attr(lw, "n_nb_median"))
}

global_moran <- function(X, Y, x, d2, style = "B") {
  lw <- make_listw(X, Y, d2, style); if (is.null(lw)) return(0)
  r <- tryCatch(spdep::moran.test(x, lw, zero.policy = TRUE), error = function(e) NULL)
  if (is.null(r)) return(0)
  v <- unname(r$estimate[1]); if (!is.finite(v)) 0 else v
}

global_getis <- function(X, Y, x, d2, style = "B") {
  lw <- make_listw(X, Y, d2, style); if (is.null(lw)) return(0)
  r <- tryCatch(spdep::globalG.test(x, lw, zero.policy = TRUE), error = function(e) NULL)
  if (is.null(r)) return(0)
  v <- unname(r$estimate[1]); if (!is.finite(v)) 0 else v
}

# ---- slide summaries: SIGN-BASED, exactly as published ---------------
# 2.3.getis_ord_gi.rmd:162-174 - no p-value threshold anywhere.
summarise_gi <- function(z) {
  z <- z[is.finite(z)]
  if (!length(z)) return(c(hotspot_intensity = 0, coldspot_intensity = 0,
                           clustering_strength = 0, clustering_heterogeneity = 0))
  hot <- z[z > 0]; cold <- z[z < 0]
  c(hotspot_intensity        = if (length(hot))  mean(hot)       else 0,
    coldspot_intensity       = if (length(cold)) mean(abs(cold)) else 0,
    clustering_strength      = mean(abs(z)),
    clustering_heterogeneity = if (length(z) > 1) stats::sd(z) else 0)
}

# 2.5.moran_241224.rmd:171-176
summarise_moran <- function(Ii) {
  Ii <- Ii[is.finite(Ii)]
  if (!length(Ii)) return(c(clustered_mean = 0, dispersed_mean = 0, abs_mean = 0,
                            abs_max = 0, clustering_heterogeneity = 0))
  clus <- Ii[Ii > 0]; disp <- Ii[Ii < 0]
  c(clustered_mean           = if (length(clus)) mean(clus)       else 0,
    dispersed_mean           = if (length(disp)) mean(abs(disp))  else 0,
    abs_mean                 = mean(abs(Ii)),
    abs_max                  = max(abs(Ii)),
    clustering_heterogeneity = if (length(Ii) > 1) stats::sd(Ii) else 0)
}

# ---- curve summarisers (Ripley L, G-function) ------------------------
# trapz is byte-identical to the published helper: NO na.rm, so one NA
# propagates to NA. Matching this is what makes Ripley reproduce exactly
# (filtering first drops reproduction from 1.000 to ~0.96).
trapz <- function(x, y) sum(diff(x) * (y[-length(y)] + y[-1]) / 2)
.safe <- function(f, v) { v <- v[is.finite(v)]; if (length(v)) f(v) else NA_real_ }

summarise_curve <- function(r, v, rmax = NA_real_) {
  keep <- is.finite(r); if (!is.na(rmax)) keep <- keep & (r <= rmax)
  r <- r[keep]; v <- v[keep]
  if (length(r) < 3)
    return(c(AUC = NA_real_, mean = NA_real_, max = NA_real_, min = NA_real_,
             max_min = NA_real_, frac_positive = NA_real_, peak_r = NA_real_,
             r_max_used = if (length(r)) max(r) else NA_real_, n_r = length(r)))
  o <- order(r); r <- r[o]; v <- v[o]
  vmax <- .safe(max, v); vmin <- .safe(min, v)
  pk <- if (all(!is.finite(v))) NA_real_ else r[which.max(replace(v, !is.finite(v), -Inf))]
  c(AUC = trapz(r, v), mean = .safe(mean, v), max = vmax, min = vmin,
    max_min = vmax - vmin, frac_positive = .safe(function(z) mean(z > 0), v),
    peak_r = pk, r_max_used = max(r), n_r = length(r))
}

# ---- association test ------------------------------------------------
# Matches 4.plotting/code/3.ecology.Rmd: zeros dropped, Wilcoxon.
# Effect = rank-biserial r = 2*AUC - 1: bounded [-1,1], sign-interpretable,
# comparable across metrics on different scales (required for the dot plot).
assoc_test <- function(v, g, drop_zeros = TRUE) {
  ok <- is.finite(v) & !is.na(g)
  if (drop_zeros) ok <- ok & (v != 0)
  v <- v[ok]; g <- g[ok]
  a <- v[g == 1]; b <- v[g == 0]
  if (length(a) < 3 || length(b) < 3 || length(unique(c(a, b))) < 2)
    return(tibble(n1 = length(a), n0 = length(b), effect = NA_real_,
                  p = NA_real_, auc = NA_real_))
  w <- suppressWarnings(stats::wilcox.test(a, b, exact = FALSE))
  auc <- as.numeric(w$statistic) / (length(a) * length(b))
  tibble(n1 = length(a), n0 = length(b), effect = 2 * auc - 1,
         p = w$p.value, auc = auc)
}

# ---- tidy output -----------------------------------------------------
as_tidy <- function(wsi, metric, feature, cell, stratum,
                    param_name, param_label, param_value, value, extra = NULL) {
  out <- tibble(wsi = wsi, metric = metric, feature = feature, cell = cell,
                stratum = stratum, param_name = param_name,
                param_label = param_label, param_value = param_value, value = value)
  if (!is.null(extra)) out <- bind_cols(out, extra)
  out
}

write_shard <- function(df, out_dir, stem) {
  tid <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID", "1")); if (is.na(tid)) tid <- 1L
  f <- file.path(out_dir, "raw", sprintf("%s_shard%03d.RDS", stem, tid))
  saveRDS(df, f); message(sprintf("[write] %s (%d rows)", f, nrow(df))); invisible(f)
}

gather_shards <- function(out_dir, stem) {
  fs <- list.files(file.path(out_dir, "raw"),
                   pattern = sprintf("^%s_shard.*\\.RDS$", stem), full.names = TRUE)
  if (!length(fs)) stop("no shards for ", stem)
  df <- bind_rows(lapply(fs, readRDS))
  f <- file.path(out_dir, sprintf("%s.RDS", stem)); saveRDS(df, f)
  message(sprintf("[gather] %d shards -> %s (%d rows)", length(fs), f, nrow(df)))
  df
}

load_sweep <- function(out_dir, stem) {
  f <- file.path(out_dir, sprintf("%s.RDS", stem))
  if (file.exists(f)) return(readRDS(f))
  r <- try(gather_shards(out_dir, stem), silent = TRUE)
  if (inherits(r, "try-error")) return(NULL)
  r
}

# ---- paired association test (intra-slide design) --------------------
# Matches 4.plotting/code/3.ecology.Rmd's analyze_attention_pairs():
# paired Wilcoxon signed-rank on attn0 vs attn1 values from the SAME
# slide, both non-zero and non-NA. Effect size here is the matched-pairs
# rank-biserial correlation (bounded [-1,1]), not the published script's
# own ad hoc "Z/sqrt(N)" approximation - kept for axis consistency with
# the unpaired assoc_test() so both designs sit on one comparable scale.
assoc_test_paired <- function(a, b, drop_zeros = TRUE) {
  ok <- is.finite(a) & is.finite(b)
  if (drop_zeros) ok <- ok & (a != 0) & (b != 0)
  a <- a[ok]; b <- b[ok]
  if (length(a) < 3) return(tibble(n_pairs = length(a), effect = NA_real_, p = NA_real_))
  d <- a - b
  nz <- d[d != 0]
  if (length(nz) < 3) return(tibble(n_pairs = length(a), effect = NA_real_, p = NA_real_))
  w <- suppressWarnings(stats::wilcox.test(a, b, paired = TRUE, exact = FALSE))
  r <- rank(abs(nz))
  effect <- (sum(r[nz > 0]) - sum(r[nz < 0])) / sum(r)
  tibble(n_pairs = length(a), effect = effect, p = w$p.value)
}

# ---- clinical join ---------------------------------------------------
norm01 <- function(x) {
  if (is.numeric(x)) return(as.integer(x))
  as.integer(dplyr::recode(as.character(x), "CO" = 1L, "NCO" = 0L,
                           "Aneuploid" = 1L, "Diploid" = 0L, .default = NA_integer_))
}

get_slide_clinical <- function(published_path) {
  m <- as.data.frame(readRDS(published_path))
  tibble(wsi = m$wsi, progression = norm01(m$progression), pred = norm01(m$pred),
         dataset = as.character(m$dataset))
}
