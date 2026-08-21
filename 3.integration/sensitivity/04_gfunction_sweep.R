# =====================================================================
# 04 - Distance-based G-function: sensitivity to the summary window
#
# This is the only sweep that has to touch the cell-level point patterns
# again, because the original pipeline cached only the four scalar
# summaries and not the Gcross curves.
#
# Scale: 777 slides x 9 pairs x 3 strata ~ 21k Gcross calls on patterns of
# median 6.4k cells. Kcross - which costs MORE than Gcross - already ran
# over this same data, so this is a short parallel job.
#
# The curves ARE cached this time (--save-curves, default on), so any
# future window sweep is free.
#
# CONFIRMED CONVENTIONS (2.7.nn_gfunction.Rmd):
#   * cell-level MICRON coordinates - so the window is genuinely 150 um,
#     unlike the tile-level d2 which is in pixels
#   * Gcross(..., correction = "km")
#   * summarise_g(g_cross, short_range = 50, medium_range = 150) declares
#     both arguments and then IGNORES them, hardcoding r <= 150 in the
#     body. short_range = 50 is therefore an author-intended value and is
#     the natural low end of the grid.
#   * the published peak_r is CENSORED at 150 by construction (computed
#     inside the window), so it cannot be used to justify the window.
#     This script records the peak over the FULL curve as well.
#
# SLURM: array over slide chunks; pairs/strata run inside each task.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R"))
source(file.path(here, "lib_sweep.R"))
suppressPackageStartupMessages(library(spatstat))

SAVE_CURVES <- !("--no-save-curves" %in% commandArgs(TRUE))
CURVE_DIR   <- file.path(OUT_DIR, "gcross_curves")
if (SAVE_CURVES) dir.create(CURVE_DIR, recursive = TRUE, showWarnings = FALSE)

# NOTE ordering: the cell table is loaded and SHARDED BEFORE the parallel
# workers are created. furrr exports globals to every worker, so holding
# the full 5.9M-row table here would put one copy in each of N workers
# (tens of GB at high core counts). After sharding, a task holds only its
# own slides, and each worker receives just one slide's rows.
cell <- as.data.frame(readRDS(PATHS$cell))
names(cell) <- sub("^Centroid\\.X\\..*m$", "Xum", names(cell))
names(cell) <- sub("^Centroid\\.Y\\..*m$", "Yum", names(cell))
cell <- cell[is.finite(cell$Xum) & is.finite(cell$Yum), ]
cell$Class <- as.character(cell$Class)
keep_cols <- intersect(c("wsi", "Xum", "Yum", "Class", ATTN_COLUMN_CELL), names(cell))
cell <- cell[, keep_cols, drop = FALSE]
message(sprintf("[data] %d cells / %d slides", nrow(cell), length(unique(cell$wsi))))

augment_classes <- function(d) {
  lp <- d[d$Class %in% c("Lymphocyte", "Plasma"), ];                 if (nrow(lp)) lp$Class <- "Lymph.Plasma"
  im <- d[d$Class %in% c("Lymphocyte", "Plasma", "OtherImmune"), ];  if (nrow(im)) im$Class <- "Immune"
  dplyr::bind_rows(d, lp, im)
}

make_ppp <- function(d) {
  win <- spatstat.geom::owin(xrange = range(d$Xum), yrange = range(d$Yum))
  spatstat.geom::ppp(x = d$Xum, y = d$Yum, marks = factor(d$Class),
                     window = win, checkdup = FALSE)
}

wsis <- sort(unique(cell$wsi))
my_wsis <- shard(wsis)
message(sprintf("[work] task %s/%s -> %d of %d slides",
                attr(my_wsis, "task_id"), attr(my_wsis, "n_tasks"),
                length(my_wsis), length(wsis)))

# Drop everything this task does not need, then split, THEN start workers.
cell <- cell[cell$wsi %in% my_wsis, , drop = FALSE]
cell_by_wsi <- split(cell, cell$wsi)
rm(cell); invisible(gc())
message(sprintf("[work] holding %d slides in this task", length(cell_by_wsi)))

setup_parallel()

run_wsi <- function(wname) {
  d_all <- augment_classes(cell_by_wsi[[wname]])
  curves <- list(); rows <- list()

  for (sname in names(STRATA)) {
    stratum <- STRATA[[sname]]
    # cell-level table: stratify on attention_score, not the tile-level
    # attn_class_extended (which does not exist here)
    d <- filter_stratum(d_all, stratum, ATTN_COLUMN_CELL, ATTN_THRESHOLD)
    if (nrow(d) < 20) next
    pp <- tryCatch(make_ppp(d), error = function(e) NULL)
    if (is.null(pp)) next
    mk <- spatstat.geom::marks(pp)

    for (pr in CELL_PAIRS) {
      ct1 <- pr[1]; ct2 <- pr[2]
      if (sum(mk == ct1) < 3 || sum(mk == ct2) < 3) next
      g <- tryCatch(spatstat.explore::Gcross(pp, i = ct1, j = ct2, correction = "km"),
                    error = function(e) NULL)
      if (is.null(g)) next

      dev <- g$km - g$theo
      pair <- paste0(ct1, "_", ct2)
      if (SAVE_CURVES)
        curves[[length(curves) + 1]] <- tibble::tibble(
          wsi = wname, stratum = sname, cell = pair, r = g$r, km = g$km, theo = g$theo)

      # full-range AUC as published, plus the UNCENSORED peak and the
      # saturation radius (95% of plateau) - both needed by
      # 08_justification.R because the published peak_r is censored at 150.
      fin <- is.finite(g$r) & is.finite(dev)
      pk_full <- if (any(fin)) g$r[which.max(replace(dev, !fin, -Inf))] else NA_real_
      km_fin <- g$km[is.finite(g$km)]
      sat <- if (length(km_fin) > 3) {
        plateau <- max(km_fin, na.rm = TRUE)
        idx <- which(is.finite(g$km) & g$km >= 0.95 * plateau)
        if (length(idx)) g$r[idx[1]] else NA_real_
      } else NA_real_
      rows[[length(rows) + 1]] <- as_tidy(
        wsi = wname, metric = "Gfunction",
        feature = c("auc_diff", "peak_r_uncensored", "saturation_r95"), cell = pair,
        stratum = sname, param_name = "window_um", param_label = "default",
        param_value = NA_real_, value = c(trapz(g$r, dev), pk_full, sat))

      for (lab in names(GFUN_WINDOW_UM)) {
        w <- GFUN_WINDOW_UM[[lab]]
        keep <- is.finite(g$r) & g$r <= w
        dv <- dev[keep]; rr <- g$r[keep]
        if (sum(is.finite(dv)) < 3) next
        rows[[length(rows) + 1]] <- as_tidy(
          wsi = wname, metric = "Gfunction",
          feature = c("mean_dev", "max_dev", "peak_r", "auc_diff"), cell = pair,
          stratum = sname, param_name = "window_um", param_label = lab, param_value = w,
          value = c(mean(dv, na.rm = TRUE),
                    max(dv, na.rm = TRUE),
                    rr[which.max(replace(dv, !is.finite(dv), -Inf))],
                    trapz(rr, dv)))
      }
    }
  }

  if (SAVE_CURVES && length(curves))
    saveRDS(dplyr::bind_rows(curves), file.path(CURVE_DIR, sprintf("gcross_%s.RDS", wname)))
  if (length(rows)) dplyr::bind_rows(rows) else NULL
}

out <- furrr::future_map_dfr(names(cell_by_wsi), run_wsi,
                             .options = furrr::furrr_options(seed = TRUE),
                             .progress = TRUE)
write_shard(out, OUT_DIR, "sweep_gfunction")
message("[done] 04_gfunction_sweep")
