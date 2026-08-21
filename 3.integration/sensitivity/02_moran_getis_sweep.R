# =====================================================================
# 02 - Moran's I and Getis-Ord Gi*: sensitivity to the neighbourhood ring
#
# Confirmed conventions (from ecology_code/, 2026-07-31):
#   * coordinates are PIXELS (X, Y on the 112 px tile lattice) - the
#     published d2 = 200 is 200 PIXELS (~91-100 um), not 200 um
#   * d2 = 200 px is exactly the 8-neighbour QUEEN ring; intermediate
#     distances give identical neighbour sets, so the grid is RINGS
#   * localG(..., GeoDa = TRUE)
#   * slide summaries are SIGN-BASED - no p-value threshold anywhere
#   * stratification on attn_class_extended
#
# Cost: 444k tiles / 777 slides, median ~500 tiles per slide. Cheap.
# SLURM: array over (ring x style x stratum); slides parallel within task.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R")); source(file.path(here, "lib_sweep.R"))
suppressPackageStartupMessages(library(spdep))
setup_parallel()

region <- as.data.frame(readRDS(PATHS$region))
region <- region[!is.na(region$X) & !is.na(region$Y), ]
for (cc in CELL_TYPES_REGION) if (cc %in% names(region)) region[[cc]][is.na(region[[cc]])] <- 0
message(sprintf("[data] %d tiles / %d slides", nrow(region), length(unique(region$wsi))))

work <- expand.grid(ring = names(D2_RINGS_PX), style = MORAN_STYLES,
                    stratum = names(STRATA), stringsAsFactors = FALSE)
work$d2_px <- unname(D2_RINGS_PX[work$ring])
my_work <- shard(seq_len(nrow(work)))
message(sprintf("[work] task %s/%s -> %d of %d settings",
                attr(my_work, "task_id"), attr(my_work, "n_tasks"),
                length(my_work), nrow(work)))

run_setting <- function(i) {
  w <- work[i, ]; d2 <- w$d2_px; style <- w$style; st <- STRATA[[w$stratum]]
  message(sprintf("[run] %s (d2=%d px) style=%s stratum=%s", w$ring, d2, style, w$stratum))
  dat <- filter_stratum(region, st, ATTN_COLUMN, ATTN_THRESHOLD)
  if (!nrow(dat)) return(NULL)
  by_wsi <- split(dat, dat$wsi)

  furrr::future_map_dfr(names(by_wsi), function(wn) {
    d <- by_wsi[[wn]]
    if (nrow(d) < 5) return(NULL)
    purrr::map_dfr(CELL_TYPES_REGION, function(cc) {
      if (!cc %in% names(d)) return(NULL)
      x <- d[[cc]]; cell <- sub("^Num\\.", "", cc)
      if (all(x == x[1])) return(NULL)          # zero variance -> undefined
      rows <- list()

      gi <- local_gi(d$X, d$Y, x, d2, style, GETIS_GEODA)
      if (!is.null(gi)) {
        s <- summarise_gi(gi$z)
        rows[[length(rows) + 1]] <- as_tidy(wn, "GetisOrd_Gi", names(s), cell, w$stratum,
          "d2_px", w$ring, d2, unname(s),
          tibble(style = style, n_nb_median = gi$n_nb_median))
      }
      mo <- local_moran(d$X, d$Y, x, d2, style)
      if (!is.null(mo)) {
        s <- summarise_moran(mo$Ii)
        rows[[length(rows) + 1]] <- as_tidy(wn, "Moran_local", names(s), cell, w$stratum,
          "d2_px", w$ring, d2, unname(s),
          tibble(style = style, n_nb_median = mo$n_nb_median))
      }
      # global statistics - GlobalMoran_Plasma_attn1 is the anchor claim
      rows[[length(rows) + 1]] <- as_tidy(wn, "Global",
        c("Moran_I", "Getis_G"), cell, w$stratum, "d2_px", w$ring, d2,
        c(global_moran(d$X, d$Y, x, d2, style), global_getis(d$X, d$Y, x, d2, style)),
        tibble(style = style, n_nb_median = NA_real_))

      bind_rows(rows)
    })
  }, .options = furrr::furrr_options(seed = TRUE))
}

out <- purrr::map_dfr(my_work, run_setting)
write_shard(out, OUT_DIR, "sweep_moran_getis")
message("[done] 02_moran_getis_sweep")
