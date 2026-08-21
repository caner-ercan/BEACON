# =====================================================================
# 13 - Local Moran's I, computed EXACTLY the way the publication does it.
#
# Scoped fix for today's deliverable only (CE, 2026-08-11): "do the
# analysis exactly the way it was done in the original publication...
# I'll fix the [02_moran_getis_sweep.R] bug later." Covers exactly the
# 2 metrics in the dual-design figure that need it (Moran_local x
# Immune/LymphoPlasma). Global Moran and Ripley are unaffected by the
# bug (independently reverified) and are NOT recomputed here.
#
# THE BUG, for the record: 02_moran_getis_sweep.R filters tiles to the
# stratum BEFORE building the neighbourhood graph, so the local statistic
# is computed on a stratum-RESTRICTED neighbourhood. Verified against
# 4.plotting/code/2.5.moran_241224.rmd (via ecology_code, same file):
# lines 138 vs 152-159 show the published pipeline computes local Moran's
# I ONCE per slide using ALL tiles (unrestricted neighbourhood), and only
# AFTERWARD splits the resulting PER-TILE Ii values by each tile's own
# attn_class_extended membership to build the attn0/attn1 "clustered_mean"
# summaries (mean of positive Ii within that class's tiles). Filtering
# before vs after changes which tiles are NEIGHBOURS of which, not just
# which tiles get averaged - that is what broke reproduction (cor 0.25-0.63
# instead of 1.000, verified against the published non-zero columns).
#
# SECOND discrepancy found after fixing the first (cor rose from ~0.3 to
# only ~0.6, not 1.0): calculate_localMoran() in the same source file,
# line 84, does `local_moran_Ii <- ifelse(p_value <= 0.05, Ii, 0)` -
# EVERY tile's own local Moran value is zeroed out if that tile's own
# local p-value is not significant, BEFORE the attn0/attn1 split. This
# is implemented directly below (not via lib_sweep.R's local_moran(),
# which discards p-values) since it is specific to this exact replication.
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

CELLS <- c("Immune" = "Num.Immune", "LymphoPlasma" = "Num.LymphoPlasma")
RINGS <- D2_RINGS_PX  # ring1_queen=200, ring2=350, ring3=500 (px), same grid as before

# calculate_localMoran(), lines 66-91: localmoran() with style="B",
# zero.policy=TRUE, then Ii zeroed wherever that tile's own p (column 5,
# the two-sided Pr(z != E(Ii))) exceeds 0.05.
local_moran_pgated <- function(X, Y, x, d2) {
  lw <- make_listw(X, Y, d2, "B")
  if (is.null(lw)) return(NULL)
  r <- tryCatch(suppressWarnings(spdep::localmoran(x, lw, zero.policy = TRUE)), error = function(e) NULL)
  if (is.null(r)) return(NULL)
  Ii <- as.numeric(r[, 1]); pv <- as.numeric(r[, 5])
  ifelse(is.finite(pv) & pv <= 0.05, Ii, 0)
}

# ---- per-slide: compute Ii ONCE on ALL tiles, then split by the tile's
# OWN attn_class_extended for the attn0/attn1 summaries. ----------------
run_slide <- function(d, d2) {
  purrr::map_dfr(names(CELLS), function(cell_label) {
    col <- CELLS[[cell_label]]
    x <- d[[col]]
    if (all(x == x[1])) return(NULL)  # zero variance -> Ii undefined

    Ii <- local_moran_pgated(d$X, d$Y, x, d2)  # full, unrestricted neighbourhood, p-gated
    if (is.null(Ii)) return(NULL)

    attn <- as_attn_numeric(d[[ATTN_COLUMN]])
    is_abn <- attn == 1

    s1 <- summarise_moran(Ii[is_abn])   # attn1: this tile's Ii, among abnormal tiles
    s0 <- summarise_moran(Ii[!is_abn])  # attn0: this tile's Ii, among normal tiles

    dplyr::bind_rows(
      as_tidy(d$wsi[1], "Moran_local", names(s1), cell_label, "abnormal",
             "d2_px", NA_character_, d2, unname(s1)),
      as_tidy(d$wsi[1], "Moran_local", names(s0), cell_label, "normal",
             "d2_px", NA_character_, d2, unname(s0))
    )
  })
}

out <- purrr::imap_dfr(as.list(RINGS), function(d2, ring_label) {
  message(sprintf("[run] %s (d2=%d px), full-neighbourhood then split", ring_label, d2))
  by_wsi <- split(region, region$wsi)
  res <- furrr::future_map_dfr(by_wsi, run_slide, d2 = d2, .options = furrr::furrr_options(seed = TRUE))
  res$param_label <- ring_label
  res
})

saveRDS(out, file.path(OUT_DIR, "local_moran_corrected.RDS"))
write.csv(out, file.path(OUT_DIR, "local_moran_corrected.csv"), row.names = FALSE)
message(sprintf("[done] %d rows", nrow(out)))

# ---- reproduction check, at the published ring (200 px) ---------------
pub <- as.data.frame(readRDS(PATHS$published)); names(pub) <- iconv(names(pub), "UTF-8", "ASCII", sub = "u")
check <- function(cell_label, stratum, pubcol) {
  x <- out %>% filter(cell == cell_label, stratum == !!stratum, param_label == "ring1_queen",
                      feature == "clustered_mean")
  j <- merge(x[, c("wsi", "value")], pub[, c("wsi", pubcol)], by = "wsi")
  names(j)[3] <- "pub"
  nz <- j[j$pub != 0 & is.finite(j$value), ]
  cat(sprintf("%-55s n=%3d cor=%+.6f exact=%d/%d\n", pubcol, nrow(nz),
              if (nrow(nz) > 2) cor(nz$value, nz$pub) else NA,
              sum(abs(nz$value - nz$pub) < 1e-9), nrow(nz)))
}
message("\n=== reproduction check (must be ~1.0, not the 0.25-0.38 the buggy version gave) ===")
check("Immune", "abnormal", "Moran_Local_Immune_Ii_clustered_mean_attn1")
check("Immune", "normal",   "Moran_Local_Immune_Ii_clustered_mean_attn0")
check("LymphoPlasma", "abnormal", "Moran_Local_LymphoPlasma_Ii_clustered_mean_attn1")
check("LymphoPlasma", "normal",   "Moran_Local_LymphoPlasma_Ii_clustered_mean_attn0")
