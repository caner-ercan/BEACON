# =====================================================================
# 99 - Reproduction check at the published parameter setting.
#
# RUN THIS BEFORE TRUSTING ANY SWEEP. The entire robustness argument
# rests on one property: at the published parameter value, this code
# reproduces the published feature values. If it does not, the sweep is
# measuring a different pipeline and the rebuttal claim is void.
#
# Status as of writing:
#   Ripley  - REPRODUCES EXACTLY (cor = 1.000, all 9 cell pairs, n = 776)
#   Getis   - NOT YET REPRODUCED. The published features in
#             merged_ecology_250812.RDS are copied verbatim from
#             calculated/getis_tile112_neighbourAttn.RDS, produced by a
#             newer pipeline that is not in the repo. Best match from a
#             d2 / threshold grid search was r = 0.886, so the newer
#             script is required to pin the true setting.
#   Moran   - same situation as Getis (moran_tile112_neighbourAttn.csv).
#
# Until the newer scripts are supplied, treat Getis/Moran sweep output as
# internally consistent but NOT anchored to the published numbers.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R"))
source(file.path(here, "lib_sweep.R"))

stopifnot(file.exists(PATHS$published))
pub <- as.data.frame(readRDS(PATHS$published))
report <- list()

check <- function(label, ours, theirs) {
  ok <- is.finite(ours) & is.finite(theirs)
  if (sum(ok) < 3) return(tibble::tibble(item = label, n = sum(ok),
                                         cor = NA_real_, maxdiff = NA_real_,
                                         exact = FALSE))
  tibble::tibble(item = label, n = sum(ok),
                 cor = stats::cor(ours[ok], theirs[ok]),
                 maxdiff = max(abs(ours[ok] - theirs[ok])),
                 exact = isTRUE(all.equal(ours[ok], theirs[ok], tolerance = 1e-8)))
}

# ---- Ripley ----------------------------------------------------------
f <- file.path(OUT_DIR, "sweep_ripley.RDS")
if (file.exists(f)) {
  sw <- readRDS(f) %>%
    filter(stratum == "all", feature == "AUC", param_label == "default")
  for (pr in unique(sw$cell)) {
    pc <- paste0("Ripley_L12_", pr, "_AUC")
    if (!pc %in% names(pub)) next
    j <- merge(sw[sw$cell == pr, c("wsi", "value")], pub[, c("wsi", pc)], by = "wsi")
    report[[length(report) + 1]] <- check(paste0("Ripley/", pr), j$value, j[[pc]])
  }
} else message("[skip] run 01_ripley_sweep.R first")

# ---- Getis / Moran ---------------------------------------------------
f <- file.path(OUT_DIR, "sweep_moran_getis.RDS")
if (!file.exists(f)) f <- try(gather_shards(OUT_DIR, "sweep_moran_getis"), silent = TRUE)
if (is.character(f) && file.exists(f)) {
  sw <- readRDS(f)
  gi <- sw %>% filter(metric == "GetisOrd_Gi", stratum == "all", style == "B",
                      param_label == "published_local", z_threshold == "p05",
                      feature == "hotspot_intensity")
  for (ct in unique(gi$cell)) {
    pc <- paste0("Gi_star_", ct, "_hotspot_intensity_attn2")
    if (!pc %in% names(pub)) next
    j <- merge(gi[gi$cell == ct, c("wsi", "value")], pub[, c("wsi", pc)], by = "wsi")
    report[[length(report) + 1]] <- check(paste0("Getis/", ct), j$value, j[[pc]])
  }
  mo <- sw %>% filter(metric == "Moran_local", stratum == "all", style == "B",
                      param_label == "published_local", z_threshold == "sign",
                      feature == "clustered_mean")
  for (ct in unique(mo$cell)) {
    pc <- paste0("Moran_Local_", ct, "_Ii_clustered_mean_attn2")
    if (!pc %in% names(pub)) next
    j <- merge(mo[mo$cell == ct, c("wsi", "value")], pub[, c("wsi", pc)], by = "wsi")
    report[[length(report) + 1]] <- check(paste0("Moran/", ct), j$value, j[[pc]])
  }
}

rep_df <- dplyr::bind_rows(report)
if (!nrow(rep_df)) { message("[validate] nothing to check yet"); quit(save = "no") }

message("\n=== reproduction at the published setting ===")
print(as.data.frame(rep_df), row.names = FALSE)
utils::write.csv(rep_df, file.path(OUT_DIR, "validation_report.csv"), row.names = FALSE)

n_bad <- sum(!is.na(rep_df$cor) & rep_df$cor < 0.9999)
if (n_bad > 0) {
  message(sprintf(
    "\n[WARNING] %d/%d checks below cor 0.9999.\nDo not present these as reproducing the published pipeline until resolved.",
    n_bad, nrow(rep_df)))
} else {
  message("\n[OK] all checked features reproduce the published values.")
}
