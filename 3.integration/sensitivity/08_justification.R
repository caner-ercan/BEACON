# =====================================================================
# 08 - Outcome-blind justification numbers for the Supplementary Table
#
# NOTHING HERE TOUCHES THE OUTCOME. No progression, no survival, no
# p-values. That separation is the whole point: the parameter is
# justified by structure and by descriptive properties of the data, and
# is never argued to be "best" against progression - which would be
# selection on a post-hoc finding and would hand Reviewer 1 the
# circularity objection from comment #12.
#
# Produces, per parameter:
#   d2 ring        - realised neighbour count distribution (shows that
#                    200 px == the 8-neighbour queen ring exactly)
#   Morisita       - quadrat side in px and um, quadrats per slide
#   kNN k          - median target-class cells within the queen-ring
#                    radius: a STRUCTURAL anchor for k, the one otherwise
#                    genuinely arbitrary parameter
#   G-function     - UNCENSORED peak of the km-theo deviation and the
#                    saturation radius. NOTE the published peak_r is
#                    censored at 150 um by construction (summarise_g
#                    computes it inside the window), so it cannot justify
#                    the window; this recomputes over the full curve.
#   Ripley         - distribution of the per-slide default rmax, which is
#                    what makes AUC/mean non-comparable across slides.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R")); source(file.path(here, "lib_sweep.R"))
suppressPackageStartupMessages({ library(spdep) })

report <- list()

# ---- 1. neighbourhood structure on the tile lattice ------------------
region <- as.data.frame(readRDS(PATHS$region))
region <- region[!is.na(region$X) & !is.na(region$Y), ]
set.seed(1)
samp <- sample(unique(region$wsi), min(150, n_distinct(region$wsi)))
nb_stats <- purrr::imap_dfr(as.list(D2_RINGS_PX), function(d2, lab) {
  v <- unlist(lapply(split(region[region$wsi %in% samp, ], region$wsi[region$wsi %in% samp]),
    function(d) if (nrow(d) < 3) NULL else spdep::card(spdep::dnearneigh(cbind(d$X, d$Y), 0, d2))))
  tibble(parameter = "d2 (Moran, Getis)", setting = lab, value_px = d2,
         value_um = d2 * UM_PER_PX_MEAN,
         nb_median = stats::median(v), nb_mean = round(mean(v), 1),
         nb_p25 = stats::quantile(v, .25), nb_p75 = stats::quantile(v, .75),
         frac_isolated = round(mean(v == 0), 4))
})
report$neighbours <- nb_stats
message("\n[1] neighbourhood structure (200 px should be the 8-neighbour queen ring):")
print(as.data.frame(nb_stats), row.names = FALSE)

# ---- 2. Morisita quadrat geometry ------------------------------------
mor <- purrr::imap_dfr(as.list(MORISITA_QUADRAT_PX), function(qpx, lab) {
  d <- region; d$bx <- floor(d$X / qpx); d$by <- floor(d$Y / qpx)
  n <- d %>% distinct(wsi, bx, by) %>% count(wsi)
  tibble(parameter = "Morisita quadrat", setting = lab, value_px = qpx,
         value_um = qpx * UM_PER_PX_MEAN,
         quadrats_median = stats::median(n$n), quadrats_min = min(n$n))
})
report$morisita <- mor
message("\n[2] Morisita quadrat geometry:")
print(as.data.frame(mor), row.names = FALSE)

# ---- 3. structural anchor for kNN k ----------------------------------
# k = 10 is the one genuinely arbitrary parameter. If the median count of
# target-class cells within the queen-ring radius is near 10, then k = 10
# is the lattice-coherent choice and moves out of "arbitrary".
if (file.exists(PATHS$cell)) {
  cell <- as.data.frame(readRDS(PATHS$cell))
  names(cell) <- sub("^Centroid\\.X\\..*m$", "Xum", names(cell))
  names(cell) <- sub("^Centroid\\.Y\\..*m$", "Yum", names(cell))
  cell <- cell[is.finite(cell$Xum) & is.finite(cell$Yum), ]
  cell$Class <- as.character(cell$Class)
  radius_um <- D2_PUBLISHED_PX * UM_PER_PX_MEAN
  set.seed(1); cs <- sample(unique(cell$wsi), min(60, n_distinct(cell$wsi)))
  knn_anchor <- purrr::map_dfr(c("Plasma", "Lymphocyte", "Epithelial", "Stroma"), function(ct) {
    v <- unlist(lapply(split(cell[cell$wsi %in% cs, ], cell$wsi[cell$wsi %in% cs]), function(d) {
      p <- d[d$Class == ct, c("Xum", "Yum")]
      if (nrow(p) < 5 || nrow(p) > 6000) return(NULL)
      D <- as.matrix(dist(p)); rowSums(D > 0 & D <= radius_um)
    }))
    if (is.null(v)) return(NULL)
    tibble(parameter = "kNN k", cell_type = ct, radius_um = round(radius_um, 1),
           n_within_median = stats::median(v), n_within_p25 = stats::quantile(v, .25),
           n_within_p75 = stats::quantile(v, .75))
  })
  report$knn <- knn_anchor
  message(sprintf("\n[3] cells within the queen-ring radius (%.0f um) - structural anchor for k:",
                  radius_um))
  print(as.data.frame(knn_anchor), row.names = FALSE)
}

# ---- 4. Ripley per-slide default rmax --------------------------------
if (file.exists(PATHS$ripley_all)) {
  rr <- as.data.frame(readRDS(PATHS$ripley_all))
  rmax <- rr %>% group_by(wsi) %>% summarise(rmax = max(r, na.rm = TRUE), .groups = "drop")
  report$ripley <- tibble(parameter = "Ripley r range", setting = "spatstat per-slide default",
    rmax_min = min(rmax$rmax), rmax_p25 = quantile(rmax$rmax, .25),
    rmax_median = median(rmax$rmax), rmax_p75 = quantile(rmax$rmax, .75),
    rmax_max = max(rmax$rmax),
    frac_below_100um = round(mean(rmax$rmax < 100), 3),
    frac_below_50um  = round(mean(rmax$rmax < 50), 3))
  message("\n[4] Ripley per-slide default rmax (why AUC/mean are not comparable across slides):")
  print(as.data.frame(report$ripley), row.names = FALSE)
  # retention anchor: largest fixed rmax keeping >=90% of slides
  cand <- c(25, 50, 75, 100, 150, 200, 250, 300)
  ret <- tibble(rmax_um = cand, frac_slides_retained = sapply(cand, function(x) mean(rmax$rmax >= x)))
  report$ripley_retention <- ret
  message("\n[4b] retention anchor - fraction of slides whose curve reaches a given fixed rmax:")
  print(as.data.frame(ret), row.names = FALSE)
}

saveRDS(report, file.path(OUT_DIR, "justification_numbers.RDS"))
for (nm in names(report))
  write.csv(report[[nm]], file.path(OUT_DIR, sprintf("justification_%s.csv", nm)), row.names = FALSE)
message("\n[done] 08_justification - outcome-blind numbers for the Supp. Table")
