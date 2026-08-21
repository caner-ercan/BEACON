# =====================================================================
# 05 - Morisita-Horn: sensitivity to quadrat size
#
# Confirmed conventions:
#   * the published quadrat is 224 px (2x2 tiles = ONE non-overlapping FM
#     window), built by expand_tiles() in 1.1.cell_dist_241106.Rmd, which
#     sums counts and takes max(attention_score) per block
#   * the index is 1 - horn_morisita (OVERLAP, not dissimilarity) - this
#     flipped sense between the repo-era and the authoritative version
#   * NOTE the published pipeline emits attn1/attn2 only; the attn0
#     (DNA-normal) stratum was silently dropped by a malformed left_join
#     at 2.6.morisita.Rmd:105. All three are computed here.
#
# Grid: 112 px (one stride) / 224 px (published, one FM tile) / 448 px.
# Cheap - single node, minutes.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R")); source(file.path(here, "lib_sweep.R"))
suppressPackageStartupMessages(library(abdiv))

region <- as.data.frame(readRDS(PATHS$region))
region <- region[!is.na(region$X) & !is.na(region$Y), ]
count_cols <- intersect(c(CELL_TYPES_REGION, "Num.Epithelial"), names(region))
for (cc in count_cols) region[[cc]][is.na(region[[cc]])] <- 0
immune_cols <- setdiff(count_cols, "Num.Epithelial")
message(sprintf("[data] %d tiles / %d slides", nrow(region), length(unique(region$wsi))))

out <- purrr::imap_dfr(as.list(MORISITA_QUADRAT_PX), function(qpx, qlab) {
  message(sprintf("[run] morisita quadrat = %d px (%s)", qpx, qlab))
  d <- region
  d$bx <- floor(d$X / qpx); d$by <- floor(d$Y / qpx)
  # attn_class_extended is a FACTOR - coerce through as.character(), never
  # as.numeric() directly (that returns level indices and inverts the class)
  d$.attn <- as_attn_numeric(d[[ATTN_COLUMN]])

  # aggregate to quadrats: counts summed, attention = max (as expand_tiles does)
  agg <- d %>%
    group_by(wsi, bx, by) %>%
    summarise(across(all_of(count_cols), ~ sum(.x, na.rm = TRUE)),
              attn = max(.attn, na.rm = TRUE), .groups = "drop")

  purrr::imap_dfr(as.list(STRATA), function(st, sname) {
    a <- if (st == 2) agg else if (st == 1) agg[agg$attn == 1, ] else agg[agg$attn == 0, ]
    if (!nrow(a)) return(NULL)
    a %>%
      group_by(wsi) %>%
      group_modify(function(g, k) {
        purrr::map_dfr(immune_cols, function(cc) {
          v <- suppressWarnings(1 - abdiv::horn_morisita(g$Num.Epithelial, g[[cc]]))
          tibble(feature = "morisita_horn", cell = sub("^Num\\.", "", cc),
                 value = ifelse(is.finite(v), v, NA_real_), n_quadrats = nrow(g))
        })
      }) %>% ungroup() %>%
      transmute(wsi, metric = "Morisita_Horn", feature, cell, stratum = sname,
                param_name = "quadrat_px", param_label = qlab, param_value = qpx,
                value, quadrat_side_um = qpx * UM_PER_PX_MEAN, n_quadrats)
  })
})

saveRDS(out, file.path(OUT_DIR, "sweep_morisita.RDS"))
message(sprintf("[done] %s (%d rows)", file.path(OUT_DIR, "sweep_morisita.RDS"), nrow(out)))
