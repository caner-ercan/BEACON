# =====================================================================
# 01 - Ripley's L: sensitivity to the r range
#
# FREE: no recomputation. The full L(r)/K(r) curves were cached by the
# original pipeline, and re-summarising them at rmax = full range
# reproduces the published Ripley_L12_*_AUC features EXACTLY (verified,
# cor = 1.0000, maxdiff = 0 on the all-tissue stratum). So this sweep
# runs against the genuinely published pipeline, not a re-implementation.
#
# Published behaviour: spatstat was called WITHOUT an explicit r, so each
# slide was integrated over its own default range (observed rmax spans
# 2.8-450 um). Imposing a common range across slides is the more
# principled choice; that fix falls out of this sweep for free.
#
# Runs in minutes on a laptop. No SLURM needed.
# =====================================================================

args <- commandArgs(trailingOnly = TRUE)
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R"))
source(file.path(here, "lib_sweep.R"))

STRATUM_FILES <- c(all = PATHS$ripley_all, abnormal = PATHS$ripley_a1, normal = PATHS$ripley_a0)

sweep_one_stratum <- function(stratum_name, path) {
  if (!file.exists(path)) {
    warning("missing cached curves: ", path); return(NULL)
  }
  message(sprintf("[ripley] %s <- %s", stratum_name, basename(path)))
  res <- as.data.frame(readRDS(path))
  if (!"wsi" %in% names(res) && "Image" %in% names(res)) res$wsi <- substr(res$Image, 1, 5)

  lcols <- grep("^Ripley_L12_", names(res), value = TRUE)
  if (!length(lcols)) { warning("no L12 columns in ", path); return(NULL) }

  by_wsi <- split(res, res$wsi)
  message(sprintf("[ripley]   %d slides, %d cell pairs", length(by_wsi), length(lcols)))

  out <- purrr::imap_dfr(by_wsi, function(d, w) {
    d <- d[order(d$r), ]
    purrr::map_dfr(lcols, function(cc) {
      pair <- sub("^Ripley_L12_", "", cc)
      purrr::imap_dfr(as.list(RIPLEY_RMAX_UM), function(rmax, lab) {
        s <- summarise_curve(d$r, d[[cc]], rmax)
        as_tidy(
          wsi = w, metric = "Ripley_L", feature = names(s), cell = pair,
          stratum = stratum_name, param_name = "rmax_um",
          param_label = lab,
          param_value = ifelse(is.na(rmax), NA_real_, rmax),
          value = unname(s)
        )
      })
    })
  })
  out
}

all_out <- purrr::imap_dfr(as.list(STRATUM_FILES), function(p, s) sweep_one_stratum(s, p))

# --- reproduction check against the published feature table -----------
if (file.exists(PATHS$published)) {
  pub <- as.data.frame(readRDS(PATHS$published))
  chk <- all_out %>%
    filter(stratum == "all", feature == "AUC", param_label == "default") %>%
    select(wsi, cell, value) %>%
    mutate(pubcol = paste0("Ripley_L12_", cell, "_AUC"))
  cors <- chk %>%
    group_by(cell, pubcol) %>%
    group_modify(function(g, k) {
      if (!k$pubcol[1] %in% names(pub)) return(tibble::tibble(cor = NA_real_, n = 0L))
      j <- merge(g, pub[, c("wsi", k$pubcol[1])], by = "wsi")
      names(j)[ncol(j)] <- "pubval"
      j <- j[complete.cases(j$value, j$pubval), ]
      tibble::tibble(cor = if (nrow(j) > 2) cor(j$value, j$pubval) else NA_real_, n = nrow(j))
    }) %>% ungroup()
  message("\n[validate] reproduction of published *_AUC at the published (default) r range:")
  print(as.data.frame(cors), row.names = FALSE)
  bad <- cors$cor[is.finite(cors$cor)] < 0.9999
  if (any(bad)) warning("some Ripley features did not reproduce exactly - inspect before using")
}

saveRDS(all_out, file.path(OUT_DIR, "sweep_ripley.RDS"))
message(sprintf("\n[done] %s  (%d rows)", file.path(OUT_DIR, "sweep_ripley.RDS"), nrow(all_out)))
