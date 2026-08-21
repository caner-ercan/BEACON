# =====================================================================
# 03 - k-nearest neighbours: sensitivity to k
#
# Replicates the original calculate_mean_knn_roi() exactly (FNN::get.knnx,
# query = cell_type2 against data = cell_type1, mean over all query points
# and all k neighbours, and the same k > n fallback), then sweeps k.
#
# The original attention filter is ASYMMETRIC - it restricts cell_type1 to
# the stratum but leaves cell_type2 unrestricted. That is kept here so the
# published setting reproduces; it is documented in the supplementary
# table rather than silently changed.
#
# k = 10 is the ONE genuinely arbitrary parameter in the pipeline: a round
# number with no structural or spatial-statistics basis (10-fold CV is a
# real convention, but it does not transfer to a nearest-neighbour
# distance measure). Reviewer 1 names kNN explicitly, so it is swept.
# 08_justification.R computes a structural anchor - the median count of
# target-class cells within the queen-ring radius - which, if near 10,
# would move k out of "arbitrary".
#
# k is reported alongside the median distance it spans, so that k maps onto
# the same physical scale as the distance grid used elsewhere.
#
# SLURM: array over strata; slides parallelised within the task.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R"))
source(file.path(here, "lib_sweep.R"))
suppressPackageStartupMessages(library(FNN))

setup_parallel()

cell <- as.data.frame(readRDS(PATHS$cell))
names(cell) <- sub("^Centroid\\.X\\..*m$", "Xum", names(cell))
names(cell) <- sub("^Centroid\\.Y\\..*m$", "Yum", names(cell))
stopifnot(all(c("Xum", "Yum", "Class", "Image", "wsi") %in% names(cell)))
cell <- cell[is.finite(cell$Xum) & is.finite(cell$Yum), ]
cell$Class <- as.character(cell$Class)
message(sprintf("[data] %d cells / %d slides", nrow(cell), length(unique(cell$wsi))))

# Composite classes, added as duplicate rows (as in the original).
augment_classes <- function(d) {
  lp <- d[d$Class %in% c("Lymphocyte", "Plasma"), ];                     if (nrow(lp)) lp$Class <- "Lymph.Plasma"
  im <- d[d$Class %in% c("Lymphocyte", "Plasma", "OtherImmune"), ];      if (nrow(im)) im$Class <- "Immune"
  dplyr::bind_rows(d, lp, im)
}

work <- expand.grid(stratum = names(STRATA), stringsAsFactors = FALSE)
my_work <- shard(seq_len(nrow(work)))

run_stratum <- function(i) {
  sname <- work$stratum[i]; stratum <- STRATA[[sname]]
  message(sprintf("[run] knn stratum=%s", sname))
  by_wsi <- split(cell, cell$wsi)

  furrr::future_map_dfr(names(by_wsi), function(wname) {
    d0 <- augment_classes(by_wsi[[wname]])
    purrr::map_dfr(CELL_PAIRS, function(pr) {
      ct1 <- pr[1]; ct2 <- pr[2]

      # asymmetric attention filter, as published: only cell_type1 is
      # restricted to the stratum. Cell-level tables carry attention_score,
      # not the tile-level attn_class_extended (see ATTN_COLUMN_CELL).
      d <- d0
      if (stratum != 2) {
        v <- d[[ATTN_COLUMN_CELL]]
        if (is.null(v)) stop("attention column not found in cell table: ", ATTN_COLUMN_CELL)
        vn <- as_attn_numeric(v)
        uv <- unique(vn[!is.na(vn)])
        hi <- if (length(uv) > 0 && all(uv %in% c(0, 1))) vn == 1 else vn >= ATTN_THRESHOLD
        hi[is.na(hi)] <- FALSE
        drop <- (d$Class == ct1) & (if (stratum == 1) !hi else hi)
        d <- d[!drop, , drop = FALSE]
      }

      c1 <- as.matrix(d[d$Class == ct1, c("Xum", "Yum")])
      c2 <- as.matrix(d[d$Class == ct2, c("Xum", "Yum")])
      min_len <- min(nrow(c1), nrow(c2))
      if (min_len < 2) return(NULL)

      purrr::map_dfr(KNN_K, function(k) {
        k_eff <- min(k, min_len - 1L, nrow(c1))
        if (k_eff < 1) return(NULL)
        nnd <- tryCatch(FNN::get.knnx(c1, c2, k = k_eff)$nn.dist, error = function(e) NULL)
        if (is.null(nnd)) return(NULL)
        as_tidy(
          wsi = wname, metric = "kNN", feature = c("mean_knn", "median_knn"),
          cell = paste0(ct1, "_", ct2), stratum = sname,
          param_name = "k", param_label = paste0("k", k), param_value = k,
          value = c(mean(nnd), stats::median(nnd)),
          extra = tibble::tibble(k_effective = k_eff, n_ct1 = nrow(c1), n_ct2 = nrow(c2))
        )
      })
    })
  }, .options = furrr::furrr_options(seed = TRUE))
}

out <- purrr::map_dfr(my_work, run_stratum)
write_shard(out, OUT_DIR, "sweep_knn")
message("[done] 03_knn_sweep")
