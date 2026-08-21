## ---------------------------------------------------------------------------
## rev1_unit_of_analysis / 00_config.R
##
## Shared paths, metric definitions, statistics helpers.
## Sourced by 01_slide_vs_patient.R and 02_plots.R.
##
## Reviewer 1, comment #3: the tile-level p-values are said to be "inflated by
## the high number of tiles analysed", and the reviewer asks whether the
## differences would still be seen at slide or patient level.
##
## The premise is wrong -- the published tests already run on one row per
## biopsy (777 WSIs), not per tile. This folder (a) documents that, and
## (b) does the patient-level re-test the reviewer asked for anyway.
## ---------------------------------------------------------------------------

suppressMessages({
  library(dplyr)
  library(tidyr)
  library(readxl)
})

options(width = 200, stringsAsFactors = FALSE)

## -- base folder ------------------------------------------------------------
## Set BE_MASTER to your BE_master checkout (a relative default here would
## depend on the invocation directory, not the script's location, so this
## is required rather than guessed).
base_folder <- Sys.getenv("BE_MASTER", unset = NA_character_)
if (is.na(base_folder) || !dir.exists(base_folder))
  stop("Set the BE_MASTER environment variable to your BE_master checkout.")
message("BE_master: ", base_folder)

## -- inputs -----------------------------------------------------------------
## The exact table 4.plotting/code/3.ecology.Rmd loads to draw Fig 3C/3D and
## Supp Fig 1. Unstandardised on purpose: we report raw median shifts, so the
## z-scored variant would make the effect-magnitude columns meaningless.
P_ECOLOGY <- file.path(base_folder, "3.integration", "integration_project",
                       "spatial_analysis", "tabular", "calculated",
                       "merged_ecology_250812_wNuc.RDS")

## wsi -> RandomID. Prefer the map rebuilt for the revision (rev1_clinical/01),
## which recovered the 75 slides convertion_sample.csv had left unmapped.
## Fall back to the raw clinical workbook if that map has not been built.
P_IDMAP  <- file.path(base_folder, "0.input", "organised_wsi_patient",
                      "wsi_randomid_map_rev1_260730.csv")
P_SRC_FU <- file.path(base_folder, "0.input", "Sample_data",
                      "barrett_dataset_withFUtimes.xlsx")

## -- outputs ----------------------------------------------------------------
STAMP <- "260801"
out_folder <- file.path(base_folder, "4.plotting", "output", "rev1_unit_of_analysis")
dir.create(out_folder, recursive = TRUE, showWarnings = FALSE)

P_MAIN_OUT  <- file.path(out_folder, sprintf("unit_of_analysis_%s.csv", STAMP))
P_SENS_OUT  <- file.path(out_folder, sprintf("unit_of_analysis_sensitivity_%s.csv", STAMP))
P_LONG_OUT  <- file.path(out_folder, sprintf("patient_level_values_%s.rds", STAMP))
P_FIG4C_OUT <- file.path(out_folder, sprintf("fig4c_outofscope_check_%s.csv", STAMP))

## -- metric definitions -----------------------------------------------------
## Verbatim from the `base_names` vector of the {r intrawsi} chunk in
## 4.plotting/code/3.ecology.Rmd -- these twelve are what Fig 3C, Fig 3D and
## Supp Fig 1 are drawn from. `panel` assigns each to its published panel from
## the Fig 3 legend; the eight "Supp Fig 1" assignments are inferred from the
## code and the Results text, NOT verified against the supplementary PDF
## (which is not in the repo). See README, "Known gaps".
METRICS <- tibble::tribble(
  ~key,             ~base,                                              ~panel,      ~label,
  "nucODmax",       "nuc_median_OD.Sum..Max_",                          "Fig 3C",    "Nucleus OD sum (max), median",
  "nucHaralick",    "nuc_median_Hematoxylin..Haralick.Contrast..F1._",  "Fig 3C",    "Haralick contrast (F1), median",
  "immuneDensity",  "Dens_Immune_",                                     "Fig 3D",    "Immune cell density",
  "moranLymphoPl",  "Moran_Local_LymphoPlasma_Ii_clustered_mean_",      "Fig 3D",    "Moran's I, lymphoplasmacytic",
  "nucHematoxSD",   "nuc_median_Hematoxylin..Std.dev._",                "Supp Fig 1", "Hematoxylin SD, median",
  "nucCircularity", "nuc_q75_Circularity_",                             "Supp Fig 1", "Nucleus circularity, q75",
  "nucAreaSD",      "nuc_sd_Area.uum.2_",                               "Supp Fig 1", "Nucleus area, SD",
  "nucODmedianSD",  "nuc_sd_OD.Sum..Median_",                           "Supp Fig 1", "Nucleus OD sum (median), SD",
  "globalMoranImm", "GlobalMoran_Immune_",                              "Supp Fig 1", "Global Moran's I, immune",
  "knnEpiImmune",   "meanKNN_Epithelial_Immune_",                       "Supp Fig 1", "Mean kNN, epithelial-immune",
  "moranImmune",    "Moran_Local_Immune_Ii_clustered_mean_",            "Supp Fig 1", "Moran's I, immune",
  "ripleyEpiImmune","Ripley_L12_Epithelial_Immune_frac_positive_",      "Supp Fig 1", "Ripley's L, epithelial-immune"
)

## Fig 4C metrics. Out of scope for reviewer comment #3 (which names Fig 3D and
## Supp Fig 1 only) -- computed for the record so the finding is documented,
## never reported. See README, "Out of scope".
METRICS_FIG4C <- tibble::tribble(
  ~key,               ~base,                                ~panel,   ~label,
  "giPlasmaClust",    "Gi_star_Plasma_clustering_strength_", "Fig 4C", "Getis-Ord Gi*, plasma clustering strength",
  "densPlasma",       "Dens_Plasma_",                        "Fig 4C", "Plasma cell density",
  "globalMoranPlasma","GlobalMoran_Plasma_",                 "Fig 4C", "Global Moran's I, plasma"
)

## -- loader -----------------------------------------------------------------
## Column names carry a literal micron sign that is not representable in every
## locale; normalise to ASCII up front so the scripts are locale-independent.
## Density columns ship as `Dens_X_0/1/2` while everything else is
## `..._attn0/1/2`; 3.ecology.Rmd harmonises these with the same two-step
## rename, so we repeat BOTH steps here (fixed 2026-08-13 -- the first version
## only had the second step, which silently left whole-slide/unstratified
## columns with no numeric suffix -- e.g. the bare `Ripley_L12_..._frac_positive`
## -- unrenamed to `_attn2` and therefore invisible to every `<base>attn2`
## lookup in this codebase; `Ripley_L12` does have a real attn2, it was just
## not being found before this fix. Genuinely absent for the nuclear features:
## `nuc_*` columns only ever have attn0/attn1, no whole-slide variant exists
## anywhere in the pipeline).
load_ecology <- function() {
  d <- readRDS(P_ECOLOGY)
  names(d) <- iconv(names(d), "UTF-8", "ASCII", sub = "u")
  ## Step 1 (3.ecology.Rmd): any column without a trailing digit is a
  ## whole-slide/unstratified value -- label it attn2 before the digit-suffix
  ## step runs, or it's silently left out of every attn2 lookup.
  id_cols <- c("wsi", "progression", "flow", "pred", "dataset")
  feat <- setdiff(names(d), id_cols)
  names(d)[names(d) %in% feat] <- ifelse(
    grepl("[0-9]$", feat), feat, paste0(feat, "_attn2")
  )
  ## Step 2: numeric suffix -> attn-prefixed suffix.
  names(d) <- gsub("_([0-9])$", "_attn\\1", names(d))

  if (file.exists(P_IDMAP)) {
    idmap <- read.csv(P_IDMAP)[, c("wsi", "rid")]
    names(idmap) <- c("wsi", "patient")
    message("ID map: wsi_randomid_map_rev1_260730.csv")
  } else {
    idmap <- read_excel(P_SRC_FU) %>%
      select(wsi = `Slide Label`, patient = RandomID) %>%
      distinct()
    message("ID map: barrett_dataset_withFUtimes.xlsx (rev1 map not found)")
  }

  d <- left_join(d, idmap, by = "wsi")
  if (any(is.na(d$patient)))
    warning(sum(is.na(d$patient)), " slides have no patient ID and are dropped")
  d
}

## -- statistics -------------------------------------------------------------

## Test selection replicated verbatim from `single_inter_wsi()` in
## 3.ecology.Rmd: Shapiro-Wilk on both groups, Welch t-test when both look
## normal, otherwise Mann-Whitney. Kept identical so the slide-level column
## reproduces the published p-values rather than a re-derived variant of them.
choose_test_unpaired <- function(a, b) {
  s_a <- tryCatch(shapiro.test(a), error = function(e) NULL)
  s_b <- tryCatch(shapiro.test(b), error = function(e) NULL)
  if (!is.null(s_a) && !is.null(s_b) && s_a$p.value > 0.05 && s_b$p.value > 0.05) {
    list(p = t.test(b, a, var.equal = FALSE)$p.value, test = "Welch t-test")
  } else {
    list(p = suppressWarnings(wilcox.test(b, a)$p.value), test = "Mann-Whitney U")
  }
}

## Paired equivalent, matching `single_intra_wsi()`.
choose_test_paired <- function(a, b) {
  s_a <- tryCatch(shapiro.test(a), error = function(e) NULL)
  s_b <- tryCatch(shapiro.test(b), error = function(e) NULL)
  if (!is.null(s_a) && !is.null(s_b) && s_a$p.value > 0.05 && s_b$p.value > 0.05) {
    list(p = t.test(a, b, paired = TRUE)$p.value, test = "Paired t-test")
  } else {
    list(p = suppressWarnings(wilcox.test(a, b, paired = TRUE)$p.value), test = "Wilcoxon signed-rank")
  }
}

## Cliff's delta: nonparametric, bounded [-1, 1], and the effect size that
## actually matches a Mann-Whitney test. Positive = group `a` (DNA content
## abnormal) tends to exceed group `b`. Implemented here rather than via
## {effsize} to keep the dependency list to what the cluster image already has.
cliffs_delta <- function(a, b) {
  o <- outer(a, b, "-")
  (sum(o > 0) - sum(o < 0)) / length(o)
}

## Percentile bootstrap CI for Cliff's delta.
cliffs_delta_ci <- function(a, b, R = 2000, seed = 1) {
  set.seed(seed)
  bs <- replicate(R, cliffs_delta(sample(a, replace = TRUE), sample(b, replace = TRUE)))
  unname(quantile(bs, c(0.025, 0.975), na.rm = TRUE))
}

## Matched-pairs rank-biserial correlation, the paired analogue of Cliff's
## delta. Positive = abnormal region exceeds normal region.
rank_biserial_paired <- function(a, b) {
  d <- a - b
  d <- d[d != 0]
  if (!length(d)) return(NA_real_)
  r <- rank(abs(d))
  (sum(r[d > 0]) - sum(r[d < 0])) / sum(r)
}

## Conventional Cliff's delta magnitude bands (Romano et al. 2006).
delta_magnitude <- function(d) {
  a <- abs(d)
  ifelse(is.na(a), NA_character_,
  ifelse(a < 0.147, "negligible",
  ifelse(a < 0.330, "small",
  ifelse(a < 0.474, "medium", "large"))))
}

fmt_p <- function(p) ifelse(is.na(p), NA_character_,
                            ifelse(p < 0.001, formatC(p, format = "e", digits = 1),
                                   formatC(p, format = "f", digits = 3)))

## Apply `fn` to each row of a metric table, returning the bound result.
## Avoids a {purrr} dependency for what is one rowwise map.
map_rows <- function(tbl, fn) {
  bind_rows(lapply(seq_len(nrow(tbl)), function(i) fn(as.list(tbl[i, ]))))
}
