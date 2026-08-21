# =====================================================================
# 11 - Dual-design robustness check: nuclear + immune-ecology metrics,
#      each tested INTER-slide and INTRA-slide, exactly matching the
#      published test constructions in 4.plotting/code/3.ecology.Rmd.
#
# SCOPE (CE, 2026-08-11) - 9 metrics, no Fig. 4C/4D plasma/progression
# metrics here (those are 09/10's separate deliverable, untouched):
#
#   Nuclear (5, no spatial parameter - single point per design):
#     nuc_median_OD.Sum..Max_                          "Nucleus Maximum Intensity Median"
#     nuc_median_Hematoxylin..Haralick.Contrast..F1._  "Nucleus Harlick Contrast Median"
#     nuc_median_Hematoxylin..Std.dev._
#     nuc_q75_Circularity_
#     nuc_sd_Area.uum.2_
#
#   Immune ecology (4, HAVE a swept spatial parameter - 3-point grid,
#   per CE's explicit instruction to keep the grid for these):
#     Moran_Local_LymphoPlasma_Ii_clustered_mean_  ("Lymphoplasmacytic Cluster Moran Mean")
#     GlobalMoran_Immune_
#     Moran_Local_Immune_Ii_clustered_mean_
#     Ripley_L12_Epithelial_Immune_frac_positive_
#
# Both designs were VERIFIED against 3.ecology.Rmd directly, not guessed:
#
#   INTER-slide ("DNA Content" Low/High in the published figure):
#     analyze_intergroup(), lines ~264-298. Uses the SAME `{base}attn1`
#     column for BOTH pred groups (does NOT switch to a whole-slide value
#     for pred==0 - CE's verbal description of that switch does not match
#     what the code does; resolved here in favour of the code, per CE's
#     own instruction to check it). All slides, split by pred, UNPAIRED
#     test, values == 0 dropped from each group before testing.
#
#   INTRA-slide ("Attention Region" Low/High in the published figure):
#     analyze_attention_pairs(), lines ~110-233, called at line 254 on
#     `merged_df %>% filter(pred == 1)` - i.e. DACOR-POSITIVE SLIDES ONLY.
#     PAIRED test on {base}attn0 vs {base}attn1 from the same slide, both
#     values non-zero and non-NA.
#
# The published code Shapiro-gates between a t-test and Wilcoxon per
# metric; kept uniformly Wilcoxon-based here (as in 06/09) so every row
# in the resulting figure sits on one comparable bounded effect-size axis
# - a deliberate, already-flagged simplification, not an oversight.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R")); source(file.path(here, "lib_sweep.R"))

# ---- inputs ------------------------------------------------------------
pub <- as.data.frame(readRDS(PATHS$published))
names(pub) <- iconv(names(pub), "UTF-8", "ASCII", sub = "u")
pub$pred_n <- norm01(pub$pred)
message(sprintf("[data] published table: %d slides, pred 1/0 = %d/%d",
                nrow(pub), sum(pub$pred_n == 1, na.rm = TRUE), sum(pub$pred_n == 0, na.rm = TRUE)))

sweeps <- purrr::map_dfr(c("sweep_ripley", "sweep_moran_getis"), function(s) {
  d <- load_sweep(OUT_DIR, s)
  if (is.null(d)) return(NULL)
  d
})

# Local Moran's I (metric == "Moran_local") from the buggy sweep is
# REPLACED with 13_local_moran_correct.R's output - see that script for
# the two discrepancies found (stratum-restricted neighbourhood; missing
# per-tile p-value gate) and their fix, both verified to reproduce the
# published columns exactly (cor = 1.000, 100% exact matches). Global
# Moran ("Global") and Ripley are unaffected by that bug and are left
# sourced from the original sweep.
local_corrected <- readRDS(file.path(OUT_DIR, "local_moran_corrected.RDS"))
message(sprintf("[data] local Moran replaced with corrected values: %d rows (was buggy in `sweeps`)",
                sum(sweeps$metric == "Moran_local")))
sweeps <- bind_rows(sweeps %>% filter(metric != "Moran_local"), local_corrected)

clin <- get_slide_clinical(PATHS$published)

# =========================================================================
# NUCLEAR features - read directly from the published table, no parameter.
# =========================================================================
NUCLEAR <- tibble::tribble(
  ~label,                                    ~base,
  "Nucleus Maximum Intensity Median",         "nuc_median_OD.Sum..Max_",
  "Nucleus Harlick Contrast Median",          "nuc_median_Hematoxylin..Haralick.Contrast..F1._",
  "Nucleus Hematoxylin Std.dev., median",     "nuc_median_Hematoxylin..Std.dev._",
  "Nucleus Circularity, q75",                 "nuc_q75_Circularity_",
  "Nucleus Area, sd"  ,                       "nuc_sd_Area.uum.2_"
)

run_nuclear <- function(row) {
  col1 <- paste0(row$base, "attn1"); col0 <- paste0(row$base, "attn0")
  stopifnot(col1 %in% names(pub), col0 %in% names(pub))

  # inter-slide: attn1 column, ALL slides, split by pred (matches
  # analyze_intergroup exactly - same column for both groups)
  inter <- assoc_test(pub[[col1]], pub$pred_n, DROP_ZEROS)
  inter <- dplyr::mutate(inter, design = "inter", label = row$label,
                         feature_type = "nuclear", param_label = "published", is_published = TRUE)

  # intra-slide: pred==1 only, paired attn0 vs attn1. Argument order
  # (abnormal, normal) so positive effect = higher in abnormal, matching
  # the inter-design sign convention from assoc_test() - not (normal,
  # abnormal), which would silently flip the sign relative to every other
  # row in this figure.
  d1 <- pub[pub$pred_n == 1 & !is.na(pub$pred_n), ]
  intra <- assoc_test_paired(d1[[col1]], d1[[col0]], DROP_ZEROS)
  intra <- dplyr::rename(intra, n1 = n_pairs) %>%
    dplyr::mutate(n0 = n1, design = "intra", label = row$label,
                  feature_type = "nuclear", param_label = "published", is_published = TRUE)

  dplyr::bind_rows(inter, intra)
}
nuclear_res <- purrr::map_dfr(seq_len(nrow(NUCLEAR)), function(i) run_nuclear(NUCLEAR[i, ]))
message(sprintf("[nuclear] %d rows (5 metrics x 2 designs)", nrow(nuclear_res)))

# =========================================================================
# ECOLOGY features - reuse the existing 3-value parameter sweeps.
# =========================================================================
ECOLOGY <- tibble::tribble(
  ~label,                                          ~metric,       ~feature,           ~cell,
  "Lymphoplasmacytic Cluster Moran Mean",           "Moran_local", "clustered_mean",   "LymphoPlasma",
  "Immune GlobalMoran",                             "Global",      "Moran_I",          "Immune",
  "Immune Moran, clustered mean",                   "Moran_local", "clustered_mean",   "Immune",
  "Epithelial-Immune Ripley, frac positive",        "Ripley_L",    "frac_positive",    "Epithelial_Immune"
)
PUBLISHED_PARAM <- c(d2_px = D2_PUBLISHED_PX)  # all 4 are ring-based (Moran/Global) or Ripley-default

run_ecology_inter <- function(row) {
  # inter-slide: stratum "abnormal" == the attn1-equivalent, ALL slides,
  # split by pred, at every grid setting - i.e. exactly what 09 already
  # does for its "pred" contrast, reused here under the new label.
  d <- sweeps %>% filter(metric == row$metric, feature == row$feature, cell == row$cell,
                         stratum == "abnormal") %>%
    inner_join(clin, by = "wsi") %>% filter(!is.na(pred))
  if (!nrow(d)) return(NULL)
  d %>% group_by(param_name, param_label, param_value) %>%
    group_modify(~ assoc_test(.x$value, .x$pred, DROP_ZEROS)) %>% ungroup() %>%
    mutate(design = "inter", label = row$label, feature_type = "ecology")
}

run_ecology_intra <- function(row) {
  # intra-slide: pred==1 only, paired normal(attn0) vs abnormal(attn1) at
  # every grid setting.
  d <- sweeps %>% filter(metric == row$metric, feature == row$feature, cell == row$cell,
                         stratum %in% c("normal", "abnormal")) %>%
    inner_join(clin, by = "wsi") %>% filter(!is.na(pred), pred == 1)
  if (!nrow(d)) return(NULL)
  wide <- d %>% select(wsi, param_name, param_label, param_value, stratum, value) %>%
    tidyr::pivot_wider(names_from = stratum, values_from = value)
  if (!all(c("normal", "abnormal") %in% names(wide))) return(NULL)
  # (abnormal, normal) order - see the matching comment in run_nuclear().
  wide %>% group_by(param_name, param_label, param_value) %>%
    group_modify(~ assoc_test_paired(.x$abnormal, .x$normal, DROP_ZEROS)) %>% ungroup() %>%
    dplyr::rename(n1 = n_pairs) %>% mutate(n0 = n1) %>%
    mutate(design = "intra", label = row$label, feature_type = "ecology")
}

ecology_res <- purrr::map_dfr(seq_len(nrow(ECOLOGY)), function(i) {
  bind_rows(run_ecology_inter(ECOLOGY[i, ]), run_ecology_intra(ECOLOGY[i, ]))
}) %>%
  # Ripley's published setting is param_value = NA ("default"), not
  # comparable to the ring-based d2_px value - a naive param_value ==
  # 200 check would wrongly flag "fixed200" as published instead.
  mutate(is_published = dplyr::case_when(
           param_name == "rmax_um" ~ param_label == "default",
           param_name == "d2_px"   ~ param_value == PUBLISHED_PARAM["d2_px"],
           TRUE ~ FALSE))
message(sprintf("[ecology] %d rows (4 metrics x 2 designs x up to 3 settings)", nrow(ecology_res)))

# =========================================================================
# combine, save
# =========================================================================
res <- bind_rows(nuclear_res, ecology_res)
saveRDS(res, file.path(OUT_DIR, "dual_design_associations.RDS"))
write.csv(res, file.path(OUT_DIR, "dual_design_associations.csv"), row.names = FALSE)

message("\n=== nuclear (published columns, no parameter to vary) ===")
print(as.data.frame(nuclear_res %>% select(label, design, n1, n0, effect, p)), row.names = FALSE)

message("\n=== ecology (3-point grid per design) ===")
print(as.data.frame(ecology_res %>% select(label, design, param_label, param_value, n1, n0, effect, p) %>%
  arrange(label, design, param_value)), row.names = FALSE)

stab <- res %>% filter(is.finite(effect)) %>%
  group_by(label, design, feature_type) %>%
  summarise(n_settings = n(), sign_consistent = all(effect > 0) | all(effect < 0),
            any_sig = any(p < 0.05, na.rm = TRUE), .groups = "drop")
message("\n=== stability per label x design ===")
print(as.data.frame(stab), row.names = FALSE)
write.csv(stab, file.path(OUT_DIR, "dual_design_stability.csv"), row.names = FALSE)
