# =====================================================================
# 06 - Progression association at every parameter setting
#
# This is the script that answers Reviewer 1 #2's second half.
#
# Conventions match 4.plotting/code/3.ecology.Rmd:
#   * SLIDE level, restricted to DACOR-abnormal slides (as Fig. 4C/4D)
#   * zeros dropped before testing (see below - this matters enormously)
#   * Wilcoxon (Shapiro-Wilk is <= 1e-5 in every group tested, so the
#     published normality branch always selects it)
#
# Effect = rank-biserial r = 2*AUC - 1. Bounded, sign-interpretable, and
# comparable across metrics on different scales, which is what lets the
# dot plot put every metric on one axis.
#
# THE CLAIM IS STABILITY, NOT SIGNIFICANCE. A metric that keeps its sign
# and rough magnitude across the grid is robust even where p drifts above
# 0.05 at an endpoint. Committing to that reading in advance is far
# stronger than being forced into it if one cell comes back non-significant.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R")); source(file.path(here, "lib_sweep.R"))

STEMS <- c("sweep_ripley", "sweep_moran_getis", "sweep_knn",
           "sweep_gfunction", "sweep_morisita")
sweeps <- purrr::map_dfr(STEMS, function(s) {
  d <- load_sweep(OUT_DIR, s)
  if (is.null(d)) { message("[skip] no results for ", s); return(NULL) }
  message(sprintf("[load] %-20s %8d rows", s, nrow(d)))
  d
})
stopifnot(nrow(sweeps) > 0)

clin <- get_slide_clinical(PATHS$published)
dat <- sweeps %>% inner_join(clin, by = "wsi") %>% filter(!is.na(progression))
if (!is.na(RESTRICT_PRED)) dat <- dat %>% filter(!is.na(pred), pred == RESTRICT_PRED)
message(sprintf("[scope] %d rows / %d slides (pred==%s)",
                nrow(dat), n_distinct(dat$wsi), RESTRICT_PRED))

# ---- isolate PARAMETER sensitivity from INCLUSION sensitivity --------
# This code computes a value for slides where the published pipeline left
# an imputed zero. Most of those are legitimate (317/505 have no plasma
# cells at all in abnormal regions, so Moran's I is undefined), but ~188
# have plasma cells and were still dropped - and recovering them changes
# the anchor p from 0.0075 to 0.054.
#
# That is an INCLUSION effect, not a parameter effect. To keep the two
# apart, MATCH_PUBLISHED_SAMPLE restricts every grid setting to the exact
# slides the published analysis used, so the ring comparison is
# apples-to-apples with the published claim.
MATCH_PUBLISHED_SAMPLE <- !identical(Sys.getenv("BEACON_MATCH_SAMPLE"), "0")
if (MATCH_PUBLISHED_SAMPLE) {
  pubm <- as.data.frame(readRDS(PATHS$published))
  usable <- pubm %>%
    select(wsi, any_of(ANCHOR_FEATURE)) %>%
    filter(.data[[ANCHOR_FEATURE]] != 0) %>% pull(wsi)
  dat_matched <- dat %>% filter(wsi %in% usable)
  message(sprintf("[scope] published-sample match: %d slides (anchor feature non-zero)",
                  n_distinct(dat_matched$wsi)))
} else dat_matched <- NULL

# Bookkeeping columns emitted by summarise_curve() alongside the real
# summaries. These are DIAGNOSTICS OF THE PARAMETER, not spatial metrics -
# r_max_used is literally the integration limit that was applied, and n_r
# the number of r values inside it - so both track the swept parameter by
# construction and are guaranteed to "flip" direction. Testing them as if
# they were metrics inflates the inconsistent count (23 -> 14 flippers,
# 84% -> 89% direction-consistent once removed). Never treat them as
# findings; they are retained in the sweep output for QC only.
DIAGNOSTIC_FEATURES <- c("r_max_used", "n_r")
dat <- dat %>% filter(!feature %in% DIAGNOSTIC_FEATURES)
message(sprintf("[scope] excluded diagnostic features (%s): %d rows / %d combos remain",
                paste(DIAGNOSTIC_FEATURES, collapse = ", "), nrow(dat),
                dplyr::n_distinct(paste(dat$metric, dat$feature, dat$cell, dat$stratum))))

group_keys <- c("metric", "feature", "cell", "stratum",
                "param_name", "param_label", "param_value")

res <- dat %>%
  group_by(across(all_of(intersect(c(group_keys, "style"), names(dat))))) %>%
  group_modify(function(g, k) assoc_test(g$value, g$progression, DROP_ZEROS)) %>%
  ungroup()

# Also compute WITHOUT zero-dropping, so the sensitivity of the published
# convention itself is visible rather than hidden.
res_wz <- dat %>%
  group_by(across(all_of(intersect(c(group_keys, "style"), names(dat))))) %>%
  group_modify(function(g, k) assoc_test(g$value, g$progression, FALSE)) %>%
  ungroup() %>%
  rename(n1_wz = n1, n0_wz = n0, effect_wz = effect, p_wz = p, auc_wz = auc)

res <- left_join(res, res_wz,
                 by = intersect(c(group_keys, "style"), names(res)))

# ---- stability summary: the headline numbers for the rebuttal --------
stability <- res %>%
  filter(is.finite(effect)) %>%
  group_by(across(all_of(intersect(c("metric", "feature", "cell", "stratum",
                                     "param_name", "style"), names(res))))) %>%
  summarise(n_settings      = n(),
            effect_min      = min(effect),
            effect_max      = max(effect),
            effect_median   = stats::median(effect),
            effect_range    = max(effect) - min(effect),
            sign_consistent = all(effect > 0) | all(effect < 0),
            n_sig           = sum(p < 0.05, na.rm = TRUE),
            frac_sig        = mean(p < 0.05, na.rm = TRUE),
            .groups = "drop")

saveRDS(res,       file.path(OUT_DIR, "associations.RDS"))
saveRDS(stability, file.path(OUT_DIR, "association_stability.RDS"))
write.csv(res,       file.path(OUT_DIR, "associations.csv"),          row.names = FALSE)
write.csv(stability, file.path(OUT_DIR, "association_stability.csv"), row.names = FALSE)
message(sprintf("[done] %d association rows, %d stability rows", nrow(res), nrow(stability)))

# ---- anchor check ----------------------------------------------------
anchor_of <- function(d) d %>% filter(metric == "Global", feature == "Moran_I",
                                      cell == "Plasma", stratum == "abnormal")
anchor <- anchor_of(res)
if (nrow(anchor)) {
  message("\n[anchor] GlobalMoran_Plasma_attn1 across rings - ALL computable slides:")
  print(as.data.frame(anchor %>%
    select(param_label, param_value, n1, n0, effect, p)), row.names = FALSE)
}

if (!is.null(dat_matched) && nrow(dat_matched)) {
  res_m <- dat_matched %>%
    group_by(across(all_of(intersect(c(group_keys, "style"), names(dat_matched))))) %>%
    group_modify(function(g, k) assoc_test(g$value, g$progression, DROP_ZEROS)) %>%
    ungroup()
  saveRDS(res_m, file.path(OUT_DIR, "associations_published_sample.RDS"))
  write.csv(res_m, file.path(OUT_DIR, "associations_published_sample.csv"), row.names = FALSE)
  am <- anchor_of(res_m)
  if (nrow(am)) {
    message("\n[anchor] same, restricted to the PUBLISHED sample (parameter effect isolated):")
    print(as.data.frame(am %>% select(param_label, param_value, n1, n0, effect, p)),
          row.names = FALSE)
  }
}

message("\n[stability] sign-consistent across the grid:")
print(as.data.frame(stability %>%
  summarise(total = n(), sign_consistent = sum(sign_consistent),
            always_sig = sum(frac_sig == 1), never_sig = sum(frac_sig == 0))),
  row.names = FALSE)
