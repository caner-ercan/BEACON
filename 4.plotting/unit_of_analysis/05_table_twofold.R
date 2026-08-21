## ---------------------------------------------------------------------------
## rev1_unit_of_analysis / 05_table_twofold.R
##
## Refined, two-fold patient-level table for a CE-specified subset of metrics
## (5 nuclear + 4 immune-ecological), following the exact group definitions in
## `4.plotting/code/3.ecology.Rmd`'s single_inter_wsi()/single_intra_wsi():
##
##   INTER-SLIDE  DACOR-positive patients: their own DACOR-abnormal (pred==1)
##                slides' attn1 (high-attention region) value.
##                DACOR-negative patients: their slides' whole-slide (attn2)
##                value -- EXCEPT the 5 nuclear metrics, which have no attn2
##                anywhere in the pipeline (only attn0/attn1 were ever
##                computed upstream); for those, CE's call (2026-08-13) is to
##                fall back to attn1-for-both-groups, matching the original
##                published method and the task-4 primary table.
##
##   INTRA-SLIDE  Within DACOR-positive (pred==1) SLIDES ONLY (matching
##                single_intra_wsi()'s `filter(pred==1)` -- a restriction the
##                task-4 primary table's `analyse_within()` did NOT apply;
##                that was every slide, this is the corrected, narrower scope
##                CE asked for here), paired attn0 (normal region) vs attn1
##                (abnormal region), same slide.
##
## Both folds use the corrected patient-level rule (restrict_to_own_class):
## for DACOR-positive patients with mixed-status biopsies, only their own
## abnormal biopsies feed the abnormal-class aggregate.
##
## Depends on the 00_config.R loader fix (2026-08-13): the original two-step
## rename (unsuffixed -> attn2, THEN digit -> attn0/1/2) is now replicated in
## full, which is what surfaces Ripley's L's true attn2 in the first place.
##
##   Rscript 05_table_twofold.R
## ---------------------------------------------------------------------------

SCRIPT_DIR <- local({
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
})
source(file.path(SCRIPT_DIR, "00_config.R"))

eco <- load_ecology()
patient_class <- function(g) as.integer(any(g == 1))

## -- the 9-metric, 2-group subset --------------------------------------------
METRICS_REFINED <- tibble::tribble(
  ~key,             ~base,                                              ~group,      ~label,                             ~has_attn2,
  "nucODmax",       "nuc_median_OD.Sum..Max_",                          "Nuclear",   "Nucleus Maximum Intensity, median", FALSE,
  "nucHaralick",    "nuc_median_Hematoxylin..Haralick.Contrast..F1._",  "Nuclear",   "Nucleus Haralick Contrast, median",  FALSE,
  "nucHematoxSD",   "nuc_median_Hematoxylin..Std.dev._",                "Nuclear",   "Hematoxylin SD, median",             FALSE,
  "nucCircularity", "nuc_q75_Circularity_",                             "Nuclear",   "Nucleus circularity, q75",           FALSE,
  "nucAreaSD",      "nuc_sd_Area.uum.2_",                               "Nuclear",   "Nucleus area, SD",                   FALSE,
  "moranLymphoPl",  "Moran_Local_LymphoPlasma_Ii_clustered_mean_",      "Ecology",   "Lymphoplasmacytic cluster Moran mean", TRUE,
  "globalMoranImm", "GlobalMoran_Immune_",                              "Ecology",   "Global Moran's I, immune",           TRUE,
  "moranImmune",    "Moran_Local_Immune_Ii_clustered_mean_",            "Ecology",   "Moran's I, immune",                  TRUE,
  "ripleyEpiImmune","Ripley_L12_Epithelial_Immune_frac_positive_",      "Ecology",   "Ripley's L, epithelial-immune",      TRUE
)

## ---------------------------------------------------------------------------
## INTER-SLIDE
## ---------------------------------------------------------------------------
analyse_interslide <- function(base, key, group, label, has_attn2) {
  col1   <- paste0(base, "attn1")
  colneg <- if (has_attn2) paste0(base, "attn2") else paste0(base, "attn1")
  if (!col1 %in% names(eco) || !colneg %in% names(eco))
    return(tibble::tibble(fold = "inter", key = key, group = group, label = label, note = "column not found"))

  d <- data.frame(x1 = eco[[col1]], xneg = eco[[colneg]], g = eco$pred, p = eco$patient) %>%
    group_by(p) %>%
    mutate(cls = patient_class(g)) %>%
    ungroup()

  ## Abnormal-class patients: only their own pred==1 slides, attn1 value.
  abn <- d %>%
    filter(cls == 1, g == 1, !is.na(x1), x1 != 0, is.finite(x1)) %>%
    group_by(p) %>% summarise(v = median(x1), .groups = "drop")

  ## Normal-class patients: all their slides are pred==0 by construction of
  ## cls==0; value column depends on has_attn2 (attn2 if available, else the
  ## attn1 fallback CE specified for the nuclear group).
  norm <- d %>%
    filter(cls == 0, !is.na(xneg), xneg != 0, is.finite(xneg)) %>%
    group_by(p) %>% summarise(v = median(xneg), .groups = "drop")

  pa <- abn$v; pb <- norm$v
  if (length(pa) < 3 || length(pb) < 3)
    return(tibble::tibble(fold = "inter", key = key, group = group, label = label, note = "insufficient data"))

  tp <- choose_test_unpaired(pa, pb)
  tibble::tibble(
    fold = "inter", key = key, group = group, label = label,
    n_abn = length(pa), n_norm = length(pb),
    test = tp$test, p_value = tp$p,
    cliffs_delta = cliffs_delta(pa, pb),
    neg_col = ifelse(has_attn2, "attn2", "attn1 (fallback)"),
    note = NA_character_
  )
}

## ---------------------------------------------------------------------------
## INTRA-SLIDE -- restricted to DACOR-positive SLIDES first, matching
## single_intra_wsi()'s filter(pred==1); the task-4 primary table's
## analyse_within() did not apply this restriction.
## ---------------------------------------------------------------------------
analyse_intraslide <- function(base, key, group, label) {
  c1 <- paste0(base, "attn1"); c0 <- paste0(base, "attn0")
  if (!all(c(c0, c1) %in% names(eco)))
    return(tibble::tibble(fold = "intra", key = key, group = group, label = label, note = "column not found"))

  sub <- eco %>% filter(pred == 1)
  a <- sub[[c1]]; b <- sub[[c0]]; p <- sub$patient
  ok <- !is.na(a) & !is.na(b) & is.finite(a) & is.finite(b) & a != 0 & b != 0 & !is.na(p)
  a <- a[ok]; b <- b[ok]; p <- p[ok]
  if (length(a) < 3)
    return(tibble::tibble(fold = "intra", key = key, group = group, label = label, note = "insufficient data"))

  pl <- data.frame(a, b, p) %>%
    group_by(p) %>% summarise(A = median(a), B = median(b), .groups = "drop")
  tp <- choose_test_paired(pl$A, pl$B)

  tibble::tibble(
    fold = "intra", key = key, group = group, label = label,
    n = nrow(pl),
    test = tp$test, p_value = tp$p,
    cliffs_delta = rank_biserial_paired(pl$A, pl$B),
    note = NA_character_
  )
}

## ---------------------------------------------------------------------------
## Run + BH within each fold's own 9-metric family
## ---------------------------------------------------------------------------
inter <- map_rows(METRICS_REFINED, function(r)
  analyse_interslide(r$base, r$key, r$group, r$label, as.logical(r$has_attn2))) %>%
  mutate(p_BH = p.adjust(p_value, "BH"))

intra <- map_rows(METRICS_REFINED, function(r)
  analyse_intraslide(r$base, r$key, r$group, r$label)) %>%
  mutate(p_BH = p.adjust(p_value, "BH"))

message("== INTER-SLIDE ==")
print(as.data.frame(inter %>% transmute(group, label, n_abn, n_norm, neg_col,
        p = fmt_p(p_value), p_BH = fmt_p(p_BH), delta = round(cliffs_delta, 2))), row.names = FALSE)

message("\n== INTRA-SLIDE ==")
print(as.data.frame(intra %>% transmute(group, label, n,
        p = fmt_p(p_value), p_BH = fmt_p(p_BH), delta = round(cliffs_delta, 2))), row.names = FALSE)

P_TWOFOLD_INTER <- file.path(out_folder, "twofold_interslide_260813.csv")
P_TWOFOLD_INTRA <- file.path(out_folder, "twofold_intraslide_260813.csv")
write.csv(inter, P_TWOFOLD_INTER, row.names = FALSE)
write.csv(intra, P_TWOFOLD_INTRA, row.names = FALSE)
message("\nwrote ", P_TWOFOLD_INTER, "\nwrote ", P_TWOFOLD_INTRA)
