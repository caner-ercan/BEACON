## ---------------------------------------------------------------------------
## rev1_unit_of_analysis / 07_replicate_published.R
##
## Exact replication of the PUBLISHED slide-level p-values, plus the adjusted
## p-values that were never computed for them -- and a decomposition of why the
## two-fold patient-level table (05/06) differs.
##
## The published Fig 3C/3D/Supp Fig 1 numbers come from two functions in
## `4.plotting/code/3.ecology.Rmd`, both of which operate on SLIDES, with no
## patient aggregation at all:
##
##   single_inter_wsi(merged_df, paste0(base,"attn1"))   [exp = "pred"]
##     -> pulls the SAME column (attn1) for both groups, splits pred==1 vs
##        pred==0 SLIDES, drops NA and exact zeros, Shapiro on each arm ->
##        Welch t-test if both normal else Mann-Whitney.
##        NOTE: attn1-vs-attn1. NOT attn1-vs-attn2.
##
##   single_intra_wsi(merged_df, base)
##     -> hardcodes `data <- merged_df %>% filter(pred == 1)` (the passed
##        `data` argument is ignored), pairs attn0 vs attn1 within each of
##        those slides, drops rows where either is NA or exactly 0, Shapiro
##        on each arm -> paired t-test if both normal else Wilcoxon signed-rank.
##
## Two independent things separate these from the 05/06 two-fold table:
##   (a) UNIT   published = slide; two-fold table = patient.
##   (b) DESIGN published inter = attn1 vs attn1; two-fold inter = attn1 vs
##       attn2 for the 4 ecology metrics (CE's redesign; nuclear metrics have
##       no attn2 and fall back to attn1-vs-attn1, so design is unchanged for
##       them and any difference there is purely (a)).
## This script quantifies (a) and (b) separately in a 2x2 for every metric.
##
##   Rscript 07_replicate_published.R
## ---------------------------------------------------------------------------

SCRIPT_DIR <- local({
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
})
source(file.path(SCRIPT_DIR, "00_config.R"))

eco <- load_ecology()
patient_class <- function(g) as.integer(any(g == 1))

## The 12 metrics of the published `base_names` vector, in its order. The
## multiplicity family for the published figures is these 12, not the 9 CE
## later selected -- both families are reported below.
PUB <- tibble::tribble(
  ~key,             ~base,                                              ~group,    ~label,                                ~in9,
  "nucODmax",       "nuc_median_OD.Sum..Max_",                          "Nuclear", "Nucleus Maximum Intensity, median",   TRUE,
  "nucHaralick",    "nuc_median_Hematoxylin..Haralick.Contrast..F1._",  "Nuclear", "Nucleus Haralick Contrast, median",   TRUE,
  "nucHematoxSD",   "nuc_median_Hematoxylin..Std.dev._",                "Nuclear", "Hematoxylin SD, median",              TRUE,
  "nucCircularity", "nuc_q75_Circularity_",                             "Nuclear", "Nucleus circularity, q75",            TRUE,
  "nucAreaSD",      "nuc_sd_Area.uum.2_",                               "Nuclear", "Nucleus area, SD",                    TRUE,
  "nucODmedianSD",  "nuc_sd_OD.Sum..Median_",                           "Nuclear", "Nucleus OD sum (median), SD",         FALSE,
  "immuneDensity",  "Dens_Immune_",                                     "Ecology", "Immune cell density",                 FALSE,
  "moranLymphoPl",  "Moran_Local_LymphoPlasma_Ii_clustered_mean_",      "Ecology", "Lymphoplasmacytic cluster Moran mean", TRUE,
  "globalMoranImm", "GlobalMoran_Immune_",                              "Ecology", "Global Moran's I, immune",            TRUE,
  "knnEpiImmune",   "meanKNN_Epithelial_Immune_",                       "Ecology", "Mean kNN, epithelial-immune",         FALSE,
  "moranImmune",    "Moran_Local_Immune_Ii_clustered_mean_",            "Ecology", "Moran's I, immune",                   TRUE,
  "ripleyEpiImmune","Ripley_L12_Epithelial_Immune_frac_positive_",      "Ecology", "Ripley's L, epithelial-immune",       TRUE
)

## ---------------------------------------------------------------------------
## Published INTER (slide level, attn1 vs attn1, split by pred)
## ---------------------------------------------------------------------------
pub_inter <- function(base, col_suffix_neg = "attn1") {
  col1   <- paste0(base, "attn1")
  colneg <- paste0(base, col_suffix_neg)
  if (!all(c(col1, colneg) %in% names(eco))) return(list(p = NA, n1 = NA, n0 = NA, test = NA, d = NA))
  d1 <- eco[[col1]][eco$pred == 1]
  d0 <- eco[[colneg]][eco$pred == 0]
  d1 <- d1[!is.na(d1) & d1 != 0]
  d0 <- d0[!is.na(d0) & d0 != 0]
  if (length(d1) < 3 || length(d0) < 3) return(list(p = NA, n1 = length(d1), n0 = length(d0), test = NA, d = NA))
  ts <- choose_test_unpaired(d1, d0)
  list(p = ts$p, n1 = length(d1), n0 = length(d0), test = ts$test, d = cliffs_delta(d1, d0))
}

## ---------------------------------------------------------------------------
## Published INTRA (slide level, paired attn0 vs attn1, pred==1 slides only)
## ---------------------------------------------------------------------------
pub_intra <- function(base) {
  c1 <- paste0(base, "attn1"); c0 <- paste0(base, "attn0")
  if (!all(c(c0, c1) %in% names(eco))) return(list(p = NA, n = NA, test = NA, d = NA))
  sub <- eco[eco$pred == 1, ]
  a <- sub[[c1]]; b <- sub[[c0]]
  ok <- !is.na(a) & !is.na(b) & a != 0 & b != 0
  a <- a[ok]; b <- b[ok]
  if (length(a) < 3) return(list(p = NA, n = length(a), test = NA, d = NA))
  ts <- choose_test_paired(a, b)
  list(p = ts$p, n = length(a), test = ts$test, d = rank_biserial_paired(a, b))
}

## ---------------------------------------------------------------------------
## Patient-level counterparts, for the unit-vs-design decomposition
## ---------------------------------------------------------------------------
pat_inter <- function(base, col_suffix_neg = "attn1") {
  col1   <- paste0(base, "attn1")
  colneg <- paste0(base, col_suffix_neg)
  if (!all(c(col1, colneg) %in% names(eco))) return(list(p = NA, d = NA))
  d <- data.frame(x1 = eco[[col1]], xneg = eco[[colneg]], g = eco$pred, p = eco$patient) %>%
    group_by(p) %>% mutate(cls = patient_class(g)) %>% ungroup()
  abn <- d %>% filter(cls == 1, g == 1, !is.na(x1), x1 != 0, is.finite(x1)) %>%
    group_by(p) %>% summarise(v = median(x1), .groups = "drop")
  nrm <- d %>% filter(cls == 0, !is.na(xneg), xneg != 0, is.finite(xneg)) %>%
    group_by(p) %>% summarise(v = median(xneg), .groups = "drop")
  if (nrow(abn) < 3 || nrow(nrm) < 3) return(list(p = NA, d = NA))
  ts <- choose_test_unpaired(abn$v, nrm$v)
  list(p = ts$p, d = cliffs_delta(abn$v, nrm$v))
}

## ---------------------------------------------------------------------------
## Run: published replication
## ---------------------------------------------------------------------------
res <- map_rows(PUB, function(r) {
  i <- pub_inter(r$base)          # published design: attn1 vs attn1
  a <- pub_intra(r$base)
  tibble::tibble(
    key = r$key, group = r$group, label = r$label, in9 = as.logical(r$in9),
    inter_n_abn = i$n1, inter_n_norm = i$n0, inter_test = i$test,
    inter_p_published = i$p, inter_delta = i$d,
    intra_n = a$n, intra_test = a$test,
    intra_p_published = a$p, intra_delta = a$d
  )
})

## Adjusted p over the published 12-metric family, and over CE's 9-metric subset.
res <- res %>%
  mutate(
    inter_BH_12   = p.adjust(inter_p_published, "BH"),
    inter_bonf_12 = p.adjust(inter_p_published, "bonferroni"),
    intra_BH_12   = p.adjust(intra_p_published, "BH"),
    intra_bonf_12 = p.adjust(intra_p_published, "bonferroni")
  )
res$inter_BH_9  <- NA_real_; res$intra_BH_9 <- NA_real_
res$inter_BH_9[res$in9]  <- p.adjust(res$inter_p_published[res$in9], "BH")
res$intra_BH_9[res$in9]  <- p.adjust(res$intra_p_published[res$in9], "BH")

message("\n=== PUBLISHED (slide-level) replication + adjusted p ===")
print(as.data.frame(res %>% transmute(
  group, label,
  inter_n = sprintf("%d/%d", inter_n_abn, inter_n_norm),
  inter_p = fmt_p(inter_p_published), inter_BH12 = fmt_p(inter_BH_12), inter_BH9 = fmt_p(inter_BH_9),
  intra_n, intra_p = fmt_p(intra_p_published), intra_BH12 = fmt_p(intra_BH_12), intra_BH9 = fmt_p(intra_BH_9)
)), row.names = FALSE)

## ---------------------------------------------------------------------------
## 2x2 decomposition: UNIT (slide/patient) x DESIGN (attn1-vs-attn1 / attn1-vs-attn2)
## ---------------------------------------------------------------------------
decomp <- map_rows(PUB, function(r) {
  has2 <- paste0(r$base, "attn2") %in% names(eco)
  tibble::tibble(
    key = r$key, group = r$group, label = r$label, has_attn2 = has2,
    slide_a1a1   = pub_inter(r$base, "attn1")$p,
    slide_a1a2   = if (has2) pub_inter(r$base, "attn2")$p else NA_real_,
    patient_a1a1 = pat_inter(r$base, "attn1")$p,
    patient_a1a2 = if (has2) pat_inter(r$base, "attn2")$p else NA_real_
  )
})

message("\n=== INTER-SLIDE 2x2: unit (slide/patient) x comparator (attn1/attn2) ===")
message("    slide_a1a1 = the published number. patient_a1a2 = the 05/06 table number (ecology).")
print(as.data.frame(decomp %>% transmute(group, label, has_attn2,
  slide_a1a1 = fmt_p(slide_a1a1), slide_a1a2 = fmt_p(slide_a1a2),
  patient_a1a1 = fmt_p(patient_a1a1), patient_a1a2 = fmt_p(patient_a1a2))), row.names = FALSE)

P_PUB_OUT    <- file.path(out_folder, "published_replication_260814.csv")
P_DECOMP_OUT <- file.path(out_folder, "interslide_decomposition_260814.csv")
write.csv(res, P_PUB_OUT, row.names = FALSE)
write.csv(decomp, P_DECOMP_OUT, row.names = FALSE)
message("\nwrote ", P_PUB_OUT, "\nwrote ", P_DECOMP_OUT)
