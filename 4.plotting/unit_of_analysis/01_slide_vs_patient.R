## ---------------------------------------------------------------------------
## rev1_unit_of_analysis / 01_slide_vs_patient.R
##
## Re-tests every Fig 3C / Fig 3D / Supp Fig 1 metric at the patient level and
## reports effect magnitude alongside significance.
##
## Two contrasts, because the manuscript presents both:
##   BETWEEN  the metric measured in the DNA content abnormal region (attn1),
##            compared between slides DACOR called abnormal vs normal
##            (`single_inter_wsi(..., exp = "pred")` in 3.ecology.Rmd).
##   WITHIN   the same slide's abnormal region (attn1) vs its own normal
##            region (attn0), paired (`single_intra_wsi()`), i.e. Supp Fig 1D.
##
## For each: slide-level (as published) and patient-level, plus Cliff's delta,
## a bootstrap CI, the raw median shift, and BH / Bonferroni adjustment across
## the twelve-metric family.
##
##   Rscript 01_slide_vs_patient.R
## ---------------------------------------------------------------------------

## Resolve this script's own folder, whether run via Rscript or source().
SCRIPT_DIR <- local({
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
})
source(file.path(SCRIPT_DIR, "00_config.R"))

eco <- load_ecology()

## Patient class for the BETWEEN contrast: a patient counts as DNA content
## abnormal if ANY of their biopsies was predicted abnormal. This is the
## manuscript's own patient-level rule ("classifying a patient as positive if
## any single biopsy was positive"), reused here so the patient-level test is
## not answering a subtly different question from the published one.
patient_class <- function(g) as.integer(any(g == 1))

## CORRECTNESS FIX (2026-08-05, CE): when a patient's biopsies have mixed
## DACOR status, only that patient's g==cls biopsies may feed the aggregate for
## their class. 38 of 60 patient-level "abnormal" (cls=1) cases are mixed (as
## few as 1 positive slide alongside up to 30 negative ones) -- pooling every
## biopsy regardless of its own status, as the previous version did, diluted
## the abnormal-region signal for exactly the patients it's meant to isolate.
## Negative (cls=0) patients are unaffected by construction: patient_class()
## is 0 only when every one of their biopsies is already g==0.
restrict_to_own_class <- function(df) {
  df %>%
    group_by(p) %>%
    mutate(cls = patient_class(g), n_biopsy_total = n()) %>%
    filter(g == cls) %>%
    ungroup()
}

## ---------------------------------------------------------------------------
## BETWEEN-slide contrast
## ---------------------------------------------------------------------------
analyse_between <- function(base, key, panel, label, agg = median, agg_name = "median") {
  col <- paste0(base, "attn1")
  if (!col %in% names(eco))
    return(tibble::tibble(key = key, panel = panel, label = label, note = "column not found"))

  x <- eco[[col]]; g <- eco$pred; p <- eco$patient
  ## `x != 0` replicates the published filter. Upstream, non-computable
  ## spatial metrics are imputed to 0, and 3.ecology.Rmd drops them; keeping
  ## the filter is what makes the slide-level column comparable to the paper.
  ok <- !is.na(x) & x != 0 & is.finite(x) & !is.na(g) & !is.na(p)
  x <- x[ok]; g <- g[ok]; p <- p[ok]

  a <- x[g == 1]; b <- x[g == 0]
  if (length(a) < 3 || length(b) < 3)
    return(tibble::tibble(key = key, panel = panel, label = label, note = "insufficient data"))
  ts <- choose_test_unpaired(a, b)

  pl <- data.frame(x, g, p) %>%
    restrict_to_own_class() %>%
    group_by(p, cls, n_biopsy_total) %>%
    summarise(v = agg(x), n_biopsy_used = n(), .groups = "drop")
  pa <- pl$v[pl$cls == 1]; pb <- pl$v[pl$cls == 0]
  tp <- choose_test_unpaired(pa, pb)
  n_mixed_abn <- sum(pl$cls == 1 & pl$n_biopsy_used < pl$n_biopsy_total)

  d  <- cliffs_delta(pa, pb)
  ci <- cliffs_delta_ci(pa, pb)

  tibble::tibble(
    key = key, panel = panel, label = label, contrast = "between", aggregation = agg_name,
    n_slide = length(x), n_slide_abn = length(a), n_slide_norm = length(b),
    test_slide = ts$test, p_slide = ts$p,
    delta_slide = cliffs_delta(a, b),
    pct_shift_slide = 100 * (median(a) - median(b)) / abs(median(b)),
    n_patient = nrow(pl), n_patient_abn = length(pa), n_patient_norm = length(pb),
    n_patient_abn_mixed = n_mixed_abn,
    test_patient = tp$test, p_patient = tp$p,
    cliffs_delta = d, delta_lo = ci[1], delta_hi = ci[2], magnitude = delta_magnitude(d),
    median_abn = median(pa), median_norm = median(pb),
    pct_shift = 100 * (median(pa) - median(pb)) / abs(median(pb)),
    note = NA_character_
  )
}

## ---------------------------------------------------------------------------
## WITHIN-slide paired contrast
## ---------------------------------------------------------------------------
analyse_within <- function(base, key, panel, label) {
  c1 <- paste0(base, "attn1"); c0 <- paste0(base, "attn0")
  if (!all(c(c0, c1) %in% names(eco)))
    return(tibble::tibble(key = key, panel = panel, label = label, note = "column not found"))

  a <- eco[[c1]]; b <- eco[[c0]]; p <- eco$patient
  ok <- !is.na(a) & !is.na(b) & is.finite(a) & is.finite(b) & a != 0 & b != 0 & !is.na(p)
  a <- a[ok]; b <- b[ok]; p <- p[ok]
  if (length(a) < 3)
    return(tibble::tibble(key = key, panel = panel, label = label, note = "insufficient data"))
  ts <- choose_test_paired(a, b)

  ## Patient level: collapse each patient's biopsies to one abnormal-region and
  ## one normal-region value, then pair those. Keeps the paired design (each
  ## patient is still their own control) while removing pseudo-replication.
  pl <- data.frame(a, b, p) %>%
    group_by(p) %>%
    summarise(A = median(a), B = median(b), n_biopsy = n(), .groups = "drop")
  tp <- choose_test_paired(pl$A, pl$B)
  r  <- rank_biserial_paired(pl$A, pl$B)

  tibble::tibble(
    key = key, panel = panel, label = label, contrast = "within", aggregation = "median",
    n_slide = length(a), n_slide_abn = length(a), n_slide_norm = length(b),
    test_slide = ts$test, p_slide = ts$p,
    delta_slide = rank_biserial_paired(a, b),
    pct_shift_slide = 100 * (median(a) - median(b)) / abs(median(b)),
    n_patient = nrow(pl), n_patient_abn = nrow(pl), n_patient_norm = nrow(pl),
    test_patient = tp$test, p_patient = tp$p,
    cliffs_delta = r, delta_lo = NA_real_, delta_hi = NA_real_, magnitude = delta_magnitude(r),
    median_abn = median(pl$A), median_norm = median(pl$B),
    pct_shift = 100 * (median(pl$A) - median(pl$B)) / abs(median(pl$B)),
    note = NA_character_
  )
}

## ---------------------------------------------------------------------------
## Run
## ---------------------------------------------------------------------------
message("\n== BETWEEN-slide contrast (patient aggregation: median) ==")
between <- map_rows(METRICS, function(r) analyse_between(r$base, r$key, r$panel, r$label))

message("== WITHIN-slide paired contrast ==")
within  <- map_rows(METRICS, function(r) analyse_within(r$base, r$key, r$panel, r$label))

res <- bind_rows(between, within)

## Multiplicity is applied WITHIN each contrast family (12 metrics each), not
## across the pooled 24 -- the two contrasts answer different questions and are
## reported as separate families in the response.
##
## p_slide_BH/p_slide_bonf: adjusted p for the ORIGINAL, published slide-level
## test. This did not previously exist anywhere -- the manuscript's own tests
## (3.ecology.Rmd) report raw p only, and our first patient-level pass only
## adjusted p_patient. Reviewer 1's complaint is specifically that the
## SLIDE-level p-values are "inflated by the high number of tiles [sic]
## analysed" -- so the slide-level family needs its own correction shown
## alongside the patient-level one, not instead of it.
res <- res %>%
  group_by(contrast) %>%
  mutate(
    p_slide_BH     = p.adjust(p_slide, "BH"),
    p_slide_bonf   = p.adjust(p_slide, "bonferroni"),
    holds_slide    = ifelse(is.na(p_slide), NA,
                     ifelse(p_slide_bonf < 0.05, "yes (Bonferroni)",
                     ifelse(p_slide_BH   < 0.05, "yes (BH only)", "no"))),
    p_patient_BH   = p.adjust(p_patient, "BH"),
    p_patient_bonf = p.adjust(p_patient, "bonferroni"),
    holds_patient  = ifelse(is.na(p_patient), NA,
                     ifelse(p_patient_bonf < 0.05, "yes (Bonferroni)",
                     ifelse(p_patient_BH   < 0.05, "yes (BH only)", "no")))
  ) %>%
  ungroup() %>%
  arrange(contrast, factor(panel, levels = c("Fig 3C", "Fig 3D", "Supp Fig 1")), key)

write.csv(res, P_MAIN_OUT, row.names = FALSE)
message("\nwrote ", P_MAIN_OUT)

print(as.data.frame(res %>% filter(contrast == "between") %>%
  transmute(panel, label, n_slide, p_slide = fmt_p(p_slide),
            p_slide_bonf = fmt_p(p_slide_bonf), holds_slide,
            n_patient, p_patient = fmt_p(p_patient), p_bonf = fmt_p(p_patient_bonf),
            delta = round(cliffs_delta, 2), magnitude,
            shift = sprintf("%+.0f%%", pct_shift), holds_patient)), row.names = FALSE)

message("\n-- within-slide paired --")
print(as.data.frame(res %>% filter(contrast == "within") %>%
  transmute(panel, label, n_slide, p_slide = fmt_p(p_slide),
            p_slide_bonf = fmt_p(p_slide_bonf), holds_slide,
            n_patient, p_patient = fmt_p(p_patient), p_bonf = fmt_p(p_patient_bonf),
            rb = round(cliffs_delta, 2), shift = sprintf("%+.0f%%", pct_shift),
            holds_patient)), row.names = FALSE)

## ---------------------------------------------------------------------------
## Sensitivity: max instead of median as the patient aggregator
##
## `max` mirrors the manuscript's "positive if any biopsy is positive" rule and
## is the natural "worst biopsy" summary. It is a sensitivity analysis, not the
## primary: biopsies per patient range from 1 to 63, and a maximum over 63
## draws is upward-biased relative to a maximum over 1.
## ---------------------------------------------------------------------------
message("\n== SENSITIVITY: patient aggregation by max ==")
## p_slide is identical to the between-contrast p_slide in `res` (the
## slide-level test doesn't depend on how patients are later aggregated) --
## adjusted here too so this file is self-contained and matches `res`.
sens <- map_rows(METRICS, function(r)
  analyse_between(r$base, r$key, r$panel, r$label, agg = max, agg_name = "max")) %>%
  mutate(p_slide_BH     = p.adjust(p_slide, "BH"),
         p_slide_bonf   = p.adjust(p_slide, "bonferroni"),
         p_patient_BH   = p.adjust(p_patient, "BH"),
         p_patient_bonf = p.adjust(p_patient, "bonferroni"))

## Redundancy diagnostic: for each metric, its strongest Spearman correlation
## with any OTHER metric in the 12-metric family (attn1, slide-level, all 777
## slides regardless of pred -- matches the ad hoc check quoted in the
## rebuttal-defense discussion: median |r|=0.16 family-wide, moranLymphoPl-
## moranImmune r=0.84). Supports the BH-over-Bonferroni argument for metrics
## like moranLymphoPl that are BH-significant but not Bonferroni: a metric
## with a highly correlated sibling in the family is not an independent test,
## so Bonferroni's independence assumption is overpaying for it specifically.
corr_cols <- paste0(METRICS$base, "attn1")
corr_cols <- corr_cols[corr_cols %in% names(eco)]
corr_mat  <- eco[, corr_cols]
corr_mat[corr_mat == 0] <- NA
names(corr_mat) <- METRICS$key[match(corr_cols, paste0(METRICS$base, "attn1"))]
cr <- cor(corr_mat, use = "pairwise.complete.obs", method = "spearman")
diag(cr) <- NA

redundancy <- tibble::tibble(
  key = colnames(cr),
  max_abs_corr_family = apply(abs(cr), 2, max, na.rm = TRUE),
  most_corr_with = colnames(cr)[apply(abs(cr), 2, which.max)]
)

sens <- sens %>% left_join(redundancy, by = "key")
write.csv(sens, P_SENS_OUT, row.names = FALSE)
print(as.data.frame(sens %>% transmute(panel, label, p_slide = fmt_p(p_slide),
        p_slide_bonf = fmt_p(p_slide_bonf), p_patient = fmt_p(p_patient),
        p_bonf = fmt_p(p_patient_bonf), delta = round(cliffs_delta, 2))), row.names = FALSE)

message("\n-- redundancy diagnostic (strongest correlated sibling, family-wide) --")
print(as.data.frame(sens %>% transmute(label,
        max_abs_corr = round(max_abs_corr_family, 2), most_corr_with)), row.names = FALSE)

## ---------------------------------------------------------------------------
## Out of scope: Fig 4C
##
## Reviewer comment #3 names Fig 3D and Supp Fig 1 only, so Fig 4C is NOT part
## of the response. Computed here purely so the finding is on the record --
## see README, "Out of scope". Do not report without a decision to widen scope.
## ---------------------------------------------------------------------------
message("\n== OUT OF SCOPE -- Fig 4C, recorded only ==")
fig4c <- map_rows(METRICS_FIG4C, function(r) {
  col <- paste0(r$base, "attn1")
  if (!col %in% names(eco)) return(tibble::tibble(key = r$key, note = "column not found"))
  x <- eco[[col]]; g <- eco$progression; p <- eco$patient
  ok <- !is.na(x) & x != 0 & is.finite(x) & !is.na(g) & !is.na(p)
  x <- x[ok]; g <- g[ok]; p <- p[ok]
  ts <- choose_test_unpaired(x[g == 1], x[g == 0])
  pl <- data.frame(x, g, p) %>%
        restrict_to_own_class() %>%
        group_by(p, cls) %>%
        summarise(v = median(x), .groups = "drop")
  pa <- pl$v[pl$cls == 1]; pb <- pl$v[pl$cls == 0]
  tp <- choose_test_unpaired(pa, pb)
  tibble::tibble(key = r$key, panel = r$panel, label = r$label,
                 n_slide = length(x), p_slide = ts$p,
                 n_patient = nrow(pl), p_patient = tp$p,
                 cliffs_delta = cliffs_delta(pa, pb),
                 pct_shift = 100 * (median(pa) - median(pb)) / abs(median(pb)))
}) %>%
  mutate(p_slide_BH     = p.adjust(p_slide, "BH"),
         p_slide_bonf   = p.adjust(p_slide, "bonferroni"),
         p_patient_BH   = p.adjust(p_patient, "BH"),
         p_patient_bonf = p.adjust(p_patient, "bonferroni"))
write.csv(fig4c, P_FIG4C_OUT, row.names = FALSE)
print(as.data.frame(fig4c %>% transmute(label, n_slide, p_slide = fmt_p(p_slide),
        n_patient, p_patient = fmt_p(p_patient), p_bonf = fmt_p(p_patient_bonf),
        delta = round(cliffs_delta, 2), shift = sprintf("%+.0f%%", pct_shift))), row.names = FALSE)

## Patient-level values for every metric, kept for 02_plots.R.
saveRDS(list(eco = eco, metrics = METRICS), P_LONG_OUT)
message("\ndone.")
