## ---------------------------------------------------------------------------
## rev1_clinical / 03_cox_clinical.R
##
## Patient-level Cox models with the SAME covariate set in both cohorts.
##
## Answers reviewer 1 comment #4: "why were 'Dysplasia' and 'BE length' not
## included as covariates in the test cohort model?" -- in the submitted code
## (4.plotting/code/0.input.Rmd, plot_forest) the discovery model used
## age + sex + dysplasia + BELength + marker while the test model used only
## age + sex + marker, because dysplasia and BE length were unavailable for the
## test cohort. Both are now populated (see 01), so the two models are aligned
## and treatment is added as a further covariate.
##
## Also emits the likelihood-ratio test of clinical-only vs clinical + marker,
## which is the first component of what reviewer 3 major comment #1 asks for.
##
## Run 01_build_clinical_tables.R first.
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_CLINICAL_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))

suppressMessages({library(survival); library(broom); library(ggplot2)})

## --- analysis options ------------------------------------------------------
## Minimum follow-up. The submitted code used two different rules in two
## places -- `fu_time > 14` (ecology Cox) and `!(event == 1 & fu_time < 30)`
## (demographics Cox). 14 days is used here as the primary rule because it also
## removes the records with fu_time <= 0 (biopsies taken at or after the event
## date); 30 is reported as a sensitivity analysis.
MIN_FU_DAYS  <- as.numeric(Sys.getenv("MIN_FU_DAYS", "14"))
SENS_FU_DAYS <- 30

## Identical in every cohort. This is the whole point of the fix.
##
## Treatment is the ever/never acid-suppression binary (revision decision).
## NOTE: it turns out to be CONSTANT -- every patient with any treatment record
## was on a PPI or an H2 blocker at baseline, at follow-up, or both, so there is
## no untreated group and no contrast to estimate. fit_one() drops zero-variance
## covariates automatically and says so. `treatment_fu` (PPI/H2/None at last
## follow-up) is the only version with any variance and is run as a sensitivity.
CLIN_COVARS      <- c("age", "sex", "dysplasia_bin", "BELength", "treatment")
CLIN_COVARS_TX3  <- c("age", "sex", "dysplasia_bin", "BELength", "treatment_fu")
MARKERS          <- c("flow", "pred", "eco_risk")

P_RES  <- file.path(out_folder, sprintf("cox_clinical_%s.csv", STAMP))
P_LRT  <- file.path(out_folder, sprintf("cox_lrt_%s.csv", STAMP))
P_LOG  <- file.path(out_folder, sprintf("cox_clinical_%s.txt", STAMP))
P_PDF  <- file.path(out_folder, sprintf("cox_forest_%s.pdf", STAMP))

## --- data ------------------------------------------------------------------
src <- if (file.exists(P_ECO_OUT)) P_ECO_OUT else P_PATIENTS_OUT
d0 <- read.csv(src, stringsAsFactors = FALSE)

d0 <- d0 %>%
  mutate(
    event = case_when(progression == "CO" ~ 1L, progression == "NCO" ~ 0L, TRUE ~ NA_integer_),
    sex           = factor(sex, levels = c("F", "M")),
    dysplasia_bin = factor(dysplasia_bin, levels = c("No Dysplasia", "Dysplasia")),
    treatment     = factor(treatment,    levels = c("No", "Yes")),
    treatment_fu  = factor(treatment_fu, levels = c("PPI", "H2", "None")),
    flow          = factor(flow, levels = c("Diploid", "Aneuploid")),
    pred          = factor(pred,  levels = c("Diploid", "Aneuploid")),
    eco_risk      = if ("eco_risk_category" %in% names(.)) factor(
                      ifelse(pred == "Diploid" | eco_risk_category == "Low Risk", "Low Risk",
                        ifelse(eco_risk_category == "High Risk", "High Risk", NA)),
                      levels = c("Low Risk", "High Risk")) else NA
  )

con <- file(P_LOG, "wt"); sink(con, split = TRUE)
on.exit({sink(); close(con)}, add = TRUE)

cat("rev1 clinical Cox --", format(Sys.time()), "\n")
cat("source:", basename(src), "\n")
cat("covariates (identical in every cohort):", paste(CLIN_COVARS, collapse = " + "), "\n")
cat("minimum follow-up:", MIN_FU_DAYS, "days\n")

prep <- function(d, cohort, min_fu = MIN_FU_DAYS) {
  if (cohort != "Whole") d <- d[d$dataset == cohort, ]
  d[!is.na(d$fu_time) & d$fu_time > min_fu & !is.na(d$event), ]
}

## --- model fitting ---------------------------------------------------------

## Keep only covariates that are present, observed, and actually vary. A
## constant factor (e.g. the ever/never treatment variable, which has no
## untreated patients) carries no information and would make coxph fail.
usable <- function(d, covars, announce = NULL) {
  ok <- vapply(covars, function(v)
    v %in% names(d) && sum(!is.na(d[[v]])) > 0 &&
      (!is.factor(d[[v]]) || n_distinct(d[[v]][!is.na(d[[v]])]) > 1), logical(1))
  if (!is.null(announce) && any(!ok))
    cat(sprintf("  [%s] dropped (no variation): %s\n", announce,
                paste(covars[!ok], collapse = ", ")))
  covars[ok]
}

fit_one <- function(d, covars, label, cohort, analysis, cluster = FALSE) {
  covars <- usable(d, covars)
  if (!length(covars)) return(NULL)

  keep <- stats::complete.cases(d[, c(covars, "fu_time", "event")])
  dd <- d[keep, ]
  if (nrow(dd) < 10 || sum(dd$event) < 3) {
    cat(sprintf("  skipped %-10s %-22s (n=%d, events=%d -- too few)\n",
                cohort, label, nrow(dd), sum(dd$event)))
    return(NULL)
  }

  rhs <- paste(covars, collapse = " + ")
  if (cluster) rhs <- paste(rhs, "+ cluster(rid)")
  fm <- as.formula(paste("Surv(fu_time, event) ~", rhs))
  m  <- try(coxph(fm, data = dd, control = coxph.control(iter.max = 200)), silent = TRUE)
  if (inherits(m, "try-error")) {
    cat(sprintf("  FAILED  %-10s %-22s\n", cohort, label)); return(NULL)
  }

  tidy(m, exponentiate = TRUE, conf.int = TRUE) %>%
    transmute(cohort, analysis, model = label, term,
              HR = estimate, CI_low = conf.low, CI_high = conf.high, p = p.value,
              n = nrow(dd), events = sum(dd$event),
              concordance = unname(summary(m)$concordance[1]))
}

results <- list(); lrts <- list()

for (cohort in c("MDA", "NU", "Whole")) {
  d <- prep(d0, cohort)
  cat(sprintf("\n=== %s: n=%d, events=%d ===\n", cohort, nrow(d), sum(d$event)))
  invisible(usable(d, CLIN_COVARS, announce = cohort))

  ## univariate, every covariate and marker on its own
  for (v in c(CLIN_COVARS, MARKERS)) {
    results[[length(results) + 1]] <- fit_one(d, v, v, cohort, "univariate")
  }

  ## clinical-only reference model
  results[[length(results) + 1]] <-
    fit_one(d, CLIN_COVARS, "clinical only", cohort, "multivariate")

  ## clinical + each marker -- the aligned models the reviewer asked for
  for (mk in MARKERS) {
    results[[length(results) + 1]] <-
      fit_one(d, c(CLIN_COVARS, mk), paste("clinical +", mk), cohort, "multivariate")
    ## same model with a within-person cluster, to show the duplicated-person
    ## records are not driving the result (see STEP 2 audit in the QC report)
    results[[length(results) + 1]] <-
      fit_one(d, c(CLIN_COVARS, mk), paste("clinical +", mk), cohort,
              "multivariate, cluster(rid)", cluster = TRUE)

    ## LRT: does the marker add anything to the clinical model?
    base_cv <- usable(d, CLIN_COVARS)
    cv <- usable(d, c(CLIN_COVARS, mk))
    keep <- stats::complete.cases(d[, c(cv, "fu_time", "event")])
    dd <- d[keep, ]
    if (nrow(dd) >= 10 && sum(dd$event) >= 3 &&
        n_distinct(dd[[mk]][!is.na(dd[[mk]])]) > 1) {
      m0 <- try(coxph(as.formula(paste("Surv(fu_time, event) ~",
                                       paste(base_cv, collapse = " + "))), data = dd), silent = TRUE)
      m1 <- try(coxph(as.formula(paste("Surv(fu_time, event) ~",
                                       paste(cv, collapse = " + "))), data = dd), silent = TRUE)
      if (!inherits(m0, "try-error") && !inherits(m1, "try-error")) {
        an <- anova(m0, m1)
        lrts[[length(lrts) + 1]] <- data.frame(
          cohort = cohort, marker = mk, n = nrow(dd), events = sum(dd$event),
          loglik_clinical = as.numeric(logLik(m0)), loglik_plus = as.numeric(logLik(m1)),
          df = an$Df[2], chisq = an$Chisq[2], p_LRT = an$`Pr(>|Chi|)`[2],
          C_clinical = unname(summary(m0)$concordance[1]),
          C_plus     = unname(summary(m1)$concordance[1]))
      }
    }
  }
}

## sensitivity: stricter follow-up cut
for (cohort in c("MDA", "NU", "Whole")) {
  d <- prep(d0, cohort, SENS_FU_DAYS)
  for (mk in MARKERS) {
    results[[length(results) + 1]] <-
      fit_one(d, c(CLIN_COVARS, mk), paste("clinical +", mk), cohort,
              sprintf("multivariate, fu>%d", SENS_FU_DAYS))
  }
}

## sensitivity: 3-level treatment factor
for (cohort in c("MDA", "NU", "Whole")) {
  d <- prep(d0, cohort)
  for (mk in MARKERS) {
    results[[length(results) + 1]] <-
      fit_one(d, c(CLIN_COVARS_TX3, mk), paste("clinical +", mk), cohort,
              "multivariate, treatment_fu")
  }
}

res <- bind_rows(results)
lrt <- bind_rows(lrts)

write.csv(res, P_RES, row.names = FALSE)
write.csv(lrt, P_LRT, row.names = FALSE)

## --- report ----------------------------------------------------------------
pretty <- res %>%
  mutate(`HR (95% CI)` = sprintf("%.2f (%.2f-%.2f)", HR, CI_low, CI_high),
         P = ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))) %>%
  select(cohort, analysis, model, term, `HR (95% CI)`, P, n, events)

hdr("MULTIVARIATE MODELS -- identical covariates in both cohorts")
print(as.data.frame(pretty %>% filter(analysis == "multivariate")), row.names = FALSE)

hdr("SENSITIVITY -- cluster(rid), i.e. one cluster per real person")
print(as.data.frame(pretty %>% filter(analysis == "multivariate, cluster(rid)")), row.names = FALSE)

hdr("SENSITIVITY -- minimum follow-up 30 days")
print(as.data.frame(pretty %>% filter(analysis == sprintf("multivariate, fu>%d", SENS_FU_DAYS))),
      row.names = FALSE)

hdr("UNIVARIATE")
print(as.data.frame(pretty %>% filter(analysis == "univariate")), row.names = FALSE)

hdr("SENSITIVITY -- treatment as PPI/H2/None at last follow-up")
print(as.data.frame(pretty %>% filter(analysis == "multivariate, treatment_fu")),
      row.names = FALSE)

hdr("LIKELIHOOD-RATIO TEST: clinical only vs clinical + marker")
if (nrow(lrt)) {
  print(as.data.frame(lrt %>% mutate(across(where(is.numeric), ~ round(.x, 4)))), row.names = FALSE)
} else cat("none estimable\n")

## ---------------------------------------------------------------------------
## Complete-case accounting. The multivariate n is well below the cohort n
## because dysplasia / BE length / treatment are still incomplete in the
## discovery cohort; a reviewer will want this stated rather than inferred.
## ---------------------------------------------------------------------------
hdr("COMPLETE-CASE ACCOUNTING")
for (cohort in c("MDA", "NU", "Whole")) {
  d <- prep(d0, cohort)
  cc <- stats::complete.cases(d[, c(usable(d, CLIN_COVARS), "fu_time", "event")])
  cat(sprintf("%-6s eligible n=%3d (events %2d) -> complete cases n=%3d (events %2d)\n",
              cohort, nrow(d), sum(d$event), sum(cc), sum(d$event[cc])))
  for (v in usable(d, CLIN_COVARS)) {
    k <- sum(is.na(d[[v]]))
    if (k) cat(sprintf("         dropped by %-14s : %d\n", v, k))
  }
}

## ---------------------------------------------------------------------------
## BE length behaves in OPPOSITE directions in the two cohorts. That is a
## substantive finding, not a bug we can quietly average over, so it gets its
## own diagnostic block: the effect is reported unadjusted and adjusted, and
## separately for the patients whose BE length came from each source.
## ---------------------------------------------------------------------------
hdr("DIAGNOSTIC -- BE length, direction of effect by cohort")
be <- list()
for (cohort in c("MDA", "NU", "Whole")) {
  d <- prep(d0, cohort)
  for (spec in c("BELength", paste(usable(d, CLIN_COVARS), collapse = "+"))) {
    cv <- strsplit(spec, "\\+")[[1]]
    r <- fit_one(d, cv, if (length(cv) == 1) "unadjusted" else "adjusted", cohort, "BE diagnostic")
    if (!is.null(r)) be[[length(be) + 1]] <- r %>% filter(term == "BELength")
  }
  ## split by provenance of the BE length value
  for (s in unique(na.omit(d$BELength_src))) {
    ds <- d[!is.na(d$BELength_src) & d$BELength_src == s, ]
    r <- fit_one(ds, "BELength", paste("unadjusted,", s), cohort, "BE diagnostic")
    if (!is.null(r)) be[[length(be) + 1]] <- r
  }
}
be <- bind_rows(be)
if (nrow(be)) {
  print(as.data.frame(be %>% mutate(
    `HR (95% CI)` = sprintf("%.3f (%.2f-%.2f)", HR, CI_low, CI_high),
    P = ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))) %>%
    select(cohort, model, `HR (95% CI)`, P, n, events)), row.names = FALSE)
  res <- bind_rows(res, be)
  write.csv(res, P_RES, row.names = FALSE)
}
cat("\nBE length distribution by cohort and source:\n")
print(as.data.frame(d0 %>% filter(!is.na(BELength)) %>% group_by(dataset, BELength_src) %>%
  summarise(n = n(), mean = round(mean(BELength), 2), median = median(BELength),
            min = min(BELength), max = max(BELength), .groups = "drop")), row.names = FALSE)

## --- forest plots ----------------------------------------------------------
plot_forest <- function(df, title) {
  df <- df %>% filter(is.finite(HR), is.finite(CI_low), is.finite(CI_high),
                      CI_high < 1e3, CI_low > 1e-3)
  if (!nrow(df)) return(NULL)
  ggplot(df, aes(x = HR, y = reorder(term, HR))) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40") +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.22, colour = "#2C3E50") +
    geom_point(size = 2.4, colour = "#2C3E50") +
    geom_text(aes(label = sprintf("%.2f (%.2f-%.2f)  p=%s", HR, CI_low, CI_high,
                                  ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))),
              hjust = 0, nudge_y = 0.28, size = 2.5) +
    scale_x_log10() +
    facet_wrap(~ paste0(cohort, "  (n=", n, ", events=", events, ")"),
               scales = "free_y", ncol = 1) +
    labs(title = title, x = "Hazard ratio (95% CI, log scale)", y = NULL) +
    theme_minimal(base_size = 9) +
    theme(panel.grid.major.y = element_blank(), strip.text = element_text(face = "bold"))
}

pdf(P_PDF, width = 8, height = 6)
for (mk in MARKERS) {
  p <- plot_forest(res %>% filter(analysis == "multivariate", model == paste("clinical +", mk)),
                   sprintf("Multivariate Cox: %s + clinical covariates", mk))
  if (!is.null(p)) print(p)
}
invisible(dev.off())

cat("\nwritten:", P_RES, "\n")
cat("written:", P_LRT, "\n")
cat("written:", P_PDF, "\n")
cat("written:", P_LOG, "\n")
