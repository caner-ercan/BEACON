## ---------------------------------------------------------------------------
## rev1_clinical / 04_cox_eco_split.R
##
## Cox models on the ECOLOGY MODEL's own train/validation splits
## (Supplementary Fig. 3B), as opposed to 03_cox_clinical.R which stratifies by
## scanning-site cohort (discovery / test).
##
## Why this exists
## ---------------
## At submission, dysplasia had to be dropped from the univariate analysis on
## the ecology validation split because the model collapsed. The cause is now
## documented rather than assumed: in the submitted data only 5 of the 22
## validation patients had a dysplasia grade at all (the test cohort had none),
## and ALL 5 of them were progressors -- complete separation, so no finite
## coefficient exists. With dysplasia now ascertained for the test cohort
## (01_build_clinical_tables.R) the validation split has patients on both sides
## of the outcome and the model is estimable again.
##
## This script also re-derives the published Supp. Fig. 3B hazard ratios as a
## check that the rebuilt clinical table reproduces the submitted analysis.
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

MIN_FU_DAYS <- as.numeric(Sys.getenv("MIN_FU_DAYS", "14"))
CLIN_COVARS <- c("age", "sex", "dysplasia_bin", "BELength", "treatment")
MARKER      <- "eco_risk"

## Published Supp. Fig. 3B values, for the reproduction check.
PUBLISHED <- data.frame(eco_split = c("train", "val"), HR = c(9.17, 5.53))

P_RES <- file.path(out_folder, sprintf("cox_ecosplit_%s.csv", STAMP))
P_LOG <- file.path(out_folder, sprintf("cox_ecosplit_%s.txt", STAMP))
P_PDF <- file.path(out_folder, sprintf("cox_ecosplit_forest_%s.pdf", STAMP))

stopifnot(file.exists(P_ECO_OUT))
d0 <- read.csv(P_ECO_OUT, stringsAsFactors = FALSE) %>%
  mutate(
    event         = case_when(progression == "CO" ~ 1L, progression == "NCO" ~ 0L, TRUE ~ NA_integer_),
    sex           = factor(sex, levels = c("F", "M")),
    dysplasia_bin = factor(dysplasia_bin, levels = c("No Dysplasia", "Dysplasia")),
    treatment     = factor(treatment, levels = c("No", "Yes")),
    ## the submitted state of the same variable, for the collapse post-mortem
    dysplasia_sub = factor(ifelse(is.na(dysplasia_orig) | dysplasia_orig == "", NA,
                            ifelse(dysplasia_orig == "N", "No Dysplasia", "Dysplasia")),
                           levels = c("No Dysplasia", "Dysplasia")),
    eco_risk      = factor(ifelse(pred == "Diploid" | eco_risk_category == "Low Risk", "Low Risk",
                            ifelse(eco_risk_category == "High Risk", "High Risk", NA)),
                           levels = c("Low Risk", "High Risk"))
  )

con <- file(P_LOG, "wt"); sink(con, split = TRUE)
on.exit({sink(); close(con)}, add = TRUE)

cat("rev1 ecology-split Cox --", format(Sys.time()), "\n")
cat("source:", basename(P_ECO_OUT), "  minimum follow-up:", MIN_FU_DAYS, "days\n")

prep <- function(s) d0[!is.na(d0$eco_split) & d0$eco_split == s &
                       !is.na(d0$fu_time) & d0$fu_time > MIN_FU_DAYS & !is.na(d0$event), ]

## ---------------------------------------------------------------------------
## 1. Separation diagnostics -- why it collapsed, and whether it still does
## ---------------------------------------------------------------------------
hdr("SEPARATION DIAGNOSTIC: dysplasia x outcome, submitted vs updated")

sep_flag <- function(tb) {
  if (any(dim(tb) < 2)) return("NOT ESTIMABLE (only one level present)")
  if (any(rowSums(tb) == 0) || any(colSums(tb) == 0)) return("NOT ESTIMABLE (empty margin)")
  if (any(tb == 0)) return("COMPLETE SEPARATION (a zero cell) -> infinite coefficient")
  if (min(tb) < 3) return(sprintf("near-separation (smallest cell = %d) -> wide CI", min(tb)))
  "estimable"
}

for (s in c("train", "val")) {
  b <- prep(s)
  subhdr("eco_split = %s   n = %d, events = %d", s, nrow(b), sum(b$event))
  for (nm in c("dysplasia_sub", "dysplasia_bin")) {
    tb <- table(b[[nm]], b$event)
    cat("\n", ifelse(nm == "dysplasia_sub", "AS SUBMITTED", "UPDATED     "),
        "  (n with a grade = ", sum(!is.na(b[[nm]])), " of ", nrow(b), ")\n", sep = "")
    print(tb)
    cat("  -> ", sep_flag(tb), "\n", sep = "")
  }
  cat("\ntreatment x outcome:\n"); print(table(b$treatment, b$event, useNA = "ifany"))
  cat("BE length available: ", sum(!is.na(b$BELength)), " of ", nrow(b), "\n", sep = "")
}

## ---------------------------------------------------------------------------
## 2. Models
## ---------------------------------------------------------------------------
usable <- function(d, covars, announce = NULL) {
  ok <- vapply(covars, function(v)
    v %in% names(d) && sum(!is.na(d[[v]])) > 0 &&
      (!is.factor(d[[v]]) || n_distinct(d[[v]][!is.na(d[[v]])]) > 1), logical(1))
  if (!is.null(announce) && any(!ok))
    cat(sprintf("  [%s] dropped (no variation): %s\n", announce, paste(covars[!ok], collapse = ", ")))
  covars[ok]
}

fit_one <- function(d, covars, label, split, analysis) {
  covars <- usable(d, covars)
  if (!length(covars)) return(NULL)
  dd <- d[stats::complete.cases(d[, c(covars, "fu_time", "event")]), ]
  if (nrow(dd) < 8 || sum(dd$event) < 3) {
    cat(sprintf("  skipped %-6s %-24s (n=%d, events=%d)\n", split, label, nrow(dd), sum(dd$event)))
    return(NULL)
  }
  m <- try(coxph(as.formula(paste("Surv(fu_time, event) ~", paste(covars, collapse = " + "))),
                 data = dd, control = coxph.control(iter.max = 200)), silent = TRUE)
  if (inherits(m, "try-error")) { cat(sprintf("  FAILED %-6s %s\n", split, label)); return(NULL) }
  tidy(m, exponentiate = TRUE, conf.int = TRUE) %>%
    transmute(eco_split = split, analysis, model = label, term,
              HR = estimate, CI_low = conf.low, CI_high = conf.high, p = p.value,
              n = nrow(dd), events = sum(dd$event),
              ## an unbounded or absurd interval is the numerical signature of
              ## the collapse this script exists to document
              unstable = !is.finite(conf.high) | conf.high > 1e3 | estimate > 1e3 | estimate < 1e-3)
}

res <- list()
for (s in c("train", "val")) {
  b <- prep(s)
  hdr("eco_split = %s   n = %d, events = %d", s, nrow(b), sum(b$event))
  invisible(usable(b, c(CLIN_COVARS, MARKER), announce = s))

  for (v in c(CLIN_COVARS, MARKER)) res[[length(res) + 1]] <- fit_one(b, v, v, s, "univariate")
  ## the submitted-state dysplasia, to show the collapse numerically
  res[[length(res) + 1]] <- fit_one(b, "dysplasia_sub", "dysplasia (as submitted)", s, "univariate")

  res[[length(res) + 1]] <- fit_one(b, CLIN_COVARS, "clinical only", s, "multivariate")
  res[[length(res) + 1]] <- fit_one(b, c(CLIN_COVARS, MARKER), paste("clinical +", MARKER), s, "multivariate")
}
res <- bind_rows(res)
write.csv(res, P_RES, row.names = FALSE)

pretty <- res %>%
  mutate(`HR (95% CI)` = sprintf("%.2f (%.2f-%.2f)", HR, CI_low, CI_high),
         P = ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)),
         flag = ifelse(unstable, "UNSTABLE", "")) %>%
  select(eco_split, analysis, model, term, `HR (95% CI)`, P, n, events, flag)

hdr("UNIVARIATE"); print(as.data.frame(filter(pretty, analysis == "univariate")), row.names = FALSE)
hdr("MULTIVARIATE"); print(as.data.frame(filter(pretty, analysis == "multivariate")), row.names = FALSE)

## ---------------------------------------------------------------------------
## 3. Reproduction check against the published Supp. Fig. 3B values
## ---------------------------------------------------------------------------
hdr("REPRODUCTION CHECK vs published Supp. Fig. 3B")
rep_chk <- res %>%
  filter(analysis == "univariate", term == "eco_riskHigh Risk") %>%
  select(eco_split, HR_now = HR, CI_low, CI_high, p) %>%
  left_join(PUBLISHED, by = "eco_split") %>%
  mutate(published = HR, diff = round(HR_now - HR, 3),
         match = ifelse(abs(HR_now - HR) < 0.05, "MATCH", "DIFFERS")) %>%
  select(eco_split, published, HR_now, CI_low, CI_high, p, diff, match)
print(as.data.frame(rep_chk %>% mutate(across(where(is.numeric), ~ round(.x, 3)))), row.names = FALSE)

## ---------------------------------------------------------------------------
pdf(P_PDF, width = 7.5, height = 5)
pl <- res %>% filter(analysis == "univariate", !unstable, is.finite(CI_high))
if (nrow(pl)) print(
  ggplot(pl, aes(x = HR, y = term)) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40") +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.2, colour = "#2C3E50") +
    geom_point(size = 2.4, colour = "#2C3E50") +
    geom_text(aes(label = sprintf("%.2f (%.2f-%.2f) p=%s", HR, CI_low, CI_high,
                                  ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))),
              hjust = 0, nudge_y = 0.3, size = 2.5) +
    scale_x_log10() +
    facet_wrap(~ paste0("eco_split = ", eco_split, "  (n=", n, ", events=", events, ")"),
               scales = "free_y", ncol = 1) +
    labs(title = "Univariate Cox, ecology model train / validation splits",
         x = "Hazard ratio (95% CI, log scale)", y = NULL) +
    theme_minimal(base_size = 9) +
    theme(panel.grid.major.y = element_blank(), strip.text = element_text(face = "bold")))
invisible(dev.off())

cat("\nwritten:", P_RES, "\n"); cat("written:", P_LOG, "\n"); cat("written:", P_PDF, "\n")
