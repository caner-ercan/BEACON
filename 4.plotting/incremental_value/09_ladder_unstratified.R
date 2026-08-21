## ---------------------------------------------------------------------------
## rev1_incremental_value / 09_ladder_unstratified.R
##
## The self-contained ladder analysis for this task -- supersedes the former
## 01_ladder.R and 03_summary_table.R (both removed; this script covers their
## ground plus more, from one set of fits rather than files joined across
## scripts) and replaces 07_panel_figure.R's former panel A. Adds a fifth
## analysis column, "Whole, unstratified", alongside "Whole, cohort-
## stratified", so the sensitivity of the pooled result to cohort handling is
## visible in one place, and ports 01_ladder.R's circularity check (below) as
## the one piece of its analysis this script didn't already cover.
##
## WHY THIS MATTERS. In the pooled cohort the two configurations disagree
## sharply for BEACON but barely at all for flow cytometry:
##
##                    clinical-adjusted     + strata(cohort)
##   flow             HR 2.78  p=0.0011     HR 2.64  p=0.0024
##   DACOR            HR 3.26  p=0.0012     HR 2.13  p=0.082
##   BEACON           HR 4.00  p=0.0003     HR 2.19  p=0.095
##
## The mechanism is where each marker's positives sit. Of 34 flow-positive
## patients 56% are in the discovery cohort; of 30 BEACON-high patients only
## 20% are (24 of 30 are in the test cohort, where 74% progress regardless).
## strata(cohort) restricts every comparison to within-cohort, so it asks
## BEACON to prove itself almost entirely inside the cohort with the least
## outcome variance -- and the test cohort contributes 31 of the 49 events,
## dominating the pooled within-cohort estimate. Both configurations are
## defensible; reporting only one is not.
##
## Everything is recomputed here in a single pass so the stratified and
## unstratified columns come from the same fits and complete-case set --
## rather than joining across files, which is what let a provenance mismatch
## into the original Table S-X (C-index taken from unstratified models and
## printed beside stratified HRs; fixed separately in 08_tables.R).
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))
suppressMessages({library(ggplot2); library(gridExtra); library(grid)})

d0 <- load_patients()

MODELS <- list(
  "1  Clinicopathological (baseline)" = CLIN,
  "2  + Flow cytometry"               = c(CLIN, "flow"),
  "3  + DACOR"                        = c(CLIN, "pred"),
  "4  + BEACON"                       = c(CLIN, "eco_risk")
)
NEED <- unique(c(CLIN, "flow", "pred", "eco_risk", "fu_time", "event"))

COLUMNS <- list(
  list(key = "discovery",   label = "Discovery (MDA)",              d = d0[d0$dataset == "MDA", ], strat = FALSE),
  list(key = "test",        label = "Test (NU)",                    d = d0[d0$dataset == "NU", ],  strat = FALSE),
  list(key = "holdout",     label = "BEACON holdout (n=22)",
       d = d0[d0$eco_fit_status == "held-out", ], strat = FALSE, base = c("age", "sex")),
  list(key = "whole_strat", label = "Whole, cohort-stratified",     d = d0, strat = TRUE),
  list(key = "whole_unstr", label = "Whole, unstratified",          d = d0, strat = FALSE)
)

fit_column <- function(col) {
  base <- if (is.null(col$base)) CLIN else col$base
  models <- lapply(MODELS, function(cv) c(base, setdiff(cv, CLIN)))
  need <- unique(c(base, "flow", "pred", "eco_risk", "fu_time", "event"))
  dd <- col$d[complete.cases(col$d[, need]), ]
  if (nrow(dd) < 8 || sum(dd$event) < 3) return(NULL)
  suffix <- if (col$strat) " + strata(dataset)" else ""

  f <- function(cv) {
    cv <- cv[vapply(cv, function(v) !is.factor(dd[[v]]) || n_distinct(dd[[v]]) > 1, logical(1))]
    coxph(as.formula(paste("Surv(fu_time, event) ~", paste(cv, collapse = "+"), suffix)), data = dd)
  }
  m0 <- f(models[[1]]); C0 <- summary(m0)$concordance[1]
  out <- list()
  for (i in seq_along(models)) {
    nm <- names(models)[i]
    marker <- setdiff(models[[i]], base)
    estimable <- !length(marker) || n_distinct(dd[[marker]]) > 1
    if (!estimable) {
      out[[length(out)+1]] <- data.frame(column = col$label, key = col$key, model = nm,
        n = nrow(dd), events = sum(dd$event), C = NA_real_, HR = NA_real_,
        CI_low = NA_real_, CI_high = NA_real_, p = NA_real_,
        LRT_chisq = NA_real_, LRT_p = NA_real_, estimable = FALSE); next
    }
    m <- f(models[[i]])
    C <- summary(m)$concordance[1]
    HR <- CI_low <- CI_high <- p <- LRT_chisq <- LRT_p <- NA_real_
    if (i > 1) {
      a <- anova(m0, m); LRT_chisq <- a$Chisq[2]; LRT_p <- a$`Pr(>|Chi|)`[2]
      co <- summary(m)$coefficients
      r <- grep(paste0("^", marker), rownames(co))[1]
      HR <- co[r, 2]; p <- co[r, 5]
      ci <- exp(confint(m))[r, ]; CI_low <- ci[1]; CI_high <- ci[2]
    }
    out[[length(out)+1]] <- data.frame(column = col$label, key = col$key, model = nm,
      n = nrow(dd), events = sum(dd$event), C = C, HR = HR, CI_low = CI_low,
      CI_high = CI_high, p = p, LRT_chisq = LRT_chisq, LRT_p = LRT_p, estimable = TRUE)
  }
  bind_rows(out)
}

con <- file(file.path(out_folder, sprintf("ladder_unstrat_%s.txt", STAMP)), "wt")
sink(con, split = TRUE); on.exit({sink(); close(con)}, add = TRUE)

hdr("LADDER incl. BOTH whole-cohort configurations (stratified and unstratified)")
cat("Every column fit independently on its own complete-case set. C-index here is\n")
cat("APPARENT and comes from the same fit as the HR beside it -- no cross-file join.\n")

res <- bind_rows(lapply(COLUMNS, fit_column))

for (col in COLUMNS) {
  x <- res[res$key == col$key, ]
  if (!nrow(x)) next
  sub("%s  (n=%d, events=%d)%s", col$label, x$n[1], x$events[1],
      if (col$strat) "  [strata(dataset)]" else "")
  pretty <- x %>% mutate(
    `HR (95% CI)` = ifelse(!estimable, "not estimable", ifelse(is.na(HR), "-- baseline --",
                      sprintf("%.2f (%.2f-%.2f)", HR, CI_low, CI_high))),
    p   = ifelse(is.na(p), "", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))),
    LRT = ifelse(is.na(LRT_p), "", sprintf("%.2f / %s", LRT_chisq,
                  ifelse(LRT_p < 0.001, "<0.001", sprintf("%.3f", LRT_p)))),
    `C (apparent)` = ifelse(is.na(C), "", sprintf("%.3f", C))
  ) %>% select(model, `HR (95% CI)`, p, `LRT chi2 / p` = LRT, `C (apparent)`)
  print(as.data.frame(pretty), row.names = FALSE)
}

hdr("SIDE-BY-SIDE: the whole cohort under both configurations")
w <- res[res$key %in% c("whole_strat", "whole_unstr") & res$estimable & !is.na(res$HR), ]
cmp <- w %>% mutate(cfg = ifelse(key == "whole_strat", "stratified", "unstratified")) %>%
  select(model, cfg, HR, p) %>%
  tidyr::pivot_wider(names_from = cfg, values_from = c(HR, p))
print(as.data.frame(cmp %>% mutate(across(where(is.numeric), ~round(.x, 4)))), row.names = FALSE)

write.csv(res, file.path(out_folder, sprintf("ladder_unstrat_%s.csv", STAMP)), row.names = FALSE)

## ---------------------------------------------------------------------------
## BEACON circularity check (ported from the former 01_ladder.R, "Table B")
##
## The "holdout" column above already restricts to eco_fit_status=="held-out"
## for the full ladder. This is the narrower, targeted question underneath
## that: does the eco_risk/BEACON marker ITSELF replicate out of sample, at
## several levels of covariate adjustment, and how does that compare to
## patients the ecology model was fit on ("in-sample")?
##
## Minimal covariate set for THIS check, not the full CLIN. Purpose here is
## isolating whether the eco_risk marker itself replicates out of sample --
## requiring BELength/dysplasia_bin completeness on top of that, in a group
## already down to ~22 patients, does not serve that purpose. It costs a lot:
## in the held-out group, missingness in BELength/dysplasia is concentrated
## in older MDA records disproportionately among ecology-Low patients, so
## full-CLIN-adjustment collapses Low Risk n from 10 to 2 (both of whom had
## early events) and FLIPS the sign (HR 0.24) -- a complete-case artifact,
## not a real reversal. Kept below as "full clinical adjustment" for
## transparency, but the univariate/age+sex rows are the ones to trust.
## ---------------------------------------------------------------------------
fit_b <- function(d, covars, label) {
  need <- unique(c(covars, "eco_risk", "fu_time", "event"))
  dd <- d[complete.cases(d[, need]), ]
  if (nrow(dd) < 8 || sum(dd$event) < 2 || n_distinct(dd$eco_risk) < 2) {
    cat(sprintf("%-46s too few / no contrast (n=%d, events=%d)\n", label, nrow(dd), sum(dd$event)))
    return(NULL)
  }
  rhs0 <- if (length(covars)) paste(covars, collapse = "+") else "1"
  m0 <- coxph(as.formula(paste("Surv(fu_time,event) ~", rhs0)), data = dd)
  m1 <- coxph(as.formula(paste("Surv(fu_time,event) ~", paste(c(covars, "eco_risk"), collapse = "+"))), data = dd)
  a <- anova(m0, m1)
  co <- summary(m1)$coefficients["eco_riskHigh Risk", ]
  data.frame(group = label, n = nrow(dd), events = sum(dd$event),
             C_clinical = summary(m0)$concordance[1], C_plus = summary(m1)$concordance[1],
             LRT_chisq = a$Chisq[2], LRT_p = a$`Pr(>|Chi|)`[2],
             HR = co[2], CI_low = exp(co[1] - 1.96 * co[3]), CI_high = exp(co[1] + 1.96 * co[3]), p = co[5])
}

hdr("BEACON circularity check (clinical -> +BEACON only)")
cat("Pooled across BOTH cohorts by eco_split membership, not by original\n")
cat("cohort -- eco_split is the patient-constrained split the ecology model\n")
cat("actually trained/validated on, and it crosses the MDA/NU boundary.\n")
cat("'in-sample' = ecology model was FIT on these patients (optimistic).\n")
cat("'held-out'  = genuine holdout for the ecology model (honest, but n~20).\n")
sub("Held-out set, by adjustment level -- univariate/age+sex are the trustworthy rows")

resB <- bind_rows(
  fit_b(d0[d0$eco_fit_status == "held-out", ], character(0), "held-out: univariate eco_risk"),
  fit_b(d0[d0$eco_fit_status == "held-out", ], c("age", "sex"), "held-out: + age + sex"),
  fit_b(d0[d0$eco_fit_status == "held-out", ], CLIN, "held-out: + full CLIN (artifact -- see note)"),
  fit_b(d0[d0$eco_fit_status == "in-sample", ], CLIN, "in-sample: + full CLIN"),
  fit_b(d0[d0$eco_fit_status %in% c("in-sample", "held-out"), ], CLIN, "combined: + full CLIN (= whole-cohort, unadjusted)")
)
if (!is.null(resB) && nrow(resB)) {
  print(as.data.frame(resB %>% mutate(
    `C (clin->plus)` = sprintf("%.3f -> %.3f", C_clinical, C_plus),
    LRT = sprintf("chi2=%.2f p=%s", LRT_chisq, ifelse(LRT_p < 0.001, "<0.001", sprintf("%.4g", LRT_p))),
    `HR (95% CI)` = sprintf("%.2f (%.2f-%.2f) p=%s", HR, CI_low, CI_high,
                             ifelse(p < 0.001, "<0.001", sprintf("%.3g", p)))
  ) %>% select(group, n, events, `C (clin->plus)`, LRT, `HR (95% CI)`)), row.names = FALSE)
}
write.csv(resB, file.path(out_folder, sprintf("ladder_circularity_%s.csv", STAMP)), row.names = FALSE)
cat("\nwritten:", file.path(out_folder, sprintf("ladder_circularity_%s.csv", STAMP)), "\n")

## --- forest plot, 5 columns ------------------------------------------------
COL_MARKER <- c("Flow cytometry" = "#DD8452", "DACOR" = "#8172B2", "BEACON" = "#C44E52")
MARKER_LABEL <- c("2  + Flow cytometry" = "Flow cytometry", "3  + DACOR" = "DACOR",
                  "4  + BEACON" = "BEACON")
LAB <- c(discovery = "Discovery\n(n=105, 18 ev)", test = "Test cohort\n(n=42, 31 ev)",
         holdout = "BEACON holdout\n(n=22, 12 ev)",
         whole_strat = "Whole, cohort-strat.\n(n=147, 49 ev)",
         whole_unstr = "Whole, UNstratified\n(n=147, 49 ev)")

fA <- res[res$model %in% names(MARKER_LABEL) & !is.na(res$HR), ]
fA$set <- factor(LAB[fA$key], levels = LAB)
fA$marker <- factor(MARKER_LABEL[fA$model], levels = c("BEACON", "DACOR", "Flow cytometry"))
fA$sig <- ifelse(fA$p < 0.05, "p<0.05", "n.s.")
fA$label <- sprintf("%.2f (%.2f-%.2f)", fA$HR, fA$CI_low, fA$CI_high)
fA$psub  <- sprintf("p=%s", ifelse(fA$p < 0.001, "<0.001", sprintf("%.3f", fA$p)))

p <- ggplot(fA, aes(x = HR, y = marker, colour = marker)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.3) +
  geom_errorbarh(aes(xmin = pmax(CI_low, 0.05), xmax = pmin(CI_high, 100)), height = 0.22, linewidth = 0.5) +
  geom_point(aes(shape = sig), size = 2) +
  geom_text(aes(label = label), hjust = 0, nudge_y = 0.30, size = 1.7, colour = "grey15") +
  geom_text(aes(label = psub), hjust = 0, nudge_y = -0.32, size = 1.7, colour = "grey40") +
  scale_colour_manual(values = COL_MARKER, guide = "none") +
  scale_shape_manual(values = c("p<0.05" = 16, "n.s." = 1), name = NULL) +
  scale_x_log10(limits = c(0.05, 300), breaks = c(0.1, 1, 10, 100)) +
  facet_wrap(~ set, ncol = 5) +
  labs(title = "Incremental hazard ratio vs. clinicopathological baseline — both whole-cohort configurations",
       subtitle = "Rightmost two panels are the SAME patients, differing only in whether cohort is stratified.",
       x = "Hazard ratio (95% CI, log scale)", y = NULL) +
  theme_bw(base_size = 8) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        strip.background = element_rect(fill = "grey92", colour = NA),
        strip.text = element_text(face = "bold", size = 7),
        plot.title = element_text(size = 9, face = "bold"),
        plot.subtitle = element_text(size = 7), legend.position = "bottom")

P <- file.path(out_folder, sprintf("figS7A_forest_ladder_unstrat_%s.pdf", STAMP))
ggsave(P, p, width = 10.5, height = 3.4)
cat("\nwritten:", P, "\n")

## --- formatted PDF table ---------------------------------------------------
THEME <- ttheme_minimal(
  core = list(fg_params = list(hjust = 0, x = 0.02, fontsize = 7.5),
             bg_params  = list(fill = c("grey98", "white"), col = NA)),
  colhead = list(fg_params = list(fontface = "bold", fontsize = 7.5, hjust = 0, x = 0.02),
                bg_params  = list(fill = "grey85", col = NA)),
  padding = unit(c(3, 2), "mm"))

dfT <- res %>% mutate(
  Set = "", Model = model,
  `HR (95% CI)` = ifelse(!estimable, "not estimable", ifelse(is.na(HR), "-- baseline --",
                    sprintf("%.2f (%.2f-%.2f)", HR, CI_low, CI_high))),
  p   = ifelse(is.na(p), "", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))),
  `LRT chi2 / p` = ifelse(is.na(LRT_p), "", sprintf("%.2f / %s", LRT_chisq,
                    ifelse(LRT_p < 0.001, "<0.001", sprintf("%.3f", LRT_p)))),
  `C (apparent)` = ifelse(is.na(C), "", sprintf("%.3f", C))) %>%
  select(Set, Model, n, events, `HR (95% CI)`, p, `LRT chi2 / p`, `C (apparent)`)
first <- !duplicated(res$key)
dfT$Set[first] <- res$column[first]

gT <- tableGrob(dfT, rows = NULL, theme = THEME)
CAP <- paste(
  "Incremental prognostic value, with the pooled cohort shown under BOTH configurations. The last two",
  "blocks contain the same 147 patients and differ only in whether cohort is entered as a stratum.",
  "Stratifying restricts every comparison to within-cohort; because 24 of 30 BEACON-high patients are in",
  "the test cohort (where 74% progress regardless) and that cohort contributes 31 of 49 events, it",
  "removes most of the contrast BEACON relies on, while flow cytometry -- split 56/44 across cohorts --",
  "is barely affected. C-index is APPARENT throughout this table and is taken from the same fit as the",
  "HR beside it (not bootstrap-corrected, and not joined from another file).", sep = " ")
cap_lines <- strwrap(CAP, width = 130)
cap <- textGrob(paste(cap_lines, collapse = "\n"), x = 0.01, y = unit(1, "npc"), hjust = 0, vjust = 1,
                gp = gpar(fontsize = 6.8, fontface = "italic", col = "grey30"))
pad <- unit(3, "mm")
hts <- unit.c(sum(gT$heights) + pad, unit(length(cap_lines) * 6.8 * 1.35, "pt") + pad)
full <- arrangeGrob(grobs = list(gT, cap), ncol = 1, heights = hts)
PT <- file.path(out_folder, sprintf("tableSX_ladder_unstrat_%s.pdf", STAMP))
pdf(PT, width = 9.2, height = sum(convertHeight(hts, "inches", valueOnly = TRUE)) + 0.3)
grid.draw(full); dev.off()
cat("written:", PT, "\n")
