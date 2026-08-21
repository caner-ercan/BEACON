# =====================================================================
# Panel B - Kaplan-Meier progression-free survival, High vs Low risk
# (each model's own threshold), split by model x split:
#   B1 LASSO / train      B2 LASSO / ecology_test
#   B3 Elastic Net / train  B4 Elastic Net / ecology_test
# Style follows create_survival_plot() in 4.eco_metrics.Rmd: survminer
# ggsurvplot, color_set 1 palette, theme_survminer with the same font
# sizing, risk table below.
# =====================================================================
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) ".")
if (!file.exists(file.path(here, "00_config.R"))) here <- "BEACON/4.plotting/code/rev1_elastic_net_fig"
source(file.path(here, "00_config.R"))
suppressPackageStartupMessages({library(survival); library(survminer); library(dplyr)})

fits <- readRDS(file.path(OUT_FIG, "fits_lasso_en.RDS"))

ggsave_workaround <- function(g) survminer:::.build_ggsurvplot(
  x = g, surv.plot.height = NULL, risk.table.height = NULL, ncensor.plot.height = NULL)

km_panel <- function(r, pat_field, title) {
  pat <- r[[pat_field]] %>%
    mutate(risk = factor(ifelse(score > r$row$threshold, "High Risk", "Low Risk"),
                         levels = c("Low Risk", "High Risk"))) %>%
    filter(fu_time > 14)   # matches create_survival_plot()'s censoring-artifact guard

  fit <- surv_fit(Surv(fu_time / 365, progression) ~ risk, data = pat)
  n_high <- sum(pat$risk == "High Risk"); n_low <- sum(pat$risk == "Low Risk")

  plot <- ggsurvplot(fit, data = pat, pval = TRUE, xlab = "Time (years)",
    risk.table = TRUE, conf.int = FALSE,
    palette = unname(PAL$risk[levels(pat$risk)]),
    legend = "none", risk.table.y.text = FALSE, tables.height = 0.3,
    font.main = c(13, "bold"), font.x = c(10, "bold"), font.y = c(8, "bold"),
    font.tickslab = c(12, "plain"), pval.size = 5, size = 1,
    risk.table.fontsize = 4, title = sprintf("%s (n=%d low, %d high)", title, n_low, n_high),
    ggtheme = theme_survminer() +
      theme(axis.line = element_line(color = "black", size = 0.8),
            axis.ticks = element_line(color = "black", size = 0.8),
            axis.text.y = element_text(size = 8)))
  ggsave_workaround(plot)
}

specs <- list(
  B1 = list(r = fits$lasso, field = "pat_train", title = "LASSO - training set"),
  B2 = list(r = fits$lasso, field = "pat_val",   title = "LASSO - held-out test set"),
  B3 = list(r = fits$en,    field = "pat_train", title = "Elastic Net (alpha=0.5) - training set"),
  B4 = list(r = fits$en,    field = "pat_val",   title = "Elastic Net (alpha=0.5) - held-out test set")
)

for (nm in names(specs)) {
  s <- specs[[nm]]
  p <- km_panel(s$r, s$field, s$title)
  model_tag <- if (grepl("LASSO", s$title)) "lasso" else "en"
  split_tag <- if (grepl("training", s$title)) "train" else "test"
  f <- file.path(OUT_FIG, sprintf("figS_%s_km_%s_%s.pdf", nm, model_tag, split_tag))
  ggsave(f, p, width = 10, height = 9, dpi = 300)
  cat("wrote", basename(f), "\n")
}
