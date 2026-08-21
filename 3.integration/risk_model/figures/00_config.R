# =====================================================================
# Task 1 figures - Elastic Net vs. LASSO (Reviewer 1, comment #10)
# Shared configuration. Reuses the analysis folder's data/fitting code
# directly (one directory up) rather than duplicating it.
# =====================================================================
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) ".")
if (!file.exists(file.path(here, "00_config.R"))) here <- "."
ANALYSIS_DIR <- Sys.getenv("BEACON_TASK1_CODE", dirname(here))
source(file.path(ANALYSIS_DIR, "00_config.R"))
source(file.path(ANALYSIS_DIR, "lib_en.R"))

OUT_FIG <- Sys.getenv("BEACON_FIG_OUT", unset = NA_character_)
if (is.na(OUT_FIG)) {
  if (is.na(BE_MASTER))
    stop("Set BEACON_FIG_OUT (or BE_MASTER, from which it's derived).")
  OUT_FIG <- file.path(BE_MASTER, "4.plotting/output/rev1_elastic_net_fig")
}
dir.create(OUT_FIG, recursive = TRUE, showWarnings = FALSE)

# House palette, read off BEACON/4.plotting/code/{4.eco_metrics.Rmd, 4.cox_plot.Rmd}
# and the published feature_plot.Rmd (mae_7_5_withnuc/).
PAL <- list(
  # Within-model risk group (Low/High), matches create_survival_plot()'s
  # color_set 1 in 4.eco_metrics.Rmd - used for every KM panel here.
  risk        = c("Low Risk" = "#4575b4", "High Risk" = "#fc8d59"),
  # Coefficient sign, matches the published feature_plot.Rmd exactly.
  coef_sign   = c("Positive" = "steelblue", "Negative" = "coral"),
  # Model identity (LASSO vs Elastic Net) - a THIRD palette, deliberately
  # not reusing risk-group blue/orange or the red/blue "risk direction"
  # colors from 4.cox_plot.Rmd, so "which model" is never visually
  # confusable with "which risk group" or "which direction of effect".
  model       = c("LASSO" = "#1b9e77", "Elastic Net (alpha=0.5)" = "#7570b3")
)

THEME_ROC <- theme_light(base_size = 14) +
  theme(panel.grid = element_blank(),
        axis.title = element_text(size = 14),
        axis.text  = element_text(size = 12))

# Which single fit these figures illustrate: the published configuration
# (mae, 7 folds, seed 5) used throughout task 1's response table. Not the
# 25-seed stability grid - these are the concrete pair of models named in
# the Results paragraph (published LASSO vs EN alpha=0.5).
FIG_SEED <- PUBLISHED_SEED
