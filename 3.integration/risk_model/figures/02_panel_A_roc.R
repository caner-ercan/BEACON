# =====================================================================
# Panel A - ROC curves, LASSO vs Elastic Net (alpha=0.5), each split
# plotted separately (A1 = train, A2 = ecology_test).
# Style follows the house ROC plot in 4.eco_metrics.Rmd: pROC + ggplot2,
# dashed diagonal, theme_light, AUC annotated as text.
# =====================================================================
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) ".")
if (!file.exists(file.path(here, "00_config.R"))) here <- "BEACON/4.plotting/code/rev1_elastic_net_fig"
source(file.path(here, "00_config.R"))
suppressPackageStartupMessages({library(pROC); library(ggplot2); library(dplyr); library(purrr)})

fits <- readRDS(file.path(OUT_FIG, "fits_lasso_en.RDS"))

roc_panel <- function(split_name, pat_field, n_patients) {
  d <- bind_rows(
    fits$lasso[[pat_field]] %>% mutate(model = "LASSO"),
    fits$en[[pat_field]]    %>% mutate(model = "Elastic Net (alpha=0.5)")
  )

  curves <- d %>% group_split(model) %>% map(~ {
    r <- roc(.x$progression, .x$score, direction = "<", quiet = TRUE)
    data.frame(specificity = rev(r$specificities), sensitivity = rev(r$sensitivities),
               model = unique(.x$model))
  }) %>% bind_rows()

  aucs <- d %>% group_split(model) %>% map_df(~ {
    a <- as.numeric(auc(roc(.x$progression, .x$score, direction = "<", quiet = TRUE)))
    data.frame(model = unique(.x$model), auc = a)
  })

  # AUC values go in the subtitle rather than an in-panel annotate(): with
  # only two curves the panel is too small for annotate() text to avoid
  # clipping against the legend at the default plot margins.
  subtitle <- paste(sprintf("%s AUC = %.3f", aucs$model, aucs$auc), collapse = "   ")

  ggplot(curves, aes(x = 1 - specificity, y = sensitivity, color = model)) +
    geom_line(linewidth = 1.2) +
    geom_abline(linetype = "dashed", color = "gray") +
    scale_color_manual(values = PAL$model) +
    labs(x = "1 - Specificity (False Positive Rate)",
         y = "Sensitivity (True Positive Rate)",
         color = NULL,
         title = sprintf("%s (n = %d patients)", split_name, n_patients),
         subtitle = subtitle) +
    THEME_ROC +
    theme(legend.position = c(0.75, 0.15),
          plot.subtitle = element_text(size = 10))
}

p_train <- roc_panel("Training set", "pat_train", EXPECT$train_patients)
p_test  <- roc_panel("Held-out test set", "pat_val", EXPECT$val_patients)

ggsave(file.path(OUT_FIG, "figS_A1_roc_train.pdf"), p_train, width = 14, height = 12, units = "cm")
ggsave(file.path(OUT_FIG, "figS_A2_roc_test.pdf"),  p_test,  width = 14, height = 12, units = "cm")
cat("wrote figS_A1_roc_train.pdf, figS_A2_roc_test.pdf\n")
