# Task 6 -- AUC across the three splits, one line per model (Panel B).
#
# This is the panel that carries the argument: every benchmark arm reaches
# 0.82-0.92 on held-out discovery-cohort data, and all of them fall away on the
# independent cohort, which was digitised on a different scanner. DACOR falls
# least and stays highest. A ROC panel alone cannot show that, because it is a
# single split.

source("00_config.R")

preds_slide   <- load_predictions()
preds_patient <- to_patient(preds_slide)

summarise_auc <- function(preds, models) {
  preds %>%
    filter(model %in% models) %>%
    droplevels() %>%
    group_by(model, split) %>%
    summarise(as.data.frame(t(auc_ci(flow, prob))), .groups = "drop")
}

make_split_plot <- function(preds, models, level_label, show_ci = TRUE) {
  df <- summarise_auc(preds, models)

  # Train and validation are BOTH the discovery cohort on the same scanner;
  # only the test split changes scanner. The bracket and divider mark that
  # boundary explicitly, between validation and test -- captioning it any other
  # way misattributes where the domain shift happens.
  y_top <- 1.10
  p <- ggplot(df, aes(split, auc, colour = model, group = model)) +
    geom_hline(yintercept = 0.5, linetype = "dotted",
               colour = "grey60", linewidth = 0.3) +
    geom_vline(xintercept = 2.5, linetype = "dashed",
               colour = "grey55", linewidth = 0.35)
  if (show_ci) {
    p <- p + geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.08,
                           linewidth = 0.3, alpha = 0.55,
                           position = position_dodge(width = 0.22))
  }
  p +
    geom_line(aes(linetype = model), linewidth = 0.6,
              position = position_dodge(width = 0.22)) +
    geom_point(size = 1.5, position = position_dodge(width = 0.22)) +
    annotate("segment", x = 0.7, xend = 2.3, y = y_top, yend = y_top,
             colour = "grey45", linewidth = 0.3) +
    annotate("text", x = 1.5, y = y_top + 0.035, label = "Discovery cohort",
             size = 2.5, colour = "grey30") +
    annotate("segment", x = 2.7, xend = 3.3, y = y_top, yend = y_top,
             colour = "grey45", linewidth = 0.3) +
    annotate("text", x = 3.0, y = y_top + 0.035, label = "Different scanner",
             size = 2.5, colour = "grey30") +
    scale_colour_manual(values = model_colours,
                        labels = model_labels, breaks = names(model_labels)) +
    scale_linetype_manual(values = model_linetypes,
                          labels = model_labels, breaks = names(model_labels)) +
    scale_x_discrete(labels = split_axis_labels) +
    scale_y_continuous(limits = c(0.35, y_top + 0.06),
                       breaks = seq(0.4, 1.0, 0.1)) +
    labs(x = NULL, y = sprintf("AUC (%s level)", level_label)) +
    theme_task6() +
    theme(legend.position = "bottom",
          legend.direction = "vertical")
}

cat("AUC by split ->", out_folder, "\n")

# Figure-bound.
save_pdf(make_split_plot(preds_slide, primary_models, "slide"),
         "auc_by_split_slide", width = 9.5, height = 11)

# Folder only.
save_pdf(make_split_plot(preds_slide, all_models, "slide"),
         "auc_by_split_slide_allarms", width = 10, height = 12)
save_pdf(make_split_plot(preds_patient, primary_models, "patient"),
         "auc_by_split_patient", width = 9.5, height = 11)

# --- the same information as an explicit drop, folder only ---------------
# Useful as a fallback if the line panel reads as too busy at final size.
drops <- summarise_auc(preds_slide, primary_models) %>%
  select(model, split, auc) %>%
  tidyr::pivot_wider(names_from = split, values_from = auc) %>%
  mutate(drop = val - test) %>%
  arrange(drop)

p_drop <- ggplot(drops, aes(reorder(model, drop), drop, fill = model)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", drop)), hjust = -0.15, size = 2.5) +
  scale_fill_manual(values = model_colours, guide = "none") +
  scale_x_discrete(labels = model_labels) +
  scale_y_continuous(limits = c(0, max(drops$drop) * 1.18), expand = c(0, 0)) +
  coord_flip() +
  labs(x = NULL, y = "AUC lost from validation to test cohort",
       subtitle = "Smaller is better: performance retained across scanners") +
  theme_task6()
save_pdf(p_drop, "val_to_test_drop_slide", width = 11, height = 6.5)

cat("\nAUC by split, slide level:\n")
print(as.data.frame(drops), digits = 3)
