# Task 6 -- ROC curves, one PDF per split, slide and patient level.
#
# Figure-bound: roc_slide_val.pdf and roc_slide_test.pdf (Panel A, two sub-panels).
# The train ROC is produced too but stays in the folder: every model is near
# ceiling there by construction, so it carries no argument -- it exists to show
# the fits were not degenerate.

source("00_config.R")

preds_slide   <- load_predictions()
preds_patient <- to_patient(preds_slide)

n_by_split <- preds_slide %>%
  filter(model == "DACOR") %>%
  count(split, name = "n") %>%
  mutate(pos = preds_slide %>% filter(model == "DACOR") %>%
           group_by(split) %>% summarise(p = sum(flow), .groups = "drop") %>% pull(p))

make_roc_plot <- function(preds, which_split, models, level_label, show_legend = TRUE) {
  df <- preds %>% filter(split == which_split, model %in% models) %>% droplevels()

  curves <- roc_points(df)
  # Legend ordered by AUC, best first: on a ROC panel the reader is matching
  # curve height to label, so ladder order would make them hunt.
  aucs <- df %>%
    group_by(model) %>%
    summarise(auc = auc_ci(flow, prob)[1], .groups = "drop") %>%
    arrange(desc(auc)) %>%
    mutate(label = sprintf("%s  %.3f", model_labels[as.character(model)], auc))

  unit <- if (identical(level_label, "patient")) "patients" else "biopsies"
  n_tot <- nrow(df) / length(unique(df$model))
  n_pos <- df %>% filter(model == df$model[1]) %>% summarise(s = sum(flow)) %>% pull(s)

  ggplot(curves, aes(fpr, tpr, colour = model, linetype = model)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dotted",
                colour = "grey60", linewidth = 0.3) +
    geom_line(linewidth = 0.6) +
    scale_colour_manual(values = model_colours, breaks = aucs$model,
                        labels = aucs$label) +
    scale_linetype_manual(values = model_linetypes, breaks = aucs$model,
                          labels = aucs$label) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    scale_x_continuous(breaks = seq(0, 1, 0.25)) +
    scale_y_continuous(breaks = seq(0, 1, 0.25)) +
    labs(x = "1 - specificity", y = "Sensitivity",
         # ASCII only: the base pdf device does not carry the middot glyph in
         # its default encoding and silently renders it as "..".
         subtitle = sprintf("%s, %d %s (%d abnormal)",
                            split_short[[which_split]], n_tot, unit, n_pos)) +
    theme_task6() +
    theme(legend.position = if (show_legend) c(0.97, 0.03) else "none",
          legend.justification = c(1, 0),
          legend.margin = margin(1, 2, 1, 2))
}

cat("ROC curves ->", out_folder, "\n")

# --- slide level ---------------------------------------------------------
for (sp in split_levels) {
  p <- make_roc_plot(preds_slide, sp, primary_models, "slide")
  save_pdf(p, paste0("roc_slide_", sp), width = 9.5, height = 10)
}

# All seven arms, including the stain-normalised variants. Folder only: the
# dashed variants sit on top of their base arms and would cost two curves for
# no information in a main figure.
for (sp in split_levels) {
  p <- make_roc_plot(preds_slide, sp, all_models, "slide")
  save_pdf(p, paste0("roc_slide_", sp, "_allarms"), width = 10.5, height = 11)
}

# --- patient level -------------------------------------------------------
for (sp in split_levels) {
  p <- make_roc_plot(preds_patient, sp, primary_models, "patient")
  save_pdf(p, paste0("roc_patient_", sp), width = 9.5, height = 10)
}
