# Task 6 -- supporting plots. Folder only; none of these is figure-bound.
#
# They exist so the claims in the response letter can be checked visually and
# so there is a fallback if a reviewer asks for something the two main panels
# do not show.

source("00_config.R")

preds_slide   <- load_predictions()
preds_patient <- to_patient(preds_slide)

cat("Supporting plots ->", out_folder, "\n")

# --- 1. Forest of test-cohort AUC with DeLong CIs ------------------------
forest_df <- function(preds, models) {
  preds %>%
    filter(split == "test", model %in% models) %>% droplevels() %>%
    group_by(model) %>%
    summarise(as.data.frame(t(auc_ci(flow, prob))), .groups = "drop")
}

make_forest <- function(df, level_label) {
  ggplot(df, aes(auc, reorder(model, auc), colour = model)) +
    geom_vline(xintercept = 0.5, linetype = "dotted",
               colour = "grey60", linewidth = 0.3) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.18, linewidth = 0.4) +
    geom_point(size = 2) +
    geom_text(aes(label = sprintf("%.3f (%.3f-%.3f)", auc, lo, hi)),
              hjust = 0, nudge_y = 0.30, size = 2.2, colour = "grey20") +
    scale_colour_manual(values = model_colours, guide = "none") +
    scale_y_discrete(labels = model_labels) +
    scale_x_continuous(limits = c(0.3, 1.02), breaks = seq(0.3, 1.0, 0.1)) +
    labs(x = sprintf("AUC, test cohort (%s level)", level_label), y = NULL,
         subtitle = "Point estimate with DeLong 95% CI; dotted line = chance") +
    theme_task6()
}

save_pdf(make_forest(forest_df(preds_slide, all_models), "slide"),
         "forest_test_slide_allarms", width = 12, height = 8)
save_pdf(make_forest(forest_df(preds_slide, primary_models), "slide"),
         "forest_test_slide", width = 12, height = 6.5)
save_pdf(make_forest(forest_df(preds_patient, primary_models), "patient"),
         "forest_test_patient", width = 12, height = 6.5)

# --- 2. Rung-3 training histories ----------------------------------------
# Shows the deep arms trained properly and were not stopped on a noise spike --
# the specific failure that the windowed checkpoint selection was added to
# prevent. Useful if a reviewer questions whether ref 34 was given a fair run.
hist_files <- list.files(results_dir, pattern = "^training_history_.*\\.csv$",
                         full.names = TRUE)
if (length(hist_files)) {
  hist <- lapply(hist_files, function(f) {
    nm <- sub("^training_history_", "", tools::file_path_sans_ext(basename(f)))
    read.csv(f, stringsAsFactors = FALSE) %>% mutate(model = nm)
  }) %>% bind_rows() %>%
    mutate(model = factor(model, levels = model_levels))

  chosen <- read.csv(file.path(results_dir, "benchmark_rung3.csv"),
                     stringsAsFactors = FALSE) %>%
    transmute(model = factor(model, levels = model_levels), epoch = selected_epoch)

  p_hist <- ggplot(hist, aes(epoch, val_auc, colour = model)) +
    geom_hline(yintercept = 0.5, linetype = "dotted",
               colour = "grey60", linewidth = 0.3) +
    geom_line(linewidth = 0.4, alpha = 0.65) +
    geom_line(aes(y = val_auc_smoothed), linewidth = 0.7, na.rm = TRUE) +
    geom_vline(data = chosen, aes(xintercept = epoch, colour = model),
               linetype = "dashed", linewidth = 0.4) +
    facet_wrap(~ model, ncol = 2,
               labeller = labeller(model = model_labels)) +
    scale_colour_manual(values = model_colours, guide = "none") +
    labs(x = "Epoch", y = "Validation AUC",
         subtitle = paste("Thin line = per-epoch, thick = 3-epoch mean used for",
                          "selection; dashed = epoch kept")) +
    theme_task6() +
    theme(strip.background = element_rect(fill = "grey92", colour = NA),
          strip.text = element_text(colour = "grey20", size = 7))
  save_pdf(p_hist, "rung3_training_history", width = 15, height = 10)
}

# --- 3. Score distributions on the test cohort ---------------------------
# Where a model has collapsed, the two label groups overlap almost completely.
# Makes the near-chance arms legible in a way an AUC number does not.
p_dist <- preds_slide %>%
  filter(split == "test", model %in% primary_models) %>% droplevels() %>%
  mutate(truth = factor(flow, levels = c(0, 1),
                        labels = c("DNA content normal", "DNA content abnormal"))) %>%
  ggplot(aes(truth, prob, fill = truth)) +
  geom_boxplot(outlier.size = 0.4, linewidth = 0.3, width = 0.6) +
  facet_wrap(~ model, ncol = 5, labeller = labeller(model = model_labels)) +
  scale_fill_manual(values = c("DNA content normal" = "#92c5de",
                               "DNA content abnormal" = "#ca0020"),
                    guide = "none") +
  labs(x = NULL, y = "Predicted probability, test cohort") +
  theme_task6() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 6),
        strip.background = element_rect(fill = "grey92", colour = NA),
        strip.text = element_text(colour = "grey20", size = 6.5))
save_pdf(p_dist, "score_distributions_test_slide", width = 18, height = 7)

cat("\ndone\n")
