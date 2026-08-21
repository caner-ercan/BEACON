# Task 6 -- supplementary tables, formatted for the response letter / supplement.
#
# NOTE ON A FIXED BUG. The pipeline's benchmark_all_splits_patient_level.csv has
# a correct AUC column but its delong_p_vs_dacor column is a copy of the
# slide-level p-values: 03_fit_evaluate.py always computed DeLong from
# slide-level predictions keyed on wsi, whichever metric function was passed.
# It matters -- e.g. D stain-normalised is AUC 0.497 at slide level and 0.660 at
# patient level, so the slide p-value does not describe the patient comparison.
# Patient-level DeLong is recomputed here from the per-patient aggregated
# scores. Slide-level values are read straight through and agree with the
# pipeline exactly.

source("00_config.R")

preds_slide   <- load_predictions()
preds_patient <- to_patient(preds_slide)

#' Paired DeLong of each model against DACOR, on the samples they share.
delong_vs_dacor <- function(preds) {
  dacor <- preds %>% filter(model == "DACOR")
  key   <- if ("wsi" %in% names(preds)) "wsi" else "patient"

  preds %>%
    filter(model != "DACOR") %>%
    group_by(model, split) %>%
    group_modify(~ {
      d <- dacor %>% filter(split == .y$split)
      j <- inner_join(.x, d, by = key, suffix = c("_m", "_d"))
      if (nrow(j) < 5 || length(unique(j$flow_d)) < 2)
        return(tibble::tibble(delong_p = NA_real_))
      r_m <- pROC::roc(j$flow_d, j$prob_m, quiet = TRUE, direction = "<")
      r_d <- pROC::roc(j$flow_d, j$prob_d, quiet = TRUE, direction = "<")
      tibble::tibble(delong_p = pROC::roc.test(r_m, r_d, method = "delong")$p.value)
    }) %>%
    ungroup()
}

fmt_p <- function(p) {
  ifelse(is.na(p), "-",
         ifelse(p < 1e-12, "<1e-12", formatC(p, format = "e", digits = 1)))
}

build_table <- function(preds, level_label) {
  metrics <- preds %>%
    group_by(model, split) %>%
    summarise(n = n(), n_abnormal = sum(flow),
              as.data.frame(t(auc_ci(flow, prob))), .groups = "drop")

  drops <- metrics %>%
    select(model, split, auc) %>%
    tidyr::pivot_wider(names_from = split, values_from = auc) %>%
    transmute(model, val_to_test_drop = val - test)

  metrics %>%
    left_join(delong_vs_dacor(preds), by = c("model", "split")) %>%
    left_join(drops, by = "model") %>%
    arrange(match(model, c("DACOR", "rung1_M", "rung2_CM", "rung3_D", "rung3_CMD",
                           "rung3_D_reinhard", "rung3_CMD_reinhard")),
            match(split, split_levels)) %>%
    transmute(
      Model      = model_labels[as.character(model)],
      Split      = c(train = "Train (discovery)", val = "Validation (discovery)",
                     test  = "Test (independent cohort)")[as.character(split)],
      n, `n abnormal` = n_abnormal,
      AUC        = sprintf("%.3f", auc),
      `95% CI`   = sprintf("%.3f-%.3f", lo, hi),
      `DeLong p vs DACOR` = fmt_p(delong_p),
      `Val-to-test AUC drop` = ifelse(split == "test",
                                      sprintf("%.3f", val_to_test_drop), "")
    )
}

cat("Tables ->", out_folder, "\n")

tbl_slide   <- build_table(preds_slide,   "slide")
tbl_patient <- build_table(preds_patient, "patient")

write.csv(tbl_slide,
          file.path(out_folder, "supp_table_benchmark_slide_level.csv"),
          row.names = FALSE)
write.csv(tbl_patient,
          file.path(out_folder, "supp_table_benchmark_patient_level.csv"),
          row.names = FALSE)
cat("  supp_table_benchmark_slide_level.csv\n")
cat("  supp_table_benchmark_patient_level.csv\n\n")

print(as.data.frame(tbl_slide), row.names = FALSE)

# Confirm the slide-level p-values still agree with the pipeline's own, so the
# recomputation above cannot be quietly diverging from what was run on the HPC.
ref <- read.csv(file.path(results_dir, "benchmark_all_splits_slide_level.csv"),
                stringsAsFactors = FALSE) %>%
  filter(!is.na(delong_p_vs_dacor)) %>%
  transmute(model, split, ref_p = delong_p_vs_dacor)
mine <- delong_vs_dacor(preds_slide) %>%
  transmute(model = as.character(model), split = as.character(split), my_p = delong_p)
chk <- inner_join(ref, mine, by = c("model", "split")) %>%
  mutate(rel_diff = abs(my_p - ref_p) / pmax(ref_p, 1e-300))
cat("\nslide-level DeLong agreement with pipeline (max relative difference):",
    signif(max(chk$rel_diff, na.rm = TRUE), 3), "\n")
