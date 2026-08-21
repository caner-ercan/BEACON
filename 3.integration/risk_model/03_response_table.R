# =====================================================================
# Task 1 - the response-letter table (Reviewer 1, comment #10).
#
# One row per model. Columns are the five metrics repeated for BOTH
# splits: ecology_train (43 patients) and ecology_test (22 patients).
#
#   AUC | CV-AUC | balanced accuracy | sensitivity | specificity | KM p
#
# Row 1 is the published LASSO, recomputed end-to-end from the exported
# per-patient scores in
#   BE_master/4.plotting/output/3.1.ecology/mae_7_5_withnuc/
# so that its KM p-value is available for BOTH splits (the published
# results CSV reported it for the test split only).
#
# ---------------------------------------------------------------------
# REPORTING RULE (CE, 2026-08-04): no absolute values in the paper.
#   Training AUC by resubstitution saturates at 1.000 for the denser
#   elastic-net fits, and 1.000 is not a quotable number. `CV_AUC_train`
#   -- cross-validated on the training folds at the selected lambda -- is
#   the honest training-performance column and never saturates. The
#   resubstitution column is retained in the CSV, flagged, but is not the
#   one to quote. Any cell that still lands on an absolute value is
#   marked with a dagger in the printed table.
# =====================================================================
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) ".")
if (!file.exists(file.path(here, "00_config.R"))) here <- "BEACON/3.integration/risk_model"
source(file.path(here, "00_config.R"))
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(pROC); library(survival); library(caret)
})

raw <- readRDS(outfile("en_vs_lasso_raw", "RDS"))
res <- raw$results

# ---- published LASSO row, recomputed from the exported scores ---------
EXPORT_DIR <- Sys.getenv("BEACON_PUBMODEL", unset = NA_character_)
if (is.na(EXPORT_DIR)) {
  if (is.na(BE_MASTER))
    stop("Set BEACON_PUBMODEL (or BE_MASTER, from which it's derived).")
  EXPORT_DIR <- file.path(BE_MASTER, "4.plotting/output/3.1.ecology/mae_7_5_withnuc")
}
pub_row <- NULL
if (file.exists(file.path(EXPORT_DIR, "predictions_with_metadata.csv"))) {
  pp  <- read.csv(file.path(EXPORT_DIR, "predictions_with_metadata.csv"))
  mp  <- readRDS(file.path(EXPORT_DIR, "model_parameters.rds"))
  one <- function(sp) {
    d <- pp %>% filter(eco_split == sp) %>%
      mutate(risk = ifelse(prediction_prob > mp$best_threshold, "High Risk", "Low Risk"))
    cm <- confusionMatrix(factor(d$risk, levels = c("Low Risk", "High Risk")),
            factor(d$progression, levels = c(0, 1), labels = c("Low Risk", "High Risk")),
            positive = "High Risk")
    list(n = nrow(d),
         auc = as.numeric(auc(roc(d$progression, d$prediction_prob, direction = "<", quiet = TRUE))),
         bal = unname(cm$byClass["Balanced Accuracy"]),
         sen = unname(cm$byClass["Sensitivity"]),
         spe = unname(cm$byClass["Specificity"]),
         km  = survdiff(Surv(fu_time, progression) ~ risk, data = d)$pvalue)
  }
  tr <- one("train"); va <- one("val")
  pub_row <- data.frame(
    model = "LASSO - PUBLISHED (manuscript Fig. 5)",
    params = sprintf("alpha=1, mae, %d folds, lambda.1se=%.5f", mp$k_fold, mp$lambda_1se),
    n_feat = PUBLISHED$n_features,
    tr_auc = tr$auc, tr_cvauc = NA_real_, tr_bal = tr$bal, tr_sen = tr$sen, tr_spe = tr$spe, tr_km = tr$km,
    te_auc = va$auc, te_bal = va$bal, te_sen = va$sen, te_spe = va$spe, te_km = va$km,
    stringsAsFactors = FALSE)
}

# ---- Elastic Net / LASSO candidate rows -------------------------------
arm_label <- c(lasso = "LASSO (alpha = 1)",
               fixed = sprintf("Elastic Net (alpha = %.1f)", ALPHA_FIXED))
set_label <- c(published = "mae, 7 folds, lambda grid 1e-3..1")

cand <- res %>%
  group_by(fit_setting, alpha_strategy) %>%
  summarise(across(c(alpha, n_features, cv_auc_train,
                     auc_train, bal_acc_train, sens_train, spec_train, logrank_p_train,
                     auc_val, bal_acc_val, sens_val, spec_val, logrank_p_val),
                   ~ median(.x, na.rm = TRUE)), .groups = "drop") %>%
  transmute(
    model  = unname(arm_label[alpha_strategy]),
    params = sprintf("alpha=%.2f, %s", alpha, unname(set_label[fit_setting])),
    n_feat = round(n_features),
    tr_auc = auc_train, tr_cvauc = cv_auc_train, tr_bal = bal_acc_train,
    tr_sen = sens_train, tr_spe = spec_train, tr_km = logrank_p_train,
    te_auc = auc_val, te_bal = bal_acc_val, te_sen = sens_val,
    te_spe = spec_val, te_km = logrank_p_val)

tbl <- bind_rows(pub_row, cand)
write.csv(tbl, outfile("en_response_table"), row.names = FALSE)

# ---- printing ---------------------------------------------------------
# Flag any cell that has landed on an absolute value (0 or 1). Per the
# reporting rule these must not be quoted in the paper.
f3 <- function(x) {
  ifelse(is.na(x), "-",
    ifelse(abs(x - 1) < 1e-9 | abs(x) < 1e-9,
           paste0(sprintf("%.3f", x), " +"), sprintf("%.3f", x)))
}
fp <- function(x) ifelse(is.na(x), "-",
  ifelse(x < 1e-4, sprintf("%.1e", x), sprintf("%.4f", x)))

pretty <- tbl %>% transmute(
  Model = model, Parameters = params, nFeat = n_feat,
  `TR CV-AUC` = f3(tr_cvauc), `TR AUC(rs)` = f3(tr_auc), `TR BalAcc` = f3(tr_bal),
  `TR Sens` = f3(tr_sen), `TR Spec` = f3(tr_spe), `TR KM p` = fp(tr_km),
  `TE AUC` = f3(te_auc), `TE BalAcc` = f3(te_bal), `TE Sens` = f3(te_sen),
  `TE Spec` = f3(te_spe), `TE KM p` = fp(te_km))

cat("\n=====================================================================\n")
cat("  Elastic Net vs LASSO - response table\n")
cat("  TR = ecology_train (n =", PUBLISHED$n_train, "patients) |",
    "TE = ecology_test (n =", PUBLISHED$n_val, "patients)\n")
cat("  run: single fit, published seed", PUBLISHED_SEED, "\n")
cat("  AUC(rs) = resubstitution, saturates -> do NOT quote. CV-AUC = cross-validated.\n")
cat("  + marks a cell at an absolute value (0.000 / 1.000); not quotable.\n")
cat("=====================================================================\n\n")
print(as.data.frame(pretty %>% select(Model, Parameters, nFeat,
        `TR CV-AUC`, `TR BalAcc`, `TR Sens`, `TR Spec`, `TR KM p`)), row.names = FALSE)
cat("\n")
print(as.data.frame(pretty %>% select(Model, Parameters,
        `TE AUC`, `TE BalAcc`, `TE Sens`, `TE Spec`, `TE KM p`)), row.names = FALSE)

abs_hits <- sum(grepl("\\+$", unlist(pretty[, grepl("^T", names(pretty))])))
cat("\ncells at an absolute value:", abs_hits, "\n")
cat("wrote:", outfile("en_response_table"), "\n")

# ---- KM (log-rank) p-values, eco_train vs eco_test side by side --------
# (merged in from the former 02_summary.R - the one other output that
# reached the paper/rebuttal, alongside the saturated-value guard below)
km <- res %>%
  mutate(arm = unname(arm_label[alpha_strategy])) %>%
  transmute(fit_setting, arm,
            KM_p_eco_train = fp(logrank_p_train),
            KM_p_eco_test  = fp(logrank_p_val),
            n_high_train, n_high_val)
write.csv(km, outfile("en_vs_lasso_km"), row.names = FALSE)
cat("\n=== KM log-rank p-values (published run: eco_test",
    sprintf("%.4f", PUBLISHED$surv_pval_val), ") ===\n")
print(as.data.frame(km), row.names = FALSE)
cat("wrote:", outfile("en_vs_lasso_km"), "\n")

# ---- GUARD: flag any saturated value (exactly 0 or 1) ------------------
# CE: no absolute numbers (1.00 AUC, 100% accuracy, 0% sensitivity) are to
# appear in the paper. This lists every cell that is exactly 0 or 1 so it
# cannot slip into a table unnoticed.
metric_cols <- grep("^(auc|bal_acc|sens|spec|acc|cv_auc)_", names(res), value = TRUE)
sat <- res %>%
  select(fit_setting, alpha_strategy, seed, all_of(metric_cols)) %>%
  tidyr::pivot_longer(all_of(metric_cols), names_to = "metric", values_to = "value") %>%
  filter(!is.na(value), value == 0 | value == 1)
write.csv(sat, outfile("en_vs_lasso_saturated"), row.names = FALSE)
cat("\n=== SATURATED VALUES (exactly 0 or 1) - do not put these in the paper ===\n")
if (nrow(sat) == 0) cat("  none\n") else
  print(as.data.frame(sat %>% count(fit_setting, alpha_strategy, metric, value)), row.names = FALSE)
cat("  -> report CV_AUC_train instead of AUC_train; held-out (ecology_test) metrics are unaffected.\n")
cat("wrote:", outfile("en_vs_lasso_saturated"), "\n")
