# =====================================================================
# Task 1 - Elastic Net vs. LASSO  (Reviewer 1, comment #10)
#
# Fits the two models this comparison is about, at the published CV
# settings (mae / 7 folds / explicit lambda grid), on the published
# train/val split, at the published fold seed:
#   lasso : alpha = 1.0 (the published model's own penalty)
#   fixed : alpha = 0.5 (the conventional elastic net)
#
# Usage:  Rscript 01_fit_en_vs_lasso.R
# =====================================================================
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) ".")
if (!file.exists(file.path(here, "00_config.R"))) here <- "BEACON/3.integration/risk_model"
source(file.path(here, "00_config.R"))
source(file.path(here, "lib_en.R"))

cat("=== Task 1: Elastic Net vs LASSO ===\n")
cat("features file :", basename(PATHS$features), "\n")

dat <- load_model_data(PATHS, EXPECT, drop_zero_var = DROP_ZERO_VAR)
cat(sprintf("slides %d (train %d / val %d) | patients %d (train %d / val %d)\n",
            nrow(dat$df), nrow(dat$x_train), nrow(dat$x_val),
            dplyr::n_distinct(dat$df$patient),
            dplyr::n_distinct(dat$meta_train$patient),
            dplyr::n_distinct(dat$meta_val$patient)))
cat(sprintf("features %d (dropped %d zero-variance on train) | p/n = %.1f\n\n",
            length(dat$features), dat$n_dropped, length(dat$features)/nrow(dat$x_train)))

cat("fits to run:", length(ALPHA_STRATEGIES), "(published fit setting, seed", PUBLISHED_SEED, ")\n\n")

fs <- FIT_SETTINGS$published; fs$label <- "published"

rows <- vector("list", length(ALPHA_STRATEGIES))
feats <- list(); risks <- list()

for (i in seq_along(ALPHA_STRATEGIES)) {
  strategy <- ALPHA_STRATEGIES[i]
  r <- fit_one(dat, fs, strategy, PUBLISHED_SEED,
               alpha_fixed = ALPHA_FIXED, predict_type = PREDICT_TYPE)

  rows[[i]] <- r$row
  key <- paste("published", strategy, PUBLISHED_SEED, sep = "|")
  feats[[key]] <- r$features
  risks[[key]] <- r$risk_val
  cat(sprintf("  %s: alpha=%.1f n_features=%d AUC train(rs)=%.3f val=%.3f\n",
              strategy, r$row$alpha, r$row$n_features, r$row$auc_train, r$row$auc_val))
}

results <- dplyr::bind_rows(rows)

saveRDS(list(results = results, features = feats, risks = risks,
             published = PUBLISHED, feature_names = dat$features,
             published_risk = dat$meta_val %>%
               dplyr::distinct(patient, risk_published)),
        outfile("en_vs_lasso_raw", "RDS"))
write.csv(results, outfile("en_vs_lasso_fits"), row.names = FALSE)

cat("\nwrote:", outfile("en_vs_lasso_fits"), "\n")
cat("      ", outfile("en_vs_lasso_raw", "RDS"), "\n")
