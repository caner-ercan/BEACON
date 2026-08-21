# =====================================================================
# Fit the two named models once (LASSO alpha=1, Elastic Net alpha=0.5;
# published CV settings, seed 5) and save per-patient scores for both
# splits, plus coefficients, for the panel scripts to consume.
# =====================================================================
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) ".")
if (!file.exists(file.path(here, "00_config.R"))) here <- "BEACON/4.plotting/code/rev1_elastic_net_fig"
source(file.path(here, "00_config.R"))

dat <- load_model_data(PATHS, EXPECT, drop_zero_var = DROP_ZERO_VAR)
cat(sprintf("data: %d train patients, %d test patients, %d features\n",
            EXPECT$train_patients, EXPECT$val_patients, length(dat$features)))

fs <- FIT_SETTINGS$published; fs$label <- "published"

fits <- list(
  lasso = fit_one(dat, fs, "lasso", FIG_SEED, ALPHA_FIXED, PREDICT_TYPE),
  en    = fit_one(dat, fs, "fixed", FIG_SEED, ALPHA_FIXED, PREDICT_TYPE)
)

for (nm in names(fits)) {
  r <- fits[[nm]]
  cat(sprintf("%-6s alpha=%.1f  n_features=%d  threshold=%.4f  AUC train(rs)=%.3f val=%.3f\n",
              nm, r$row$alpha, r$row$n_features, r$row$threshold,
              r$row$auc_train, r$row$auc_val))
}

# Sanity check against the response table already reported to CE.
stopifnot(abs(fits$lasso$row$auc_val - 0.8166667) < 1e-4)
stopifnot(abs(fits$en$row$auc_val    - 0.8000000) < 1e-4)

saveRDS(fits, file.path(OUT_FIG, "fits_lasso_en.RDS"))
cat("\nwrote:", file.path(OUT_FIG, "fits_lasso_en.RDS"), "\n")
