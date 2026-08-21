# =====================================================================
# Task 1 - shared data loading, fitting and evaluation helpers.
# =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(glmnet)
  library(pROC); library(caret); library(survival)
})

# ---------------------------------------------------------------------
# load_model_data()
#   Assembles the exact modelling table behind Fig. 5: DACOR-positive
#   slides carrying the PUBLISHED eco_split.
#
#   Note on fu_time > 0: the published pipeline applied this filter before
#   splitting. We do not re-apply it - the 202 slides that carry an
#   eco_split are already exactly the survivors of that filter (verified:
#   all have fu_time > 0). Slides whose patient has no eco_split are the
#   ones the published run dropped.
# ---------------------------------------------------------------------
load_model_data <- function(paths, expect, drop_zero_var = TRUE) {
  feats <- readRDS(paths$features)
  ms    <- read.csv(paths$samples)
  eco   <- read.csv(paths$eco)

  meta_cols <- c("wsi", "progression", "flow", "pred", "dataset")
  stopifnot(all(meta_cols %in% colnames(feats)))

  df <- feats %>%
    left_join(ms %>% select(wsi, patient), by = "wsi") %>%
    left_join(eco %>% select(patient, eco_split, fu_time,
                             progression_pt = progression,
                             risk_published = risk_category),
              by = "patient") %>%
    filter(pred == 1, !is.na(eco_split), eco_split %in% c("train", "val"))

  # --- integrity checks: fail loudly rather than silently model the wrong set
  stopifnot(nrow(df) == expect$slides)
  stopifnot(n_distinct(df$patient) == expect$patients)
  stopifnot(sum(df$eco_split == "train") == expect$train_slides)
  stopifnot(sum(df$eco_split == "val")   == expect$val_slides)
  stopifnot(n_distinct(df$patient[df$eco_split == "train"]) == expect$train_patients)
  stopifnot(n_distinct(df$patient[df$eco_split == "val"])   == expect$val_patients)
  # slide-level 0/1 label must agree with the patient-level CO/NCO label
  stopifnot(all((df$progression == 1) == (df$progression_pt == "CO")))
  stopifnot(all(df$fu_time > 0))
  # a patient must never straddle the split
  stopifnot(nrow(distinct(df, patient, eco_split)) == expect$patients)

  feature_cols <- setdiff(colnames(feats), meta_cols)

  # Zero-variance columns are dropped using the TRAINING slides only, so no
  # information from the validation patients reaches the feature selection.
  # Applied identically to every arm, so it cannot bias the comparison.
  tr <- df$eco_split == "train"
  if (drop_zero_var) {
    sdv <- vapply(df[tr, feature_cols], sd, numeric(1))
    keep <- feature_cols[is.finite(sdv) & sdv > 0]
  } else {
    # The published run passed all 1538 columns (nvars = 1538 in
    # fitted_model.rds), so matching it means NOT filtering.
    keep <- feature_cols
  }

  list(
    df = df,
    features = keep,
    n_dropped = length(feature_cols) - length(keep),
    x_train = as.matrix(df[tr,  keep]),
    y_train = df$progression[tr],
    x_val   = as.matrix(df[!tr, keep]),
    y_val   = df$progression[!tr],
    meta_train = df[tr,  c("wsi","patient","progression","fu_time","dataset","risk_published")],
    meta_val   = df[!tr, c("wsi","patient","progression","fu_time","dataset","risk_published")]
  )
}

# ---------------------------------------------------------------------
# aggregate_patient(): slide scores -> one score per patient by max,
# exactly as the published pipeline did.
# ---------------------------------------------------------------------
aggregate_patient <- function(meta, scores) {
  meta %>%
    mutate(score = as.vector(scores)) %>%
    group_by(patient, fu_time, progression) %>%
    summarise(score = max(score), risk_published = first(risk_published), .groups = "drop")
}

# ---------------------------------------------------------------------
# patient_metrics(): patient-level AUC / balanced accuracy / sens / spec /
# accuracy and the log-rank p for the resulting risk groups.
# ---------------------------------------------------------------------
patient_metrics <- function(pat, threshold) {
  pat <- pat %>% mutate(risk = ifelse(score > threshold, "High Risk", "Low Risk"))

  auc <- tryCatch(as.numeric(auc(roc(pat$progression, pat$score,
                                     direction = "<", quiet = TRUE))),
                  error = function(e) NA_real_)

  cm <- tryCatch(
    confusionMatrix(
      factor(pat$risk, levels = c("Low Risk", "High Risk")),
      factor(pat$progression, levels = c(0, 1), labels = c("Low Risk", "High Risk")),
      positive = "High Risk"),
    error = function(e) NULL)

  lr <- tryCatch({
    if (n_distinct(pat$risk) < 2) NA_real_
    else survdiff(Surv(fu_time, progression) ~ risk, data = pat)$pvalue
  }, error = function(e) NA_real_)

  list(
    n_patients = nrow(pat),
    auc      = auc,
    bal_acc  = if (is.null(cm)) NA_real_ else unname(cm$byClass["Balanced Accuracy"]),
    sens     = if (is.null(cm)) NA_real_ else unname(cm$byClass["Sensitivity"]),
    spec     = if (is.null(cm)) NA_real_ else unname(cm$byClass["Specificity"]),
    accuracy = if (is.null(cm)) NA_real_ else unname(cm$overall["Accuracy"]),
    logrank_p = lr,
    n_high   = sum(pat$risk == "High Risk"),
    risk_vec = setNames(pat$risk, pat$patient)
  )
}

# ---------------------------------------------------------------------
# fit_one(): one complete model fit + evaluation.
#
#   alpha_strategy:
#     "fixed" - alpha = alpha_fixed (the elastic net arm)
#     "lasso" - alpha = 1 (the published model's own penalty)
# ---------------------------------------------------------------------
fit_one <- function(dat, fit_setting, alpha_strategy, seed,
                    alpha_fixed, predict_type = "response") {

  # One fold assignment per seed, shared by both alphas. This pairs the
  # arms: LASSO and the fixed-alpha elastic net see identical folds, so
  # any difference between them cannot be an artefact of fold assignment.
  set.seed(seed)
  foldid <- sample(rep(seq_len(fit_setting$nfolds), length.out = nrow(dat$x_train)))

  cvfit_at <- function(a) {
    args <- list(x = dat$x_train, y = dat$y_train, alpha = a,
                 family = "binomial", type.measure = fit_setting$type.measure,
                 foldid = foldid)
    if (!is.null(fit_setting$lambda)) args$lambda <- fit_setting$lambda
    do.call(cv.glmnet, args)
  }

  alpha <- if (alpha_strategy == "lasso") 1.0 else alpha_fixed
  cv_model <- cvfit_at(alpha)

  lambda <- cv_model$lambda.1se
  fit <- glmnet(x = dat$x_train, y = dat$y_train, alpha = alpha,
                family = "binomial", lambda = lambda)

  # Cross-validated AUC on the training folds at the selected lambda, using
  # the SAME folds. This is the internal-validation estimate: unlike the
  # in-sample training AUC it is not driven to 1.0 by overfitting, so it is
  # the honest number to quote for training performance.
  cv_auc_train <- tryCatch({
    a <- list(x = dat$x_train, y = dat$y_train, alpha = alpha,
              family = "binomial", type.measure = "auc", foldid = foldid)
    if (!is.null(fit_setting$lambda)) a$lambda <- fit_setting$lambda
    cva <- do.call(cv.glmnet, a)
    cva$cvm[which.min(abs(cva$lambda - lambda))]
  }, error = function(e) NA_real_)

  co <- as.matrix(coef(fit))
  nz <- rownames(co)[co[, 1] != 0]
  nz <- setdiff(nz, "(Intercept)")

  s_train <- predict(fit, newx = dat$x_train, type = predict_type)
  s_val   <- predict(fit, newx = dat$x_val,   type = predict_type)

  pat_train <- aggregate_patient(dat$meta_train, s_train)
  pat_val   <- aggregate_patient(dat$meta_val,   s_val)

  # Threshold is chosen on the TRAINING patients only (Youden's J).
  thr <- tryCatch(
    coords(roc(pat_train$progression, pat_train$score, quiet = TRUE),
           "best", ret = "threshold", best.method = "youden")$threshold[1],
    error = function(e) 0.5)
  if (!is.finite(thr)) thr <- 0.5

  m_tr <- patient_metrics(pat_train, thr)
  m_va <- patient_metrics(pat_val,   thr)

  list(
    row = data.frame(
      fit_setting    = fit_setting$label,
      alpha_strategy = alpha_strategy,
      seed           = seed,
      alpha          = alpha,
      lambda         = lambda,
      n_features     = length(nz),
      threshold      = thr,
      # cross-validated on the training folds; quote this, NOT auc_train,
      # which is resubstitution and saturates at 1.0 for the denser fits
      cv_auc_train   = cv_auc_train,
      # train (resubstitution - saturates, see note above)
      auc_train      = m_tr$auc,     bal_acc_train = m_tr$bal_acc,
      sens_train     = m_tr$sens,    spec_train    = m_tr$spec,
      acc_train      = m_tr$accuracy, logrank_p_train = m_tr$logrank_p,
      n_high_train   = m_tr$n_high,  n_pat_train   = m_tr$n_patients,
      # ecology test split (the held-out estimate)
      auc_val        = m_va$auc,     bal_acc_val   = m_va$bal_acc,
      sens_val       = m_va$sens,    spec_val      = m_va$spec,
      acc_val        = m_va$accuracy, logrank_p_val = m_va$logrank_p,
      n_high_val     = m_va$n_high,  n_pat_val     = m_va$n_patients,
      stringsAsFactors = FALSE),
    features  = nz,
    coefficients = setNames(co[nz, 1], nz),
    risk_val  = m_va$risk_vec,
    pat_train = pat_train,
    pat_val   = pat_val
  )
}
