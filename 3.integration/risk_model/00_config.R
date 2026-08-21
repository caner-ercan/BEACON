# =====================================================================
# Task 1 - Elastic Net vs. LASSO  (Reviewer 1, comment #10)
# Shared configuration.
#
# DESIGN PRINCIPLE
#   Exactly ONE thing changes between arms: the glmnet `alpha` penalty
#   mixing parameter. Data, split, aggregation, thresholding and metrics
#   are held byte-identical across every arm, so any difference in the
#   reported numbers is attributable to alpha and nothing else.
#
# THE SPLIT IS READ, NOT REGENERATED
#   The published patient-level train/val assignment is stored in
#   eco_patients.csv (`eco_split`), together with the published per-patient
#   risk labels (`risk_category`). We read it. The RNG that produced it is
#   NOT reproducible (seed=5 in the surviving .Rmd gives a 47/18 patient
#   split, not the published 43/22 - a dplyr/R version row-ordering
#   difference), and the production script `grid_search_lasso.R` is lost.
#   Reading the split removes that dependency entirely.
# =====================================================================

## Set BE_MASTER to your BE_master checkout (a relative default here would
## depend on the invocation directory, not the script's location, so this
## is required rather than guessed) - or set BEACON_DATA/BEACON_CLIN
## directly to override just one of the two paths below.
BE_MASTER <- Sys.getenv("BE_MASTER", unset = NA_character_)
if (is.na(BE_MASTER) && !nzchar(Sys.getenv("BEACON_DATA")))
  stop("Set the BE_MASTER environment variable to your BE_master checkout ",
       "(or BEACON_DATA/BEACON_CLIN directly).")
DATA_ROOT <- Sys.getenv("BEACON_DATA",
  file.path(BE_MASTER, "3.integration/integration_project/spatial_analysis/tabular"))
CLIN_ROOT <- Sys.getenv("BEACON_CLIN",
  file.path(BE_MASTER, "0.input/organised_wsi_patient"))

OUT_DIR <- file.path(DATA_ROOT, "ecomerged_lasso_model", "rev1_elastic_net")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PATHS <- list(
  # Feature matrix. RESOLVED 2026-08-04 by the reproduction gate: refitting
  # at the exported lambda.1se reproduces the published coefficient vector
  # to machine precision (cor = 1, max abs diff 1.3e-15) with the _250812
  # file, and NOT with the undated one (cor 0.982, 34 vs 28 nonzero).
  features   = file.path(DATA_ROOT, "calculated",
                 Sys.getenv("BEACON_FEATURES", "merged_ecology_wNuc_standardised_250812.RDS")),
  # wsi -> patient map (777 slides, 0 unmatched).
  samples    = file.path(CLIN_ROOT, "MIL_samples.csv"),
  # PUBLISHED eco_split + PUBLISHED per-patient risk_category + fu_time.
  eco        = file.path(CLIN_ROOT, "eco_patients.csv")
)

# ---- recovered published model ---------------------------------------
# BE_master/4.plotting/output/3.1.ecology/mae_7_5_withnuc/ holds the actual
# fitted objects from the published run. From model_parameters.rds and the
# cv.glmnet call recorded in cv_model.rds:
#   alpha=1, family="binomial", type.measure="mae", nfolds=7,
#   lambda = exp(seq(log(0.001), log(1), length.out=100)),  NO `alignment`
#   lambda.1se = 0.04641589, lambda.min = 0.001
#   best_threshold = 0.6103361, seed=5, nuc_type="withnuc", fu_time_type=0
#   nobs = 138 slides, nvars = 1538 -> NO zero-variance filtering was applied
PUBLISHED_LAMBDA_1SE <- 0.04641589
PUBLISHED_THRESHOLD  <- 0.6103361
PUBLISHED_LAMBDA_SEQ <- exp(seq(log(0.001), log(1), length.out = 100))
# The published run passed all 1538 columns. Match it.
DROP_ZERO_VAR <- FALSE

# ---- the published model, for context in the output table ------------
# Source: ecomerged_lasso_model/results_250815_binary_newsplit_fu0withnuc.csv
#   row: measure="mae", k_fold=7, seed=5, nuc_type="withnuc", fu_time_type=0
# Identified uniquely: both the AUC pair and the balanced-accuracy pair
# match the Results text ("86.7% and 81.7% balanced accuracy and
# 0.902, 0.817 AUC on training and validation subsets").
# Values recomputed end-to-end from predictions_with_metadata.csv (the
# published per-patient scores) and verified against the published results
# CSV row. This supplies the KM p-value for BOTH splits, which the results
# CSV reported for the test split only.
PUBLISHED <- list(
  n_train = 43, n_val = 22,
  auc_train = 0.9021739, bal_acc_train = 0.8663043,
  sens_train = 0.7826087, spec_train = 0.9500000, km_p_train = 1.037156e-06,
  auc_val   = 0.8166667, bal_acc_val   = 0.8166667,
  sens_val  = 0.8333333, spec_val      = 0.8000000, km_p_val   = 0.01366343,
  acc_val = 0.8181818, best_threshold = 0.6103361,
  n_features = 28, lambda_1se = 0.04641589,
  surv_pval_val = 0.01366343
)

# ---- expected geometry; asserted at load time ------------------------
EXPECT <- list(slides = 202, patients = 65,
               train_slides = 138, train_patients = 43,
               val_slides = 64,  val_patients = 22)

# ---- the two models compared --------------------------------------
# (1) FIT SETTING: the published model's cross-validation settings.
#     type.measure="mae", nfolds=7, and the explicit lambda sequence
#     stated in Methods ("100 logarithmically spaced values (0.001-1.0)").
#     lambda.1se is the published selection rule, applied identically to
#     both arms below.
FIT_SETTINGS <- list(
  published = list(type.measure = "mae", nfolds = 7,
                   lambda = PUBLISHED_LAMBDA_SEQ)
)

# (2) ALPHA: the two arms this comparison is about.
#   lasso : alpha = 1.0, the published model's own penalty. Refit here
#           (rather than only read from disk) so LASSO and Elastic Net are
#           measured within one implementation, on identical folds.
#   fixed : alpha = 0.5, the conventional elastic net.
ALPHA_STRATEGIES <- c("fixed", "lasso")
ALPHA_FIXED <- 0.5

# (3) Seed. The published LASSO run is measure="mae", k_fold=7, seed=5,
#     nuc_type="withnuc" - matched here exactly.
#
#     IMPORTANT: seed=5 here is NOT the same random draw as seed=5 in the
#     original script. There, set.seed(5) was consumed first by the
#     train/val split sampling and only then by cv.glmnet's folds. Here the
#     split is read from disk, so the RNG stream starts at a different
#     position and the fold assignment differs. Pinning the seed makes THIS
#     run reproducible; it does not recreate the original's folds.
PUBLISHED_SEED <- 5

outfile <- function(stem, ext = "csv") file.path(OUT_DIR, sprintf("%s_seed5.%s", stem, ext))

# ---- fixed protocol (copied from the published pipeline) -------------
PREDICT_TYPE <- "response"  # published best_threshold in [0.52,0.67] => probability scale
AGGREGATION  <- "max"       # slide scores -> patient score by maximum
THRESHOLD    <- "youden"    # cut chosen on the TRAINING patient-level ROC
