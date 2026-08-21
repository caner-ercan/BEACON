# Task 6 -- export slide-level labels and DACOR probabilities to CSV.
#
# The only R step in this task: DACOR's slide-level probabilities live inside an
# RDS, and the clinical table is a CSV. Both are pulled out once here so the rest
# of the pipeline is plain Python.
#
# Output: interim/slide_labels.csv with one row per slide --
#   wsi, patient, dataset, split, flow, dacor_pred, dacor_prob
#
# dacor_prob is DACOR's own output and is used ONLY as the comparator ROC.
# It never enters the baseline models.

suppressPackageStartupMessages(library(dplyr))

base <- Sys.getenv("BEACON_BASE", unset = NA_character_)
if (is.na(base)) stop("Set BEACON_BASE to the directory containing BE_master/ and BEACON/.")
be_master <- file.path(base, "BE_master")

clinical_csv <- file.path(
  be_master, "0.input/organised_wsi_patient/MIL_samples_rev1_260730.csv"
)
cell_level_rds <- file.path(
  be_master,
  "3.integration/integration_project/spatial_analysis/tabular/merged/cell_level_250808.RDS"
)
out_dir <- file.path(be_master, "revision/task6_benchmark/interim")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

clinical <- read.csv(clinical_csv, stringsAsFactors = FALSE) %>%
  select(wsi, patient, dataset, split, flow)

stopifnot(!any(is.na(clinical$flow)))
stopifnot(!any(duplicated(clinical$wsi)))

# One row per slide; pred_prob is constant within a slide.
cells <- readRDS(cell_level_rds)
dacor <- cells %>%
  select(wsi, pred, pred_prob) %>%
  distinct() %>%
  rename(dacor_pred = pred, dacor_prob = pred_prob)

if (any(duplicated(dacor$wsi))) {
  stop("pred_prob is not constant within wsi -- inspect before proceeding")
}

labels <- clinical %>% left_join(dacor, by = "wsi")

cat("slides:", nrow(labels), "\n")
cat("missing DACOR probability:", sum(is.na(labels$dacor_prob)), "\n")
print(table(labels$dataset, labels$split, useNA = "ifany"))
print(table(labels$dataset, labels$flow, useNA = "ifany"))

# Sanity check: DACOR on the test cohort should reproduce the published 0.825.
if (requireNamespace("pROC", quietly = TRUE)) {
  te <- labels %>% filter(dataset == "NU", !is.na(dacor_prob))
  auc <- pROC::auc(pROC::roc(te$flow, te$dacor_prob, quiet = TRUE))
  cat(sprintf("DACOR test-cohort AUC: %.3f (published: 0.825)\n", as.numeric(auc)))
}

write.csv(labels, file.path(out_dir, "slide_labels.csv"), row.names = FALSE)
cat("wrote", file.path(out_dir, "slide_labels.csv"), "\n")
