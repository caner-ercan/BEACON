## ---------------------------------------------------------------------------
## rev1_incremental_value / 00_config.R
## Shared paths for the reviewer-3-major-#1 incremental-value analysis.
## Sourced by 01_ladder.R, 02_reclassification.R.
## ---------------------------------------------------------------------------

suppressMessages({library(dplyr); library(survival)})
options(width = 200, stringsAsFactors = FALSE)

## Set BE_MASTER to your BE_master checkout (a relative default here would
## depend on the invocation directory, not the script's location, so this
## is required rather than guessed).
base_folder <- Sys.getenv("BE_MASTER", unset = NA_character_)
if (is.na(base_folder) || !dir.exists(base_folder))
  stop("Set the BE_MASTER environment variable to your BE_master checkout.")
message("BE_master: ", base_folder)

csv_folder <- file.path(base_folder, "0.input", "organised_wsi_patient")
P_ECO_IN <- file.path(csv_folder, "eco_patients_rev1_260730.csv")

out_folder <- file.path(base_folder, "4.plotting", "output", "rev1_incremental_value")
dir.create(out_folder, recursive = TRUE, showWarnings = FALSE)
STAMP <- format(Sys.Date(), "%y%m%d")

## -- shared analysis constants ------------------------------------------------
CLIN <- c("age", "sex", "BELength", "dysplasia_bin")
MIN_FU_DAYS <- 14   # matches rev1_clinical; removes fu_time<=0 (biopsy at/after event)

## -- shared data prep ---------------------------------------------------------
## `treatment` is intentionally excluded from CLIN: it is constant (every
## patient with a record was on a PPI or H2 blocker at some point -- see
## rev1_clinical/README.md), so it carries no information and coxph would drop
## it silently. Documented as a footnote, not fit as a covariate.
load_patients <- function() {
  d <- read.csv(P_ECO_IN, stringsAsFactors = FALSE) %>%
    mutate(
      event         = case_when(progression == "CO" ~ 1L, progression == "NCO" ~ 0L, TRUE ~ NA_integer_),
      sex           = factor(sex, levels = c("F", "M")),
      dysplasia_bin = factor(dysplasia_bin, levels = c("No Dysplasia", "Dysplasia")),
      flow          = factor(flow, levels = c("Diploid", "Aneuploid")),
      pred          = factor(pred, levels = c("Diploid", "Aneuploid")),
      ## BEACON = DACOR-abnormal AND ecology-scored. High Risk vs everything
      ## else (Diploid + ecology-Low), 1 df, directly comparable to `pred`.
      eco_risk      = factor(
        ifelse(pred == "Diploid" | eco_risk_category == "Low Risk", "Low Risk",
          ifelse(eco_risk_category == "High Risk", "High Risk", NA)),
        levels = c("Low Risk", "High Risk")),
      ## Circularity flag for the eco_risk marker specifically (see README).
      ## Only meaningful for pred==Aneuploid patients (eco_split is NA for
      ## everyone else, i.e. it was never a candidate for the ecology model).
      eco_fit_status = case_when(
        pred != "Aneuploid"    ~ NA_character_,
        eco_split == "train"   ~ "in-sample",
        eco_split == "val"     ~ "held-out",
        TRUE                   ~ "excluded"   # Aneuploid but not in either ecology split
      )
    )
  d[!is.na(d$fu_time) & d$fu_time > MIN_FU_DAYS & !is.na(d$event), ]
}

hdr <- function(...) cat("\n", strrep("=", 78), "\n", sprintf(...), "\n", strrep("=", 78), "\n", sep = "")
sub <- function(...) cat("\n--- ", sprintf(...), " ---\n", sep = "")
