## ---------------------------------------------------------------------------
## rev1_clinical / 00_config.R
## Shared paths, constants and helpers for the revision clinical-data rebuild.
## Sourced by 01_build_clinical_tables.R, 02_table1.R, 03_cox_clinical.R.
## ---------------------------------------------------------------------------

suppressMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
})

## Wide enough that the HR / CI / p columns survive print() into the log files.
options(width = 200, stringsAsFactors = FALSE)

## -- base folder ------------------------------------------------------------
## Set BE_MASTER to your BE_master checkout (a relative default here would
## depend on the invocation directory, not the script's location, so this
## is required rather than guessed).
base_folder <- Sys.getenv("BE_MASTER", unset = NA_character_)
if (is.na(base_folder) || !dir.exists(base_folder))
  stop("Set the BE_MASTER environment variable to your BE_master checkout.")
message("BE_master: ", base_folder)

## -- inputs -----------------------------------------------------------------
csv_folder <- file.path(base_folder, "0.input", "organised_wsi_patient")

# Current merged tables. `_tx_260317` already carries the treatment merge that
# was done ad hoc in March 2026; it is the newest state and our starting point.
P_SAMPLES_IN  <- file.path(csv_folder, "MIL_samples_tx_260317.csv")
P_PATIENTS_IN <- file.path(csv_folder, "MIL_patients_tx_260317.csv")
P_ECO_IN      <- file.path(csv_folder, "eco_patients.csv")
P_CONV        <- file.path(csv_folder, "convertion_sample.csv")

# Authoritative wsi -> RandomID source. Covers all 777 study slides, whereas
# convertion_sample.csv leaves 75 of them without an ID (see 01, step 1).
P_SRC_FU <- file.path(base_folder, "0.input/Sample_data/barrett_dataset_withFUtimes.xlsx")

# Discovery/MDA clinicopathology (BELength, GradeOfDysplasia), one row per
# slide x level.
P_MDA_DETAIL <- file.path(
  base_folder,
  "0.input/Sample_data/finalUpdatedDetailed/raw_receivedfromTom",
  "Dysplasia Study Detailed Data.xlsx"
)

# Revision additions.
P_NU_HISTO <- file.path(base_folder, "revision", "NU slide histology data for Caner 060926.xlsx")
P_TREAT    <- file.path(base_folder, "revision", "treatment_260316",
                        "Acid med summary for Caner 03042026.xlsx")

## -- outputs ----------------------------------------------------------------
STAMP <- "260730"

P_SAMPLES_OUT  <- file.path(csv_folder, sprintf("MIL_samples_rev1_%s.csv", STAMP))
P_PATIENTS_OUT <- file.path(csv_folder, sprintf("MIL_patients_rev1_%s.csv", STAMP))
P_ECO_OUT      <- file.path(csv_folder, sprintf("eco_patients_rev1_%s.csv", STAMP))
P_IDMAP_OUT    <- file.path(csv_folder, sprintf("wsi_randomid_map_rev1_%s.csv", STAMP))

out_folder <- file.path(base_folder, "4.plotting", "output", "rev1_clinical")
dir.create(out_folder, recursive = TRUE, showWarnings = FALSE)

P_QC_OUT <- file.path(out_folder, sprintf("qc_report_%s.txt", STAMP))

## -- coding dictionaries ----------------------------------------------------
## AtypismTypeOdze (NU revision file), per the legend embedded in the sheet:
##   1 = NDBE, 2 = Indefinite for dysplasia, 3 = Low-grade dysplasia,
##   4 = High-grade dysplasia, 5 = EAC
NU_ATYPISM <- c("1" = "N", "2" = "IND", "3" = "LGD", "4" = "HGD", "5" = "EAC")

## Harmonised ordinal grade shared by both cohorts. The discovery cohort's
## mixed grades collapse onto HGD: the manuscript only ever uses the binary
## contrast, which is invariant to that collapse.
DYS_LEVELS <- c("N", "IND", "LGD", "HGD", "EAC")

MDA_TO_GRADE <- c(
  "N" = "N", "LGD" = "LGD", "LGD/HGD" = "HGD", "HGD/LGD" = "HGD", "HGD" = "HGD"
)

## Binary variable used by the manuscript: everything above NDBE counts as
## dysplasia (user decision, revision round 1 — matches the original MS, where
## all dysplasia levels were merged into a single `Dysplasia` category).
DYS_NONDYSPLASTIC <- "N"

## Treatment: values that are placeholders rather than observations.
TX_NOT_OBSERVED <- c("No Data", "")

## -- helpers ----------------------------------------------------------------

## max() over a discrete grade, NA-safe. The original patient-level rollup used
## a bare max(), which returns NA whenever *any* slide is missing -- see the QC
## report for the 5 discovery patients this silently dropped.
max_grade <- function(x, levels = DYS_LEVELS) {
  i <- match(as.character(x), levels)
  i <- i[!is.na(i)]
  if (!length(i)) return(NA_character_)
  levels[max(i)]
}

max_num <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) return(NA_real_)
  max(x)
}

## Single non-missing value, warning if the group disagrees with itself.
one_of <- function(x, what = "value", id = "") {
  u <- unique(x[!is.na(x)])
  if (!length(u)) return(NA)
  if (length(u) > 1) {
    warning(sprintf("%s: %d distinct %s in one group (%s) -- taking the first",
                    id, length(u), what, paste(u, collapse = "/")), call. = FALSE)
  }
  u[1]
}

`%||%` <- function(a, b) if (is.null(a)) b else a

## Tee console output to the QC report.
qc_con <- NULL
qc_start <- function(path) {
  qc_con <<- file(path, open = "wt")
  sink(qc_con, split = TRUE)
}
qc_end <- function() {
  if (!is.null(qc_con)) {
    sink()
    close(qc_con)
    qc_con <<- NULL
  }
}

hdr <- function(...) cat("\n", strrep("=", 74), "\n", sprintf(...), "\n",
                         strrep("=", 74), "\n", sep = "")
## NB: named subhdr(), not sub() -- a helper called sub() would shadow base
## R's sub() for every script that sources this file, silently breaking any
## regex substitution downstream.
subhdr <- function(...) cat("\n--- ", sprintf(...), " ---\n", sep = "")
