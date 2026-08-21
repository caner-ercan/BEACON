## ---------------------------------------------------------------------------
## rev1_clinical / 01_build_clinical_tables.R
##
## Rebuilds the universal slide- and patient-level clinical tables in
##   BE_master/0.input/organised_wsi_patient/
## so BE length, dysplasia grade and acid-suppression treatment are populated
## in BOTH cohorts. Addresses reviewer 1 comment #4, and supplies the covariate
## table needed for reviewer 3 major comment #1.
##
## Inputs are never modified; outputs are new `*_rev1_<STAMP>.csv` files
## alongside the existing ones.
##
## Design constraints
## ------------------
## * `patient` is a random 5-character hash keyed on (RandomID, fu_time,
##   Dataset), minted by stri_rand_strings() in
##   0.input/organised_wsi_patient/wsi_patient_data.Rmd under set.seed(123).
##   Re-running that Rmd would generate NEW hashes and silently invalidate
##   every downstream table, so the existing grouping is read from disk and
##   treated as immutable. Every external join goes through `wsi` or through
##   the recovered RandomID -- never through `patient`.
## * Patient-level fields that are already published (age, sex, progression,
##   fu_time, split, flow, pred) are carried over verbatim from
##   MIL_patients_tx_260317.csv rather than re-derived, so nothing in Table 1
##   moves unless we intend it to.
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_CLINICAL_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))

qc_start(P_QC_OUT)
on.exit(qc_end(), add = TRUE)

cat("rev1 clinical table rebuild --", format(Sys.time()), "\n")
cat("base_folder:", base_folder, "\n")

samples_in  <- read.csv(P_SAMPLES_IN,  stringsAsFactors = FALSE)
patients_in <- read.csv(P_PATIENTS_IN, stringsAsFactors = FALSE)
conv        <- read.csv(P_CONV,        stringsAsFactors = FALSE)
samples_in$X <- NULL; patients_in$X <- NULL

stopifnot(nrow(samples_in) == 777L, nrow(patients_in) == 191L)

## ===========================================================================
## STEP 1  complete the wsi -> RandomID map
## ===========================================================================
hdr("STEP 1  wsi -> RandomID map")

src_map <- read_excel(P_SRC_FU) %>%
  rename(wsi = `Slide Label`, rid_src = RandomID) %>%
  distinct(wsi, rid_src)
stopifnot(!any(duplicated(src_map$wsi)))

idmap <- samples_in %>%
  select(patient, wsi, dataset) %>%
  left_join(conv %>% select(patient, wsi, rid_conv = patient_old), by = c("patient", "wsi")) %>%
  left_join(src_map, by = "wsi") %>%
  mutate(rid = ifelse(is.na(rid_conv), rid_src, rid_conv),
         rid_source = ifelse(is.na(rid_conv), "barrett_dataset_withFUtimes", "convertion_sample"))

cat("slides:", nrow(idmap), "\n")
cat("  RandomID present in convertion_sample.csv :", sum(!is.na(idmap$rid_conv)), "\n")
cat("  RandomID recovered from the FU workbook   :", sum(is.na(idmap$rid_conv)), "\n")
cat("  still without any RandomID                :", sum(is.na(idmap$rid)), "\n")

disagree <- idmap %>% filter(!is.na(rid_conv), !is.na(rid_src), rid_conv != rid_src)
cat("  conflicts between the two sources         :", nrow(disagree), "\n")
stopifnot(nrow(disagree) == 0L, !any(is.na(idmap$rid)))

rec <- idmap %>% filter(rid_source == "barrett_dataset_withFUtimes") %>% distinct(patient, rid)
cat("\nhashes rescued by the recovery route:", n_distinct(rec$patient),
    "-> distinct RandomIDs:", n_distinct(rec$rid), "\n")

write.csv(idmap %>% select(patient, wsi, dataset, rid, rid_source), P_IDMAP_OUT, row.names = FALSE)
cat("written:", P_IDMAP_OUT, "\n")

samples <- samples_in %>% left_join(idmap %>% select(patient, wsi, rid), by = c("patient", "wsi"))
stopifnot(nrow(samples) == 777L)

## ===========================================================================
## STEP 2  patient grouping audit  (reported, deliberately NOT altered)
## ===========================================================================
hdr("STEP 2  patient grouping audit")

grp  <- samples %>% distinct(patient, rid, dataset, fu_time)
dupe <- grp %>% count(rid, name = "n_hash") %>% filter(n_hash > 1)

cat("hashed patients        :", n_distinct(grp$patient), "\n")
cat("distinct RandomIDs     :", n_distinct(grp$rid), "\n")
cat("RandomIDs with >1 hash :", nrow(dupe), "\n\n")

dupe_detail <- grp %>% filter(rid %in% dupe$rid) %>%
  group_by(rid) %>%
  summarise(n_hash = n_distinct(patient),
            cohorts = paste(sort(unique(dataset)), collapse = "+"),
            fu_times = paste(sort(unique(fu_time)), collapse = ","), .groups = "drop") %>%
  arrange(desc(n_hash))
print(as.data.frame(dupe_detail), row.names = FALSE)

cat("\nsame person in BOTH cohorts   :", sum(dupe_detail$cohorts == "MDA+NU"), "\n")
cat("same person split in 1 cohort :", sum(dupe_detail$cohorts != "MDA+NU"), "\n")

subhdr("hashed patients vs distinct people, by cohort")
print(as.data.frame(samples %>% group_by(dataset) %>%
  summarise(slides = n(), hashed_patients = n_distinct(patient),
            distinct_people = n_distinct(rid), .groups = "drop")), row.names = FALSE)

subhdr("non-positive fu_time (biopsy at or after the event date)")
cat("slides fu_time < 0 :", sum(samples$fu_time < 0, na.rm = TRUE),
    " == 0 :", sum(samples$fu_time == 0, na.rm = TRUE),
    " NA :", sum(is.na(samples$fu_time)), "\n")
cat("hashed patients affected:",
    n_distinct(samples$patient[!is.na(samples$fu_time) & samples$fu_time <= 0]), "\n")

## ===========================================================================
## STEP 3  discovery cohort (MDA): BELength + GradeOfDysplasia
## ===========================================================================
hdr("STEP 3  discovery cohort clinicopathology")

mda_raw <- suppressWarnings(read_excel(P_MDA_DETAIL))

## BELength is one constant value per patient in this workbook -> fills by rid.
mda_belen <- mda_raw %>% filter(!is.na(BELength)) %>%
  group_by(rid = RandomID) %>%
  summarise(BELength_mda = as.numeric(one_of(BELength, "BELength")), .groups = "drop")
cat("RandomIDs with a BELength:", nrow(mda_belen), "\n")

## Dysplasia is graded per slide x tissue level. Join on (wsi, selected_tisue),
## the key the original build used, so the grade belongs to the tissue block
## that was actually imaged.
mda_dys <- mda_raw %>%
  mutate(selected_tisue = as.numeric(substr(Slide, nchar(Slide), nchar(Slide)))) %>%
  filter(!is.na(GradeOfDysplasia)) %>%
  group_by(wsi = `Slide #`, selected_tisue) %>%
  summarise(dys_mda = max_grade(unname(MDA_TO_GRADE[as.character(GradeOfDysplasia)])),
            .groups = "drop")
cat("(wsi, tissue) keys with a grade:", nrow(mda_dys), "\n")

## ===========================================================================
## STEP 4  test cohort (NU): dysplasia + BE length from the revision file
## ===========================================================================
hdr("STEP 4  test cohort clinicopathology (revision file)")

nu_raw <- suppressWarnings(read_excel(P_NU_HISTO))
names(nu_raw)[1] <- "wsi"
nu_raw <- nu_raw %>% select(wsi, rid = RandomID, EndoDate, Diagnosis, LES, OS, AtypismTypeOdze)

cat("rows:", nrow(nu_raw), " slides:", n_distinct(nu_raw$wsi),
    " RandomIDs:", n_distinct(nu_raw$rid), "\n")

nu_study <- nu_raw %>% filter(wsi %in% samples$wsi)
n_nu <- n_distinct(samples$wsi[samples$dataset == "NU"])
cat("rows matching study slides:", nrow(nu_study),
    " slides covered:", n_distinct(nu_study$wsi), "/", n_nu, "\n")

odd <- setdiff(unique(na.omit(nu_study$AtypismTypeOdze)), as.numeric(names(NU_ATYPISM)))
if (length(odd)) cat("!! undocumented AtypismTypeOdze codes treated as NA:",
                     paste(odd, collapse = ", "), "\n")

## Slide grade = worst grade over that slide's levels ("max" rollup, as
## everywhere else in this pipeline).
nu_dys <- nu_study %>%
  mutate(g = unname(NU_ATYPISM[as.character(AtypismTypeOdze)])) %>%
  group_by(wsi) %>% summarise(dys_nu = max_grade(g), .groups = "drop")
subhdr("NU slide-level dysplasia grade"); print(table(nu_dys$dys_nu, useNA = "ifany"))

## BE length = LES - OS, one value per slide (i.e. per endoscopy).
nu_belen <- nu_study %>% distinct(wsi, rid, EndoDate, LES, OS) %>%
  mutate(BELength_nu = LES - OS)
stopifnot(!any(duplicated(nu_belen$wsi)))
cat("\nNU slides with LES/OS:", nrow(nu_belen),
    " missing:", sum(is.na(nu_belen$BELength_nu)), "\n")
print(summary(nu_belen$BELength_nu))

subhdr("does BE length vary within a patient, across endoscopies?")
nu_var <- nu_belen %>% group_by(rid) %>%
  summarise(n_endo = n_distinct(EndoDate), n_val = n_distinct(BELength_nu),
            lo = min(BELength_nu), hi = max(BELength_nu), .groups = "drop")
print(table(`distinct values per patient` = nu_var$n_val))
if (any(nu_var$n_val > 1)) {
  cat("\n"); print(as.data.frame(filter(nu_var, n_val > 1)), row.names = FALSE)
}

subhdr("validation: LES - OS vs the discovery workbook's own BELength")
xval <- nu_belen %>% group_by(rid) %>% summarise(les_os = max(BELength_nu), .groups = "drop") %>%
  inner_join(mda_belen, by = "rid")
if (nrow(xval) > 1) {
  print(as.data.frame(xval), row.names = FALSE)
  cat("\nn =", nrow(xval),
      "| Pearson r =", round(cor(xval$les_os, xval$BELength_mda, use = "complete.obs"), 3),
      "| exact =", sum(xval$les_os == xval$BELength_mda, na.rm = TRUE),
      "| median |diff| =", median(abs(xval$les_os - xval$BELength_mda), na.rm = TRUE), "cm\n")
}

## ===========================================================================
## STEP 5  acid-suppression treatment
## ===========================================================================
hdr("STEP 5  acid-suppression treatment")

tx_raw <- read_excel(P_TREAT)
names(tx_raw) <- c("rid", "tx_baseline", "tx_fu")
stopifnot(!any(duplicated(tx_raw$rid)))

tx <- tx_raw %>% mutate(across(c(tx_baseline, tx_fu),
  ~ ifelse(.x %in% TX_NOT_OBSERVED, NA_character_, as.character(.x))))

cat("RandomIDs in workbook:", nrow(tx),
    "| study RandomIDs covered:", length(intersect(unique(samples$rid), tx$rid)),
    "/", n_distinct(samples$rid), "\n")
subhdr("baseline");             print(table(tx$tx_baseline, useNA = "ifany"))
subhdr("follow-up maximum");    print(table(tx$tx_fu,       useNA = "ifany"))

## ===========================================================================
## STEP 6  slide-level assembly
## ===========================================================================
hdr("STEP 6  slide-level assembly")

samples_out <- samples %>%
  rename(BELength_orig = BELength, dysplasia_orig = dysplasia, treatment_orig = treatment) %>%
  left_join(mda_belen, by = "rid") %>%
  left_join(mda_dys,   by = c("wsi", "selected_tisue")) %>%
  left_join(nu_belen %>% select(wsi, EndoDate, BELength_nu), by = "wsi") %>%
  left_join(nu_dys,    by = "wsi") %>%
  left_join(tx,        by = "rid")
stopifnot(nrow(samples_out) == 777L)

samples_out <- samples_out %>%
  mutate(
    ## -- BE length: keep what was already there, else the cohort's source.
    BELength_src = case_when(
      !is.na(BELength_orig)                  ~ "original",
      dataset == "NU" & !is.na(BELength_nu)  ~ "NU_LES_minus_OS",
      !is.na(BELength_mda)                   ~ "MDA_workbook",
      TRUE                                   ~ NA_character_),
    BELength = case_when(
      !is.na(BELength_orig)                  ~ as.numeric(BELength_orig),
      dataset == "NU" & !is.na(BELength_nu)  ~ as.numeric(BELength_nu),
      !is.na(BELength_mda)                   ~ BELength_mda,
      TRUE                                   ~ NA_real_),

    ## -- dysplasia: harmonised ordinal grade N < IND < LGD < HGD < EAC.
    ##    Discovery mixed grades (LGD/HGD, HGD/LGD) collapse onto HGD; the
    ##    manuscript only uses the binary contrast, which is invariant to that.
    dysplasia_src = case_when(
      dataset == "NU" & !is.na(dys_nu)                ~ "NU_AtypismTypeOdze",
      !is.na(dysplasia_orig) & dysplasia_orig != ""   ~ "MDA_original",
      !is.na(dys_mda)                                 ~ "MDA_workbook",
      TRUE                                            ~ NA_character_),
    dysplasia_grade = case_when(
      dataset == "NU" & !is.na(dys_nu)                ~ dys_nu,
      !is.na(dysplasia_orig) & dysplasia_orig != ""   ~ unname(MDA_TO_GRADE[dysplasia_orig]),
      !is.na(dys_mda)                                 ~ dys_mda,
      TRUE                                            ~ NA_character_),

    ## -- treatment: collapsed to ever / never on acid suppression (revision
    ##    decision). Agent class and baseline-vs-follow-up are both dropped:
    ##    with ~10 untreated patients in total the 3-level factor separates in
    ##    each cohort on its own. The source columns are kept for audit only.
    treatment = case_when(
      is.na(tx_fu) & is.na(tx_baseline)                    ~ NA_character_,
      tx_fu %in% c("PPI", "H2") | tx_baseline %in% c("PPI", "H2") ~ "Yes",
      TRUE                                                 ~ "No"),
    treatment_fu       = tx_fu,
    treatment_baseline = tx_baseline
  ) %>%
  mutate(
    ## Analysis variable: every grade above NDBE counts as dysplasia
    ## (revision decision -- matches the original MS, which merged all
    ##  dysplasia levels into a single `Dysplasia` category).
    dysplasia_bin = case_when(
      is.na(dysplasia_grade)               ~ NA_character_,
      dysplasia_grade == DYS_NONDYSPLASTIC ~ "No Dysplasia",
      TRUE                                 ~ "Dysplasia"),
    ## `dysplasia` keeps GRADE semantics so that pre-existing downstream code
    ## (which does its own case_when on N/LGD/HGD) behaves exactly as before.
    ## New analyses must use `dysplasia_bin` -- see README.
    dysplasia = dysplasia_grade
  ) %>%
  select(patient, rid, wsi, dataset, split, selected_tisue,
         age, sex, progression, fu_time, EndoDate,
         dysplasia, dysplasia_grade, dysplasia_bin, dysplasia_orig, dysplasia_src,
         BELength, BELength_src, BELength_orig,
         treatment, treatment_fu, treatment_baseline, treatment_orig,
         flow, pred)

miss <- function(x) is.na(x) | x == ""
subhdr("slide-level missingness, before -> after")
print(data.frame(
  variable = rep(c("dysplasia", "BELength", "treatment"), each = 2),
  cohort   = rep(c("MDA", "NU"), 3),
  before   = c(tapply(miss(samples_out$dysplasia_orig), samples_out$dataset, sum),
               tapply(miss(samples_out$BELength_orig),  samples_out$dataset, sum),
               tapply(miss(samples_out$treatment_orig) | samples_out$treatment_orig == "No Data",
                      samples_out$dataset, sum)),
  after    = c(tapply(miss(samples_out$dysplasia), samples_out$dataset, sum),
               tapply(miss(samples_out$BELength),  samples_out$dataset, sum),
               tapply(miss(samples_out$treatment), samples_out$dataset, sum))
), row.names = FALSE)

subhdr("BE length provenance");  print(table(samples_out$BELength_src,  samples_out$dataset, useNA = "ifany"))
subhdr("dysplasia provenance");  print(table(samples_out$dysplasia_src, samples_out$dataset, useNA = "ifany"))
subhdr("dysplasia grade x cohort"); print(table(samples_out$dysplasia_grade, samples_out$dataset, useNA = "ifany"))
subhdr("treatment: ever/never vs the March merge")
print(table(March = samples_out$treatment_orig, rebuilt = samples_out$treatment, useNA = "ifany"))
subhdr("treatment: ever/never vs its two sources")
print(table(follow_up = samples_out$treatment_fu, ever = samples_out$treatment, useNA = "ifany"))
print(table(baseline = samples_out$treatment_baseline, ever = samples_out$treatment, useNA = "ifany"))

## sanity: we only ever filled gaps, never overwrote an existing value
chk <- samples_out %>% filter(!is.na(BELength_orig))
stopifnot(all(chk$BELength == chk$BELength_orig))

## ===========================================================================
## STEP 7  patient-level rollup
## ===========================================================================
hdr("STEP 7  patient-level rollup")

## Slide -> patient rollup for the three revision variables only. Everything
## else is carried over from the published patient table untouched.
##
## NOTE. The original rollup used a bare max(as.numeric(dysplasia)), which
## returns NA when ANY slide of a patient is ungraded, so patients with partial
## data were dropped rather than resolved. max_grade() is NA-safe; the diff is
## quantified below.
roll <- samples_out %>%
  arrange(patient, EndoDate) %>%
  group_by(patient) %>%
  summarise(
    rid                = one_of(rid, "rid"),
    n_slides           = n(),
    dysplasia_grade    = max_grade(dysplasia_grade),
    ## BE length at the earliest endoscopy of the group (identical to the only
    ## value for every discovery patient and for 22/31 test patients).
    BELength           = if (all(is.na(BELength))) NA_real_ else BELength[!is.na(BELength)][1],
    BELength_src       = one_of(BELength_src, "BELength_src"),
    treatment          = one_of(treatment, "treatment"),
    treatment_fu       = one_of(treatment_fu, "treatment_fu"),
    treatment_baseline = one_of(treatment_baseline, "treatment_baseline"),
    .groups = "drop"
  ) %>%
  mutate(dysplasia_bin = case_when(
    is.na(dysplasia_grade)               ~ NA_character_,
    dysplasia_grade == DYS_NONDYSPLASTIC ~ "No Dysplasia",
    TRUE                                 ~ "Dysplasia"),
    dysplasia = dysplasia_grade)

patients_out <- patients_in %>%
  rename(BELength_orig = BELength, dysplasia_orig = dysplasia, treatment_orig = treatment) %>%
  left_join(roll, by = "patient") %>%
  select(patient, rid, dataset, split, n_slides,
         age, sex, progression, fu_time,
         dysplasia, dysplasia_grade, dysplasia_bin, dysplasia_orig,
         BELength, BELength_src, BELength_orig,
         treatment, treatment_fu, treatment_baseline, treatment_orig,
         flow, pred)
stopifnot(nrow(patients_out) == 191L)

subhdr("patient-level missingness, before -> after")
print(data.frame(
  variable = rep(c("dysplasia", "BELength", "treatment"), each = 2),
  cohort   = rep(c("MDA", "NU"), 3),
  before   = c(tapply(miss(patients_out$dysplasia_orig), patients_out$dataset, sum),
               tapply(miss(patients_out$BELength_orig),  patients_out$dataset, sum),
               tapply(miss(patients_out$treatment_orig) | patients_out$treatment_orig == "No Data",
                      patients_out$dataset, sum)),
  after    = c(tapply(miss(patients_out$dysplasia), patients_out$dataset, sum),
               tapply(miss(patients_out$BELength),  patients_out$dataset, sum),
               tapply(miss(patients_out$treatment), patients_out$dataset, sum))
), row.names = FALSE)

subhdr("patients recovered purely by the NA-safe rollup (discovery cohort)")
fixed <- patients_out %>%
  filter(dataset == "MDA", miss(dysplasia_orig), !miss(dysplasia)) %>%
  select(patient, rid, n_slides, dysplasia_grade)
cat("n =", nrow(fixed), "\n")
if (nrow(fixed)) print(as.data.frame(fixed), row.names = FALSE)

subhdr("published BELength preserved where it existed?")
c2 <- patients_out %>% filter(!is.na(BELength_orig))
cat(ifelse(all(c2$BELength == c2$BELength_orig), "yes", "NO -- INVESTIGATE"), "\n")

subhdr("patient-level dysplasia (binary) x cohort")
print(table(patients_out$dysplasia_bin, patients_out$dataset, useNA = "ifany"))
subhdr("patient-level treatment x cohort")
print(table(patients_out$treatment, patients_out$dataset, useNA = "ifany"))
subhdr("patient-level BE length by cohort")
print(patients_out %>% group_by(dataset) %>%
  summarise(n = sum(!is.na(BELength)), mean = round(mean(BELength, na.rm = TRUE), 2),
            sd = round(sd(BELength, na.rm = TRUE), 2),
            min = min(BELength, na.rm = TRUE), max = max(BELength, na.rm = TRUE),
            .groups = "drop") %>% as.data.frame(), row.names = FALSE)

## ===========================================================================
## STEP 8  ecology patient table
## ===========================================================================
hdr("STEP 8  ecology patient table")

if (file.exists(P_ECO_IN)) {
  eco_in <- read.csv(P_ECO_IN, stringsAsFactors = FALSE); eco_in$X <- NULL
  eco_out <- eco_in %>%
    select(patient, eco_split, risk_category, eco_risk_category) %>%
    right_join(patients_out, by = "patient") %>%
    relocate(eco_split, risk_category, eco_risk_category, .after = split)
  stopifnot(nrow(eco_out) == 191L)
  write.csv(eco_out, P_ECO_OUT, row.names = FALSE)
  cat("written:", P_ECO_OUT, "\n")
  cat("patients with an eco_split:", sum(!is.na(eco_out$eco_split)), "/", nrow(eco_out), "\n")
  subhdr("eco risk category x cohort")
  print(table(eco_out$eco_risk_category, eco_out$dataset, useNA = "ifany"))
} else {
  cat("!! eco_patients.csv not found -- skipped\n")
}

## ===========================================================================
hdr("WRITE")
write.csv(samples_out,  P_SAMPLES_OUT,  row.names = FALSE)
write.csv(patients_out, P_PATIENTS_OUT, row.names = FALSE)
cat("written:", P_SAMPLES_OUT,  "(", nrow(samples_out),  "rows )\n")
cat("written:", P_PATIENTS_OUT, "(", nrow(patients_out), "rows )\n")
cat("QC report:", P_QC_OUT, "\n")
cat("\ndone --", format(Sys.time()), "\n")
