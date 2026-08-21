## ---------------------------------------------------------------------------
## rev1_clinical / 02_table1.R
##
## Regenerates the Table 1 demographics/clinicopathology summary from the rev1
## clinical tables, so BE length, dysplasia and treatment are now populated for
## the test cohort as well as the discovery cohort.
##
## Differences from the published Table 1, all traceable to 01:
##   * Dysplasia and BE length now have Test-column values (were blank/NA).
##   * Treatment is complete for 183/191 patients (was 164/191) and "No Data"
##     is missing rather than its own category.
##   * 5 discovery patients gained a dysplasia grade from the NA-safe rollup.
## Run 01_build_clinical_tables.R first.
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_CLINICAL_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))

P_T1_CSV <- file.path(out_folder, sprintf("table1_rev1_%s.csv", STAMP))
P_T1_TXT <- file.path(out_folder, sprintf("table1_rev1_%s.txt", STAMP))

stopifnot(file.exists(P_PATIENTS_OUT))
pat <- read.csv(P_PATIENTS_OUT, stringsAsFactors = FALSE) %>%
  mutate(cohort = recode(dataset, MDA = "Discovery", NU = "Test"),
         fu_month = fu_time * 12 / 365.25)

## --- summary helpers -------------------------------------------------------

fmt_cont <- function(d, v) {
  s <- d %>% group_by(cohort) %>%
    summarise(n = sum(!is.na(.data[[v]])),
              m = mean(.data[[v]], na.rm = TRUE),
              s = sd(.data[[v]], na.rm = TRUE), .groups = "drop") %>%
    mutate(txt = sprintf("%.1f (%.1f)", m, s))
  setNames(s$txt, s$cohort)
}

fmt_cat <- function(d, v, lvl) {
  d <- d[!is.na(d[[v]]), ]
  out <- list()
  for (co in c("Discovery", "Test")) {
    x <- d[[v]][d$cohort == co]
    tot <- length(x)
    out[[co]] <- setNames(
      vapply(lvl, function(l) {
        k <- sum(x == l)
        if (tot == 0) "0 (0%)" else sprintf("%d (%.1f%%)", k, 100 * k / tot)
      }, character(1)), lvl)
  }
  out
}

## Two-group test. Continuous -> Welch t-test; categorical -> chi-squared, with
## Fisher's exact when any expected cell is small.
ptest <- function(d, v, type) {
  if (type == "continuous") {
    a <- d[[v]][d$cohort == "Discovery"]; a <- a[!is.na(a)]
    b <- d[[v]][d$cohort == "Test"];      b <- b[!is.na(b)]
    if (length(a) < 2 || length(b) < 2) return(NA_character_)
    return(sprintf("%.2e", t.test(a, b)$p.value))
  }
  tb <- table(d$cohort, d[[v]])
  if (any(dim(tb) < 2)) return(NA_character_)
  p <- if (any(suppressWarnings(chisq.test(tb)$expected) < 5)) {
    fisher.test(tb, simulate.p.value = TRUE, B = 1e5)$p.value
  } else chisq.test(tb)$p.value
  sprintf("%.2e", p)
}

rows <- list()
add <- function(var, disc = "", test = "", p = NA) {
  rows[[length(rows) + 1]] <<- data.frame(Variable = var, Discovery = disc,
                                          Test = test, p_value = p,
                                          stringsAsFactors = FALSE)
}
add_block <- function(label, v, lvl, d = pat) {
  add(label, "", "", ptest(d, v, "categorical"))
  cc <- fmt_cat(d, v, lvl)
  for (l in lvl) add(paste0("  ", l), cc$Discovery[[l]], cc$Test[[l]], NA)
}
add_cont <- function(label, v, d = pat) {
  x <- fmt_cont(d, v)
  add(label, x[["Discovery"]], x[["Test"]], ptest(d, v, "continuous"))
}

## --- build -----------------------------------------------------------------

n <- pat %>% count(cohort)
add("Patients (hashed records), n",
    as.character(n$n[n$cohort == "Discovery"]), as.character(n$n[n$cohort == "Test"]), NA)
np <- pat %>% group_by(cohort) %>% summarise(k = n_distinct(rid), .groups = "drop")
add("  distinct individuals, n",
    as.character(np$k[np$cohort == "Discovery"]), as.character(np$k[np$cohort == "Test"]), NA)

add_cont("Age, mean (SD)", "age")
add_block("Sex, n (%)", "sex", c("F", "M"))
add_block("Progression, n (%)", "progression", c("NCO", "CO"))
add_block("Flow cytometry, n (%)", "flow", c("Diploid", "Aneuploid"))
add_block("DACOR prediction, n (%)", "pred", c("Diploid", "Aneuploid"))
add_block("Dysplasia, n (%)", "dysplasia_bin", c("No Dysplasia", "Dysplasia"))
add_block("Dysplasia grade, n (%)", "dysplasia_grade", DYS_LEVELS)
add_cont("Follow-up months, mean (SD)", "fu_month")
add_cont("BE length cm, mean (SD)", "BELength")
add_block("Acid suppression (max on follow-up), n (%)", "treatment", c("None", "H2", "PPI"))
add_block("Acid suppression (baseline), n (%)", "treatment_baseline", c("None", "H2", "PPI"))
add_cont("Biopsies per patient, mean (SD)", "n_slides")

t1 <- bind_rows(rows)

## Missingness footnote -- a reviewer will ask, and the Test column is only
## meaningful alongside it.
mrow <- function(label, v) {
  d <- pat %>% group_by(cohort) %>%
    summarise(k = sum(is.na(.data[[v]])), n = n(), .groups = "drop") %>%
    mutate(txt = sprintf("%d/%d", k, n))
  add(label, d$txt[d$cohort == "Discovery"], d$txt[d$cohort == "Test"], NA)
}
rows <- list()
add("MISSING DATA, n/total", "", "", NA)
mrow("  Dysplasia", "dysplasia_bin")
mrow("  BE length", "BELength")
mrow("  Acid suppression (follow-up)", "treatment")
mrow("  Acid suppression (baseline)", "treatment_baseline")
t1 <- bind_rows(t1, bind_rows(rows))

write.csv(t1, P_T1_CSV, row.names = FALSE)

con <- file(P_T1_TXT, "wt"); sink(con, split = TRUE)
cat("Table 1 (revision 1) --", format(Sys.time()), "\n")
cat("source:", basename(P_PATIENTS_OUT), "\n\n")
print(t1, row.names = FALSE, right = FALSE)
cat("\nNotes\n")
cat("* 'Patients' counts hashed patient records, the unit used throughout the\n",
    "  manuscript. A hash is keyed on (RandomID, fu_time, cohort), so a person\n",
    "  biopsied at several timepoints contributes several records -- hence the\n",
    "  'distinct individuals' row. See the STEP 2 audit in the QC report.\n", sep = "")
cat("* Dysplasia is the binary variable used by the manuscript: every grade\n",
    "  above NDBE (indefinite, LGD, HGD, EAC) counts as dysplasia.\n", sep = "")
cat("* Test-cohort dysplasia is derived from AtypismTypeOdze and BE length from\n",
    "  LES - OS, both from the revision histology file; see 01 for validation.\n", sep = "")
sink(); close(con)

cat("\nwritten:", P_T1_CSV, "\n")
cat("written:", P_T1_TXT, "\n")
