## ---------------------------------------------------------------------------
## rev1_clinical / 06_forest_grid.R
##
## The complete forest-plot grid for the revision, in one browsable folder, so
## panels can be picked per figure rather than being pre-assigned to one.
## Supersedes the earlier per-figure drivers (06_suppfig3b / 07_suppfig2d /
## 08_fig5b); it is a strict superset of what those produced.
##
## Grid
## ----
##   flow, DACOR   x  discovery / test / whole COHORT   (scanning-site cohorts)
##   BEACON        x  training / validation / whole SPLIT (ecology model's own
##                                                         patient-constrained split)
##   x  univariate and multivariate
##
## The two markers use different population axes on purpose. Flow cytometry and
## DACOR are evaluated across the scanning-site cohorts, which is how the
## manuscript reports them. The ecology (BEACON) risk model was *fitted* on its
## own train/validation split, so that split -- not the scanning site -- is the
## meaningful axis for it.
##
## Variants
## --------
##   full     every clinicopathological covariate (age, sex, dysplasia,
##            BE length, acid suppression) plus the marker.
##   reduced  the covariate set actually used at submission, for the two
##            populations where dysplasia was unavailable or degenerate then:
##            the test cohort (0/42 had a grade) and the ecology splits.
##   as-submitted  audit reproduction of the published whole-dataset model,
##            whose complete-case population turns out to be discovery-only.
##
## Acid suppression is requested everywhere and is estimable only in the whole
## population; elsewhere it renders as an explicit "not estimable" row. See
## 05_forest_style.R for the style and estimability rules.
##
## Run 01_build_clinical_tables.R first.
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_CLINICAL_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))
source(file.path(REV1_DIR, "05_forest_style.R"))

MIN_FU_DAYS <- as.numeric(Sys.getenv("MIN_FU_DAYS", "14"))
FIG_DIR <- file.path(out_folder, "figures", "all_forests")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

d0   <- read.csv(P_ECO_OUT, stringsAsFactors = FALSE) %>% prep_cox_data()
elig <- d0[!is.na(d0$fu_time) & d0$fu_time > MIN_FU_DAYS & !is.na(d0$event), ]

CLIN_FULL    <- c("age", "sex", "dysplasia_bin", "BELength", "treatment_fu")
CLIN_REDUCED <- c("age", "sex", "BELength")            # as submitted: no dysplasia, no treatment
CLIN_ASSUB   <- c("age", "sex", "dysplasia_sub", "BELength_orig")

## marker key -> column, the label used for its plotted ROW (must match
## VAR_LABELS, which follows the submitted figure's wording), the name used in
## subtitles/manifest, and the filename stem.
MARKERS <- list(
  flow   = list(col = "flow",     row = "Flow cytometry", name = "Flow cytometry",
                stem = "flow"),
  dacor  = list(col = "pred",     row = "DACOR",          name = "DACOR",
                stem = "dacor"),
  beacon = list(col = "eco_risk", row = "Ecology model",  name = "BEACON (ecology model)",
                stem = "beacon")
)

## population key -> (subset expression, display name, file stem, MS figure)
POPS <- list(
  discovery  = list(f = function(x) x[x$dataset == "MDA", ], label = "Discovery cohort",  stem = "discovery", fig = "Supp. Fig. 2D"),
  test       = list(f = function(x) x[x$dataset == "NU",  ], label = "Test cohort",       stem = "test",      fig = "Supp. Fig. 2D"),
  trainsplit = list(f = function(x) x[!is.na(x$eco_split) & x$eco_split == "train", ], label = "Training set",   stem = "trainsplit", fig = "Supp. Fig. 3B"),
  valsplit   = list(f = function(x) x[!is.na(x$eco_split) & x$eco_split == "val",   ], label = "Validation set", stem = "valsplit",   fig = "Supp. Fig. 3B"),
  whole      = list(f = function(x) x, label = "Whole dataset", stem = "whole", fig = "Fig. 5B")
)

## which populations each marker is run over
GRID <- list(
  flow   = c("discovery", "test", "whole"),
  dacor  = c("discovery", "test", "whole"),
  beacon = c("trainsplit", "valsplit", "whole")
)

## populations whose submitted analysis lacked dysplasia -> also get a "reduced"
REDUCED_POPS <- c("test", "trainsplit", "valsplit")

con <- file(file.path(FIG_DIR, "generation_log.txt"), "wt"); sink(con, split = TRUE)
on.exit({sink(); close(con)}, add = TRUE)
cat("Forest-plot grid --", format(Sys.time()), "\n")
cat("minimum follow-up:", MIN_FU_DAYS, "days\n")

manifest <- list()

## Pull the marker's own (non-reference) row out of a rendered panel, for the
## manifest. Markers are 2-level factors, so there is exactly one such row.
marker_stat <- function(res, label) {
  if (is.null(res) || !nrow(res$rows)) return(NULL)
  r <- res$rows[res$rows$variable == label & res$rows$kind %in% c("value", "not_estimable"), ]
  if (!nrow(r)) return(NULL)
  r[1, ]
}

emit <- function(mk, pk, analysis, variant, clin) {
  m <- MARKERS[[mk]]; p <- POPS[[pk]]
  b <- p$f(elig)
  vars <- c(clin, m$col)
  fn <- sprintf("%s_%s_%s%s.pdf", m$stem, p$stem, analysis,
                if (variant == "full") "" else paste0("_", variant))
  path <- file.path(FIG_DIR, fn)
  extra <- if (variant == "full") m$name else sprintf("%s, %s", m$name, variant)
  ttl <- fm_title(if (analysis == "univariate") "Univariate" else "Multivariate", p$label)
  sub <- fm_sub(nrow(b), sum(b$event), extra)
  tag <- sprintf("%s/%s/%s/%s", mk, pk, analysis, variant)

  res <- if (analysis == "univariate")
    render_univariate(b, vars, path, ttl, sub, announce = tag)
  else
    render_multivariate(b, vars, path, ttl, sub, announce = tag)

  st <- marker_stat(res, m$row)
  manifest[[length(manifest) + 1]] <<- data.frame(
    file = fn, ms_figure = p$fig, marker = m$name, population = p$label,
    analysis = analysis, variant = variant,
    n = if (is.null(res)) NA_integer_ else res$n,
    events = if (is.null(res)) NA_integer_ else res$events,
    marker_HR     = if (is.null(st)) NA_real_ else st$HR,
    marker_CI_low = if (is.null(st)) NA_real_ else st$CI_low,
    marker_CI_high= if (is.null(st)) NA_real_ else st$CI_high,
    marker_p      = if (is.null(st)) NA_real_ else st$p,
    stringsAsFactors = FALSE)
  invisible(res)
}

for (mk in names(GRID)) {
  hdr("%s", MARKERS[[mk]]$name)
  for (pk in GRID[[mk]]) {
    b <- POPS[[pk]]$f(elig)
    cat(sprintf("\n-- %s: n = %d, events = %d\n", POPS[[pk]]$label, nrow(b), sum(b$event)))
    for (an in c("univariate", "multivariate")) {
      emit(mk, pk, an, "full", CLIN_FULL)
      if (pk %in% REDUCED_POPS) emit(mk, pk, an, "reduced", CLIN_REDUCED)
    }
  }
}

## ---------------------------------------------------------------------------
## As-submitted reproduction of the published whole-dataset BEACON model
## ---------------------------------------------------------------------------
hdr("AS SUBMITTED -- published whole-dataset model reproduced")
sub_pop <- elig[stats::complete.cases(elig[, CLIN_ASSUB]), ]
cat("complete-case population:", nrow(sub_pop), "patients,", sum(sub_pop$event), "events\n")
print(table(sub_pop$dataset))
cat("--> the published 'whole dataset' model was discovery-only.\n")

for (an in c("univariate", "multivariate")) {
  fn <- sprintf("beacon_whole_%s_assubmitted.pdf", an)
  st_sub <- sprintf("N = %d patients, %d events - BEACON (ecology model), as submitted (discovery cohort only)",
                    nrow(sub_pop), sum(sub_pop$event))
  res <- if (an == "univariate")
    render_univariate(sub_pop, c(CLIN_ASSUB, "eco_risk"), file.path(FIG_DIR, fn),
                      fm_title("Univariate", "As submitted"), st_sub, announce = paste("assub", an))
  else
    render_multivariate(elig, c(CLIN_ASSUB, "eco_risk"), file.path(FIG_DIR, fn),
                        fm_title("Multivariate", "As submitted"), st_sub, announce = paste("assub", an))
  st <- marker_stat(res, "Ecology model")
  manifest[[length(manifest) + 1]] <- data.frame(
    file = fn, ms_figure = "Fig. 5B (audit)", marker = "BEACON (ecology model)",
    population = "As submitted (discovery only)", analysis = an, variant = "as-submitted",
    n = if (is.null(res)) NA_integer_ else res$n,
    events = if (is.null(res)) NA_integer_ else res$events,
    marker_HR = if (is.null(st)) NA_real_ else st$HR,
    marker_CI_low = if (is.null(st)) NA_real_ else st$CI_low,
    marker_CI_high = if (is.null(st)) NA_real_ else st$CI_high,
    marker_p = if (is.null(st)) NA_real_ else st$p,
    stringsAsFactors = FALSE)
  if (an == "multivariate" && !is.null(st))
    cat(sprintf("\n  reproduced BEACON HR %.2f (%.2f-%.2f) p=%.4f   [published: 6.64 (1.35-32.7)]\n",
                st$HR, st$CI_low, st$CI_high, st$p))
}

## ---------------------------------------------------------------------------
mf <- bind_rows(manifest)
write.csv(mf, file.path(FIG_DIR, "manifest.csv"), row.names = FALSE)

hdr("MANIFEST -- marker estimate in every panel")
print(as.data.frame(mf %>% mutate(
  `HR (95% CI)` = ifelse(is.na(marker_HR), "-",
                         sprintf("%.2f (%.2f-%.2f)", marker_HR, marker_CI_low, marker_CI_high)),
  p = ifelse(is.na(marker_p), "-", sprintf("%.4f", marker_p))) %>%
  select(file, ms_figure, analysis, variant, n, events, `HR (95% CI)`, p)), row.names = FALSE)

cat("\nfiles:", nrow(mf), " ->", FIG_DIR, "\n")
cat("\nCaution: the validation-split multivariate panels rest on ~12 events against\n",
    "up to 6 parameters. Their univariate counterparts are the defensible ones.\n", sep = "")
