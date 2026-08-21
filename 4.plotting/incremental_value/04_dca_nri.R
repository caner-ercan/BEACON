## ---------------------------------------------------------------------------
## rev1_incremental_value / 04_dca_nri.R
##
## Completes the three metrics Reviewer 3 named: likelihood-ratio testing (01),
## risk reclassification (02, descriptive) and now decision-curve analysis and
## a formal net reclassification improvement.
##
## Censoring is handled properly throughout -- both metrics are defined for
## binary outcomes and must be adapted for time-to-event data:
##   DCA : Vickers, Cronin, Elkin & Gonen (BMC Med Inform Decis Mak 2008) --
##         event probability among test-positives from Kaplan-Meier at the
##         horizon, not from naive counting (which would treat censored
##         patients as non-progressors and understate risk).
##   NRI : Pencina, Steyerberg & D'Agostino (Stat Med 2011) KM-based
##         estimator, same reason.
##
## Also computes bootstrap optimism-corrected C-index and net benefit
## (Harrell's procedure: refit within each bootstrap resample, evaluate on
## both the resample and the original data, subtract the mean difference).
## Every model here is fit and evaluated on the same patients, so the
## uncorrected values are optimistic; the corrected ones are what to report.
##
## Primary set is the discovery cohort: cohort-free by construction. The
## combined set is better powered but pools two cohorts with different
## progression rates by design, so it is reported as secondary only. The test
## cohort (NU) is also run -- archival only, per CE request (2026-08-06): kept
## on disk for transparency, deliberately excluded from the panel figure since
## it is already known to be uninformative (74% pre-progressed, no variance
## for any marker in the ladder). The spatial-ecology holdout (n=22, 12
## events) is too small for either metric and is deliberately not run.
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))

set.seed(20260801)
HORIZON   <- 5 * 365.25      # 5-year progression risk
THRESHOLDS <- seq(0.01, 0.40, by = 0.005)
N_BOOT    <- 500
## Risk categories for the categorical NRI, on 5-year predicted risk.
## Bracket the observed ~12% discovery risk; sensitivity scheme reported too.
NRI_CATS      <- c(0, 0.05, 0.15, 1)
NRI_CATS_SENS <- c(0, 0.10, 0.25, 1)

## "clinical + flow + BEACON" deliberately dropped (CE, 2026-08-06): BEACON
## already contains DACOR, so a joint flow+BEACON model doesn't answer a
## clean question the way each marker added singly does (matches the ladder
## in 01/03, which never fits DACOR and BEACON as separate terms either).
MODELS <- list(
  "clinical"          = CLIN,
  "clinical + flow"   = c(CLIN, "flow"),
  "clinical + DACOR"  = c(CLIN, "pred"),
  "clinical + BEACON" = c(CLIN, "eco_risk")
)
## Markers to run the formal NRI for. Both use "clinical" as the reference
## model; extends the original clinical -> +BEACON NRI to also give
## clinical -> +DACOR (CE, 2026-08-06).
NRI_MARKERS <- c("clinical + DACOR", "clinical + BEACON")

## --- predicted absolute risk at the horizon ---------------------------------
## S(T|x) = exp(-H0(T) * exp(lp)), with H0 the uncentred baseline cumulative
## hazard and lp the linear predictor referenced to zero. Validated against
## survfit() below before being used inside the bootstrap (where it is used
## for speed).
##
## H0(T) is a STEP function (cumulative hazard only changes at event times),
## so it must be read off via the last value at-or-before T -- method =
## "constant", f = 0. The original version used approx()'s default LINEAR
## interpolation, which is wrong for a step function and silently passed
## validation only because the discovery/whole horizons happened to fall near
## an existing grid point; the test cohort's horizon didn't, exposing a
## systematic ~0.03 (3 percentage point) overestimate of risk. Confirmed
## exact match to survfit() after this fix (see git history / session notes).
risk_at <- function(fit, newdata, T = HORIZON) {
  bh <- basehaz(fit, centered = FALSE)
  H0 <- stats::approx(bh$time, bh$hazard, xout = T, method = "constant", f = 0, rule = 2)$y
  lp <- as.numeric(predict(fit, newdata = newdata, type = "lp", reference = "zero"))
  1 - exp(-H0 * exp(lp))
}

km_risk <- function(time, event, T = HORIZON) {
  if (!length(time)) return(NA_real_)
  s <- summary(survfit(Surv(time, event) ~ 1), times = T, extend = TRUE)
  1 - s$surv[1]
}

## --- decision curve (survival form) -----------------------------------------
net_benefit <- function(risk, time, event, thresholds = THRESHOLDS, T = HORIZON) {
  n <- length(risk)
  vapply(thresholds, function(pt) {
    pos <- risk >= pt
    if (!any(pos)) return(0)
    r_pos <- km_risk(time[pos], event[pos], T)
    if (is.na(r_pos)) return(NA_real_)
    (sum(pos) / n) * (r_pos - (1 - r_pos) * (pt / (1 - pt)))
  }, numeric(1))
}
net_benefit_all <- function(time, event, thresholds = THRESHOLDS, T = HORIZON) {
  r <- km_risk(time, event, T)
  r - (1 - r) * (thresholds / (1 - thresholds))
}

## --- KM-based categorical NRI (Pencina 2011) --------------------------------
nri_km <- function(risk_old, risk_new, time, event, cats = NRI_CATS, T = HORIZON) {
  c_old <- cut(risk_old, breaks = cats, include.lowest = TRUE, labels = FALSE)
  c_new <- cut(risk_new, breaks = cats, include.lowest = TRUE, labels = FALSE)
  up <- c_new > c_old; down <- c_new < c_old
  n <- length(time)
  p_ev  <- km_risk(time, event, T)          # P(event by T) overall
  if (is.na(p_ev) || p_ev <= 0 || p_ev >= 1) return(c(NRI = NA, NRI_ev = NA, NRI_nonev = NA,
                                                      n_up = sum(up), n_down = sum(down)))
  r_up   <- if (any(up))   km_risk(time[up],   event[up],   T) else 0
  r_down <- if (any(down)) km_risk(time[down], event[down], T) else 0
  ## events: (#up x P(event|up) - #down x P(event|down)) / (n x P(event))
  nri_ev    <- (sum(up) * r_up - sum(down) * r_down) / (n * p_ev)
  ## non-events: (#down x P(no event|down) - #up x P(no event|up)) / (n x P(no event))
  nri_nonev <- (sum(down) * (1 - r_down) - sum(up) * (1 - r_up)) / (n * (1 - p_ev))
  c(NRI = nri_ev + nri_nonev, NRI_ev = nri_ev, NRI_nonev = nri_nonev,
    n_up = sum(up), n_down = sum(down))
}

## --- analysis ---------------------------------------------------------------
NEED <- unique(c(CLIN, "flow", "pred", "eco_risk", "fu_time", "event"))
d0 <- load_patients()

con <- file(file.path(out_folder, sprintf("dca_nri_%s.txt", STAMP)), "wt")
sink(con, split = TRUE); on.exit({sink(); close(con)}, add = TRUE)

cat(sprintf("horizon = %.1f years | thresholds %.2f-%.2f | %d bootstrap resamples\n",
            HORIZON / 365.25, min(THRESHOLDS), max(THRESHOLDS), N_BOOT))

SETS <- list(
  list(key = "discovery", label = "Discovery (MDA) -- PRIMARY, cohort-free",
       d = d0[d0$dataset == "MDA", ], in_figure = TRUE),
  list(key = "whole", label = "Combined -- SECONDARY, pools two cohorts with different progression rates",
       d = d0, in_figure = TRUE),
  list(key = "test", label = "Test cohort (NU) -- TERTIARY, archival only, not in the panel figure",
       d = d0[d0$dataset == "NU", ], in_figure = FALSE)
)

all_dca <- list(); all_nri <- list(); all_opt <- list()

for (S in SETS) {
  dd <- S$d[complete.cases(S$d[, NEED]), ]
  hdr("%s  (n=%d, events=%d)", S$label, nrow(dd), sum(dd$event))
  cat(sprintf("observed %.0f-year KM progression risk: %.3f\n\n", HORIZON/365.25,
              km_risk(dd$fu_time, dd$event)))

  fits <- lapply(MODELS, function(cv)
    coxph(as.formula(paste("Surv(fu_time, event) ~", paste(cv, collapse = "+"))),
          data = dd, x = TRUE))

  ## one-off validation of the fast risk function against survfit
  f1 <- fits[[1]]
  sf <- summary(survfit(f1, newdata = dd[1:5, ]), times = HORIZON, extend = TRUE)
  chk <- max(abs((1 - as.numeric(sf$surv)) - risk_at(f1, dd[1:5, ])))
  cat(sprintf("risk_at() vs survfit() max abs diff on 5 subjects: %.3g\n", chk))
  if (chk > 1e-6) warning("risk_at() disagrees with survfit() -- check baseline hazard handling")

  ## ---- decision curves
  nb <- lapply(fits, function(f) net_benefit(risk_at(f, dd), dd$fu_time, dd$event))
  nb_all  <- net_benefit_all(dd$fu_time, dd$event)
  nb_none <- rep(0, length(THRESHOLDS))

  ## ---- bootstrap optimism, for C-index and for net benefit
  opt_C  <- setNames(numeric(length(fits)), names(fits))
  opt_NB <- lapply(fits, function(...) numeric(length(THRESHOLDS)))
  names(opt_NB) <- names(fits)
  ok <- 0
  for (b in seq_len(N_BOOT)) {
    idx <- sample(nrow(dd), replace = TRUE)
    db  <- dd[idx, ]
    fb <- try(lapply(MODELS, function(cv)
      coxph(as.formula(paste("Surv(fu_time, event) ~", paste(cv, collapse = "+"))),
            data = db, x = TRUE)), silent = TRUE)
    if (inherits(fb, "try-error")) next
    bad <- any(vapply(fb, function(f) any(!is.finite(coef(f))), logical(1)))
    if (bad) next
    ok <- ok + 1
    for (nm in names(fits)) {
      ## C-index: on the resample (optimistic) minus on the original data
      c_boot <- concordance(fb[[nm]], newdata = db)$concordance
      c_orig <- concordance(fb[[nm]], newdata = dd)$concordance
      opt_C[nm] <- opt_C[nm] + (c_boot - c_orig)
      ## net benefit: same logic, using each fit's own predictions
      nb_boot <- net_benefit(risk_at(fb[[nm]], db), db$fu_time, db$event)
      nb_orig <- net_benefit(risk_at(fb[[nm]], dd), dd$fu_time, dd$event)
      opt_NB[[nm]] <- opt_NB[[nm]] + (nb_boot - nb_orig)
    }
  }
  opt_C <- opt_C / ok
  opt_NB <- lapply(opt_NB, function(x) x / ok)
  cat(sprintf("bootstrap resamples used: %d/%d\n\n", ok, N_BOOT))

  sub("C-index: apparent vs optimism-corrected")
  cstat <- data.frame(
    set = S$key, model = names(fits),
    C_apparent = vapply(fits, function(f) summary(f)$concordance[1], numeric(1)),
    optimism   = opt_C[names(fits)],
    row.names = NULL)
  cstat$C_corrected <- cstat$C_apparent - cstat$optimism
  print(as.data.frame(cstat %>% mutate(across(where(is.numeric), ~round(.x, 4)))), row.names = FALSE)
  all_opt[[length(all_opt) + 1]] <- cstat

  sub("Decision curve: net benefit at selected thresholds (corrected in brackets)")
  show_t <- c(0.05, 0.10, 0.15, 0.20, 0.30)
  ## nearest-index, NOT match(): THRESHOLDS is built by seq(by=0.005) so values
  ## like 0.10 and 0.15 are not exactly representable and match() returns NA.
  ix <- vapply(show_t, function(t) which.min(abs(THRESHOLDS - t)), integer(1))
  tab <- data.frame(threshold = show_t,
                    treat_all = round(nb_all[ix], 4),
                    treat_none = 0)
  for (nm in names(fits))
    tab[[nm]] <- sprintf("%.4f (%.4f)", nb[[nm]][ix], (nb[[nm]] - opt_NB[[nm]])[ix])
  print(tab, row.names = FALSE)

  dca_long <- do.call(rbind, lapply(names(fits), function(nm) data.frame(
    set = S$key, threshold = THRESHOLDS, model = nm,
    net_benefit = nb[[nm]], net_benefit_corrected = nb[[nm]] - opt_NB[[nm]])))
  dca_long <- rbind(dca_long,
    data.frame(set = S$key, threshold = THRESHOLDS, model = "treat all",
               net_benefit = nb_all, net_benefit_corrected = nb_all),
    data.frame(set = S$key, threshold = THRESHOLDS, model = "treat none",
               net_benefit = nb_none, net_benefit_corrected = nb_none))
  all_dca[[length(all_dca) + 1]] <- dca_long

  ## range of thresholds where BEACON beats the clinical model and both defaults
  ## Report the actual CONTIGUOUS intervals where BEACON wins. min()-max() of
  ## the winning set would paper over gaps -- and there is a real gap here at
  ## low thresholds, so a single min-max range would misstate the result.
  gain <- (nb[["clinical + BEACON"]] - opt_NB[["clinical + BEACON"]]) -
          (nb[["clinical"]] - opt_NB[["clinical"]])
  win <- which(gain > 0 &
               (nb[["clinical + BEACON"]] - opt_NB[["clinical + BEACON"]]) > pmax(nb_all, 0))
  fmt_runs <- function(idx, x) {
    if (!length(idx)) return("no threshold")
    brk <- c(0, which(diff(idx) != 1), length(idx))
    paste(vapply(seq_len(length(brk) - 1), function(k) {
      r <- idx[(brk[k] + 1):brk[k + 1]]
      sprintf("%.1f%%-%.1f%%", 100 * x[r[1]], 100 * x[r[length(r)]])
    }, character(1)), collapse = ", ")
  }
  cat(sprintf("\ncorrected net benefit of +BEACON exceeds clinical AND both defaults over: %s\n",
              fmt_runs(win, THRESHOLDS)))

  for (mk in NRI_MARKERS) {
    sub("Net reclassification improvement, clinical -> %s", mk)
    r_old <- risk_at(fits[["clinical"]], dd)
    r_new <- risk_at(fits[[mk]], dd)
    for (lab in c("primary <5/5-15/>15%", "sensitivity <10/10-25/>25%")) {
      cats <- if (grepl("primary", lab)) NRI_CATS else NRI_CATS_SENS
      est <- nri_km(r_old, r_new, dd$fu_time, dd$event, cats)
      bs <- replicate(N_BOOT, {
        i <- sample(nrow(dd), replace = TRUE); db <- dd[i, ]
        f0 <- try(coxph(as.formula(paste("Surv(fu_time,event)~", paste(MODELS[["clinical"]], collapse="+"))), data=db), silent=TRUE)
        f1 <- try(coxph(as.formula(paste("Surv(fu_time,event)~", paste(MODELS[[mk]], collapse="+"))), data=db), silent=TRUE)
        if (inherits(f0,"try-error")||inherits(f1,"try-error")) return(NA_real_)
        nri_km(risk_at(f0, db), risk_at(f1, db), db$fu_time, db$event, cats)["NRI"]
      })
      ci <- quantile(bs, c(.025,.975), na.rm = TRUE)
      cat(sprintf("  %-30s NRI=%+.3f (95%% CI %+.3f to %+.3f) | events %+.3f, non-events %+.3f | up=%d down=%d\n",
          lab, est["NRI"], ci[1], ci[2], est["NRI_ev"], est["NRI_nonev"], est["n_up"], est["n_down"]))
      all_nri[[length(all_nri) + 1]] <- data.frame(
        set = S$key, marker = mk, categories = lab, NRI = est["NRI"], CI_low = ci[1], CI_high = ci[2],
        NRI_events = est["NRI_ev"], NRI_nonevents = est["NRI_nonev"],
        n_up = est["n_up"], n_down = est["n_down"], row.names = NULL)
    }
  }
}

write.csv(bind_rows(all_dca), file.path(out_folder, sprintf("dca_curves_%s.csv", STAMP)), row.names = FALSE)
write.csv(bind_rows(all_nri), file.path(out_folder, sprintf("nri_%s.csv", STAMP)), row.names = FALSE)
write.csv(bind_rows(all_opt), file.path(out_folder, sprintf("optimism_cindex_%s.csv", STAMP)), row.names = FALSE)
cat("\nwritten: dca_curves / nri / optimism_cindex CSVs in", out_folder, "\n")
