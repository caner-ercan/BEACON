## ---------------------------------------------------------------------------
## rev1_incremental_value / 06_dca_summary.R
##
## Re-derives the decision-curve summary from the saved curves. Exists because
## the summary printed inside 04_dca_nri.R had two display bugs (now fixed
## there): thresholds were looked up with match() against a seq(by=0.005) grid,
## so 0.10/0.15 silently became NA; and the winning threshold region was
## printed as min-max of a possibly non-contiguous set, which would overstate
## it. The curves themselves were always correct, so this reads the CSV rather
## than repeating the bootstrap.
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))

f <- sort(list.files(out_folder, pattern = "^dca_curves_.*\\.csv$", full.names = TRUE), decreasing = TRUE)[1]
if (is.na(f)) stop("no dca_curves_*.csv found -- run 04_dca_nri.R first")
cat("reading:", basename(f), "\n")
d <- read.csv(f, stringsAsFactors = FALSE)

fmt_runs <- function(th) {
  if (!length(th)) return("no threshold")
  i <- order(th); th <- th[i]
  brk <- c(0, which(round(diff(th), 6) > 0.0051), length(th))
  paste(vapply(seq_len(length(brk) - 1), function(k) {
    r <- th[(brk[k] + 1):brk[k + 1]]
    sprintf("%.1f%%-%.1f%%", 100 * r[1], 100 * r[length(r)])
  }, character(1)), collapse = ", ")
}

for (s in unique(d$set)) {
  w <- d[d$set == s, ]
  th <- sort(unique(w$threshold))
  get <- function(m) w$net_benefit_corrected[w$model == m][order(w$threshold[w$model == m])]

  hdr("%s -- optimism-corrected net benefit", s)
  show_t <- c(0.05, 0.10, 0.15, 0.20, 0.30)
  ix <- vapply(show_t, function(t) which.min(abs(th - t)), integer(1))
  tab <- data.frame(threshold = sprintf("%.0f%%", 100 * th[ix]))
  for (m in c("treat all", "clinical", "clinical + flow", "clinical + DACOR", "clinical + BEACON")) {
    v <- get(m); if (!length(v)) next
    tab[[m]] <- round(v[ix], 4)
  }
  print(tab, row.names = FALSE)

  cB <- get("clinical + BEACON"); cC <- get("clinical")
  cF <- get("clinical + flow"); cD <- get("clinical + DACOR")
  cA <- get("treat all")
  cat("\n  +BEACON beats clinical AND both defaults : ",
      fmt_runs(th[cB > cC & cB > pmax(cA, 0)]), "\n", sep = "")
  cat("  +BEACON beats clinical + flow            : ",
      fmt_runs(th[cB > cF]), "\n", sep = "")
  cat("  +DACOR beats clinical AND both defaults  : ",
      fmt_runs(th[cD > cC & cD > pmax(cA, 0)]), "\n", sep = "")
  cat("  +flow beats clinical AND both defaults   : ",
      fmt_runs(th[cF > cC & cF > pmax(cA, 0)]), "\n", sep = "")
}
