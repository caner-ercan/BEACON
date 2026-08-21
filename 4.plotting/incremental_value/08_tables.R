## ---------------------------------------------------------------------------
## rev1_incremental_value / 08_tables.R
##
## Formatted PDF version of Table S-Y (reclassification + formal NRI). Built
## with gridExtra::tableGrob (gt/kableExtra are not installed in this
## environment; tableGrob needs no LaTeX/webshot dependency).
##
## The former Table S-X here (the ladder, with optimism-corrected C-index
## folded in, read from 03_summary_table.R's output) is superseded by
## 09_ladder_unstratified.R's own formatted table, which covers the same
## ground plus a 5th (unstratified whole-cohort) column from a single set of
## fits rather than a cross-file join. 03_summary_table.R and Table S-X here
## were removed together (CE packaging decision).
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))
suppressMessages({library(gridExtra); library(grid)})

read_latest <- function(pattern) {
  f <- sort(list.files(out_folder, pattern = pattern, full.names = TRUE), decreasing = TRUE)[1]
  if (is.na(f)) stop("no file matching ", pattern, " in ", out_folder)
  message("reading: ", basename(f))
  read.csv(f, stringsAsFactors = FALSE)
}

THEME <- ttheme_minimal(
  core = list(fg_params = list(hjust = 0, x = 0.02, fontsize = 7.5),
             bg_params  = list(fill = c("grey98", "white"), col = NA)),
  colhead = list(fg_params = list(fontface = "bold", fontsize = 7.5, hjust = 0, x = 0.02),
                bg_params  = list(fill = "grey85", col = NA)),
  rowhead = list(fg_params = list(fontface = "bold", fontsize = 7.5, hjust = 0, x = 0)),
  padding = unit(c(3, 2), "mm")
)

## Stacks a list of grobs (table titles, tableGrobs, caption) vertically,
## sizing each ROW to that grob's own natural height rather than a guessed
## cm value -- guessed heights caused a table's rows to overflow into the
## label above it on the first attempt. Canvas height is then set to the sum
## of natural heights, so there is no leftover blank canvas either.
## grobHeight() under-reports a tableGrob's true height (confirmed: content
## clipped when relied on for a 9-row table) -- a gtable's real height is the
## sum of its own row-height slots, which is what the grid device actually
## uses when drawing it. Branch on class rather than trust grobHeight() for
## everything.
grob_height <- function(g) {
  if (inherits(g, "gtable")) sum(g$heights) else grobHeight(g)
}

save_table <- function(grobs, path, width, caption, margin_in = 0.15) {
  CAP_SIZE <- 6.8
  cap_lines <- strwrap(caption, width = round(width * 14))
  cap_txt <- paste(cap_lines, collapse = "\n")
  ## y must be pinned to the viewport's top explicitly -- vjust=1 alone only
  ## changes justification around the DEFAULT y=0.5 anchor, it does not move
  ## the anchor itself. Without `y = unit(1, "npc")` the multi-line caption
  ## grows from vertical centre, pushing its lower half below the row (and,
  ## being the last row, off the bottom of the page) -- confirmed by a minimal
  ## repro before finding this.
  cap <- textGrob(cap_txt, x = 0.01, y = unit(1, "npc"), hjust = 0, vjust = 1,
                  gp = gpar(fontsize = CAP_SIZE, fontface = "italic", col = "grey30"))
  ## grobHeight() under-measures multi-line text too (same failure mode as the
  ## tableGrob case below, confirmed: caption was clipping mid-sentence at the
  ## bottom of the page). A deterministic line-count x pitch estimate is more
  ## reliable here than trusting grid's own sizing.
  cap_height <- unit(length(cap_lines) * CAP_SIZE * 1.35, "pt")

  all_grobs <- c(grobs, list(cap))
  pad <- unit(3, "mm")
  hts <- unit.c(
    do.call(unit.c, lapply(grobs, function(g) grob_height(g) + pad)),
    cap_height + pad
  )

  full <- arrangeGrob(grobs = all_grobs, ncol = 1, heights = hts)
  total_in <- sum(convertHeight(hts, "inches", valueOnly = TRUE)) + 2 * margin_in

  ## cairo_pdf needs X11/cairo, not installed on this machine -- the base pdf()
  ## device needs neither and renders identically for vector text + lines.
  pdf(path, width = width, height = total_in)
  grid.draw(full)
  dev.off()
  if (!file.exists(path)) stop("pdf() device failed to write ", path)
  message("written: ", path, sprintf(" (%.1fx%.1fin, %d KB)", width, total_in, round(file.info(path)$size / 1024)))
}

## =============================================================================
## Table S-Y -- reclassification + NRI
## =============================================================================
recl <- read_latest("^reclassification_.*\\.csv$")
nri  <- read_latest("^nri_.*\\.csv$")

r1 <- recl[recl$group %in% c("all scored", "held-out (eco_split==val)"), ]
r1$Group <- ifelse(r1$group == "all scored", "All BEACON-scored", "Ecology-model holdout only")
r1$Cell  <- paste(r1$flow, "x", r1$eco_risk)
dfY1 <- data.frame(Group = r1$Group, `Flow x BEACON` = r1$Cell, n = r1$n, Events = r1$events,
                   `Progression rate` = sprintf("%.0f%%", 100 * r1$rate),
                   stringsAsFactors = FALSE, check.names = FALSE)
dfY1 <- dfY1[order(dfY1$Group == "Ecology-model holdout only", dfY1$`Flow x BEACON`), ]
dfY1$Group[duplicated(dfY1$Group)] <- ""

## Test cohort's NRI is archival only (see 04/05 headers) -- excluded here so
## Table S-Y's scope matches the submission figure's (discovery + combined).
nri <- nri[nri$set %in% c("discovery", "whole"), ]
nri$Set <- ifelse(nri$set == "discovery", "Discovery", "Whole, cohort-adj.")
nri$Marker <- ifelse(nri$marker == "clinical + DACOR", "+ DACOR", "+ BEACON")
nri <- nri[order(nri$Set != "Discovery", nri$Marker != "+ DACOR", nri$categories), ]
dfY2 <- data.frame(
  Set = nri$Set, Marker = nri$Marker, `Risk categories` = nri$categories,
  NRI = sprintf("%+.3f (%+.3f, %+.3f)", nri$NRI, nri$CI_low, nri$CI_high),
  `NRI events` = sprintf("%+.3f", nri$NRI_events), `NRI non-events` = sprintf("%+.3f", nri$NRI_nonevents),
  `n up / down` = sprintf("%d / %d", nri$n_up, nri$n_down),
  stringsAsFactors = FALSE, check.names = FALSE)

gY1 <- tableGrob(dfY1, rows = NULL, theme = THEME)
gY2 <- tableGrob(dfY2, rows = NULL, theme = THEME)
lab1 <- textGrob("Risk reclassification: observed 5-year progression by flow cytometry x BEACON",
                 x = 0.01, hjust = 0, gp = gpar(fontsize = 8, fontface = "bold"))
lab2 <- textGrob("Net reclassification improvement, clinical -> clinical + marker (KM-based, censoring-aware)",
                 x = 0.01, hjust = 0, gp = gpar(fontsize = 8, fontface = "bold"))
CAPTION_Y <- paste(
  "Table S-Y. Reclassification analysis (Reviewer 3, major comment #1). Top: observed 5-year",
  "progression rate by flow cytometry x BEACON status, for all scored patients and restricted to",
  "patients excluded from ecology-model fitting. Bottom: net reclassification improvement (Pencina",
  "et al. 2011, Kaplan-Meier-based to account for censoring) moving from the clinicopathological",
  "model to clinical + DACOR or clinical + BEACON, at 5 years, under two risk-category schemes;",
  "95% CI from 500 bootstrap resamples. All intervals shown include zero. Test-cohort NRI is",
  "archival only (see Methods) and not shown here.", sep = " ")

save_table(list(lab1, gY1, lab2, gY2),
          file.path(out_folder, sprintf("tableSY_reclassification_%s.pdf", STAMP)),
          width = 8.6, caption = CAPTION_Y)
