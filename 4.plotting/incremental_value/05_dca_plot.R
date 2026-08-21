## ---------------------------------------------------------------------------
## rev1_incremental_value / 05_dca_plot.R
##
## ARCHIVAL overview -- all three analysis sets (discovery, combined, test),
## kept on disk for transparency (CE, 2026-08-06) but NOT the submission
## figure. The submission-ready figure is `07_panel_figure.R`'s
## figS7_incremental_value_*.pdf (discovery + combined only, matching colours
## with the forest-ladder panel). This script differs from that one only in
## including the test cohort, which is known to be uninformative for every
## marker (74% pre-progressed) and was excluded from the manuscript figure
## for exactly that reason -- it is here so the full picture is on record.
##
## Plots the OPTIMISM-CORRECTED net benefit (the apparent curve is shown
## faint for comparison) against the two default strategies, treat-all and
## treat-none.
##
## Reading a decision curve: a model is worth using at a given threshold
## probability only where its curve sits above BOTH defaults. Threshold
## probability is the progression risk at which a clinician would switch to
## intensive surveillance, so only the clinically plausible range matters.
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))
suppressMessages(library(ggplot2))

f <- sort(list.files(out_folder, pattern = "^dca_curves_.*\\.csv$", full.names = TRUE), decreasing = TRUE)[1]
if (is.na(f)) stop("no dca_curves_*.csv found -- run 04_dca_nri.R first")
message("reading: ", f)
d <- read.csv(f, stringsAsFactors = FALSE)

SET_LABEL <- c(discovery = "Discovery cohort (n=105, 18 events) -- PRIMARY",
               whole     = "Combined cohorts (n=147, 49 events) -- secondary",
               test      = "Test cohort NU (n=42, 31 events) -- archival only, not in the MS figure")
LEVELS <- c("treat all", "treat none", "clinical", "clinical + flow",
            "clinical + DACOR", "clinical + BEACON")
## Same palette as 07_panel_figure.R's COL_MODEL, so this archival version and
## the submission figure are directly comparable.
PALETTE <- c("treat all" = "grey55", "treat none" = "grey25",
             "clinical" = "grey45", "clinical + flow" = "#DD8452",
             "clinical + DACOR" = "#8172B2", "clinical + BEACON" = "#C44E52")

d$model <- factor(d$model, levels = LEVELS)
## Discovery first (primary, cohort-free), then combined, then test last (the
## archival addition) -- default alphabetical facet order would scramble this.
d$set_label <- factor(SET_LABEL[d$set], levels = SET_LABEL[c("discovery", "whole", "test")])

## y-limit: net benefit below zero is never a reason to use a model, and a few
## deeply negative points at high thresholds otherwise squash the whole plot.
ymax <- max(d$net_benefit_corrected[is.finite(d$net_benefit_corrected)], na.rm = TRUE)
ylo  <- -0.02

p <- ggplot(d, aes(threshold, net_benefit_corrected, colour = model)) +
  geom_line(aes(y = net_benefit), linewidth = 0.35, alpha = 0.30, na.rm = TRUE) +
  geom_line(linewidth = 0.85, na.rm = TRUE) +
  geom_hline(yintercept = 0, colour = "grey25", linewidth = 0.3) +
  facet_wrap(~ set_label, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = PALETTE, name = NULL) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  coord_cartesian(ylim = c(ylo, ymax * 1.05)) +
  labs(
    x = "Threshold probability of progression within 5 years",
    y = "Net benefit",
    title = "Decision-curve analysis: does BEACON change management?",
    subtitle = "Bold = optimism-corrected (500 bootstrap resamples); faint = apparent (in-sample)"
  ) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold", hjust = 0))

P <- file.path(out_folder, sprintf("dca_curves_%s.pdf", STAMP))
ggsave(P, p, width = 7.2, height = 7.6)
message("written: ", P)

## Winning threshold regions are reported by 06_dca_summary.R, which formats
## genuinely contiguous runs. Deliberately not duplicated here -- the earlier
## version of this block printed min-max of a non-contiguous set and so
## overstated the region (discovery is 4.0% plus 6.0-40.0%, not 4.0-40.0%).
message("run 06_dca_summary.R for the winning threshold intervals")
