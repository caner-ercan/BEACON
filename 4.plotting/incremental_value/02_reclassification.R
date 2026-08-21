## ---------------------------------------------------------------------------
## rev1_incremental_value / 02_reclassification.R
##
## Risk reclassification: flow cytometry x BEACON, observed progression rate
## per cell. This is where flow and BEACON legitimately appear together -- as
## a cross-tabulation, not as two terms in one Cox model (BEACON already
## contains DACOR, so a joint model would be collinear; see 01_ladder.R).
##
## Reported twice for the same reason as Table B in 01_ladder.R: the primary
## table pools all BEACON-scored patients (some of which the ecology model
## was fit on), the held-out version restricts to eco_split=="val" only. The
## held-out table will be sparse (~20 patients total) -- report it anyway,
## flagged, rather than let the primary table imply an evaluation it isn't.
## ---------------------------------------------------------------------------

REV1_DIR <- Sys.getenv("REV1_DIR", unset = NA_character_)
if (is.na(REV1_DIR)) {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  REV1_DIR <- if (length(a)) dirname(sub("^--file=", "", a[1])) else "."
}
source(file.path(REV1_DIR, "00_config.R"))

d0 <- load_patients()
NEED <- c("flow", "eco_risk", "fu_time", "event")

reclass_table <- function(d, label) {
  dd <- d[complete.cases(d[, NEED]), ]
  if (!nrow(dd)) { cat(sprintf("%-40s no complete cases\n", label)); return(NULL) }
  t <- dd %>%
    group_by(flow, eco_risk) %>%
    summarise(n = n(), events = sum(event), rate = round(mean(event), 3), .groups = "drop") %>%
    mutate(group = label, .before = 1)
  cat(sprintf("\n%s  (n=%d, events=%d)\n", label, nrow(dd), sum(dd$event)))
  print(as.data.frame(t %>% select(-group)), row.names = FALSE)
  t
}

con <- file(file.path(out_folder, sprintf("reclassification_%s.txt", STAMP)), "wt")
sink(con, split = TRUE); on.exit({sink(); close(con)}, add = TRUE)

hdr("RECLASSIFICATION -- flow cytometry x BEACON, observed progression rate")

sub("Primary: all BEACON-scored patients (includes ecology-model training data)")
t_all <- reclass_table(d0, "all scored")

sub("Held-out only: eco_split == val (genuine ecology-model holdout, ~20 patients -- sparse)")
## Circularity tracks DACOR (pred), not flow: a patient only enters the
## ecology model, and so only has a fitting-circularity risk, if pred ==
## Aneuploid. Restricting by flow instead (an earlier version of this script
## did) misses 13 flow-Diploid/DACOR-Aneuploid discordant patients who WERE
## used to fit the ecology model -- pred is the correct filter.
d_held <- d0[d0$pred != "Aneuploid" | d0$eco_fit_status == "held-out", ]
t_held <- reclass_table(d_held, "held-out (DACOR-abnormal patients restricted to eco_split==val)")

sub("By cohort (primary table, for reference)")
t_mda <- reclass_table(d0[d0$dataset == "MDA", ], "Discovery (MDA)")
t_nu  <- reclass_table(d0[d0$dataset == "NU", ],  "Test (NU)")

out <- bind_rows(
  t_all %>% mutate(group = "all scored"),
  t_held %>% mutate(group = "held-out (eco_split==val)"),
  t_mda, t_nu
)
write.csv(out, file.path(out_folder, sprintf("reclassification_%s.csv", STAMP)), row.names = FALSE)
cat("\nwritten:", file.path(out_folder, sprintf("reclassification_%s.csv", STAMP)), "\n")
