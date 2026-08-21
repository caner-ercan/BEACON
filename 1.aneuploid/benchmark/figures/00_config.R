# Task 6 benchmark figures -- shared configuration.
#
# Source of truth: round 3 (2026-08-07), step c_fit_all_models. This is the only
# round in which every model was fitted on the same 340-slide training split, so
# train/val/test are comparable across models. Earlier rounds fitted rungs 1-2 on
# train+val pooled and must not be mixed in.
#
# Principle: every plot is written to OUT_DIR as PDF, including the ones that
# will not reach the main figure. The figure is assembled by picking from there.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(pROC)
})

# --- paths ---------------------------------------------------------------
# Set BE_MASTER to your BE_master checkout. results_dir now reads from
# BE_master/revision/task6_benchmark/results/, the synced, complete copy of
# round 3's results (previously read from rev_exp_run_scripts/ at the
# project root, outside both this repo and BE_master).
be_master     <- Sys.getenv("BE_MASTER", unset = NA_character_)
if (is.na(be_master)) stop("Set the BE_MASTER environment variable to your BE_master checkout.")
results_dir   <- file.path(be_master, "revision", "task6_benchmark", "results")
labels_csv    <- file.path(be_master, "revision", "task6_benchmark",
                           "interim", "slide_labels.csv")
out_folder    <- file.path(be_master, "4.plotting", "output",
                           "rev1_task6_figures")
dir.create(out_folder, recursive = TRUE, showWarnings = FALSE)

stopifnot(dir.exists(results_dir), file.exists(labels_csv))

# --- model presentation --------------------------------------------------
# Ordered as Yu et al.'s ablation ladder, DACOR last so it draws on top.
model_levels <- c("rung1_M", "rung2_CM", "rung3_D", "rung3_CMD",
                  "rung3_D_reinhard", "rung3_CMD_reinhard", "DACOR")

model_labels <- c(
  rung1_M            = "Morphology (M)",
  rung2_CM           = "Subtypes + morphology (C+M)",
  rung3_D            = "Deep features (D)",
  rung3_CMD          = "Full hybrid (C+M+D)",
  rung3_D_reinhard   = "D, stain-normalised",
  rung3_CMD_reinhard = "C+M+D, stain-normalised",
  DACOR              = "DACOR (this study)"
)

# House red for DACOR (matches 4.cox_plot.Rmd's #d32f2f); a blue ramp for the
# handcrafted ladder, purple for the deep-only arm since it is a different
# feature family rather than one more step along the same axis.
model_colours <- c(
  rung1_M            = "#9ecae1",
  rung2_CM           = "#4292c6",
  rung3_D            = "#6a51a3",
  rung3_CMD          = "#08519c",
  rung3_D_reinhard   = "#9e9ac8",
  rung3_CMD_reinhard = "#6baed6",
  DACOR              = "#d32f2f"
)

# The stain-normalised arms sit almost exactly on their base arms, so they are
# dashed where shown and excluded from the figure-bound panels entirely.
model_linetypes <- c(
  rung1_M = "solid", rung2_CM = "solid", rung3_D = "solid", rung3_CMD = "solid",
  rung3_D_reinhard = "dashed", rung3_CMD_reinhard = "dashed", DACOR = "solid"
)

# Five primary models = DACOR + Yu's four published ablation arms.
primary_models <- c("DACOR", "rung1_M", "rung2_CM", "rung3_D", "rung3_CMD")
all_models     <- model_levels

split_levels <- c("train", "val", "test")
# Axis form (two lines, for the by-split panel) and a compact inline form
# (for plot subtitles, which have to fit a ~9.5 cm panel without clipping).
split_labels <- c(train = "Train\n(discovery, n=340)",
                  val   = "Validation\n(discovery, n=82)",
                  test  = "Test\n(independent cohort, n=355)")
# Short axis form: the long labels above collide at a 9.5 cm panel width, and
# the cohort each split belongs to is carried by the bracket annotation instead.
split_axis_labels <- c(train = "Train\nn=340",
                       val   = "Validation\nn=82",
                       test  = "Test\nn=355")
split_short  <- c(train = "Discovery, training",
                  val   = "Discovery, validation",
                  test  = "Independent test cohort")

# --- theme ---------------------------------------------------------------
# Follows 1.mil.Rmd: theme_light, no panel grid, modest base size.
theme_task6 <- function(base_size = 9) {
  theme_light(base_size = base_size) +
    theme(
      panel.grid       = element_blank(),
      panel.border     = element_rect(colour = "grey30", fill = NA, linewidth = 0.4),
      axis.title       = element_text(size = base_size),
      axis.text        = element_text(size = base_size - 1),
      legend.title     = element_blank(),
      legend.key.size  = unit(0.4, "cm"),
      legend.text      = element_text(size = base_size - 1),
      legend.background = element_rect(fill = alpha("white", 0.85), colour = NA),
      legend.key       = element_blank(),
      plot.title       = element_text(size = base_size + 1, face = "bold"),
      plot.subtitle    = element_text(size = base_size - 1, colour = "grey30")
    )
}

# --- data ----------------------------------------------------------------

#' Long table of per-slide predictions for every model, all three splits.
load_predictions <- function() {
  labels <- read.csv(labels_csv, stringsAsFactors = FALSE)

  dacor <- labels %>%
    transmute(model = "DACOR", wsi, patient, split,
              flow = as.integer(flow), prob = as.numeric(dacor_prob))

  files <- list.files(results_dir, pattern = "^predictions_.*\\.csv$", full.names = TRUE)
  rungs <- lapply(files, function(f) {
    nm <- sub("^predictions_", "", tools::file_path_sans_ext(basename(f)))
    d  <- read.csv(f, stringsAsFactors = FALSE)
    if (!"split" %in% names(d)) {
      warning("skipping ", basename(f), ": no split column (pre-fix file)")
      return(NULL)
    }
    d %>% transmute(model = nm, wsi, patient, split,
                    flow = as.integer(flow), prob = as.numeric(prob))
  })

  bind_rows(dacor, bind_rows(rungs)) %>%
    filter(split %in% split_levels) %>%
    mutate(model = factor(model, levels = model_levels),
           split = factor(split, levels = split_levels))
}

#' Slide scores -> patient scores, by max within split (as DACOR aggregates).
to_patient <- function(preds) {
  preds %>%
    group_by(model, split, patient) %>%
    summarise(prob = max(prob), flow = max(flow), .groups = "drop")
}

#' AUC with DeLong CI.
#'
#' direction is pinned to "<" rather than left on pROC's default "auto".
#' With "auto", pROC silently flips any curve that falls below chance so it
#' reports >= 0.5 -- which would turn the deep-only arm's test AUC of 0.439
#' into 0.561 and disagree with the pipeline's own numbers.
auc_ci <- function(flow, prob) {
  if (length(unique(flow)) < 2) return(c(NA, NA, NA))
  r  <- pROC::roc(flow, prob, quiet = TRUE, direction = "<")
  ci <- as.numeric(pROC::ci.auc(r, method = "delong"))
  c(auc = ci[2], lo = ci[1], hi = ci[3])
}

#' ROC coordinates for ggplot, one model per group.
roc_points <- function(df) {
  df %>%
    group_by(model) %>%
    group_modify(~ {
      r <- pROC::roc(.x$flow, .x$prob, quiet = TRUE, direction = "<")
      tibble::tibble(fpr = rev(1 - r$specificities),
                     tpr = rev(r$sensitivities))
    }) %>%
    ungroup()
}

#' Consistent PDF export. Everything lands in out_folder.
#'
#' The base pdf device, not cairo_pdf: this machine has no X11/cairo, and
#' ggsave silently falls back with a warning otherwise. Output is vector either
#' way; only the font backend differs.
save_pdf <- function(plot, name, width, height) {
  path <- file.path(out_folder, paste0(name, ".pdf"))
  ggsave(path, plot, width = width, height = height, units = "cm",
         device = "pdf", useDingbats = FALSE)
  cat(sprintf("  %-46s %4.1f x %4.1f cm\n", basename(path), width, height))
  invisible(path)
}
