## ---------------------------------------------------------------------------
## rev1_clinical / 05_forest_style.R
##
## Shared Cox-fitting and forest-plot rendering for the revised Supp. Fig. 3B,
## Supp. Fig. 2D and Fig. 5B panels.
##
## STYLE. Five columns, left to right:
##   1. Variable name (shown once, on the variable's first row)
##   2. Level, with sample size on the line below: "Female\n(N=31)"; for a
##      continuous variable there is no level, just "(N=141)"
##   3. "HR (95% CI)" text, or "reference" for a factor's baseline level, or
##      "not estimable" where the contrast could not be fit
##   4. the forest itself, on a log hazard axis with a dashed line at HR = 1;
##      a reference level is drawn as a filled SQUARE sitting exactly on that
##      line (no error bar, since it is not an estimate), everything else as a
##      point + 95% CI whisker
##   5. p-value, blank for reference and not-estimable rows
## Rows are banded in alternating shading BY VARIABLE, not by row, so a
## variable's reference + level rows read as one group. This matches
## `forestmodel::forest_model()`'s default layout (imported in
## `4.cox_plot.Rmd`), reproduced here by hand so an inestimable contrast can be
## shown explicitly rather than erroring or silently distorting the axis.
##
## Every output is a single standalone PDF -- nothing is combined into
## multi-panel composites.
##
## ESTIMABILITY. Judged by fitting, not by counting cells: a zero cell in a
## level-by-outcome table does not by itself break a Cox model (the partial
## likelihood conditions on risk sets), so e.g. Sex in the ecology validation
## split (Female 0:2) still estimates finitely. What breaks it is monotone
## likelihood -- the coefficient runs to +/-Inf and the interval is unbounded --
## and that is judged PER CONTRAST, because a 3-level factor can have one
## estimable contrast and one that is not (acid suppression in the discovery
## cohort: H2 has zero events and is not estimable, PPI is fine).
##
## TREATMENT. Acid-suppression therapy is a 3-level factor with "None" as the
## reference, so contrasts read as treated-vs-untreated. The "H2" and "PPI"
## rows are rendered in EVERY panel regardless of whether they can be
## estimated there, so the table layout is identical across populations; only
## the reported numbers (or "not estimable") differ, following the data:
##   discovery   None 7:1, H2 7:0, PPI 86:17 -> H2 not estimable, PPI is
##   test        None 1:0, H2 1:2, PPI 9:28  -> neither (reference has no events)
##   training    None 0:0, H2 0:0, PPI 19:22 -> neither (only PPI is present)
##   validation  None 1:0, H2 0:2, PPI 8:9   -> neither
##   whole       None 8:1, H2 8:2, PPI 95:45 -> both
## Because the two contrasts have no single p-value, the overall drop-one LRT
## for treatment is appended to the subtitle of every panel that fits it.
## ---------------------------------------------------------------------------

suppressMessages({
  library(survival); library(broom); library(ggplot2); library(dplyr)
  library(showtext); library(sysfonts)
})

## Arial, embedded as vector outlines via showtext rather than relied on as a
## PDF device font: this machine has no Cairo/X11, so grDevices::pdf() cannot
## look Arial up by name, and showtext sidesteps that by drawing every glyph
## as a path. showtext_auto() applies to any device opened afterwards,
## including the one ggsave() opens internally, so nothing further is needed
## at each plot call site.
ARIAL_TTF <- "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD_TTF <- "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ARIAL_ITALIC_TTF <- "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
if (file.exists(ARIAL_TTF)) {
  sysfonts::font_add(family = "Arial", regular = ARIAL_TTF,
                     bold = if (file.exists(ARIAL_BOLD_TTF)) ARIAL_BOLD_TTF else ARIAL_TTF,
                     italic = if (file.exists(ARIAL_ITALIC_TTF)) ARIAL_ITALIC_TTF else ARIAL_TTF)
  showtext::showtext_auto()
  FONT_FAMILY <- "Arial"
} else {
  warning("Arial not found on this system; falling back to the device default font.")
  FONT_FAMILY <- ""
}

## Publication sizing (CE spec): 3.4 in wide, 5.5 pt Arial throughout. geom_text
## takes its `size` in mm, not pt, hence the .pt conversion (a ggplot2
## constant, 72.27/25.4); theme() text elements take pt directly.
FIG_WIDTH  <- 3.6
FONT_PT    <- 6.5
FONT_MM    <- FONT_PT / .pt

## Row labels, matching the submitted figure's wording.
VAR_LABELS <- c(
  age           = "Age",
  sex           = "Sex",
  dysplasia_bin = "Dysplasia",
  dysplasia_sub = "Dysplasia",
  BELength      = "BE Length",
  BELength_orig = "BE Length",
  treatment_fu  = "Acid suppression",
  treatment     = "Acid suppression",
  flow          = "DNA content",
  pred          = "DNA content",
  eco_risk      = "Ecology model"
)

FOREST_COL <- "#2c3e50"

## Reference levels chosen so the plotted contrast reads the way the submitted
## figure does: "Sex: Male", "Dysplasia: Yes", "Ecology model: High Risk".
prep_cox_data <- function(d) {
  d %>% mutate(
    event         = case_when(progression == "CO" ~ 1L, progression == "NCO" ~ 0L, TRUE ~ NA_integer_),
    sex           = factor(sex, levels = c("F", "M"), labels = c("Female", "Male")),
    dysplasia_bin = factor(dysplasia_bin, levels = c("No Dysplasia", "Dysplasia"),
                           labels = c("No", "Yes")),
    treatment     = factor(treatment, levels = c("No", "Yes")),
    ## "None" is the reference so the contrasts read as treated-vs-untreated,
    ## the clinically conventional direction. This is a reparameterisation only:
    ## the model fit, every other covariate and the overall treatment LRT are
    ## identical whichever level is chosen as baseline.
    treatment_fu  = factor(treatment_fu, levels = c("None", "H2", "PPI")),
    flow          = factor(flow, levels = c("Diploid", "Aneuploid"), labels = c("Normal", "Abnormal")),
    pred          = factor(pred, levels = c("Diploid", "Aneuploid"), labels = c("Normal", "Abnormal")),
    ## Derived from `eco_risk_category` ALONE, never from `pred`. `pred` is
    ## relabelled a couple of lines above and mutate() evaluates sequentially,
    ## so a test like `pred == "Diploid"` silently becomes always-FALSE the
    ## moment the labels change -- which drops every DNA-content-normal patient
    ## to NA and collapsed the whole-dataset multivariate n from 141 to 44.
    ## `eco_risk_category` already encodes "Diploid" for exactly those patients
    ## (verified: 0 rows disagree with pred), so keying off it makes this
    ## immune to future relabelling.
    eco_risk      = factor(ifelse(is.na(eco_risk_category), NA_character_,
                            ifelse(eco_risk_category %in% c("Diploid", "Low Risk"), "Low Risk",
                            ifelse(eco_risk_category == "High Risk", "High Risk", NA_character_))),
                           levels = c("Low Risk", "High Risk")),
    ## submission-state columns, for the audit reproductions
    dysplasia_sub = factor(ifelse(is.na(dysplasia_orig) | dysplasia_orig == "", NA,
                            ifelse(dysplasia_orig == "N", "No", "Yes")),
                           levels = c("No", "Yes")),
    BELength_orig = as.numeric(BELength_orig)
  )
}

## ---------------------------------------------------------------------------
## Estimability
## ---------------------------------------------------------------------------

COEF_MAX <- 15      # |log HR| ~ 15  =>  HR 3e6 / 3e-7
CI_MAX   <- 1e4

## Per-contrast test: TRUE if this coefficient's estimate or CI has run to
## infinity (or effectively so).
coef_degenerate <- function(est, hi) {
  !is.finite(est) | !is.finite(hi) | hi > CI_MAX | est > CI_MAX | est < 1 / CI_MAX
}

## Which contrasts of `m` are degenerate? Named logical over model terms.
degenerate_terms <- function(m) {
  s <- summary(m); ci <- s$conf.int
  if (!nrow(ci)) return(logical(0))
  setNames(coef_degenerate(ci[, 1], ci[, "upper .95"]), rownames(ci))
}

## Variable-level verdict for deciding model INCLUSION: degenerate only if
## every contrast is (an all-collapsed variable contributes nothing and is
## dropped from the formula; a partly-collapsed one is kept, with its bad
## contrast(s) shown as "not estimable" by var_rows()).
is_degenerate <- function(m) {
  dg <- degenerate_terms(m)
  if (!length(dg)) return(TRUE)
  all(dg)
}

## Keep covariates that are present, vary, and estimate finitely on their own.
## Returns the usable subset; records why anything was dropped.
usable_vars <- function(data, vars, event = "event", announce = NULL) {
  keep <- character(0); notes <- character(0)
  for (v in vars) {
    if (!v %in% names(data)) { notes <- c(notes, sprintf("%s [absent]", v)); next }
    dd <- data[!is.na(data[[v]]) & !is.na(data[[event]]) & !is.na(data$fu_time), ]
    if (!nrow(dd)) { notes <- c(notes, sprintf("%s [no data]", v)); next }

    if (is.factor(dd[[v]])) {
      f <- droplevels(dd[[v]])
      if (nlevels(f) < 2) {
        notes <- c(notes, sprintf("%s [constant: %s]", v,
                    paste(sprintf("%s=%d", levels(f), table(f)), collapse = ", ")))
        next
      }
      dd[[v]] <- f
    }
    m <- try(coxph(as.formula(sprintf("Surv(fu_time, event) ~ %s", v)), data = dd), silent = TRUE)
    if (inherits(m, "try-error")) { notes <- c(notes, sprintf("%s [fit failed]", v)); next }
    if (is_degenerate(m)) {
      det <- if (is.factor(dd[[v]])) {
        tb <- table(dd[[v]], dd[[event]])
        sprintf(": %s", paste(sprintf("%s %d:%d", rownames(tb), tb[, 1], tb[, 2]), collapse = ", "))
      } else ""
      notes <- c(notes, sprintf("%s [all contrasts collapsed%s]", v, det))
      next
    }
    part <- degenerate_terms(m)
    if (any(part))
      notes <- c(notes, sprintf("%s [partly collapsed: %s not estimable, others kept]",
                                v, paste(names(part)[part], collapse = ", ")))
    keep <- c(keep, v)
  }
  if (!is.null(announce) && length(notes))
    cat(sprintf("  [%s] dropped -> %s\n", announce, paste(notes, collapse = "; ")))
  attr(keep, "dropped") <- notes
  keep
}

## Overall (drop-one) likelihood-ratio test for one term in a fitted model.
## A factor with >2 levels has no single p-value -- the displayed contrasts are
## each level against the reference -- so the overall test is what should be
## quoted alongside them.
lrt_drop <- function(m, dd, vars, term) {
  rest <- setdiff(vars, term)
  f0 <- if (length(rest))
          as.formula(sprintf("Surv(fu_time, event) ~ %s", paste(rest, collapse = " + ")))
        else as.formula("Surv(fu_time, event) ~ 1")
  m0 <- try(coxph(f0, data = dd), silent = TRUE)
  if (inherits(m0, "try-error")) return(NA_real_)
  a <- try(anova(m0, m), silent = TRUE)
  if (inherits(a, "try-error")) return(NA_real_)
  a[["Pr(>|Chi|)"]][2]
}

## Which fitted terms are multi-level factors needing an overall LRT?
multilevel_terms <- function(dd, vars)
  vars[vapply(vars, function(v) is.factor(dd[[v]]) && nlevels(droplevels(dd[[v]])) > 2, logical(1))]

fmt_lrt <- function(v, p) {
  lab <- if (v %in% names(VAR_LABELS)) VAR_LABELS[[v]] else v
  txt <- if (is.na(p)) "n/a" else if (p < 0.001) "<.001" else sub("^0", "", sprintf("%.3f", p))
  sprintf("%s overall LRT p = %s", tolower(lab), txt)
}

## ---------------------------------------------------------------------------
## Row builder: one row per variable (continuous) or per level (factor,
## including the reference), whether or not the variable made it into the
## fitted model.
## ---------------------------------------------------------------------------

## `data` is the exact population the row's N is drawn from -- for a variable
## that IS in the model, its own complete-case data; for one that was dropped
## entirely, the model's complete-case data on everything else (so a level
## with truly zero patients reads as "(N=0)", which is self-explanatory).
## `td` is broom::tidy() output of the fitted model, or NULL if this variable
## was not part of the formula at all.
var_rows <- function(v, data, td) {
  lab <- if (v %in% names(VAR_LABELS)) VAR_LABELS[[v]] else v

  if (!is.factor(data[[v]])) {
    n <- sum(!is.na(data[[v]]))
    r <- if (is.null(td)) NULL else td[td$term == v, ]
    if (is.null(r) || !nrow(r) || coef_degenerate(r$estimate, r$conf.high))
      return(data.frame(variable = lab, level = "", n = n, kind = "not_estimable",
                        HR = NA_real_, CI_low = NA_real_, CI_high = NA_real_, p = NA_real_,
                        stringsAsFactors = FALSE))
    return(data.frame(variable = lab, level = "", n = n, kind = "continuous",
                      HR = r$estimate, CI_low = r$conf.low, CI_high = r$conf.high,
                      p = r$p.value, stringsAsFactors = FALSE))
  }

  ## factor: one row for the reference level, then one per remaining level, in
  ## the variable's defined level order (so a level absent from this
  ## population -- N = 0 -- still gets its row).
  ## `variable` is filled on every row of the group -- draw_forest() decides
  ## on its own which row prints it, so lookups (e.g. this variable's own
  ## estimate) never depend on row position.
  lv <- levels(data[[v]])
  out <- vector("list", length(lv))
  for (i in seq_along(lv)) {
    l <- lv[i]
    n <- sum(data[[v]] == l, na.rm = TRUE)
    if (i == 1) {
      out[[i]] <- data.frame(variable = lab, level = l, n = n, kind = "reference",
                             HR = NA_real_, CI_low = NA_real_, CI_high = NA_real_, p = NA_real_,
                             stringsAsFactors = FALSE)
      next
    }
    r <- if (is.null(td)) NULL else td[td$term == paste0(v, l), ]
    if (is.null(r) || !nrow(r) || coef_degenerate(r$estimate, r$conf.high)) {
      out[[i]] <- data.frame(variable = lab, level = l, n = n, kind = "not_estimable",
                             HR = NA_real_, CI_low = NA_real_, CI_high = NA_real_, p = NA_real_,
                             stringsAsFactors = FALSE)
    } else {
      out[[i]] <- data.frame(variable = lab, level = l, n = n, kind = "value",
                             HR = r$estimate, CI_low = r$conf.low, CI_high = r$conf.high,
                             p = r$p.value, stringsAsFactors = FALSE)
    }
  }
  bind_rows(out)
}

## ---------------------------------------------------------------------------
## Rendering
## ---------------------------------------------------------------------------

## HR/CI text is two lines, matching the requested "HR\n(95% CI)" layout.
fmt_hr2 <- function(hr, lo, hi) sprintf("%.2f\n(%.2f-%.2f)", hr, lo, hi)

## Level + sample size, e.g. "Female\n(N=31)"; "(N=141)" alone for a
## continuous row, which has no level text.
fmt_level_n <- function(level, n)
  ifelse(level == "", sprintf("(N=%d)", n), sprintf("%s\n(N=%d)", level, n))

## P-values as in the submitted figure: leading zero dropped, floored at
## "<.001", trailing " *" when significant.
fmt_p <- function(p) {
  s <- formatC(p, format = "f", digits = 3)
  s <- ifelse(substr(s, 1, 2) == "0.", substring(s, 2), s)
  s <- ifelse(!is.na(p) & p < 0.001, "<.001", s)
  paste0(s, ifelse(!is.na(p) & p < 0.05, " *", ""))
}

## Draws the 5-column banded forest table. Rows are plotted top-to-bottom in
## the order given; shading alternates by `variable` GROUP (every row of one
## variable shares a band), not by individual row.
##
## `title` is accepted for call-site compatibility but is not drawn -- the
## descriptive title line ("Multivariate Cox Regression Analysis (X)") is
## dropped per the publication spec; only the N/events/marker subtitle remains.
draw_forest <- function(rows, path, title = NULL, subtitle, width = FIG_WIDTH) {
  ## `variable` is filled on every row of a group (var_rows() no longer blanks
  ## it), so group membership is just `variable` itself; only the FIRST row of
  ## each run gets the variable name printed, decided from top-to-bottom order
  ## before the y-axis reversal below.
  rows$group    <- rows$variable
  rows$show_var <- !duplicated(rows$group)

  rows <- rows[nrow(rows):1, , drop = FALSE]   # ggplot y increases upward
  rows$y <- seq_len(nrow(rows))
  n <- nrow(rows)

  val <- rows[rows$kind %in% c("value", "continuous"), , drop = FALSE]
  ref <- rows[rows$kind == "reference", , drop = FALSE]
  ne  <- rows[rows$kind == "not_estimable", , drop = FALSE]

  ## forest extent in log10 space, with the reference line always visible
  lv <- c(val$CI_low, val$CI_high, 1)
  lv <- lv[is.finite(lv) & lv > 0]
  if (!length(lv)) lv <- c(0.5, 2)
  fmin <- log10(min(lv)); fmax <- log10(max(lv))
  if (fmax - fmin < 0.6) { mid <- (fmin + fmax) / 2; fmin <- mid - 0.3; fmax <- mid + 0.3 }
  pad <- 0.10 * (fmax - fmin)
  fmin <- fmin - pad; fmax <- fmax + pad
  fw <- fmax - fmin

  ## Column x-positions, held to FIXED PHYSICAL WIDTHS (inches) rather than
  ## multiples of `fw`. `fw` is the forest's own log-HR range, which differs
  ## panel to panel -- deriving text-column widths from it (as an earlier
  ## version did) makes fixed-size 5.5pt text overflow its column whenever a
  ## panel's HR range is narrow, since the same multiplier then buys fewer
  ## physical inches. Every text column is a constant width in every panel;
  ## only the forest column's *content* (not its width) varies with the data.
  MARGIN_L_IN <- 0.02; MARGIN_R_IN <- 0.03
  ## Column widths scale with FONT_PT (calibrated at 5.5pt -> the widths below)
  ## so a caller raising FONT_PT gets proportionally more room instead of the
  ## same fixed inches. `width` (from FIG_WIDTH) is used exactly as given --
  ## no silent override -- so both knobs actually do something; if the two are
  ## set in conflict (bigger font, narrower figure) the forest column just
  ## gets tighter, which is the caller's call to make, not this function's.
  FONT_SCALE <- FONT_PT / 5.5
  W_VAR_IN <- 0.80 * FONT_SCALE; W_LVL_IN <- 0.62 * FONT_SCALE
  W_HRC_IN <- 0.70 * FONT_SCALE; W_P_IN   <- 0.35 * FONT_SCALE
  panel_w_in  <- width - MARGIN_L_IN - MARGIN_R_IN
  forest_w_in <- panel_w_in - W_VAR_IN - W_LVL_IN - W_HRC_IN - W_P_IN
  if (forest_w_in < 0.5)
    warning(sprintf("forest column is only %.2fin wide (width=%.2gin, FONT_PT=%.1f) -- may look cramped",
                    forest_w_in, width, FONT_PT))
  du_per_in <- fw / forest_w_in   # data (log10 HR) units per physical inch

  x_hrc  <- fmin - W_HRC_IN * du_per_in
  x_lvl  <- x_hrc - W_LVL_IN * du_per_in
  x_var  <- x_lvl - W_VAR_IN * du_per_in
  x_left <- x_var - MARGIN_L_IN * du_per_in
  x_p    <- fmax
  x_right<- x_p + (W_P_IN + MARGIN_R_IN) * du_per_in

  ## axis breaks: only the conventional ones that fall inside the forest.
  ## Widened range (was 0.01-100) so very wide CIs -- e.g. the ecology
  ## validation split -- still hit this branch instead of pretty()'s
  ## scientific-notation fallback, which was unreadable at this column width.
  cand <- c(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
           1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
  brk <- cand[log10(cand) >= fmin & log10(cand) <= fmax]
  if (length(brk) > 6) brk <- brk[round(seq(1, length(brk), length.out = 6))]
  if (length(brk) < 3) brk <- signif(10^pretty(c(fmin, fmax), 3), 1)

  ## band alternates by variable GROUP, not by row
  grp_order <- unique(rows$group[order(-rows$y)])
  shaded <- grp_order[seq(1, length(grp_order), by = 2)]
  bands <- rows %>% filter(group %in% shaded) %>% group_by(group) %>%
    summarise(ymin = min(y) - 0.5, ymax = max(y) + 0.5, .groups = "drop")

  p <- ggplot() +
    geom_rect(data = bands, aes(xmin = x_left, xmax = x_right, ymin = ymin, ymax = ymax),
              fill = "grey90") +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey35", linewidth = 0.3) +
    geom_text(data = rows[rows$show_var, ], aes(x = x_var, y = y, label = variable),
              hjust = 0, size = FONT_MM, family = FONT_FAMILY, colour = "black",
              fontface = "bold", vjust = 0.5) +
    geom_text(data = rows, aes(x = x_lvl, y = y, label = fmt_level_n(level, n)),
              hjust = 0, size = FONT_MM, family = FONT_FAMILY, colour = "black",
              lineheight = 0.85, vjust = 0.5) +
    coord_cartesian(xlim = c(x_left, x_right), ylim = c(0.35, n + 1.05), clip = "off") +
    scale_x_continuous(breaks = log10(brk),
                       labels = format(brk, scientific = FALSE, drop0trailing = TRUE),
                       expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(subtitle = subtitle, x = "Hazard Ratio (95% CI)", y = NULL)

  if (nrow(val)) {
    val$lx <- log10(val$HR); val$llo <- log10(val$CI_low); val$lhi <- log10(val$CI_high)
    p <- p +
      geom_errorbarh(data = val, aes(y = y, xmin = llo, xmax = lhi),
                     height = 0.20, colour = FOREST_COL, linewidth = 0.35) +
      geom_point(data = val, aes(x = lx, y = y), size = 1.1, colour = FOREST_COL) +
      geom_text(data = val, aes(x = x_hrc, y = y, label = fmt_hr2(HR, CI_low, CI_high)),
                hjust = 0, size = FONT_MM, family = FONT_FAMILY, lineheight = 0.85, vjust = 0.5) +
      geom_text(data = val, aes(x = x_p, y = y, label = fmt_p(p)),
                hjust = 0, size = FONT_MM, family = FONT_FAMILY, vjust = 0.5)
  }
  if (nrow(ref)) {
    ## the reference level: a filled square exactly on the HR=1 line, no
    ## error bar (it is not an estimate), "reference" in place of HR/CI text,
    ## and no p-value
    p <- p +
      geom_point(data = ref, aes(x = 0, y = y), shape = 15, size = 1.3, colour = FOREST_COL) +
      geom_text(data = ref, aes(x = x_hrc, y = y, label = "reference"),
                hjust = 0, size = FONT_MM, family = FONT_FAMILY, colour = "grey30",
                fontface = "italic", vjust = 0.5)
  }
  if (nrow(ne)) {
    p <- p +
      geom_text(data = ne, aes(x = 0, y = y, label = "not estimable"),
                colour = "grey45", fontface = "italic", size = FONT_MM, family = FONT_FAMILY,
                hjust = 0.5, vjust = 0.5)
  }

  ## column headings, sitting just above the top row
  hdr <- data.frame(x = c(x_hrc, x_p), lab = c("HR\n(95% CI)", "P-value"))
  p <- p +
    geom_text(data = hdr, aes(x = x, y = n + 0.80, label = lab),
              hjust = 0, size = FONT_MM, family = FONT_FAMILY, colour = "grey20",
              fontface = "bold", lineheight = 0.85)

  p <- p + theme_minimal(base_size = FONT_PT) +
    theme(
      text = element_text(family = FONT_FAMILY, size = FONT_PT),
      panel.grid = element_blank(),
      axis.text.y = element_blank(), axis.ticks.y = element_blank(),
      axis.text.x = element_text(size = FONT_PT, family = FONT_FAMILY, colour = "grey25"),
      axis.title.x = element_text(size = FONT_PT, family = FONT_FAMILY, colour = "grey15",
                                  margin = margin(t = 2)),
      plot.subtitle = element_text(size = FONT_PT, family = FONT_FAMILY, colour = "grey35",
                                   hjust = 0, margin = margin(b = 3)),
      ## l/r reuse MARGIN_L_IN/MARGIN_R_IN (in points: 1in = 72.27pt) so the
      ## panel-width arithmetic used to place the columns above is honoured by
      ## the actual rendered margins, not just assumed.
      plot.margin = margin(t = 3, r = MARGIN_R_IN * 72.27, b = 3, l = MARGIN_L_IN * 72.27)
    )

  ggsave(path, p, width = width, height = 0.20 * n + 0.55, device = "pdf")
  invisible(p)
}

## ---------------------------------------------------------------------------
## Drivers
## ---------------------------------------------------------------------------

## Univariate: one model per covariate, stacked -- reference row, level rows,
## sample sizes and all, in the format above.
render_univariate <- function(data, vars, path, title, subtitle, announce = NULL) {
  ok <- usable_vars(data, vars, announce = announce)
  dropped <- attr(ok, "dropped")
  dd <- data[!is.na(data$fu_time) & !is.na(data$event), ]

  rows <- list(); lrt_notes <- character(0)
  for (v in vars) {
    if (v %in% ok) {
      d1 <- dd[!is.na(dd[[v]]), ]
      m <- coxph(as.formula(sprintf("Surv(fu_time, event) ~ %s", v)), data = d1)
      td <- broom::tidy(m, exponentiate = TRUE, conf.int = TRUE)
      rows[[length(rows) + 1]] <- var_rows(v, d1, td)
      ## a >2-level factor fitted alone: the model-level LRT IS its overall test
      if (length(multilevel_terms(d1, v)))
        lrt_notes <- c(lrt_notes, fmt_lrt(v, unname(summary(m)$logtest["pvalue"])))
    } else {
      ## not fit at all: N per level still comes from the shared eligible
      ## population, restricted to the outcome/follow-up being present
      rows[[length(rows) + 1]] <- var_rows(v, dd, NULL)
    }
  }
  rows <- bind_rows(rows)
  if (!nrow(rows)) { cat("  nothing to plot:", basename(path), "\n"); return(invisible(NULL)) }
  if (length(lrt_notes)) subtitle <- paste0(subtitle, "; ", paste(lrt_notes, collapse = "; "))
  draw_forest(rows, path, title, subtitle)
  invisible(list(rows = rows, n = nrow(dd), events = sum(dd$event)))
}

## Multivariate: one joint model on the estimable covariates.
render_multivariate <- function(data, vars, path, title, subtitle, announce = NULL) {
  ok <- usable_vars(data, vars, announce = announce)
  dropped <- attr(ok, "dropped")
  if (!length(ok)) { cat("  no estimable covariates:", basename(path), "\n"); return(invisible(NULL)) }
  dd <- data[stats::complete.cases(data[, c(ok, "fu_time", "event")]), ]
  m <- coxph(as.formula(sprintf("Surv(fu_time, event) ~ %s", paste(ok, collapse = " + "))),
             data = dd, control = coxph.control(iter.max = 200))
  td <- broom::tidy(m, exponentiate = TRUE, conf.int = TRUE)

  rows <- list()
  for (v in vars)
    rows[[length(rows) + 1]] <- var_rows(v, dd, if (v %in% ok) td else NULL)
  rows <- bind_rows(rows)

  ml <- multilevel_terms(dd, ok)
  if (length(ml)) {
    notes <- vapply(ml, function(v) fmt_lrt(v, lrt_drop(m, dd, ok, v)), character(1))
    subtitle <- paste0(subtitle, "; ", paste(notes, collapse = "; "))
  }
  cc <- sprintf("N = %d patients, %d events", nrow(dd), sum(dd$event))
  subtitle <- if (grepl("^N = [0-9]+ patients, [0-9]+ events", subtitle))
    sub("^N = [0-9]+ patients, [0-9]+ events", cc, subtitle) else paste0(cc, " - ", subtitle)
  draw_forest(rows, path, title, subtitle)
  invisible(list(model = m, n = nrow(dd), events = sum(dd$event), rows = rows))
}

## Consistent wording, matching the submitted figure.
fm_title <- function(kind, set) sprintf("%s Cox Regression Analysis (%s)", kind, set)
fm_sub   <- function(n, ev, extra = NULL)
  paste0(sprintf("N = %d patients, %d events", n, ev),
         if (!is.null(extra)) paste0(" - ", extra) else "")
