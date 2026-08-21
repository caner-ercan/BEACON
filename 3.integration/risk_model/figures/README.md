# Task 1 figures — Elastic Net vs. LASSO

Panel figure supporting `paper/rev1/REVISION_MASTER.md` §1 (Reviewer 1, comment #10). Reuses the
data/fitting code in `BEACON/3.integration/code/rev1_elastic_net/` directly (`00_config.R`, `lib_en.R`) —
nothing here refits the model differently; it only visualises the same LASSO (alpha=1) and Elastic Net
(alpha=0.5) fits at the published CV settings and seed 5 that are already in the response table
(`en_response_table_seed5.csv`).

## Files

| | |
|---|---|
| `00_config.R` | sources task 1's config/lib, sets output dir + house palette |
| `01_prepare_fits.R` | fits both models once, saves per-patient scores (both splits) + coefficients |
| `02_panel_A_roc.R` | Panel A — ROC, train and test plotted separately |
| `03_panel_B_km.R` | Panel B — Kaplan-Meier, both models x both splits (4 files) |
| `04_panel_C_features.R` | Panel C — LASSO feature bar chart, colored by EN retention |

Run in order (`01` before `02`/`03`/`04`; `02`–`04` are independent of each other):

```bash
Rscript BEACON/4.plotting/code/rev1_elastic_net_fig/01_prepare_fits.R
Rscript BEACON/4.plotting/code/rev1_elastic_net_fig/02_panel_A_roc.R
Rscript BEACON/4.plotting/code/rev1_elastic_net_fig/03_panel_B_km.R
Rscript BEACON/4.plotting/code/rev1_elastic_net_fig/04_panel_C_features.R
```

Output → `BE_master/4.plotting/output/rev1_elastic_net_fig/` — 7 panel PDFs, `figS_C_feature_comparison_data.csv`
(the full coefficient table behind Panel C), `fits_lasso_en.RDS` (both fitted models, for re-plotting).

## Style

Read off `BEACON/4.plotting/code/{4.eco_metrics.Rmd, 4.cox_plot.Rmd}` and the published
`.../mae_7_5_withnuc/feature_plot.Rmd`:

- ROC: `pROC` + `ggplot2`, dashed diagonal, `theme_light`, matches `4.eco_metrics.Rmd`'s ROC chunk.
- KM: `survminer::ggsurvplot`, `color_set 1` palette (`#4575b4`/`#fc8d59`), `theme_survminer`, risk table —
  matches `create_survival_plot()` in `4.eco_metrics.Rmd` exactly (font sizes, `fu_time > 14` guard, the
  `ggsave_workaround` for `survminer`'s combined grob).
- Feature bars: steelblue/coral sign fill — matches the published `feature_plot.Rmd` exactly.
- Individual PDFs per panel, not a combined patchwork figure — matches every existing multi-panel output in
  this repo (e.g. `mae_7_5_withnuc/{cm_discovery,cm_val,roc_curves,feature_importance}.pdf`), for assembly
  in Illustrator/Affinity.
- One departure from house style, deliberate: Panels A and B needed a "which model" color dimension that
  doesn't already carry a meaning elsewhere in the house palette (blue/orange = risk group; red/blue =
  effect direction in the Cox plots). Panel A/B's teal/purple (`#1b9e77`/`#7570b3`) is new and used only for
  "LASSO vs Elastic Net" — see `PAL$model` in `00_config.R`.

## A rendering pitfall worth knowing about

Panel C's stroma marker was originally a unicode dagger (`†`). R's default `pdf()` device uses a
non-embedded, Custom-encoded Helvetica font that cannot represent it — instead of erroring, it silently
substitutes the literal string `"..."`, which corrupted exactly the 7 stroma-flagged axis labels and the
subtitle. Caught by rasterizing the PDF and reading it back, not by the R session (no warning was printed).
Fixed by switching to a plain ASCII marker (`" *"`). If any future panel needs a non-ASCII glyph, use
`ggsave(..., device = cairo_pdf)` instead of relying on the default device.

## Verification

`01_prepare_fits.R` asserts its own AUCs against the values already reported in the response table
(`stopifnot` on val AUC = 0.8167 / 0.8000) — if task 1's underlying data or fitting code ever changes, this
script fails loudly rather than silently drawing a stale figure.
