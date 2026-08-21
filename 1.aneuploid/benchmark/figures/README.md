# Task 6 benchmark — figures and tables

Answers **Reviewer 1, comment #6** (does DACOR outperform predefined
nuclear-feature methods?). Renders the round-3 results into publication PDFs.

```bash
Rscript 01_roc_curves.R
Rscript 02_auc_by_split.R
Rscript 03_supporting_plots.R
Rscript 04_tables.R
```

Everything writes to `BE_master/4.plotting/output/rev1_task6_figures/`.

## Data source — read before regenerating

`00_config.R` points at **round 3, step `c_fit_all_models`**:

```
rev_exp_run_scripts/runs/round3_2026-08-07_005808/c_fit_all_models/results/
```

This is the only round in which every model was fitted on the same 340-slide
training split. Rounds 1 and 2 fitted rungs 1–2 on train+val pooled (422
slides), so their train/val/test values are not comparable across models and
**must not be mixed in**. If a later round supersedes this one, change `run_id`
in `00_config.R` and rerun — nothing else is path-dependent.

## Principle: every plot is a PDF in the output folder

Including the ones that will not reach the main figure. The figure is assembled
by picking from the folder, so a panel that turns out to be needed is already
rendered rather than requiring a code change.

| File | Figure-bound? | What it shows |
|---|---|---|
| `roc_slide_val.pdf` | **yes — Panel A left** | ROC, discovery validation split, 5 primary models |
| `roc_slide_test.pdf` | **yes — Panel A right** | ROC, independent test cohort, 5 primary models |
| `auc_by_split_slide.pdf` | **yes — Panel B** | AUC across train/val/test, one line per model |
| `roc_slide_train.pdf` | no | Train ROC; every model near ceiling, shows fits were not degenerate |
| `roc_slide_*_allarms.pdf` | no | Same three splits with the two stain-normalised arms added |
| `roc_patient_{train,val,test}.pdf` | no | Patient-level ROCs (max aggregation) |
| `auc_by_split_slide_allarms.pdf` | no | Panel B including stain-normalised arms |
| `auc_by_split_patient.pdf` | no | Panel B at patient level |
| `val_to_test_drop_slide.pdf` | no | The transfer loss as a single ranked bar chart — fallback if Panel B reads busy |
| `forest_test_slide.pdf` | no | Test AUC with DeLong CIs, 5 primary models |
| `forest_test_slide_allarms.pdf` | no | Same, all 7 arms |
| `forest_test_patient.pdf` | no | Same, patient level |
| `rung3_training_history.pdf` | no | Per-epoch validation AUC for the 4 deep arms |
| `score_distributions_test_slide.pdf` | no | Predicted probability by true label, test cohort |
| `supp_table_benchmark_slide_level.csv` | **table** | Supplementary Table Y |
| `supp_table_benchmark_patient_level.csv` | no | Patient-level counterpart |

## Design decisions

**DACOR is the house red (`#d32f2f`, as in `4.cox_plot.Rmd`).** The handcrafted
ladder gets a light-to-dark blue ramp in increasing complexity; the deep-only
arm is purple, because it is a different feature family rather than one more
step along the same axis.

**The stain-normalised arms are excluded from figure-bound panels.** They sit
almost exactly on their base arms (C+M+D 0.571 vs 0.571) and would cost two
curves for no information. They remain in the `_allarms` variants and in both
tables.

**Panel B marks the scanner boundary between validation and test**, not between
train and validation. Train and validation are both discovery-cohort slides on
the same scanner; only the test cohort changes scanner. Captioning it any other
way misattributes where the domain shift happens.

**ROC legends are ordered by AUC, best first** — on a ROC panel the reader is
matching curve height to label, so ladder order would make them hunt.

## Two things that would silently corrupt these plots

**`pROC` direction must stay pinned to `"<"`.** With the default
`direction = "auto"`, pROC flips any curve that falls below chance so it reports
≥ 0.5 — which would turn the deep-only arm's test AUC of 0.439 into 0.561 and
disagree with the pipeline. All 21 AUCs were checked against
`benchmark_all_splits_slide_level.csv` and match exactly.

**The base `pdf` device, not `cairo_pdf`.** No X11/cairo on this machine;
`ggsave` otherwise warns and falls back. Also means non-ASCII glyphs (a middot,
for instance) render as `..`, so plot text is kept to ASCII.

## A pipeline bug fixed here, not upstream

`benchmark_all_splits_patient_level.csv` has correct AUCs but its
`delong_p_vs_dacor` column is a **copy of the slide-level p-values** —
`03_fit_evaluate.py` always computes DeLong from slide-level predictions keyed
on `wsi`, whichever metric function it was given. `04_tables.R` recomputes
patient-level DeLong from the aggregated per-patient scores.

It changes conclusions. At slide level, Morphology vs DACOR is p = 6.3×10⁻³;
at patient level the same comparison is **p = 0.15**, i.e. not significant with
64 patients. Any patient-level claim must use the table from `04_tables.R`, not
the pipeline CSV. Fixing `03_fit_evaluate.py` upstream is still worth doing.

As a guard, `04_tables.R` recomputes the *slide-level* p-values too and prints
their maximum relative deviation from the pipeline's (currently 2.6%, from
`pROC`'s variance estimator differing slightly from the pipeline's numpy
implementation). If that number grows, the two have diverged.
