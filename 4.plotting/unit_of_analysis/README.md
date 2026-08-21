# rev1_unit_of_analysis — Reviewer 1, comment #3

> **MS display item settled (CE, 2026-08-14).** The only item from this task that goes into the
> manuscript is the shared figure
> `BE_master/3.integration/integration_project/spatial_analysis/tabular/rev1_sensitivity/fig/SuppFig_dual_design_robustness.pdf`
> (built in task 3's tree; carries both the R1#2 parameter axis and this task's inter/intra design
> axis). **Every table produced here is now internal record**, not MS-bound — kept as the numerical
> backing for the rebuttal letter. Its point estimates are numerically identical to
> `twofold_table_publication_260813`. Drafted MS text and legend live in `REVISION_MASTER.md` §4.
> Caveat that binds the figure: in the **inter-slide** panel both Moran metrics flip sign and go
> non-significant at the 2-/3-ring settings, so "stable across parameter settings" cannot be written
> as a blanket claim.

> "Various parameters measured in the slides, such as immune cell density or lymphoplasmacytic
> cluster Moran mean, as shown in Fig 3D (and nearly all parameters in Supplementary Fig 1) are
> claimed to differ between normal and abnormal DNA content **tiles**. While the p-values are
> indeed significant, they are inflated by the high number of tiles analysed. […] when examining
> the actual shift in mean/median, this is barely noticeable. Thus, how biologically meaningful
> are these differences? Would significant differences still be seen at slide level or at
> patient level?"

Two separate questions are bundled here: a **unit-of-analysis** question and an **effect-magnitude**
question. They need different answers.

---

## 1. The premise is factually wrong — and that is the strongest point in the response

The published tests are **not** tile-level. `merged_ecology_250812_wNuc.RDS` is **777 rows = 777
biopsies**, one row per WSI. Every feature is already a per-slide summary over that slide's cells
or tiles — `nuc_median_*`, `nuc_q75_*`, `nuc_sd_*`, `Moran_Local_*_clustered_mean`, `Dens_*`. The
tests in `4.plotting/code/3.ecology.Rmd` (`single_inter_wsi`, `single_intra_wsi`) run Mann-Whitney
or Welch t-tests on n ≈ 600–780 **slides**.

So "would differences still be seen at slide level?" — **that is the slide level**, and it is what
was published.

The confusion is our fault, not the reviewer's: the Results text says "tiles" and "regions", and
no figure legend in Fig 3 or Supp Fig 1 states the unit of analysis or gives a denominator.
**Action: add "n = X biopsies" to every affected figure legend.** That is the root-cause fix and it
prevents the same comment in round 2.

## 2. The patient-level re-test was run anyway, and it strengthens the results

Aggregating 777 biopsies → 149 patients (median over each patient's biopsies; a patient counts as
DNA content abnormal if any biopsy was, matching the manuscript's own patient-level rule).

> **Correctness fix, 2026-08-05 (CE):** the aggregation above has a second, sharper rule that the
> first pass missed. When a patient's biopsies have mixed DACOR status (some abnormal, some not —
> **38 of 60 patients called abnormal, 63%**, are mixed like this, with as few as 1 abnormal biopsy
> alongside up to 30 normal ones), only that patient's own DACOR-**abnormal** biopsies may feed
> their abnormal-class aggregate value. The original code pooled every biopsy regardless of its own
> call, which diluted the abnormal-region signal for the majority of abnormal patients — exactly
> the patients the test is meant to isolate. Fixed in `01_slide_vs_patient.R`
> (`restrict_to_own_class()`). Numbers below are post-fix.

**All four main-figure metrics survive, and survive Bonferroni across the 12-metric family — and
the fix makes them slightly *stronger*, not weaker** (e.g. nucleus OD max δ 0.65→0.70, Moran's I
lymphoplasmacytic δ −0.30→−0.34). Effect sizes are still roughly *double* the slide-level ones —
the opposite of what an n-inflation artefact would do.

| Panel | Metric | n slides | p slide | n pts | p patient | p Bonf | Cliff's δ | median shift |
|---|---|---|---|---|---|---|---|---|
| Fig 3C | Nucleus OD sum (max) | 691 | 3.0e-15 | 140 | 1.5e-12 | 1.8e-11 | **0.70** large | +39% |
| Fig 3C | Haralick contrast F1 | 685 | 2.5e-04 | 140 | 3.6e-06 | 4.3e-05 | 0.46 medium | +67% |
| Fig 3D | Immune cell density | 646 | 4.8e-06 | 137 | 9.0e-07 | 1.1e-05 | **0.49** large | +86% |
| Fig 3D | Moran's I lymphoplasmacytic | 642 | 0.006 | 140 | 6.5e-04 | 0.008 | −0.34 medium | −26% |
| Supp 1 | Nucleus circularity q75 | 691 | 3.2e-05 | 140 | 3.4e-06 | 4.1e-05 | −0.46 medium | **−2%** |
| Supp 1 | Hematoxylin SD | 690 | 2.5e-04 | 140 | 2.1e-05 | 2.5e-04 | 0.42 medium | +26% |
| Supp 1 | Moran's I immune | 657 | 0.003 | 141 | 9.5e-04 | 0.011 | −0.33 small | −24% |
| Supp 1 | Global Moran's I immune | 424 | 0.014 | 107 | 0.012 | 0.140 | 0.28 small | +13% |
| Supp 1 | Ripley's L epi–immune | 605 | 0.032 | 140 | 0.015 | 0.185 | 0.24 small | +1% |
| Supp 1 | Mean kNN epi–immune | 777 | 5.8e-04 | 149 | 0.115 | 1.00 | −0.15 | −19% |
| Supp 1 | Nucleus OD sum (median) SD | 673 | 0.027 | 140 | 0.297 | 1.00 | 0.10 | +10% |
| Supp 1 | Nucleus area SD | 673 | 7.5e-04 | 140 | 0.476 | 1.00 | 0.07 | +4% |

**7 of 12 clear Bonferroni** (unchanged from before the fix); **2 more now cross BH** (Global
Moran's I immune, Ripley's L epi–immune — both moved from clearly null to weakly significant).
3 remain fully null: mean kNN, nucleus OD median SD, nucleus area SD.

Five still fail Bonferroni — all in Supp Fig 1. Reporting these honestly is the right call: they
are supplementary, no claim in the paper leans on them, and conceding them buys credibility for
the seven that hold.

## 3. Effect magnitude — the half of the comment that p-values do not answer

The reviewer's *first* question is about magnitude, not significance. Returning only statistical
refinements would answer the second question and dodge the first. Both figures and the output
table therefore carry Cliff's δ (with bootstrap CI) and the raw median shift alongside every p.

**Circularity is the case the reviewer is describing**, and it needs a decision:
Bonferroni p = 1.1e-04, but the median shift is **−1%**. Note that the two magnitude measures
disagree here — Cliff's δ = −0.44 is "medium", because circularity is tightly bounded
(≈0.78–0.83) so a small absolute shift is a large *rank* separation. Defensible either way, but
pick one story and state it: either report δ and explain why it is the fairer summary for a
bounded metric, or drop the panel. Do not report the p alone.

## 4. Refined two-fold table — 9 CE-specified metrics, inter-slide vs. intra-slide

CE narrowed the metric list (5 nuclear + 4 immune-ecological, dropping immune density, mean kNN,
and nucleus OD-median-SD from the original 12) and specified two folds, matching the exact group
definitions in `3.ecology.Rmd`'s `single_inter_wsi()`/`single_intra_wsi()`:

- **Inter-slide**: DACOR-abnormal patients' own abnormal biopsies (attn1) vs. DACOR-normal
  patients' biopsies (attn2, whole-slide).
- **Intra-slide**: within DACOR-abnormal (`pred==1`) biopsies only, attn0 vs. attn1, paired.

### A loader bug surfaced while building this, now fixed

`load_ecology()` only replicated the second of `3.ecology.Rmd`'s two rename steps. The first step
labels any column *without* a trailing digit as the whole-slide value (`_attn2`) before the
digit-suffix step runs; skipping it left unsuffixed whole-slide columns invisible to every
`<base>attn2` lookup in this codebase. Fixed 2026-08-13. Concretely this recovers **Ripley's L's**
true attn2 (a bare `Ripley_L12_..._frac_positive` column was sitting there unrenamed) — it was
previously reported as "no attn2 available," which was wrong.

### A real, separate data gap: nuclear features have no whole-slide value anywhere

Checked directly against the raw column names: every `nuc_*` metric has only `_attn0`/`_attn1`,
no unsuffixed or `_2`-suffixed variant exists in `merged_ecology_250812_wNuc.RDS` at all — this
was never computed upstream (nuclear features come from a separate HPC pipeline that only ran the
attn0/attn1 split). **CE's call (2026-08-13):** fall back to attn1-for-both-groups for the 5
nuclear metrics' inter-slide fold, matching the original published method and the task's primary
table. The 4 ecology metrics use the true attn1-vs-attn2 design.

### Results worth flagging

The new inter-slide design (attn1 vs. attn2, not attn1 vs. attn1) changes the ecology-metric
numbers substantially — not just noise:

| Metric | Old (attn1 vs attn1) δ | New (attn1 vs attn2) δ | Read |
|---|---|---|---|
| Ripley's L, epi-immune | 0.24 | **0.74** | Much cleaner separation — the new design is doing what it's supposed to. |
| Moran's I, immune | -0.33 | **-0.67** | Same, direction unchanged. |
| Moran's I, lymphoplasmacytic | -0.34 | **-0.67** | Same direction unchanged — the still-unresolved Moran-direction discrepancy (see below) is now a *bigger* number, not a smaller problem. |
| Global Moran's I, immune | 0.28 (p=0.012) | **0.01 (p=0.817)** | Goes to essentially null. The earlier "BH-significant" finding for this metric doesn't survive the more biologically appropriate comparator. |

**Intra-slide, properly restricted to `pred==1` slides**, also changes things: n drops from 149
(all slides, the task's primary table) to 50–58 (DACOR-abnormal-only). Global Moran's I immune
drops out of significance entirely (p=0.077) at this reduced n — it was apparently relying on
DACOR-normal slides' attn0-vs-attn1 contrast for its earlier significance, which is not the
biologically intended comparison. Ripley's L intra-slide is very large (δ=0.85, p=4.5e-08) —
checked this isn't a repeat of the kNN region-partition artifact flagged in section 3: only 86% of
slides move the same direction here (vs. 99.4% for kNN), consistent with real biology rather than
a stratum-definition artifact, but still the single largest effect in either fold and worth a
second look before leaning on it.

## 5. Replication of the published slide-level p-values + their adjusted p (`07_replicate_published.R`)

The published Fig 3C/3D/Supp Fig 1 numbers are **slide-level with no patient aggregation**, and the
published inter-slide fold is **attn1 vs attn1** (`single_inter_wsi(merged_df, paste0(base,"attn1"))`
pulls the *same* column for both `pred` groups). `single_intra_wsi()` hardcodes
`filter(pred == 1)` and ignores its own `data` argument. `07_replicate_published.R` reproduces both
exactly.

**Replication verified against known published values:** Global Moran's I immune reproduces to the
digit in both folds — inter 0.014, intra 0.009.

### All published findings survive BH; several do not survive Bonferroni

| Fold | max BH (12-metric family) | all < 0.05? | max Bonferroni | all < 0.05? |
|---|---|---|---|---|
| Inter-slide | 0.032 | **yes** | 0.380 | no |
| Intra-slide | 0.047 | **yes** | 0.566 | no |

This is the strongest single fact for the R1#3 rebuttal: **every published p-value survives
BH-FDR correction in both folds**, which is exactly the correction argued for in section "How to
decide between BH and Bonferroni". Bonferroni failures (inter: Ripley 0.380, nucODmedianSD 0.326,
globalMoranImm 0.173, moranLymphoPl 0.067; intra: circularity 0.566, immuneDensity 0.430) are the
expected over-penalty on a correlated family.

### Why the two-fold table (05/06) differs — decomposed 2×2

`interslide_decomposition_260814.csv` separates the two confounded changes: **unit**
(slide → patient) and **comparator** (attn1-vs-attn1 → attn1-vs-attn2). They do not act the same
way, and the two metrics CE queried fail for opposite reasons:

| Metric | slide, a1-a1 (published) | slide, a1-a2 | patient, a1-a1 | patient, a1-a2 (05/06) | Cause |
|---|---|---|---|---|---|
| Global Moran's I, immune | **0.014** | 0.148 | 0.009 | **0.817** | **Comparator.** Already lost at slide level once attn2 is the comparator; patient aggregation alone *keeps* it (0.009). |
| Nucleus area, SD | **7.5e-04** | — (no attn2) | **0.534** | — | **Unit.** Design is identical (nuclear falls back to a1-a1); 673 slides → 139 patients kills it outright. |

So nucleus area SD is a genuine pseudo-replication case — its slide-level significance was carried
by patients contributing many biopsies, which is precisely what R1#3 suspected. Global Moran's I
immune is *not* a pseudo-replication casualty; it is sensitive to which region the DACOR-normal
comparator is drawn from.

### ⚠ Two metrics point in opposite directions in the two folds

Now that both folds sit side by side in one table, a discrepancy is visible that was not before.
Sign of the effect size, published slide-level:

| Metric | Inter-slide δ | Intra-slide rb | |
|---|---|---|---|
| Nucleus area, SD | **+0.16** | **−0.34** | opposite |
| Global Moran's I, immune | **+0.14** | **−0.19** | opposite |
| *(other 7)* | — | — | consistent |

Both are significant after BH in *both* folds, so the table shows two starred cells whose signs
disagree: DNA-content-abnormal biopsies have *higher* nucleus-area SD than normal biopsies, but
*within* an abnormal biopsy the abnormal region has *lower* area SD than that same biopsy's normal
region. Not necessarily wrong — between-biopsy and within-biopsy contrasts genuinely can diverge —
but it needs a one-line interpretation in the legend or it reads as an inconsistency. Same for
Global Moran's I immune. **Owner: CE**, alongside the Moran-direction item below.

### Latent bug in the published `single_inter_wsi()` (did not affect these 12)

`p` is only assigned inside `if (pval < p_threshold)`, but the function still `return(p)`s
unconditionally — for a non-significant metric it therefore returns whatever `p` is in the calling
environment, which in the `{r intrawsi}` loop is **the intra-slide plot from the same iteration**,
and saves it into `interwsi/`. Harmless for the published 12 (every inter p < 0.05, max 0.032), but
any rerun over a wider metric set will silently mislabel plots.

---

## Notes and open items

### ⚠ Moran direction may contradict the Fig 3D legend — UNRESOLVED, owner: CE

`Moran_Local_LymphoPlasma_Ii_clustered_mean` comes out **lower** in DNA content abnormal tissue:
Cliff's δ = −0.30, median shift −25% (patient level); −0.13 at slide level. The same negative sign
appears in *both* the between-slide and the within-slide paired contrast, so it is not an artefact
of the contrast chosen.

The Fig 3D legend says the opposite: *"DNA content abnormal regions exhibited **elevated**
concentrations of lymphoplasmacytic cell clusters, as quantified by Moran's I spatial
autocorrelation analysis, indicating enhanced local immune activation"*, and the Results text says
*"Moran's local I clustering score of plasma cells was significantly **higher** in DNA content
abnormal samples"*.

Either the wrong Moran variant is being read here (four exist: `_clustered_mean`, `_abs_mean`,
`_abs_max`, `_clustering_heterogeneity`) or the direction in the text is wrong. **Must be settled
before this table is submitted** — reviewer 1 comment #7 already flags a metric-labelling error in
Fig 4C, and a second one would be costly. Numbers in this folder are reported as computed; they
have not been reconciled with the figure.

### Out of scope: Fig 4C

Comment #3 names Fig 3D and Supp Fig 1 only, so **Fig 4C is not part of this response** and is not
to be reported. Computed once for the record (`fig4c_outofscope_check_*.csv`), because the same
design underlies it:

| Metric | n slides | p slide | n pts | p patient |
|---|---|---|---|---|
| Getis-Ord Gi*, plasma clustering strength | 487 | 2.0e-04 | 115 | 0.549 |
| Plasma cell density | 333 | 3.6e-08 | 84 | 0.029 |
| Global Moran's I, plasma | 272 | 0.003 | 73 | 0.152 |

Two of three lose significance at patient level. Mitigating: n collapses to 73–115 within the DNA
content abnormal subgroup, so this is as much a power limit as a null result. Relevant to reviewer
1 comment #9, which independently asks for that claim to be toned down — the toning-down is
happening anyway, so this does not need to be volunteered, but the numbers should be known before
deciding how far to soften.

### ⚠ Do not report the within-slide kNN or Ripley comparison

The within-slide paired contrast (`contrast == "within"` in the output) is sound for most metrics
but **not** for `meanKNN_Epithelial_Immune`: attn1 median = 418 vs attn0 median = 70.6, with
772/777 slides going the same direction (rank-biserial = 1.00, p = 1e-128). A metric where 99.4%
of slides move one way is describing the region partition — the two strata differ in area and
cell count, and mean kNN distance is not comparable across them without normalisation — not
biology. `Ripley_L12` within-slide (rb = 0.72, p = 6e-43) is suspect for the same reason.

The **between-slide contrast is the sound one** and is what Fig 3D actually plots. Prefer it.

### Sensitivity: max vs median aggregation

`unit_of_analysis_sensitivity_*.csv` repeats the between contrast using `max` over each patient's
**own DACOR-abnormal** biopsies (the "worst biopsy" rule, matching the manuscript's patient-level
DACOR rule; post-fix, this now correctly excludes each patient's normal biopsies too).

**Median is the primary, deliberately** — but note the previous justification here was itself an
artefact of the aggregation bug and has been corrected. Pre-fix, `max` appeared to invert the
downward-shifting metrics (Moran's I lymphoplasmacytic → p = 0.92, circularity → p = 0.65),
because `max` was being taken over a mix of abnormal *and* normal biopsies and tended to land on
one of the (higher) normal-biopsy values. Post-fix, restricted to each patient's own abnormal
biopsies, `max` agrees with `median` in direction and mostly in significance (Moran's I
lymphoplasmacytic: p = 0.007, Bonf = 0.083, δ = −0.27; circularity: p = 0.001, Bonf = 0.014,
δ = −0.32). The real reason to prefer median as primary is narrower: the number of a patient's own
abnormal biopsies ranges 1–33 (median 3), so `max` is still upward-biased for patients with many
abnormal biopsies relative to patients with only one.

### Known gaps

- **Supp Fig 1 panel assignment is inferred, not verified.** The twelve metrics are taken verbatim
  from the `base_names` vector of the `{r intrawsi}` chunk in `3.ecology.Rmd`; the four Fig 3C/3D
  assignments come from the Fig 3 legend. The remaining eight are *assumed* to be Supp Fig 1. The
  supplementary PDF is not in the repo, and the reviewer says "nearly all parameters in
  Supplementary Fig 1" — if the real Supp Fig 1 has more panels, this table under-covers it.
- Slide-level p-values here reproduce the published pipeline (same `!= 0` filter, same
  Shapiro-then-t-test-or-Mann-Whitney selection) but were not diffed against the archived figure
  outputs value-by-value.
- Patient count is 149, not the 191 of Table 1: the ecology table covers 777 slides, and `RandomID`
  collapses the 191 records to 149 people (see `beacon_file_locations` — `patient` in the clinical
  tables is a hash keyed on `(RandomID, fu_time, cohort)`).

---

## Files

| | |
|---|---|
| `00_config.R` | Paths, the 12-metric definition table, statistics helpers. Resolves `BE_master` across cluster/mount/local, or `$BE_MASTER`. |
| `01_slide_vs_patient.R` | Both contrasts at both units + effect sizes + multiplicity, at both slide and patient level. Writes the main, sensitivity (+ redundancy diagnostic) and Fig-4C tables. Contains `restrict_to_own_class()` — see the 2026-08-05 fix note above. |
| `02_plots.R` | Fig S-A and Fig S-B. **⚠ Built against the pre-fix (2026-08-01) patient-level numbers — stale, not regenerated yet.** Do not use until rerun against the corrected CSV; run `01` first when it is. |
| `03_table.R` | **Internal/diagnostic table** — full column set (n incl. mixed-status count, raw + Bonferroni p at both levels, δ + magnitude, % shift). Our own audit trail and rebuttal-letter source; not for the manuscript. Run `01` first. |
| `04_table_publication.R` | **Publication table (12-metric, single-fold)** — the 5-column version CE uses in the MS/supplement: Metric, n (abn./norm.), p-value, p-adjusted (BH), Cliff's δ. Run `01` first. |
| `05_table_twofold.R` | Refined 9-metric, two-fold (inter-slide/intra-slide) analysis — CE's narrowed metric list and exact `single_inter_wsi()`/`single_intra_wsi()` group definitions. Independent of `01` — has its own metric table and loads `eco` directly. Contains the `load_ecology()` two-step-rename fix (2026-08-13). |
| `06_table_twofold_publication.R` | **Publication table (9-metric, two-fold)** — nested Inter-slide/Intra-slide header, rows grouped Nuclear / Immune ecological. `BASIS <- "published"` (default, 2026-08-14) builds it from the **published slide-level** numbers via `07`; `BASIS <- "patient"` rebuilds the earlier patient-level rendering from `05`. |
| `07_replicate_published.R` | Exact replication of the **published slide-level** p-values (both folds, as `3.ecology.Rmd` computes them), their never-before-computed BH/Bonferroni adjustment, and the unit-vs-comparator 2×2 decomposition. Standalone. See section 5 above. |

Outputs → `BE_master/4.plotting/output/rev1_unit_of_analysis/`

| | |
|---|---|
| `unit_of_analysis_260801.csv` | Primary table: 12 metrics × 2 contrasts, slide and patient level, δ + CI, shifts, BH and Bonferroni **at both levels** (added 2026-08-06 — the slide-level family didn't have adjusted p before). Post-fix as of the 2026-08-05 rerun (filename timestamp is the script's original write date, not the fix date — check file mtime / git history for provenance). |
| `unit_of_analysis_sensitivity_260801.csv` | Same, `max` aggregation, plus a redundancy diagnostic (`max_abs_corr_family`, `most_corr_with` — each metric's strongest Spearman correlation with another metric in the family; e.g. moranLymphoPl↔moranImmune r=0.84, added 2026-08-08). Sensitivity only — see caveat above. Also post-fix. |
| `fig4c_outofscope_check_260801.csv` | Fig 4C. Record only, not for reporting. Also post-fix (same `restrict_to_own_class()` applied, though moot for `progression`, which doesn't vary within a patient). |
| `unit_of_analysis_table_internal_260805.tex/.pdf` | Full diagnostic table — 12 metrics, between-slide contrast, all columns. **Not the manuscript table** — kept for our own reference and as the rebuttal-letter numbers source. (Archived from its original name `unit_of_analysis_table_260805` on 2026-08-11 when the publication version was split out.) |
| `unit_of_analysis_table_publication_260811.tex/.pdf` | The 12-metric, single-fold manuscript table. 5 columns, per CE's original spec. |
| `twofold_interslide_260813.csv` / `twofold_intraslide_260813.csv` | Raw results behind the two-fold table — 9 metrics × {inter, intra}, own BH family per fold. |
| `twofold_table_publication_260813.tex/.pdf` | **The current manuscript table.** 9-metric, two-fold, **built on the published slide-level numbers** (rebased 2026-08-14 — raw p reproduce the Fig 3C/3D/Supp Fig 1 panels exactly, BH added). All 9 metrics significant after BH in both folds. `n` column and its footnote text dropped 2026-08-16 (CE) — sample sizes reported in text/legend instead. |
| `twofold_table_patientlevel_260813.tex/.pdf` | The earlier **patient-level** rendering of the same 9 metrics (attn1-vs-attn2 for ecology), preserved when the file above was rebased. Regenerate with `BASIS <- "patient"` in `06`. Still the only place the "effect sizes grow at patient level" argument is visible. |
| `published_replication_260814.csv` | Published slide-level p (both folds) reproduced exactly, + BH and Bonferroni over the 12-metric published family and CE's 9-metric subset. Section 5. |
| `interslide_decomposition_260814.csv` | Inter-slide 2×2: unit (slide/patient) × comparator (attn1/attn2), per metric. Section 5. |
| `figSA_patient_level_260801.pdf/.png` | Patient-level distributions, all 12 metrics. **Stale — see `02_plots.R` note.** |
| `figSB_slide_vs_patient_260801.pdf/.png` | Effect size slide → patient. **Stale — see `02_plots.R` note.** |

```bash
Rscript 01_slide_vs_patient.R && Rscript 04_table_publication.R
## 03_table.R (internal) is not chained here by default — run it separately when the
## audit-trail/rebuttal-letter numbers need refreshing.
## 02_plots.R intentionally not chained here until it's rerun against the fix.

Rscript 05_table_twofold.R && Rscript 06_table_twofold_publication.R
## Independent pipeline (different metric list, different fold design) -- does not
## depend on or feed into 01-04.
```

Needs `dplyr`, `tidyr`, `readxl`, `ggplot2` (01/02); `latexmk`/`xelatex` on `$PATH` (03/04). Cliff's
δ is implemented locally rather than via `{effsize}`, and the figures use facets rather than
`{patchwork}`, to avoid adding dependencies.
