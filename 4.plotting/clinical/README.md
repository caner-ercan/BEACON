# rev1_clinical — universal clinical table rebuild (revision round 1)

Rebuilds the shared slide- and patient-level clinical tables so that **BE
length, dysplasia grade and acid-suppression treatment are populated in both
cohorts**, then regenerates Table 1 and the patient-level Cox models on top of
them.

Addresses **reviewer 1 comment #4** (dysplasia and BE length were missing from
the test-cohort multivariate model) and produces the covariate table needed for
**reviewer 3 major comment #1** (incremental value over clinical factors).

## Run order

```bash
cd BEACON/4.plotting/code/rev1_clinical
Rscript 01_build_clinical_tables.R   # writes the tables + QC report
Rscript 02_table1.R                  # Table 1
Rscript 03_cox_clinical.R            # Cox by cohort (discovery/test/whole), LRT, forest plots
Rscript 04_cox_eco_split.R           # Cox by ecology train/val split (Supp. Fig. 3B)
Rscript 06_forest_grid.R             # every forest panel (sources 05_forest_style.R)
```

`03` stratifies by **scanning-site cohort**; `04` stratifies by the **ecology
model's own train/validation split** and additionally documents why dysplasia
collapsed on the validation split at submission. Both are needed — they answer
different questions and are not substitutes.

## Publication-style forest plots

`05_forest_style.R` holds the shared Cox-fitting and rendering code. Layout,
five columns left to right:

1. **Variable name** — shown once, on the variable's first row.
2. **Level + N** — e.g. `Female\n(N=31)`; for a continuous variable there is
   no level, just `(N=141)`.
3. **HR (95% CI)** as two lines, or `reference` for a factor's baseline level,
   or `not estimable` where the contrast could not be fit.
4. **The forest** — point + 95% CI whisker on a log hazard axis with a dashed
   line at HR = 1. A reference level is drawn as a filled **square sitting
   exactly on that line**, with no error bar (it isn't an estimate).
5. **P-value** — blank for reference and not-estimable rows.

Rows are banded in alternating shading **by variable**, not by row, so a
variable's reference + level rows read as one group. This is
`forestmodel::forest_model()`'s default layout (the package `4.cox_plot.Rmd`
imports) reproduced by hand, so an inestimable contrast can be shown
explicitly rather than erroring or distorting the axis.

**Estimability is decided by fitting, not by counting cells**, and **per
contrast**: a 3-level factor can have one estimable level and one that is not
(acid suppression in the discovery cohort — H2 has zero events and is not
estimable, PPI is fine). `usable_vars()` fits each covariate and keeps it if
*any* contrast is finite; `var_rows()` then marks the individual degenerate
contrast as "not estimable" rather than dropping the whole variable.

**Acid suppression** is a 3-level factor with **`None` as the reference**. The
`H2` and `PPI` rows are rendered in *every* panel — including with `(N=0)`
where a level is entirely absent from that population (the ecology training
split is 100% PPI) — so the table layout is identical across populations; only
the numbers (or "not estimable") follow the data:

| population | None | H2 | PPI | estimable? |
|---|---|---|---|---|
| discovery | 7:1 | 7:0 | 86:17 | PPI only — H2 has no events |
| test | 1:0 | 1:2 | 9:28 | neither — the reference has no events |
| training split | 0:0 | 0:0 | 19:22 | neither — everyone is on PPI |
| validation split | 1:0 | 0:2 | 8:9 | neither |
| whole | 8:1 | 8:2 | 95:45 | both |

(cells are non-event:event)

Because the two contrasts have no single p-value, the **overall drop-one
likelihood-ratio test** for treatment is appended to the subtitle of any panel
that fits it. This is the drop-one (Type II) test — treatment adjusted for
everything else including the marker — not `anova()`'s sequential test, which
enters treatment before the marker and gives a different p-value.

## The grid

`06_forest_grid.R` writes **28 standalone PDFs plus `manifest.csv`** to
`BE_master/4.plotting/output/rev1_clinical/figures/all_forests/`. One folder, so
panels can be picked per figure rather than being pre-assigned to one.

| marker | populations | why that axis |
|---|---|---|
| Flow cytometry | discovery / test / whole **cohort** | scanning-site cohorts, as the manuscript reports them |
| DACOR | discovery / test / whole **cohort** | same |
| BEACON (ecology model) | training / validation / whole **split** | the model was *fitted* on its own patient-constrained split, so that — not scanning site — is the meaningful axis |

× univariate and multivariate. Filenames are
`{marker}_{population}_{analysis}[_variant].pdf`.

Variants beyond `full`:

- **`_reduced`** — the covariate set actually used at submission (age, sex, BE
  length; no dysplasia, no treatment), for the three populations where
  dysplasia was unavailable or degenerate then: test cohort, and both ecology
  splits. Note that in *univariate* panels the marker estimate is identical
  between full and reduced — each variable is fitted alone, so only the set of
  displayed rows differs. The reduced univariate panels exist to reproduce the
  submitted row set exactly, not to give a different number.
- **`_assubmitted`** — audit reproduction of the published whole-dataset model,
  whose complete-case population turns out to be discovery-only (n=100).

`manifest.csv` lists every panel with its MS figure, population, n, events and
the marker's own HR/CI/p — use it to choose without opening each PDF.

Needs `readxl`, `dplyr`, `tidyr`, `survival`, `broom`, `ggplot2`. Deliberately
**not** `survminer` — `ggforest` is replaced with a plain ggplot forest plot, so
there is one less dependency and the output is legible at figure size.

`00_config.R` resolves `BE_master` from a candidate list (cluster path, mounted
share, local checkout); override with the `BE_MASTER` environment variable.

## Outputs

In `BE_master/0.input/organised_wsi_patient/`:

| file | rows | what |
|---|---|---|
| `MIL_samples_rev1_260730.csv` | 777 | slide level |
| `MIL_patients_rev1_260730.csv` | 191 | patient level |
| `eco_patients_rev1_260730.csv` | 191 | patient level + ecology risk category |
| `wsi_randomid_map_rev1_260730.csv` | 777 | complete `wsi → RandomID` map |

In `BE_master/4.plotting/output/rev1_clinical/`: `qc_report_*.txt`,
`table1_rev1_*.{csv,txt}`, `cox_clinical_*.{csv,txt}`, `cox_lrt_*.csv`,
`cox_forest_*.pdf`.

Nothing existing is overwritten. The inputs are `MIL_*_tx_260317.csv`, which
already carried the March 2026 treatment merge.

## Column contract

Three dysplasia columns, because downstream code disagrees about what
`dysplasia` means:

| column | values | use |
|---|---|---|
| `dysplasia_bin` | `No Dysplasia` / `Dysplasia` | **use this for all analysis** |
| `dysplasia_grade` | `N` < `IND` < `LGD` < `HGD` < `EAC` | harmonised ordinal |
| `dysplasia` | = `dysplasia_grade` | back-compatibility only |
| `dysplasia_orig` | discovery cohort's original label | audit trail |

`dysplasia` keeps *grade* semantics so that pre-existing scripts doing their own
`case_when(dysplasia %in% c("LGD","LGD/HGD","HGD/LGD","HGD"))` behave exactly as
before. Those scripts will return `NA` for the test cohort's `IND` and `EAC`
slides — they had no test-cohort dysplasia at all previously, so this is not a
regression, but new work must use `dysplasia_bin`.

Per the revision decision, **every grade above NDBE counts as dysplasia**
(indefinite, LGD, HGD and EAC alike), matching the original manuscript, which
merged all dysplasia levels into one `Dysplasia` category. No separate rule was
introduced for EAC.

`BELength_src` / `dysplasia_src` record provenance; `*_orig` columns preserve
the pre-revision value so any change is auditable.

Treatment: `treatment` is **ever / never on acid suppression**, pooled across
baseline and follow-up (revision decision — agent class and timepoint are both
collapsed). `treatment_fu` (`PPI`/`H2`/`None` at last follow-up),
`treatment_baseline` and `treatment_orig` (the March column) are retained for
audit. `"No Data"` is `NA` rather than a level.

**`treatment` (ever/never) is constant.** Every one of the 188 patients with a
treatment record was on a PPI or an H2 blocker at baseline, at follow-up, or
both, so that binary has no untreated group and no contrast to estimate.
`03_cox_clinical.R` therefore drops it.

Splitting the agent class (`treatment_fu`: PPI / H2 / None) does give variation
and is what the figure scripts use — but it only survives in the whole-cohort
model; see the forest-plot section above for the per-population verdict. Either
way, note the treatment contrasts in the whole-cohort model are **not** a
credible causal effect: they rest on 9 untreated patients with 1 event, and
point the "wrong" way (treated patients have the higher hazard), which is the
signature of confounding by indication — patients with longer segments and
worse disease are the ones who get treated. Treatment is in the model to adjust
for it, not to make a claim about it.

## Where the new data came from

* **Test-cohort dysplasia** — `BE_master/revision/NU slide histology data for
  Caner 060926.xlsx`, column `AtypismTypeOdze` (1 NDBE, 2 indefinite, 3 LGD,
  4 HGD, 5 EAC). Keyed on the unnamed first column, which is the digital slide
  ID. Covers **355/355 test slides and 31/31 test individuals**. Graded per
  slide × level, rolled up with the "worst grade" rule used elsewhere in the
  pipeline. One slide has no grade at any level.
* **Test-cohort BE length** — `LES − OS` from the same file, present for all 355
  slides. See the caveat below.
* **Treatment** — `BE_master/revision/treatment_260316/Acid med summary for
  Caner 03042026.xlsx`.
* **Discovery gaps** — `Dysplasia Study Detailed Data.xlsx`, joined on
  `(wsi, selected_tisue)` for dysplasia (the key the original build used, so the
  grade belongs to the tissue block that was imaged) and on RandomID for
  BE length (constant per patient there).

## Things that needed deciding, and what was decided

**The `wsi → RandomID` map was incomplete.** `convertion_sample.csv` leaves 75
slides / 26 hashed patients without a RandomID, which is why the March merge
could not reach them. `barrett_dataset_withFUtimes.xlsx` maps all 777 slides,
with **zero conflicts** where both sources have a value, so it is used to
complete the map. This alone recovered treatment for 19 patients.

**The patient-level rollup was dropping data.** The original used a bare
`max(as.numeric(dysplasia))`, which returns `NA` if *any* slide of a patient is
ungraded — so patients with partial data were discarded rather than resolved.
The NA-safe rollup here recovers 5 discovery patients, one of them HGD. This
changes Table 1's discovery dysplasia counts.

## Caveats worth reading before writing the rebuttal

**BE length points in opposite directions in the two cohorts.** Discovery
HR 1.21 per cm (95% CI 1.07–1.38, p = 0.003); test HR 0.76 (0.62–0.93,
p = 0.006). Both survive adjustment and both are significant. Because the two
cohorts use *different measurement sources* — the workbook's own `BELength` for
discovery, `LES − OS` for the test cohort — cohort and measurement method are
perfectly confounded and cannot be separated with these data. Validation on the
13 individuals present in both sources gives r = 0.93 but only 2/13 exact
matches (typical disagreement 1–3 cm), so `LES − OS` is a good proxy, not the
same quantity. `03_cox_clinical.R` prints a dedicated diagnostic block for this.

**Hashed patients are not people.** `patient` is a random 5-character hash keyed
on `(RandomID, fu_time, cohort)`, so one person biopsied at several timepoints
becomes several "patients". The 191 records are **149 distinct individuals**;
in the test cohort, **64 records are 31 individuals** (one contributes 9).
Seven individuals appear in *both* cohorts. Table 1 reports both counts, and
`03_cox_clinical.R` runs every model a second time with `cluster(rid)` so the
duplication can be shown not to drive the result. The grouping itself is left
untouched — it is what every existing downstream table is keyed on, and
`wsi_patient_data.Rmd` must not be re-run, since the hashes are seeded and
would be regenerated differently.

**33 slides have negative `fu_time`** and 100 have zero — biopsies at or after
the event date. The analysis scripts drop `fu_time <= 14` days, which removes
them; this subsumes the concurrent-malignancy exclusion described in the
Methods.

**Discovery data is still incomplete.** After the rebuild, 22/127 discovery
patients lack a dysplasia grade and 15/127 lack BE length (both were absent on
every slide of those patients, in every available source). Complete-case
multivariate models therefore run on n = 100 of 124 eligible discovery patients.
`03_cox_clinical.R` prints the accounting.
