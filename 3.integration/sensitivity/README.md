# Parameter sensitivity for spatial ecology metrics

Revision task 3 — **Reviewer 1, comment #2**:

> "No parameter specifications are detailed for Ripley's L, Moran's I, k nearest neighbors etc.
> What happens if you vary the parameters, are the associations with progression risk maintained?"

All parameter values below were **read off the authoritative pipeline** in
`3.integration/ecology_code/` (2026-07-31). Nothing here is assumed.

---

## Framing

The comment has two halves, answered by two separate deliverables that are **never mixed**:

**(a) "No parameter specifications are detailed"** → **Supp. Table, outcome-blind.**
Parameter, value, `basis` (structural / convention / judgement), and one *descriptive* number.
**No inferential statistics.** Built by `08_justification.R`.

**(b) "What happens if you vary the parameters"** → **Supp. Figure, robustness.**
Effect size per grid value. Built by `06_associations.R` → `07_figure.R`.

**A sweep cannot prove a parameter "correct."** No statistical, biological or clinical criterion
exists for that, and plasma-cell clustering was a *finding*, not a pre-specified endpoint — so
choosing a parameter by how well it separates progressors would be doubly circular and would hand
Reviewer 1 the objection from comment #12. The supportable claim is narrower and is all that was
asked: **the choice was principled a priori, and the conclusions do not hinge on it.**

Methods line: *"parameters were fixed a priori on structural and conventional grounds and were not
tuned against outcome."*

---

## The parameters, and how arbitrary each really is

| Tier | Parameter | Value | Basis |
|---|---|---|---|
| Forced by architecture | tile stride | 112 px (~51–56 µm) | 224 px FM window at 50% overlap |
| Forced by architecture | Morisita quadrat | 224 px (~102–112 µm) | one non-overlapping FM window |
| Lattice convention | Moran/Getis `d2` | 200 px | **all touching neighbours (queen contiguity)** — 8 neighbours |
| Software convention | Ripley `r` | spatstat per-slide default | standard quarter-window rule |
| Statistical convention | Gi\*/Moran summaries | sign-based | no threshold exists to justify |
| Judgement | G-function window | 150 µm | author-chosen; `short_range = 50` sits declared-but-unused |
| **Genuinely arbitrary** | kNN `k` | 10 | round number; 10-fold CV is a real convention but does not transfer to a nearest-neighbour distance |

Only **one** parameter is properly arbitrary. State `d2` as *"all touching neighbours"* rather than
"200 px" — it is unit-free, so it also sidesteps the unresolved µm/px question entirely.

### Grids (3 values each, published in bold)

| Metric | Grid | Why these endpoints |
|---|---|---|
| Moran / Getis `d2` | **200 px** (1 ring, 8 nb) / 350 px (2 rings) / 500 px (3 rings) | complete enumeration of lattice rings; intermediate distances give *identical* neighbour sets |
| Ripley `r` | **per-slide default** / fixed 100 µm / fixed 200 µm | 100 µm is lattice-coherent with the queen ring |
| kNN `k` | 5 / **10** / 20 | brackets the published value |
| G-function window | 50 µm / **150 µm** / 300 µm | 50 µm is the author-declared `short_range` |
| Morisita quadrat | 112 px / **224 px** / 448 px | one stride / one FM tile / two FM tiles |

---

## The load-bearing property: reproduction

At the published setting this code must reproduce the published values, or the sweep describes a
different pipeline. Verified:

| Metric | Status |
|---|---|
| **Ripley's L** | ✅ exact — cor = 1.000, all 9 cell pairs, n = 776 |
| **Global Moran (anchor)** | ✅ exact — `GlobalMoran_Plasma_attn1`, **272/272** slides where the published column is non-zero |

### Three traps that reproduction depends on

1. **`trapz` has no `na.rm`.** One NA propagates to NA, exactly as published; the aggregate
   summaries use `na.rm = TRUE`. Filtering non-finite points *before* integrating drops Ripley
   reproduction from 1.000 to ~0.96.
2. **`attn_class_extended` is a FACTOR with levels `"0","1"`.** `as.numeric()` on it returns level
   *indices* (1, 2), silently **inverting the stratum**. Always coerce via `as.character()` —
   `as_attn_numeric()` in `lib_sweep.R` does this.
3. **`localG(..., GeoDa = TRUE)`** and the pixel coordinate system. `d2 = 200` is 200 **pixels**.

### The pipeline mixes units

Tile-level metrics (Moran, Getis, Morisita) use **pixel** coordinates; cell-level metrics
(Ripley, kNN, G-function) use **micron** coordinates. `d2 = 200` and `r <= 150` are *not* the same
scale. This must be stated explicitly in the Supp. Table.

---

## Analysis conventions (match `4.plotting/code/3.ecology.Rmd`)

- **Slide level**, restricted to DACOR-abnormal slides, as Fig. 4C/4D.
- **Zeros dropped** before testing. This matters enormously — see below.
- **Wilcoxon.** Shapiro-Wilk is ≤1e-5 in every group tested, so the published normality branch
  always selects it.
- **Effect = rank-biserial r = 2·AUC − 1.** Bounded, sign-interpretable, comparable across metrics —
  which is what lets the dot plot put every metric on one axis.

### Anchor claim

Fig. 4C bottom = **`GlobalMoran_Plasma_attn1`** (author-identified). Slide level:

| Treatment | n | Wilcoxon p |
|---|---|---|
| all values | 174 / 87 | 0.0234 |
| **zeros dropped** (as published) | 85 / 34 | **0.0075** |
| published | — | 0.003 |

Residual gap is most likely the concurrent-malignancy exclusion (22 test + 2 discovery patients),
not applied here.

### ⚠ 65% of the anchor feature is imputed zeros

`3.0.merge_normalize.Rmd` imputes non-computable values to 0. **505 of 777 slides** have
`GlobalMoran_Plasma_attn1 == 0`, yet this code computes a real value for all of them.
`3.ecology.Rmd` correctly drops them for the figure — **the LASSO feature table keeps them as real
measurements.** So Fig. 4C and the Fig. 5 model are fitted on different effective samples.
This is a task-1 issue, recorded here because it was found here.

---

## Files

| File | Purpose | Cost |
|---|---|---|
| `00_config.R` | paths, grids, confirmed conventions | — |
| `lib_sweep.R` | spdep/spatstat wrappers, summarisers, sharding | — |
| `00_claim_list.R` | derives which metrics the robustness figure covers | minutes |
| `01_ripley_sweep.R` | r-range sweep by re-summarising **cached** curves | free |
| `02_moran_getis_sweep.R` | ring sweep (local + global) | ~1 h, 13 workers |
| `03_knn_sweep.R` | k sweep | array, hours |
| `04_gfunction_sweep.R` | window sweep; **caches Gcross curves** | array, longest |
| `05_morisita_sweep.R` | quadrat sweep | minutes |
| `06_associations.R` | effect size per setting (± zero-dropping) | minutes |
| `07_figure.R` | the dot plot + companion table | minutes |
| `08_justification.R` | **outcome-blind** numbers for the Supp. Table | minutes |
| `99_validate_published.R` | reproduction check — run first | minutes |

---

## Running

Everything except `04` runs comfortably on a laptop (R 4.5.2 + `spdep`, `spatstat`, `FNN`, `furrr`,
`abdiv`). For the cluster, the container is built from `3.integration/code/Dockerfile`
(`rocker/geospatial:4.3.3`, which supplies `sf`/`spdep`/`spatstat`).

```bash
export BEACON_DATA=/path/to/spatial_analysis/tabular
export BEACON_CLIN=/path/to/0.input/organised_wsi_patient

Rscript 01_ripley_sweep.R        # free; validates the harness
Rscript 08_justification.R       # Supp. Table numbers
Rscript 02_moran_getis_sweep.R
Rscript 05_morisita_sweep.R
Rscript 03_knn_sweep.R
Rscript 04_gfunction_sweep.R     # heaviest — cluster
Rscript 06_associations.R
Rscript 07_figure.R
```

On SLURM use `./submit_sweep.sh <stage>` with `BEACON_IMAGE` set to the Singularity `.sif`.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `BEACON_DATA` | — | root of `spatial_analysis/tabular` |
| `BEACON_CLIN` | — | root of `0.input/organised_wsi_patient` |
| `BEACON_ATTN_COL` | `attn_class_extended` | stratification column |
| `BEACON_UMPX` | `code` | `code` = NU 0.4548 / MDA 0.5013; `manuscript` = the reverse |

---

## Reading the output

`06_associations.R` writes `association_stability.csv`. The columns that answer the reviewer are
**`sign_consistent`** and **`effect_min` / `effect_max`** — not `frac_sig`.

**Report stability, not significance.** A metric holding its sign and rough magnitude across the
grid is robust even where p drifts above 0.05 at an endpoint, and the figure legend commits to that
reading in advance.

---

## Known discrepancies (recorded, deliberately not acted on)

- **µm/pixel factors appear swapped** — `1.1.cell_dist_241106.Rmd:141-142` assigns NU = 0.4548 /
  MDA = 0.5013; the Methods state the reverse. **Deferred by the author.** Lower stakes now that
  `d2` is stated as queen contiguity, but it still sets the reported µm for Ripley/kNN/G-function.
- **Morisita `_attn0` never existed** — a malformed `left_join` at `2.6.morisita.Rmd:105` dropped the
  DNA-normal stratum. The Supp. Table must not claim three strata. This code computes all three.
- **Ripley integrates over a slide-dependent domain** — `Kcross` called without `r`; observed rmax
  spans 2.8–450 µm, so `_AUC` and `_mean` are not strictly comparable across slides. Deferred;
  `08_justification.R` quantifies it.
- **`summarise_g` arguments are inert** — `short_range`/`medium_range` declared, 150 hardcoded.
- **Published `peak_r` is censored at 150 µm** by construction, so it cannot justify the window.
  `04` records an uncensored peak and a saturation radius instead.
- **Script/data mismatch** — `2.5.moran_241224.rmd:292` stratifies the *global* Moran on raw
  `attention_score`, but the delivered values match `attn_class_extended` exactly. The later block
  (line 362+) is presumably what ran.
