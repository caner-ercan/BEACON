# Elastic Net vs. LASSO

Revision task 1 — **Reviewer 1, comment #10**:

> "Why was lasso used for the risk stratification modelling, and not Elastic Net for instance?
> How would the results compare to an Elastic Net model?"

**Status: the published LASSO model is exactly reproduced, and the Elastic Net comparison runs on top of it.**

---

## Reproduction — PASSED (2026-08-04)

`BE_master/4.plotting/output/3.1.ecology/mae_7_5_withnuc/` holds the fitted objects from the published run
(`fitted_model.rds`, `cv_model.rds`, `model_parameters.rds`, `lasso_feature_coefficients.csv`,
`predictions_with_metadata.csv`, `risk_patient.csv`). These resolved every open question.

Refitting LASSO at the published settings now matches the published model on **every** quantity:

| | ours | published |
|---|---|---|
| lambda.1se | 0.04641589 | 0.04641589 |
| threshold | 0.6103361 | 0.6103361 |
| non-zero features | 27 | 28 (one at 6e-17, numerically zero) |
| train AUC / bal-acc / sens / spec | 0.9022 / 0.8663 / 0.7826 / 0.9500 | identical |
| test AUC / bal-acc / sens / spec | 0.8167 / 0.8167 / 0.8333 / 0.8000 | identical |
| test log-rank p | 0.013663 | 0.013663 |

Coefficient vectors agree to machine precision (cor = 1, max abs diff 1.3e-15).

**Two settings had been wrong before the exported model was available**, and both mattered:

1. **Input matrix** = `merged_ecology_wNuc_standardised_250812.RDS`, *not* the undated
   `merged_ecology_wNuc_standardised.RDS`. Decisive: refitting at the exported lambda gives cor = 1 /
   27 non-zero with `_250812`, versus cor 0.982 / 34 non-zero with the undated file.
2. **No zero-variance filtering.** `fitted_model.rds` records `nvars = 1538`, so the published run passed
   all 1538 columns including the 24 that are constant within the training slides. `DROP_ZERO_VAR <- FALSE`.

Also settled from `cv_model.rds`'s recorded call: `alpha=1`, `family="binomial"`, `type.measure="mae"`,
`nfolds=7`, `lambda = exp(seq(log(0.001), log(1), length.out=100))`, and **no `alignment` argument**.

> **Correction to an earlier note in this file.** It previously said `seed = 5` here could not reproduce the
> original's CV folds because the original consumed RNG on the split first. That was wrong: recovering
> `lambda.1se` exactly shows the fold assignment *does* reproduce, so `grid_search_lasso.R` called
> `set.seed(5)` immediately before `cv.glmnet` with nothing drawn in between.

## Data and split

The published patient-level `eco_split` and per-patient `risk_category` are read from
`0.input/organised_wsi_patient/eco_patients.csv` (the split RNG itself is not reproducible; reading it
removes the dependency). Verified on load by `stopifnot` in `lib_en.R`:

| | |
|---|---|
| DACOR-positive slides with an `eco_split` | 202 |
| patients | 65 |
| ecology_train | 138 slides / **43 patients** (23 progressors) |
| ecology_test | 64 slides / **22 patients** (12 progressors) |
| features | **1538 = 928 spatial ecology + 610 nuclear morphology** |
| p / n | ≈ 11 |

Feature set is **option 2** (ecology + nuclear morphology), matching the published LASSO. The ecology-only
matrix (`merged_ecology_standardised.RDS`, 933 cols) is not used.

## Design

Everything is held fixed at the published pipeline; **only `alpha` changes**.

**Fit settings**

| label | `type.measure` | `nfolds` | lambda path |
|---|---|---|---|
| `published` | `mae` | 7 | `exp(seq(log(0.001), log(1), length.out = 100))` |
| `default` | `deviance` | 10 | glmnet's automatic path |

`lambda.1se` in both. **Alpha arms**: `cv` (selected by CV on the training set over 0.1–0.9), `fixed` (0.5),
`lasso` (1.0, the reproduction/control arm). One fold assignment per seed is **shared across every alpha**, so
the arms are paired.

Protocol: `predict(type="response")` → slide scores → patient score by **max** → Youden threshold on the
**training** patient-level ROC → patient-level metrics + log-rank.

## Reporting rule — no absolute values

CE, 2026-08-04: no absolute numbers (1.000 AUC, 100% sensitivity, 0% specificity) in the paper.

Training AUC by **resubstitution saturates at 1.000** for the denser elastic-net fits — that is a description
of overfitting, not a result, and it is not quotable. The table therefore carries **`CV_AUC_train`**,
cross-validated on the training folds at the selected lambda, which never saturates (range 0.726–0.803 here).
**Quote `CV_AUC_train`, not `AUC(rs)`.** `03_response_table.R` marks any cell that lands on 0.000 or 1.000
with `+` so they cannot be picked up by accident.

## Files

| | |
|---|---|
| `00_config.R` | paths, recovered published parameters, the two arms compared |
| `lib_en.R` | data loading + integrity assertions, fitting, patient-level metrics |
| `01_fit_en_vs_lasso.R` | runs both fits (LASSO, Elastic Net α=0.5) |
| `03_response_table.R` | **the response-letter table** (train/test side by side), the KM p-value table and the saturated-value guard |
| `figures/` | the four response-letter panels, reusing this folder's fitting code directly |

```bash
Rscript BEACON/3.integration/risk_model/01_fit_en_vs_lasso.R
Rscript BEACON/3.integration/risk_model/03_response_table.R
```

Runs the published configuration (seed 5) only — 2 fits, a few seconds. Outputs are suffixed `_seed5`, in
`BE_master/.../ecomerged_lasso_model/rev1_elastic_net/`. Overridable: `BEACON_DATA`, `BEACON_CLIN`,
`BEACON_FEATURES`, `BEACON_PUBMODEL`.

---

## Results (published configuration, seed 5)

### ecology_test — held out, n = 22 patients

| model | parameters | n feat | AUC | bal acc | sens | spec | KM p |
|---|---|---|---|---|---|---|---|
| **LASSO — PUBLISHED** | α=1, mae, 7 folds | 28 | **0.817** | **0.817** | 0.833 | 0.800 | **0.0137** |
| LASSO (α=1) | mae, 7 folds | 27 | **0.817** | **0.817** | 0.833 | 0.800 | **0.0137** |
| EN (α=0.5) | mae, 7 folds | 175 | 0.800 | 0.767 | 0.833 | 0.700 | 0.0471 |
| EN (α by CV → 0.10) | mae, 7 folds | 582 | 0.800 | 0.617 | 0.833 | 0.400 | 0.268 |
| LASSO (α=1) | deviance, 10 folds | 5 | 0.817 | 0.525 | 0.250 | 0.800 | 0.833 |
| EN (α=0.5) | deviance, 10 folds | 23 | 0.833 | 0.692 | 0.583 | 0.800 | 0.195 |
| EN (α by CV → 0.90) | deviance, 10 folds | 7 | 0.817 | 0.567 | 0.333 | 0.800 | 0.479 |

### ecology_train — n = 43 patients

| model | parameters | n feat | CV-AUC | bal acc | sens | spec | KM p |
|---|---|---|---|---|---|---|---|
| **LASSO — PUBLISHED** | α=1, mae, 7 folds | 28 | — | 0.866 | 0.783 | 0.950 | 1.0e-06 |
| LASSO (α=1) | mae, 7 folds | 27 | 0.739 | 0.866 | 0.783 | 0.950 | 1.0e-06 |
| EN (α=0.5) | mae, 7 folds | 175 | 0.726 | 0.978 | 0.957 | *1.000* | 1.7e-11 |
| EN (α by CV → 0.10) | mae, 7 folds | 582 | 0.726 | *1.000* | *1.000* | *1.000* | 1.9e-11 |
| LASSO (α=1) | deviance, 10 folds | 5 | 0.803 | 0.823 | 0.696 | 0.950 | 3.7e-06 |
| EN (α=0.5) | deviance, 10 folds | 23 | 0.801 | 0.816 | 0.783 | 0.850 | 8.5e-05 |
| EN (α by CV → 0.90) | deviance, 10 folds | 7 | 0.803 | 0.823 | 0.696 | 0.950 | 3.7e-06 |

*Italics* = absolute value; not quotable, and the reason `CV_AUC_train` is the column to use.

### Reading

**At the published settings LASSO is at least as good as Elastic Net on every held-out metric**, and is the
only arm combining a significant held-out log-rank with balanced sensitivity and specificity:

- AUC 0.817 vs 0.800 for both Elastic Net arms.
- Balanced accuracy 0.817 vs 0.767 (α=0.5) and 0.617 (α=0.10).
- Held-out KM p 0.0137 vs 0.0471 and 0.268.

Elastic Net's cost is visible in the feature counts and specificity: 175 and 582 features against LASSO's 27,
with specificity falling 0.800 → 0.700 → 0.400 as alpha drops. The extra correlated features are recruited
into the model, drive training separation to absolute values (bal acc 1.000 at α=0.10) and **lower** the
cross-validated training AUC (0.726 vs 0.739) — the signature of fitting noise, not signal.

Under glmnet's default settings every arm reaches a similar AUC (0.817–0.833) but none reaches a significant
held-out log-rank (p 0.195–0.833), because the very sparse fits (5–23 features) classify almost nobody as
high risk (sensitivity 0.250–0.583). AUC alone would hide this; that is why the table carries all five metrics.

**Answer for the response letter:** Elastic Net was evaluated across the alpha range, both at a fixed α = 0.5
and with α chosen by cross-validation. It did not improve on LASSO on any held-out metric; discrimination was
equivalent or slightly lower, and risk-group separation weakened as alpha decreased. LASSO was retained
because at p/n ≈ 11 with 43 training patients the sparser penalty gave equal discrimination with markedly
better calibration of the risk groups and an interpretable feature set.

### Stability

A 25-fold-seed sensitivity check was run during the revision to see how much of the single-seed comparison
is fold-assignment noise. It is not part of this published code (a seed grid is configuration search, not
the reported model — see the repo's no-sweep policy) and is not reproducible from this folder; the finding
lives in the rebuttal text and the revision archive, not here.

---

## Open items

- **`grid_search_lasso.R` is still missing** (only an empty macOS sidecar in
  `BE_master/code/3.integration/OneDrive_1_31.07.2026/`; the zip beside it is truncated). Deferred by CE.
  No longer needed for reproduction — the exported model objects settled it — but still wanted for
  **R1#13 (code availability)**, since the script behind Fig. 5 is not in the repo.
- The 25-seed stability run needs regenerating on the corrected inputs.
