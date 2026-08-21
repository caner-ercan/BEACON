# HPC run brief — Task 6 benchmark

**Paste this file to the SSH session as its starting prompt.**

> ## RERUN 2 (supersedes RERUN 1 below) — read this first
>
> Steps 1–2 and 4 have already been run and their outputs are on disk and unchanged.
> **Three commands need running, in this order:**
>
> ```
> python 03_fit_evaluate.py   # CPU, minutes — fixes rungs 1-2 (see why, below)
> python 05_rung3_deep.py     # GPU, ~2 hours — unrelated fix from RERUN 1, still pending
> python 03_fit_evaluate.py   # CPU, minutes — re-run to fold in rung 3's new predictions
> ```
>
> Do **not** rerun steps 1, 2 or 4 — the nucleus table, feature matrices and crop
> cache are unaffected by this fix and regenerating the crops would waste hours.
>
> **Why step 3 changed:** rungs 1–2 previously fit on all 422 MDA slides (DACOR's
> train **and** val split pooled), which both leaked validation data into
> hyperparameter selection and made a val-split score impossible to report. They
> now fit on the 340-slide training split alone — matching DACOR and rung 3 — and
> report train/val/test together. The first `03` call will print `SKIPPING
> rung3_*` for the predictions files currently on disk (from before RERUN 1's
> fix); that is expected, not an error — those files have void numbers and no
> `split` column. Running `05` produces compliant ones, and the second `03` call
> picks them up automatically.
>
> **Expect the first `03` run's rung1/rung2 test-cohort AUCs to differ from any
> number you've seen quoted before** — the old pooled-fitting numbers are
> superseded, not a rerun of the same thing.
>
> **Primary output to bring back:** `results/benchmark_all_splits_slide_level.csv`
> and `..._patient_level.csv` — one row per (model, split) for every model, not
> just test. Everything under "Checks to perform" below still applies; add:
> confirm every model in the output has all three of train/val/test rows.
>
> <details><summary>RERUN 1 (2026-08-03) — original text, for reference</summary>
>
> Steps 1–4 have already been run and their outputs are on disk. Only two steps
> needed rerunning: `05_rung3_deep.py` (GPU) then `03_fit_evaluate.py` (CPU, folds
> rung 3 into the combined table + ROC). `05` now trains four models instead of
> two (each arm with and without stain normalisation), roughly 2× the previous
> wall time (previously 39 minutes, so budget ~2 hours). On its first execution
> `05` makes one pass over the crop cache to compute per-slide colour statistics,
> writing `interim/slide_lab_stats.csv` (cached for subsequent runs). This step is
> still pending — RERUN 2 above folds it into the same session.
>
> </details>

Your job is to write SLURM submission scripts and run this pipeline. You do not
need to change the analysis code, interpret the results, or edit the manuscript.
The scientific decisions are already made; the output files are brought back to a
local session for interpretation.

---

## What this experiment is

A manuscript reviewer asked whether our deep-learning model (DACOR, which predicts
DNA content abnormality from H&E images) actually outperforms older published
methods that use predefined nuclear morphology features.

To answer it we rebuild those older methods on our own data, on the same slides
and the same train/test split as DACOR, and compare AUCs. Four models are trained,
increasing in complexity:

| Rung | Features | Compute |
|---|---|---|
| 1 | Pooled nuclear morphology | CPU |
| 2 | Nuclei clustered into 15 subtypes → subtype ratios + per-subtype morphology | CPU |
| 3 `D` | DenseNet-121 + transformer attention MIL over nucleus image crops | **GPU** |
| 3 `CMD` | Rung 2 features + rung 3 deep features (full published architecture) | **GPU** |

DACOR's own scores are read from an existing file and used only as the comparison
line — no DACOR retraining, no DACOR inference.

## Scripts, in run order

Located in `BEACON/1.aneuploid/rev1_task6_benchmark/`.

| Step | Script | Runtime | Resources |
|---|---|---|---|
| 1 | `01_build_nucleus_table.py` | heavy, parallel | many CPUs, ~32 GB |
| 2 | `02_build_features.py` | ~10–30 min | 4–8 CPUs, ~64 GB |
| 3 | `03_fit_evaluate.py` | ~5–15 min | 4 CPUs, ~16 GB |
| 4 | `04_extract_crops.py` | heavy, parallel, IO-bound | many CPUs, ~32 GB, **~25 GB disk** |
| 5 | `05_rung3_deep.py` | hours | **1 GPU**, 4–8 CPUs, ~32 GB |
| 6 | `03_fit_evaluate.py` again | ~5–15 min | 4 CPUs, ~16 GB |

Steps 1–3 are self-contained and answer the reviewer's question on their own.
Steps 4–6 add the GPU rung. **Run 0–3 first and confirm they succeed before
submitting 4–5** — they are much cheaper and will catch any path problem.

Step 6 is the same script as step 3, re-run after rung 3 exists so the final table
and ROC figure include all four models.

### Dependency between steps

Strictly sequential — each step reads the previous step's output. Do not run them
as a parallel array. Use SLURM job dependencies (`--dependency=afterok:<jobid>`)
or submit them one at a time.

`01` and `04` parallelise internally across `BEACON_N_JOBS` within a single node.
Do not split them into array jobs.

## Environment

- **Python** (steps 1–5): `conda` environment `beacon`
  Needs `pandas`, `pyarrow`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.
  Step 4 additionally needs `tifffile`, `pillow`, **`imagecodecs`** (the images are
  JPEG-compressed pyramid TIFFs and will not decode without it).
  Step 5 additionally needs `torch`, `torchvision` with CUDA.

Step 5 downloads ImageNet DenseNet-121 weights on first use. If compute nodes have
no internet, pre-download on a login node (`torchvision.models.densenet121(weights="IMAGENET1K_V1")`)
so the cache in `~/.cache/torch` is populated, or set `TORCH_HOME`.

## Configuration

Everything is driven by environment variables. Set these in every job script:

```bash
export BE_MASTER="/work/FAC/FBM/DBC/mrapsoma/prometex/cercan/projects/BE/aneuploid/mda/out/BE_Master"   # dir containing BE_Master/ and BEACON/
export BEACON_N_JOBS=<N_CPUS>                  # match --cpus-per-task
export BEACON_DEVICE=cuda                      # step 5 only
```

`BE_MASTER` is the only path that should need setting — the server layout
replicates the local one. Everything else is derived in `config.py`. Do not
hardcode paths inside the analysis scripts.

Run scripts from their own directory (they import `config.py` and `lib.py` as
sibling modules), or set `PYTHONPATH` to that directory.

## Inputs (read-only, must exist)

`BE_Master` is located here: `/work/FAC/FBM/DBC/mrapsoma/prometex/cercan/projects/BE/aneuploid/mda/out/`
Relative to that:

- `BE_Master/2.intermediate/2.1.nucleus/2_inference_quantification/inference_project/nuc_feature_export/` — 781 TSVs, ~3.1 GB
- `BE_Master/0.input/organised_wsi_patient/MIL_samples_rev1_260730.csv`
- `BE_Master/3.integration/integration_project/spatial_analysis/tabular/merged/cell_level_250808.RDS`
- `BE_Master/0.input/1.qp_wsi_projects/{mda,nu}/ROI_pyramid/` — pyramid TIFFs (step 4 only)

## Outputs (all written under `BE_Master/revision/task6_benchmark/`)

- `interim/` — intermediate parquet files and the crop cache (large; keep on scratch if needed)
- `results/` — CSV tables, model coefficients, predictions
- `figures/` — ROC figure

**Bring back `results/` and `figures/` only.** `interim/` is large and
regenerable — it does not need to be transferred.

## Checks to perform

On RERUN 2, checks 1–2 don't apply (steps 1 and 4 are not rerun); 3, 4, 5, 6 do.

1. **After step 1** — should report ~2.86M nuclei across ~781 slides. A much
   smaller number means the filter or input path is wrong.
2. **After step 4** — the printed micron-per-pixel values should be only
   **0.5013** and **0.4548**. Any other value means a slide's resolution metadata
   is unexpected; report it rather than ignoring it.
3. **Step 5** prints a stain-target line, then per arm an in-cohort (validation)
   AUC and a cross-cohort (test) AUC. Report both numbers for all four arms. A
   large gap between them is an expected finding here, not a failure — do not
   tune anything in response to it.
4. **Step 5** must produce four rows in `results/benchmark_rung3.csv`. Fewer
   means an arm crashed; report which and include its traceback.
5. **Every `03` run prints a "DACOR reference" block first**, with train/val/test
   AUC. The test row must read **0.825** (95% CI band around it is fine, the
   point estimate should not move). If it does not, stop — the comparator is
   wrong and nothing downstream should be trusted.
6. **The final `results/benchmark_all_splits_slide_level.csv` must have exactly
   three rows (train/val/test) for DACOR, rung1_M, and rung2_CM**, and three
   rows each for whichever rung-3 arms are present. Fewer than three rows for
   any model, or a model silently missing, means something upstream did not
   produce a compliant `split`-column predictions file — report which model and
   check that model's own log output from `03` (it prints per-model feature
   counts and split sizes before fitting).

## What to do if something fails

Report the failure with the log. Do not modify the analysis code, change
hyperparameters, alter the train/test split, or drop slides to make a step pass.
Fixing paths, module loads, resource requests and SLURM directives is expected and
fine. Changing anything inside `lib.py`, `config.py`'s modelling constants, or the
model definitions is not.

One exception: if a small number of slides fail in step 4 with a read error, that
is handled — the scripts log them and continue. Report how many failed.

## Deliverable

A short summary of: which steps ran, wall time and resources used, the four
verification checks above, any slides that failed, and the contents of
`results/benchmark_slide_level.csv`. Then the `results/` and `figures/`
directories transferred back.
