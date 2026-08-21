# Task 6 — Benchmark DACOR against prior nuclear-feature aneuploidy methods

Answers **Reviewer 1, comment #6**: *"Previous studies utilized predefined nuclear morphological features for aneuploid prediction (33,34)" — does DACOR outperform such methods?*

Written to run on the HPC. Nothing here has been executed locally.

---

## What ref 34 actually is

Yu F, Wang X, Sali R, Li R. *Single-Cell Heterogeneity-Aware Transformer-Guided Multiple Instance Learning for Cancer Aneuploidy Prediction From Whole Slide Histopathology Images.* IEEE JBHI 2024;28(1):134–144.

It is **not** a predefined-nuclear-feature method. It is a hybrid of three feature blocks concatenated into a 588-D vector → one fully-connected layer → sigmoid:

| Block | Size | Content |
|---|---|---|
| **C** | 14-D | Cell-subtype ratios (K-means, 15 clusters; one dropped since ratios sum to 1) |
| **M** | 510-D | Handcrafted morphology: 34 features (12 shape + 10 colour + 12 texture) × 15 subtypes |
| **D** | 64-D | DenseNet-121 + one-layer transformer attention MIL over 200 sampled cells |

Their ablation, test-set AUC (LUAD / HNSC):

| Arm | LUAD | HNSC |
|---|---|---|
| M only | 0.689 | 0.747 |
| C+M | 0.693 | 0.789 |
| D only | 0.713 | 0.587 |
| **C+M+D (full)** | **0.818** | **0.827** |

Their task is arm-level SCNA in **invasive** LUAD and HNSC at 40×. Ours is flow-cytometric DNA content abnormality in a **precancerous** lesion. Cross-study AUC comparison is not meaningful — hence this same-cohort reimplementation.

### Points that were non-obvious on a first read

1. **There are two unrelated 64-D vectors.** The CAE 64-D (§III-B-4) is used *only* as the input space for K-means; its values never enter the classifier, only the resulting cluster ID does. The D-block 64-D is a separate DenseNet-121 + transformer output. Fig. 2 also mislabels the backbone "ResNet18" — the text and all main results use DenseNet-121 (ResNet-18 is an ablation, Tables VI–VII).
2. **Cluster assignments are used three times**: the C block (ratios), the *structure* of the M block (features pooled within each cluster), and the MIL sampler.
3. **How handcrafted features become a fixed-length slide vector**: hierarchical pooling, cell → subtype → tumour. Average the 34 features within each of the 15 clusters and concatenate → 510-D, regardless of how many cells the slide has.
4. **How the 200 cells are chosen**: randomly resampled *every epoch*, preserving the slide's subtype distribution — this doubles as augmentation.
5. **How those 200 embeddings are summarised**: not averaged — a one-layer transformer encoder with a `[CLS]` pseudo-cell token; the CLS output is the attention-weighted aggregate.

Consequence: reproducing C and M needs **no neural network at all** — only a per-nucleus feature vector to cluster on.

### Ref 33 is mischaracterised in our Methods

Fu et al. (Nat Cancer 2020) is tile-based deep learning, not predefined nuclear features. The sentence *"Previous studies utilized predefined nuclear morphological features for aneuploid prediction (33,34)"* needs correcting regardless of how this experiment turns out.

---

## The ladder

| Rung | Content | Yu's equivalent | Compute |
|---|---|---|---|
| 1 | Pooled nuclear morphology | M | CPU |
| 2 | Subtype ratios + per-subtype morphology | C+M | CPU |
| 3 `D` | DenseNet-121 + transformer MIL | D | GPU |
| 3 `CMD` | Rung 2 + deep features | C+M+D | GPU |
| — | DACOR (reference line) | — | — |

Rungs 1–2 answer the reviewer's question on their own and need no images. Rung 3 reproduces ref 34's full architecture and needs a crop per nucleus.

Rung 3 uses **ImageNet-pretrained DenseNet-121**, which is what Yu et al. used — faithful, off-the-shelf, and outside this paper's narrative. Their CAE is a small from-scratch conv autoencoder, not a foundation model. Deliberately *not* CellViT/Virchow: it would be less faithful to ref 34, and it appears later than this benchmark in the manuscript's chronology.

### Rung 3 specifics

**Crops are defined in microns, not pixels.** The two scanners differ, and we train on one cohort and test on the other, so a fixed pixel box would confound scanner with cohort. `CROP_UM = 25.0` matches Yu's 100 px at 40× (0.25 µm/px); crops are resampled to `CROP_PX = 100`, their input size.

**The µm/px factor is read from each TIFF's own resolution tag, never assumed from cohort.** This matters: *the manuscript Methods has the two scanner factors swapped.* Checked against nucleus coordinate extents — for MDA slides, 0.4548 places nuclei outside the image bounds while 0.5013 fits exactly; NU is the reverse. The `1.1.cell_dist_241106.Rmd` values were right and the Methods text is wrong. **This needs fixing in the manuscript independently of this task.**

**Images are located via the TSV's own `Image` column**, not a filename pattern — the cohorts differ (`D0050_ROI.tif` vs `D1021-2020-11-17_16.44.48_ROI.tif`).

**Crops are pooled, not exhaustive.** Caching all ~2.86M would be ~86 GB, so `POOL_PER_SLIDE = 1000` nuclei are cached per slide, stratified by subtype so rare subtypes survive (~25 GB). Each epoch then samples `CELLS_PER_BAG = 200` from the pool, reweighted to the slide's *true* subtype distribution — preserving Yu's distribution-preserving sampler and its per-epoch resampling, which doubles as augmentation.

**Rung 3 uses DACOR's literal train/val split** (`split` column: MDA train=340, val=82) for early stopping, so the deep model is fitted on exactly the slides DACOR was fitted on. Rungs 1–2 use patient-grouped CV across all 422 MDA slides instead, since they need no early-stopping set.

**No image augmentation**, because Yu et al. apply none. Their only augmentation is the per-epoch cell resampling, which they say so explicitly in their Discussion and which their own ablation supports (fixing the 200 cells costs 1.9%/2.1% AUC). That resampling is implemented here.

### Revisions after the first run

The first run gave rung 3 a near-chance test AUC (D 0.496, C+M+D 0.574) against strong in-cohort scores. Three changes followed.

**Evaluation sampling was broken.** Evaluation took the `CELLS_PER_BAG` *highest-weight* cells. Every cell in a subtype shares one weight, so bags filled from the largest subtypes first. On a realistic slide profile this yields **4 of 15 subtypes, with 99% of the bag from three of them** — total absolute deviation from the slide's true subtype distribution 1.13, against 0.13 for correct sampling. Training used proper weighted sampling, so the model was trained on diverse bags and scored on lopsided ones. Evaluation now uses the same distribution-preserving sampler, seeded per (slide, bag index) so draws are deterministic and identical across epochs, with each slide's score averaged over `EVAL_BAGS` draws. This matters most for the test cohort, because the stain shift moves NU nuclei into different subtypes and therefore changes which cells a biased bag would contain.

**Stain normalisation** (`STAIN_NORM_VARIANTS`). Each arm is trained twice — once as published, once with per-slide LAB mean/SD matched to a training-cohort target (Reinhard) — so both can be reported side by side. Motivation: hematoxylin mean is 0.352 in MDA and 0.623 in NU, and every stain-derived feature separates the cohorts at AUC 0.82–0.95 while nuclear area, which does not differ between scanners (AUC 0.53), is the one feature whose predictive value transfers. The LAB conversion is written out in `lib.py` rather than adding a scikit-image dependency; its round-trip is bit-exact.

**Checkpoint selection** (`SELECTION_WINDOW`). Each epoch is scored by a centred moving average of validation AUC and the middle epoch of the best window is kept. With 82 validation slides (~24 positive) a single epoch's AUC swings on luck — rung3_D scored 0.879 at epoch 3 and 0.506 at epoch 4 — so taking the single best epoch tends to save a lucky bounce. Only rung3_D was materially affected; rung3_CMD's curve was already smooth (SD 0.030).

Not changed: the seed stays 42 (fixed within the experiment is what matters), and rungs 1–2 get no cohort-wise standardisation — their stain dependence is a real property of the method and is noted as a limitation instead.

Hyperparameters follow the paper (§IV-C): Adam lr 5e-4, β (0.9, 0.999), ε 1e-8, weight decay 0.01, 50 epochs, batch 4 bags, gradient-norm clip 5.0, plateau LR ÷10 with patience 10; transformer with 4 heads, feedforward 128, dropout 0.1, ReLU, LayerNorm ε 1e-5; handcrafted block BatchNormed before concatenation.

### Revision 2 — rungs 1–2 were fitted on train+val pooled

Rungs 1–2 originally fitted on all 422 MDA slides (DACOR's train **and** val split pooled) and reported only a test-cohort number, with an in-cohort figure coming from cross-validation over that pooled 422. DACOR and rung 3 use a different, stricter protocol: fit on the 340-slide train split alone, with the 82-slide val split held out and never touched during fitting. The two were not comparable — rungs 1–2 had no genuine held-out val score, and their CV figure was optimistic relative to DACOR/rung 3 by construction (some of what they call "training" data, the val slides, DACOR never fits on at all).

Fixed: rungs 1–2 now fit on the training split only — standardisation, patient-grouped CV for hyperparameter selection, and the final refit all use only the 340 slides — then report train, val and test metrics under one protocol shared with DACOR and rung 3. `lib.fit_select_evaluate` takes `X_val`/`y_val` alongside train and test; `03_fit_evaluate.py` derives `train`/`val`/`test` from the `split` column rather than from `dataset == COHORT_TRAIN`. `05_rung3_deep.py` now saves predictions for all three splits (a `split` column), not test alone, so every model can go into one table. Predictions files from before this fix have no `split` column and are skipped by `03` with a printed warning rather than silently treated as compliant.

---

## Design decisions

**Every DACOR-derived quantity is excluded.** DACOR is the model being benchmarked against, so `attention_score` and the `attn0`/`attn1` region strata never touch the baseline — see `NON_FEATURE_COLS` in `config.py` and the explicit guard in `01`. This is also why the published slide-level `nuc_*` columns are not reused: they are stratified by DACOR attention. Everything is recomputed from the per-nucleus exports.

**The Barrett-epithelium filter uses two columns.** `Class == "epithelial"` **and** `Parent == "epithelium_area"`. They disagree often — in D0050, 333 epithelial-classified nuclei sit in stroma and 190 non-epithelial nuclei sit inside epithelium — so filtering on either alone is wrong. Yields ~2.86M nuclei across 781 slides, median 2,871 per slide; only 6 slides fall below 200.

**The data partition and fitting protocol are DACOR's own.** Train = MDA's 340-slide training split, val = MDA's 82-slide validation split (scored, never fitted on), test = NU (355 slides, DACOR's test set). Every model — DACOR, rungs 1–2, rung 3 — fits on train alone; any AUC difference is therefore attributable to the method, not the split or a difference in how much data each model saw. Hyperparameters for rungs 1–2 are chosen by 5-fold CV grouped by patient, entirely within the training split, so no patient spans folds and val is never used for model selection either. (See "Revision 2" below — this was fixed after an earlier version pooled train+val for rungs 1–2.)

**The classifier head matches Yu et al.** — a single fully-connected layer, i.e. logistic regression. Elastic-net regularisation is added because our *n* is small relative to *p*; the L1/L2 mix is tuned rather than assumed.

**Clustering space is a stated deviation.** Yu clustered in a CAE latent space learned from nucleus images. Here `CLUSTER_SPACE = "raw"` clusters in the standardised handcrafted feature space; `"pca"` is a sensitivity check. The raw variant defines clusters in the same space later pooled per cluster — a mild redundancy that should be stated in the Methods, with the PCA run reported if it changes the answer.

**All slides are kept in the main comparison** so DACOR and the baseline are scored on an identical test set. `MIN_NUCLEI_SENSITIVITY` is for a secondary analysis only.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, the epithelium filter, modelling constants |
| `lib.py` | Name canonicalisation, slide summarisation, fitting, DeLong |
| `00_export_labels.R` | Labels + DACOR probabilities → `interim/slide_labels.csv` |
| `01_build_nucleus_table.py` | 781 TSVs → filtered per-nucleus parquet |
| `02_build_features.py` | Slide-level feature matrices for rungs 1 and 2 |
| `03_fit_evaluate.py` | Fit, evaluate, compare to DACOR, ROC figure |
| `04_extract_crops.py` | Nucleus crop cache for rung 3 (CPU, parallel) |
| `05_rung3_deep.py` | Rung 3 training and evaluation (GPU) |
| `HPC_RUN_BRIEF.md` | Operational brief for the session that runs this on the cluster |

## Running

Only `BEACON_BASE` should need changing; the HPC layout replicates the local one.

```bash
export BEACON_BASE=/path/to/root      # dir containing BE_master/ and BEACON/
export BEACON_N_JOBS=32
Rscript 00_export_labels.R
python 01_build_nucleus_table.py
python 02_build_features.py
python 03_fit_evaluate.py             # rungs 1-2: already answers the reviewer

python 04_extract_crops.py            # rung 3, CPU-heavy, ~25 GB of crops
python 05_rung3_deep.py               # rung 3, GPU, hours
python 03_fit_evaluate.py             # re-run to fold rung 3 into table + ROC
```

Steps 0–3 are CPU-only and self-contained. `01` and `04` are the heavy steps and parallelise internally over `BEACON_N_JOBS` — they should not be split into array jobs. Everything is strictly sequential.

Requires `pandas`, `pyarrow`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`; R needs `dplyr` (`pROC` optional). Rung 3 adds `tifffile`, `pillow`, **`imagecodecs`** (the pyramid TIFFs are JPEG-compressed and will not decode without it — `04` checks this up front), plus `torch` and `torchvision` with CUDA.

`00_export_labels.R` prints DACOR's test-cohort AUC — it should read **0.825**, matching the manuscript. If it does not, stop: the comparator is wrong.

## Outputs

Written under `BE_master/revision/task6_benchmark/`:

- `results/benchmark_all_splits_slide_level.csv` — **primary output.** One row per (model, split), split ∈ {train, val, test}, for every model with compliant predictions: AUC with DeLong CIs, accuracy, F1, kappa, and a paired DeLong test against DACOR computed independently per split. Model-level fitting details (feature count, CV AUC, selected hyperparameters) are attached to the train row only.
- `results/benchmark_all_splits_patient_level.csv` — same, patients aggregated by max within each split.
- `results/benchmark_slide_level.csv`, `benchmark_patient_level.csv` — test-split-only views in the original wide format, kept for filenames already cited elsewhere (response letter, `REVISION_MASTER.md`); derived from the files above, not computed separately.
- `results/predictions_*.csv` — now carry all three splits (a `split` column), one file per model.
- `results/coefficients_*.csv`, `cv_grid_*.csv`, `cluster_sizes_*.csv`
- `figures/roc_benchmark.{pdf,png}` — test-cohort ROC sanity check, not the response-letter figure (that is a separate, not-yet-started task)

## Interpreting the outcome

If the baseline loses clearly, the reviewer's question is answered directly and the Discussion's *"significant performance improvement through automated feature learning"* is substantiated.

If it wins or comes close — plausible, since Fig. 3C already shows these features separate DNA-content status — the numbers stand and the argument changes: DACOR localises abnormality spatially (which the entire ecology analysis depends on), needs no per-slide feature engineering, and produces the heatmaps. In that case the "performance improvement" sentence should be softened rather than defended.
