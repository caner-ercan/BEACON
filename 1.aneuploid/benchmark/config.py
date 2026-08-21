"""
Task 6 -- benchmark DACOR against prior predefined-nuclear-feature methods.
Central configuration: paths, the epithelium filter, and modelling constants.

Set the BE_MASTER environment variable to your BE_master checkout.
"""

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

if "BE_MASTER" not in os.environ:
    sys.exit("Set the BE_MASTER environment variable to your BE_master checkout.")
BE_MASTER = Path(os.environ["BE_MASTER"])

# Per-nucleus QuPath exports: 781 TSVs, one per slide, 70 columns each.
NUC_EXPORT_DIR = (
    BE_MASTER
    / "2.intermediate/2.1.nucleus/2_inference_quantification/inference_project/nuc_feature_export"
)

# Slide-level clinical table (patient, wsi, dataset, split, flow).
CLINICAL_CSV = BE_MASTER / "0.input/organised_wsi_patient/MIL_samples_rev1_260730.csv"

# DACOR slide-level probabilities live inside this RDS; 00_export_labels.R
# pulls them out together with the clinical columns.
CELL_LEVEL_RDS = (
    BE_MASTER
    / "3.integration/integration_project/spatial_analysis/tabular/merged/cell_level_250808.RDS"
)

# ROI pyramid images (rung 3 only). Per-cohort folders; filenames differ between
# cohorts (MDA 'D0050_ROI.tif', NU 'D1021-2020-11-17_16.44.48_ROI.tif'), so the
# image is located via the TSV's own 'Image' column rather than a name pattern.
def _roi_dir(cohort: str):
    """Locate a cohort's ROI pyramid folder.

    The folder is named 'ROI_pyramid' locally but 'ROIpyramid' for the NU
    cohort on the HPC copy, so accept either spelling rather than assuming one.
    """
    parent = BE_MASTER / "0.input/1.qp_wsi_projects" / cohort
    for name in ("ROI_pyramid", "ROIpyramid"):
        if (parent / name).is_dir():
            return parent / name
    return parent / "ROI_pyramid"


ROI_DIRS = {
    "MDA": _roi_dir("mda"),
    "NU": _roi_dir("nu"),
}

OUT_DIR = BE_MASTER / "revision/task6_benchmark"
INTERIM_DIR = OUT_DIR / "interim"
RESULTS_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"

SLIDE_LABELS_CSV = INTERIM_DIR / "slide_labels.csv"      # written by 00_export_labels.R
NUCLEI_PARQUET = INTERIM_DIR / "nuclei_filtered.parquet"  # written by 01
FEATURES_DIR = INTERIM_DIR / "features"                   # written by 02
CLUSTERS_PARQUET = INTERIM_DIR / "nucleus_clusters.parquet"  # written by 02 (rung 3 needs these)
CROPS_DIR = INTERIM_DIR / "crops"                         # written by 04

for _d in (INTERIM_DIR, RESULTS_DIR, FIG_DIR, FEATURES_DIR, CROPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# The Barrett-epithelium filter
# --------------------------------------------------------------------------
# Two independent columns, and they disagree often enough that filtering on
# either alone is wrong (e.g. D0050: 333 epithelial-classified nuclei sit in
# stroma, 190 non-epithelial nuclei sit inside epithelium):
#   Class  -- the nucleus model's cell-type call
#   Parent -- the tissue-segmentation region
CLASS_KEEP = "epithelial"
PARENT_KEEP = "epithelium_area"

# Columns that are NOT nuclear features. Everything else in the TSV is one of
# the 61 features.
#
# attention_score is DACOR's own output. DACOR is the model being benchmarked
# against, so every DACOR-derived quantity is excluded by construction -- the
# baseline never sees it, at any stage. This is what makes the comparison an
# honest ablation rather than a circular one.
NON_FEATURE_COLS = [
    "Image",
    "Object ID",
    "Name",
    "Class",
    "Parent",
    "ROI",
    "Centroid X µm",
    "Centroid Y µm",
    "attention_score",
]

# Columns carried alongside the features in the nucleus parquet. Not features,
# and not DACOR-derived: the slide id, the ROI filename (used to locate the
# image in rung 3), and the centroid in microns (used to place crops).
NUCLEUS_META_COLS = ["wsi", "image_file", "centroid_x_um", "centroid_y_um"]

EXPECTED_N_FEATURES = 61
EXPECTED_N_SLIDES = 781

# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

# Rung 1 (Yu's "M"): per-slide summary of each nuclear feature.
# Five statistics, matching the existing nuc_* summarisation in the pipeline.
SUMMARY_STATS = ("mean", "std", "median", "q25", "q75")

# Rung 2 (Yu's "C+M"): nuclei clustered into subtypes, then
#   C = subtype ratios (N_CLUSTERS - 1, since ratios sum to 1)
#   M = per-subtype mean of each feature, concatenated
N_CLUSTERS = 15                  # Yu et al. use 15
KMEANS_SUBSAMPLE = 500_000       # nuclei sampled to fit k-means (train slides only)
CLUSTER_SPACE = "raw"            # "raw" = the 61 standardised features; "pca" = sensitivity

PCA_COMPONENTS = 20              # only used when CLUSTER_SPACE == "pca"

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
# Yu et al.'s classifier head is a single fully-connected layer on the
# concatenated feature vector, i.e. logistic regression. We keep that, adding
# elastic-net regularisation because our n is small relative to p.

COHORT_TRAIN = "MDA"   # discovery -- DACOR's train + val
COHORT_TEST = "NU"     # test -- never touched during fitting or tuning

CV_FOLDS = 5           # GroupKFold, grouped by patient (no patient spans folds)
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
L1_RATIO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_ITER = 5000

# Patient-level score aggregation, mirroring DACOR (max over a patient's slides).
PATIENT_AGG = "max"

# Secondary analysis only -- the main comparison keeps every slide so that
# DACOR and the baseline are scored on an identical test set.
MIN_NUCLEI_SENSITIVITY = 200

RANDOM_SEED = 42
N_JOBS = int(os.environ.get("BEACON_N_JOBS", os.cpu_count() or 4))

# --------------------------------------------------------------------------
# Rung 3 -- deep block (Yu et al.'s "D"), GPU
# --------------------------------------------------------------------------
# Crops are defined in MICRONS and resampled to a fixed pixel size. The two
# scanners differ (MDA 0.5013, NU 0.4548 um/px), so a fixed pixel box would mean
# different physical extents per cohort -- and we train on one and test on the
# other. CROP_UM matches Yu's 100 px at 40x (0.25 um/px) = 25 um; CROP_PX matches
# their network input size.
#
# The um/px factor is read from each TIFF's own resolution tag, not assumed.
# (Verified: the manuscript Methods has the two scanner factors swapped -- for
# MDA, 0.4548 places nuclei outside the image bounds while 0.5013 fits.)
CROP_UM = 25.0
CROP_PX = 100

# Storing a crop for all ~2.86M nuclei would be ~86 GB. Instead a bounded pool
# is cached per slide, stratified by subtype so rare subtypes survive; each epoch
# then samples CELLS_PER_BAG from the pool, reweighted to the slide's true
# subtype distribution. This keeps Yu's per-epoch resampling (which doubles as
# augmentation) at ~23 GB.
POOL_PER_SLIDE = 1000
CELLS_PER_BAG = 200          # Yu et al.'s n = 200

# Yu et al.'s training configuration (paper section IV-C).
DEEP_EMBED_DIM = 64          # DenseNet 1024-D -> 64-D cell embeddings
TRANSFORMER_HEADS = 4
TRANSFORMER_FF_DIM = 128
TRANSFORMER_DROPOUT = 0.1
DEEP_EPOCHS = 50
DEEP_BATCH_SIZE = 4          # bags (slides) per step
DEEP_LR = 5e-4
DEEP_WEIGHT_DECAY = 0.01
DEEP_ADAM_EPS = 1e-8
DEEP_ADAM_BETAS = (0.9, 0.999)
DEEP_GRAD_CLIP = 5.0
DEEP_LR_PATIENCE = 10        # plateau scheduler, factor 10 reduction
DEEP_EARLY_STOP_PATIENCE = 15

# Which rung-3 arms to train. "D" reproduces Yu's deep-only ablation,
# "CMD" their full hybrid.
DEEP_ARMS = ("D", "CMD")

# Stain normalisation variants, each trained separately so the two can be
# reported side by side. None = exactly as published (Yu et al. apply no colour
# processing); "reinhard" = per-slide LAB mean/std matched to a training-cohort
# target, which is the cheapest standard remedy for the scanner colour shift
# between the two cohorts (MDA hematoxylin mean 0.352 vs NU 0.623).
STAIN_NORM_VARIANTS = (None, "reinhard")

# Pixels at or above this value in all three channels are treated as slide
# background and excluded when estimating a slide's colour statistics --
# otherwise the white padding dominates the mean.
STAIN_BG_THRESHOLD = 220

# Evaluation bags. Bags are drawn with the SAME distribution-preserving sampler
# as training (previously evaluation took the highest-weight cells, which filled
# a bag from the one or two largest subtypes and mismatched training). Draws are
# seeded per (slide, bag index), so they are deterministic and identical across
# epochs -- an epoch-to-epoch change in validation score then reflects the model
# rather than resampling. Final train/test scores average over EVAL_BAGS draws;
# per-epoch monitoring uses a single fixed draw to keep epochs cheap.
EVAL_BAGS = 5

# Checkpoint selection. Each epoch is scored by a centred moving average of
# validation AUC over this many epochs, and the middle epoch of the best window
# is kept. With 82 validation slides (~24 positive) a single epoch's AUC swings
# on luck alone -- rung3_D scored 0.879 at epoch 3 and 0.506 at epoch 4 -- so
# taking the single best epoch tends to save a lucky bounce.
SELECTION_WINDOW = 3

# Rung 3 uses DACOR's own train/val split for early stopping, so the deep model
# is fitted on exactly the slides DACOR was fitted on.
DEEP_DEVICE = os.environ.get("BEACON_DEVICE", "cuda")
