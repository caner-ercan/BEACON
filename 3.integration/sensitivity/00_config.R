# =====================================================================
# Task 3 - Parameter sensitivity for spatial metrics (Reviewer 1, #2)
# Shared configuration.
#
# All values below were READ OFF the authoritative pipeline in
# 3.integration/ecology_code/ (2026-07-31), not assumed. See README.
#
# DESIGN PRINCIPLE
#   Grids are anchored to STRUCTURE (lattice rings, FM tile multiples) or
#   to STATED CONVENTION, never to multipliers of the published value.
#   The published setting is a member of every grid so that reproduction
#   can be checked at the anchor point.
# =====================================================================

# Set BE_MASTER to your BE_master checkout (a relative default here would
# depend on the invocation directory, not the script's location, so this
# is required rather than guessed) - or set BEACON_DATA/BEACON_CLIN
# directly to override just one of the two paths below.
BE_MASTER <- Sys.getenv("BE_MASTER", unset = NA_character_)
if (is.na(BE_MASTER) && !nzchar(Sys.getenv("BEACON_DATA")))
  stop("Set the BE_MASTER environment variable to your BE_master checkout ",
       "(or BEACON_DATA/BEACON_CLIN directly).")
DATA_ROOT <- Sys.getenv("BEACON_DATA",
  file.path(BE_MASTER, "3.integration/integration_project/spatial_analysis/tabular"))
CLIN_ROOT <- Sys.getenv("BEACON_CLIN",
  file.path(BE_MASTER, "0.input/organised_wsi_patient"))

# Output lives at <tabular>/rev1_sensitivity. Earlier runs wrote to
# <tabular>/calculated/rev1_sensitivity, so both are accepted: an explicit
# BEACON_OUT wins, otherwise whichever already exists, otherwise the new
# top-level location.
OUT_DIR <- local({
  env <- Sys.getenv("BEACON_OUT")
  if (nzchar(env)) return(env)
  top <- file.path(DATA_ROOT, "rev1_sensitivity")
  old <- file.path(DATA_ROOT, "calculated", "rev1_sensitivity")
  if (dir.exists(top)) top else if (dir.exists(old)) old else top
})
for (d in c(OUT_DIR, file.path(OUT_DIR, "raw"), file.path(OUT_DIR, "fig")))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

PATHS <- list(
  region     = file.path(DATA_ROOT, "merged", "region_level_processed_250808.RDS"),
  cell       = file.path(DATA_ROOT, "merged", "cell_level_250808.RDS"),
  ripley_all = file.path(DATA_ROOT, "intermediate", "ripley_results.RDS"),
  ripley_a0  = file.path(DATA_ROOT, "intermediate", "ripley_results_attn0.RDS"),
  ripley_a1  = file.path(DATA_ROOT, "intermediate", "ripley_results_attn1.RDS"),
  published  = file.path(DATA_ROOT, "calculated", "merged_ecology_250812_wNuc.RDS"),
  wsi_map    = file.path(CLIN_ROOT, "wsi_randomid_map_rev1_260730.csv")
)

# ---- geometry (confirmed) -------------------------------------------
# Region tiles are on a 112 px lattice: the FM takes 224x224 px input and
# local prediction used a 50% overlap sliding window -> 112 px stride.
TILE_PX  <- 112
# Morisita runs on the EXPANDED tile: 2x2 aggregation = 224 px = one full
# non-overlapping FM window.
MORISITA_BASE_FACTOR <- 2

# NOTE (deferred discrepancy): the code assigns NU = 0.4548 and
# MDA = 0.5013; the manuscript Methods state the opposite. Used only to
# report um equivalents - the lattice grid itself is unit-free.
UM_PER_PX <- if (identical(Sys.getenv("BEACON_UMPX"), "manuscript"))
  c(MDA = 0.4548, NU = 0.5013) else c(NU = 0.4548, MDA = 0.5013)
UM_PER_PX_MEAN <- mean(unname(UM_PER_PX))

# ---- stratification (confirmed) -------------------------------------
# The delivered GlobalMoran_*_attn1 values match attn_class_extended far
# better than raw attention_score (cor 0.977 vs 0.641), so the extended
# perilesional class is what actually ran.
ATTN_COLUMN    <- Sys.getenv("BEACON_ATTN_COL", "attn_class_extended")
ATTN_THRESHOLD <- 0.5
STRATA <- c(all = 2, abnormal = 1, normal = 0)

# CELL-level tables carry only the continuous `attention_score` - the
# extended class is a TILE-level construct and does not exist there. The
# published cell-level scripts (2.4.knn, 2.7.nn_gfunction) accordingly
# stratify on attention_score >= 0.5. Using the tile column here silently
# yields NULL and drops the abnormal/normal strata entirely.
ATTN_COLUMN_CELL <- Sys.getenv("BEACON_ATTN_COL_CELL", "attention_score")

# =====================================================================
# PARAMETER GRIDS  (published value marked; 3 values each)
# =====================================================================

# ---- Moran / Getis: neighbourhood, in PIXELS on the tile lattice -----
# Published d2 = 200 px. On a 112 px lattice: orthogonal neighbours at
# 112, diagonal at 158.4, next ring at 224. So 200 px == the 8-neighbour
# QUEEN ring exactly. Intermediate distances give identical neighbour
# sets, so the grid is expressed as RINGS, not as a distance sweep.
D2_RINGS_PX <- c(ring1_queen = 200, ring2 = 350, ring3 = 500)
D2_PUBLISHED_PX <- 200
MORAN_STYLES <- c("B")            # published; "W" optional, see README

# ---- Getis-Ord: variant ---------------------------------------------
# Published: localG(..., GeoDa = TRUE). Slide summaries are SIGN-BASED,
# with no p-value threshold anywhere in the published pipeline.
GETIS_GEODA <- TRUE

# ---- Ripley's L: r range (MICRONS - cell level) ----------------------
# Published: Kcross called WITHOUT r, so spatstat's per-slide default
# applies (observed rmax 2.8-450 um). "default" reproduces that exactly;
# the fixed values impose a common domain across slides.
# 100 um is lattice-coherent: it matches the queen ring (~91-100 um),
# putting cell-level and tile-level statistics on the same scale.
RIPLEY_RMAX_UM <- c(default = NA_real_, fixed100 = 100, fixed200 = 200)

# ---- kNN: k (cell level) --------------------------------------------
# Published k = 10 (fallback min_n - 1). The only genuinely arbitrary
# parameter in the pipeline; 08_justification.R computes a structural
# anchor (median target-class cells within the queen-ring radius).
KNN_K <- c(5, 10, 20)
KNN_K_PUBLISHED <- 10

# ---- G-function: summary window (MICRONS - cell level) --------------
# Published r <= 150 um. NOTE summarise_g() declares short_range = 50 and
# medium_range = 150 but ignores both and hardcodes 150 in the body.
# 50 um is therefore an author-intended value.
# WARNING: the published peak_r is CENSORED at 150 by construction, so it
# cannot justify the window. 08_justification.R recomputes the peak and
# the saturation point over the FULL curve.
GFUN_WINDOW_UM <- c(short = 50, published = 150, long = 300)
GFUN_WINDOW_PUBLISHED <- 150

# ---- Morisita-Horn: quadrat (PIXELS) --------------------------------
# Published quadrat = 224 px (one FM window). Grid = one stride / one FM
# tile / two FM tiles.
MORISITA_QUADRAT_PX <- c(stride = 112, published = 224, double = 448)
MORISITA_QUADRAT_PUBLISHED <- 224

# ---- cell pairs / types (as published) ------------------------------
CELL_PAIRS <- list(
  c("Epithelial", "Stroma"), c("Epithelial", "Immune"),
  c("Epithelial", "Lymph.Plasma"), c("Epithelial", "Lymphocyte"),
  c("Epithelial", "Plasma"), c("Immune", "Stroma"),
  c("Lymph.Plasma", "Stroma"), c("Lymphocyte", "Stroma"),
  c("Plasma", "Stroma")
)
CELL_TYPES_REGION <- c("Num.Lymphocyte", "Num.Plasma", "Num.LymphoPlasma", "Num.Immune")

# =====================================================================
# ANALYSIS CONVENTIONS  (must match 4.plotting/code/3.ecology.Rmd)
# =====================================================================
# Slide level (settled with the author).
ANALYSIS_LEVEL <- "slide"
# 3.ecology.Rmd drops zero values before testing. This matters enormously:
# 3.0.merge_normalize.Rmd imputes non-computable values to 0, and >50% of
# slides have GlobalMoran_Plasma_attn1 == 0.
DROP_ZEROS <- TRUE
# Shapiro-Wilk is <= 1e-5 in every group tested, so the published
# normality branch always selects Wilcoxon. Fixed here for determinism.
TEST_METHOD <- "wilcox"
# Restrict to DACOR-abnormal slides, as Fig. 4C/4D do.
RESTRICT_PRED <- 1

# ---- anchor claim ----------------------------------------------------
# Fig. 4C bottom, author-identified. Slide level, zeros dropped:
# Wilcoxon p = 0.0075 (published 0.003; residual likely the
# concurrent-malignancy exclusion, not applied here).
ANCHOR_FEATURE <- "GlobalMoran_Plasma_attn1"

message(sprintf("[config] DATA_ROOT=%s", DATA_ROOT))
message(sprintf("[config] attn=%s | level=%s | drop_zeros=%s | anchor=%s",
                ATTN_COLUMN, ANALYSIS_LEVEL, DROP_ZEROS, ANCHOR_FEATURE))
