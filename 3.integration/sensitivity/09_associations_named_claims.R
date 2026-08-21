# =====================================================================
# 09 - Association tests for the NAMED manuscript claims only, each
#      against the CORRECT contrast for that claim.
#
# 06_associations.R tests everything against PROGRESSION, within
# DACOR-abnormal cases - that is the Fig. 4C/4D contrast. Fig. 3D and
# Supp. Fig. 1C do NOT make a progression claim: they compare DNA-content
# ABNORMAL vs NORMAL slides (the `pred` variable), all 777 slides, not
# restricted to pred==1. Testing those metrics against progression (as an
# earlier due-diligence pass in this task did) checks the wrong thing and
# will show "not significant" regardless of whether the published finding
# is robust - that is a scope error, not a replication failure.
#
# This script computes both contrasts for EXACTLY the 4 metrics CE named
# (see NAMED_CLAIMS below), and nothing else - it is deliberately NOT a
# sweep over the full feature space (that is 06/07's job, archived) and
# deliberately NOT "every significant panel in Fig. 3D/Supp. Fig. 1C"
# either. An earlier version of this script pulled in a wider Fig 3D/Supp
# Fig 1C candidate list on its own initiative; CE narrowed it back to the
# 4 explicitly requested. Do not re-widen this list without being asked.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R")); source(file.path(here, "lib_sweep.R"))

sweeps <- purrr::map_dfr(c("sweep_ripley", "sweep_moran_getis", "sweep_knn",
                          "sweep_gfunction", "sweep_morisita"), function(s) {
  d <- load_sweep(OUT_DIR, s)
  if (is.null(d)) return(NULL)
  d
})
clin <- get_slide_clinical(PATHS$published)

# ---- published-sample restriction, generalised across all 8 claims ----
# CRITICAL, not optional: our recomputation returns far FEWER exact zeros
# than the published table for every metric except kNN (kNN is imputed to
# the column MAX for non-computable slides, everything else to 0 - see
# 3.0.merge_normalize.Rmd's impute_na()). Checked directly for all 8 claims:
#   Plasma GlobalMoran:        0/460 zero here   vs 505/777 published
#   Plasma Getis clustering:   0/460 zero here   vs 290/777 published
#   Plasma-Stroma Ripley:     15/331 zero here   vs 461/777 published
#   LymphoPlasma Moran:        3/716 zero here   vs 135/777 published
#   Immune GlobalMoran:        0/717 zero here   vs 353/777 published
#   Immune Moran:              2/717 zero here   vs 120/777 published
#   Epithelial-Immune kNN:     0/777 zero here   vs   0/777 published (clean)
#   Epithelial-Immune Ripley: 46/651 zero here   vs 172/777 published
# Comparing "everything we can compute" against "what the published table
# could compute" mixes an INCLUSION effect (which slides are in the test)
# into what is supposed to be a pure PARAMETER effect (does ring/k/window
# choice change the answer) - exactly the confound worked through earlier
# for the anchor claim alone. Generalised here to all 8, not left as a
# one-off fix, because the same gap is present in 7 of them.
pub <- as.data.frame(readRDS(PATHS$published))
names(pub) <- iconv(names(pub), "UTF-8", "ASCII", sub = "u")

published_nonzero_wsi <- function(pubcol) {
  if (!pubcol %in% names(pub)) { warning("published column not found: ", pubcol); return(NULL) }
  pub$wsi[!is.na(pub[[pubcol]]) & pub[[pubcol]] != 0]
}

# ---- the named claims, each tagged with its correct contrast ----------
# contrast = "progression" -> Fig. 4C/4D: within DACOR-abnormal slides,
#            progressor vs non-progressor.
# contrast = "pred"        -> Fig. 3D / Supp. Fig. 1C: DNA-content
#            abnormal vs normal, all slides, stratum = "abnormal" (the
#            "_attn1" suffix - metric computed on abnormal-region tiles,
#            then compared BETWEEN pred groups, exactly as the published
#            single_inter_wsi() test and task 4's replication of it do).
# SCOPE, tightened per CE 2026-08-10: exactly these 4 metrics, no more.
# An earlier version of this list added 4 further Fig 3D/Supp Fig 1C
# candidates pulled from rev1_unit_of_analysis's own [INF]-flagged (i.e.
# INFERRED, not verified) metric table, without being asked to - CE caught
# this as scope creep and named the 4 that belong here. The 4th (immune
# Moran, clustered mean) is now additionally VERIFIED against the actual
# Supp. Fig. 1 PDF (CE-supplied): Panel C, "DNA Content" Low/High,
# P=.003, tested on the _attn1-suffixed column - exactly the "pred"
# contrast already used below. No change needed to how it's tested, only
# to which other rows are (not) included.
NAMED_CLAIMS <- tibble::tribble(
  ~label,                                             ~metric,       ~feature,             ~cell,               ~stratum,   ~contrast,    ~pubcol,
  "Fig 4C bottom: plasma GlobalMoran (anchor)",        "Global",      "Moran_I",            "Plasma",            "abnormal", "progression", "GlobalMoran_Plasma_attn1",
  "Fig 4D: plasma Getis-Ord clustering strength",      "GetisOrd_Gi", "clustering_strength","Plasma",            "abnormal", "progression", "Gi_star_Plasma_clustering_strength_attn1",
  "Fig 4D: plasma-stroma attraction (Ripley frac+)",   "Ripley_L",    "frac_positive",      "Plasma_Stroma",     "abnormal", "progression", "Ripley_L12_Plasma_Stroma_frac_positive_attn1",
  "Supp Fig 1C: immune Moran, clustered mean",         "Moran_local", "clustered_mean",     "Immune",            "abnormal", "pred",        "Moran_Local_Immune_Ii_clustered_mean_attn1"
)

message(sprintf("[claims] %d named metrics: %d progression (Fig 4C/4D), %d pred (Fig 3D/Supp Fig 1C)",
                nrow(NAMED_CLAIMS), sum(NAMED_CLAIMS$contrast == "progression"),
                sum(NAMED_CLAIMS$contrast == "pred")))

# ---- run each claim across its parameter grid --------------------------
run_claim <- function(row) {
  usable <- published_nonzero_wsi(row$pubcol)
  d <- sweeps %>%
    filter(metric == row$metric, feature == row$feature, cell == row$cell, stratum == row$stratum) %>%
    inner_join(clin, by = "wsi")
  if (!nrow(d)) { warning("no sweep rows for ", row$label); return(NULL) }
  if (!is.null(usable)) d <- d %>% filter(wsi %in% usable)

  if (row$contrast == "progression") {
    d <- d %>% filter(!is.na(pred), pred == 1, !is.na(progression)) %>% mutate(.g = progression)
  } else {
    d <- d %>% filter(!is.na(pred)) %>% mutate(.g = pred)
  }

  d %>%
    group_by(param_name, param_label, param_value) %>%
    group_modify(~ assoc_test(.x$value, .x$.g, DROP_ZEROS)) %>%
    ungroup() %>%
    mutate(label = row$label, contrast = row$contrast,
           metric = row$metric, feature = row$feature, cell = row$cell, stratum = row$stratum)
}

res <- purrr::map_dfr(seq_len(nrow(NAMED_CLAIMS)), function(i) run_claim(NAMED_CLAIMS[i, ]))

saveRDS(res, file.path(OUT_DIR, "named_claims_associations.RDS"))
write.csv(res, file.path(OUT_DIR, "named_claims_associations.csv"), row.names = FALSE)

message("\n=== named claims, every parameter setting ===")
print(as.data.frame(res %>% select(label, contrast, param_label, param_value, n1, n0, effect, p) %>%
  arrange(contrast, label, param_value)), row.names = FALSE)

# ---- regression check: must reproduce the already-published anchor ----
# The anchor's ring1_queen number was independently derived and vetted
# earlier (r=0.315, p=0.0075, n=85/34) and is already written into the
# Results text and rebuttal. If this script's generalised published-sample
# logic does not reproduce it, something is wrong with the generalisation
# and the other 7 claims should not be trusted either.
anchor_check <- res %>% filter(label == "Fig 4C bottom: plasma GlobalMoran (anchor)", param_label == "ring1_queen")
ok <- nrow(anchor_check) == 1 && abs(anchor_check$effect - 0.3148789) < 1e-4 &&
      abs(anchor_check$p - 0.007505769) < 1e-4 && anchor_check$n1 == 85 && anchor_check$n0 == 34
if (!ok) {
  message("\n[REGRESSION CHECK FAILED] anchor does not reproduce the vetted numbers:")
  print(as.data.frame(anchor_check))
  stop("Fix before trusting the other 7 claims.")
}
message("\n[regression check OK] anchor reproduces r=0.315, p=0.0075, n=85/34 exactly.")

stab <- res %>% filter(is.finite(effect)) %>%
  group_by(label, contrast) %>%
  summarise(n_settings = n(), sign_consistent = all(effect > 0) | all(effect < 0),
            effect_min = min(effect), effect_max = max(effect),
            any_sig = any(p < 0.05, na.rm = TRUE), .groups = "drop")
message("\n=== stability per named claim ===")
print(as.data.frame(stab), row.names = FALSE)
write.csv(stab, file.path(OUT_DIR, "named_claims_stability.csv"), row.names = FALSE)
