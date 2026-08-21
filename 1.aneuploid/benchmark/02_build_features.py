"""
Task 6, step 2 -- build the slide-level feature matrices for rungs 1 and 2.

Rung 1 (Yu et al.'s "M"): each of the 61 nuclear features summarised per slide
by mean/sd/median/q25/q75 -> 305 features.

Rung 2 (Yu et al.'s "C+M"): nuclei clustered into 15 subtypes, then
    C = subtype ratios                      -> 14 features (one dropped; ratios sum to 1)
    M = per-subtype mean of each feature     -> 61 x 15 = 915 features
  giving 929 features.

Leakage control: standardisation and k-means are fitted on TRAINING-cohort
nuclei only, then applied to the test cohort.

Deviation from Yu et al., stated for the Methods: they clustered in a
convolutional-autoencoder latent space learned from nucleus images. Clustering
here uses the handcrafted feature space itself (CLUSTER_SPACE='raw') or its
principal components ('pca'). The raw variant means clusters are defined in the
same space that is then pooled per cluster; the PCA variant is provided as a
sensitivity check on whether that matters.

Run:  python 02_build_features.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config as cfg
import lib


def _load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not cfg.NUCLEI_PARQUET.exists():
        sys.exit(f"missing {cfg.NUCLEI_PARQUET} -- run 01_build_nucleus_table.py first")
    if not cfg.SLIDE_LABELS_CSV.exists():
        sys.exit(f"missing {cfg.SLIDE_LABELS_CSV} -- run 00_export_labels.R first")

    nuclei = pd.read_parquet(cfg.NUCLEI_PARQUET)
    labels = pd.read_csv(cfg.SLIDE_LABELS_CSV)
    feature_cols = [c for c in nuclei.columns if c not in cfg.NUCLEUS_META_COLS]

    known = set(labels["wsi"])
    unlabelled = sorted(set(nuclei["wsi"].astype(str)) - known)
    if unlabelled:
        print(f"WARNING: {len(unlabelled)} slides have nuclei but no label row; "
              f"dropping: {unlabelled}")
        nuclei = nuclei[nuclei["wsi"].astype(str).isin(known)]

    return nuclei, labels, feature_cols


def build_rung1(nuclei: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Pooled per-slide summary statistics."""
    return lib.summarise_by_slide(nuclei, feature_cols)


def build_rung2(
    nuclei: pd.DataFrame, feature_cols: list[str], train_wsi: set[str]
) -> tuple[pd.DataFrame, dict]:
    """Cluster nuclei into subtypes, then build subtype ratios + per-subtype means."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    X = nuclei[feature_cols].to_numpy(dtype=np.float64)
    # k-means cannot take NaN; impute with the training-cohort column median.
    is_train = nuclei["wsi"].astype(str).isin(train_wsi).to_numpy()
    med = np.nanmedian(X[is_train], axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(med, np.where(nan_mask)[1])

    scaler = StandardScaler().fit(X[is_train])
    Z = scaler.transform(X)

    reducer = None
    if cfg.CLUSTER_SPACE == "pca":
        reducer = PCA(n_components=cfg.PCA_COMPONENTS, random_state=cfg.RANDOM_SEED)
        reducer.fit(Z[is_train])
        Z = reducer.transform(Z)
        print(f"clustering in PCA space: {cfg.PCA_COMPONENTS} components, "
              f"explained variance {reducer.explained_variance_ratio_.sum():.3f}")
    elif cfg.CLUSTER_SPACE != "raw":
        sys.exit(f"unknown CLUSTER_SPACE: {cfg.CLUSTER_SPACE}")

    train_idx = np.flatnonzero(is_train)
    if len(train_idx) > cfg.KMEANS_SUBSAMPLE:
        train_idx = rng.choice(train_idx, size=cfg.KMEANS_SUBSAMPLE, replace=False)
    print(f"fitting k-means (k={cfg.N_CLUSTERS}) on {len(train_idx):,} training nuclei")

    km = MiniBatchKMeans(
        n_clusters=cfg.N_CLUSTERS,
        random_state=cfg.RANDOM_SEED,
        batch_size=10_000,
        n_init=10,
        max_iter=500,
    ).fit(Z[train_idx])

    cluster = km.predict(Z)
    work = pd.DataFrame({"wsi": nuclei["wsi"].astype(str).to_numpy(), "cluster": cluster})
    work[feature_cols] = nuclei[feature_cols].to_numpy()

    # C: subtype ratios, one column dropped (ratios sum to 1).
    counts = pd.crosstab(work["wsi"], work["cluster"])
    counts = counts.reindex(columns=range(cfg.N_CLUSTERS), fill_value=0)
    ratios = counts.div(counts.sum(axis=1), axis=0)
    ratios.columns = [f"ratio_c{c}" for c in ratios.columns]
    C = ratios.iloc[:, :-1]

    # M: per-subtype mean of each feature, concatenated across subtypes.
    per = work.groupby(["wsi", "cluster"], observed=False)[feature_cols].mean()
    M = per.unstack("cluster")
    M.columns = [f"c{int(c)}_{f}" for f, c in M.columns]
    M = M.reindex(sorted(M.columns), axis=1)

    # A slide need not contain every subtype. Fill absent subtype/feature cells
    # with the training-cohort mean for that cell, so the vector stays fixed-length.
    train_rows = M.index.isin(train_wsi)
    fill = M.loc[train_rows].mean()
    n_missing = int(M.isna().sum().sum())
    if n_missing:
        print(f"absent subtype cells filled with training means: {n_missing:,} "
              f"({n_missing / M.size:.2%} of the M block)")
    M = M.fillna(fill)

    features = pd.concat([C, M], axis=1)

    diagnostics = {
        "cluster_sizes": pd.Series(np.bincount(cluster, minlength=cfg.N_CLUSTERS),
                                   name="n_nuclei"),
        "slides_missing_any_subtype": int((counts == 0).any(axis=1).sum()),
        "cluster_space": cfg.CLUSTER_SPACE,
        # Per-nucleus assignments, in the row order of the LABELLED subset of
        # nuclei_filtered.parquet (slides absent from slide_labels.csv are
        # dropped in _load above, so this is shorter than the full table).
        # Rung 3 needs these for stratified crop pooling and the MIL sampler.
        "assignments": cluster.astype("int16"),
    }
    return features, diagnostics


def main() -> int:
    nuclei, labels, feature_cols = _load()
    train_wsi = set(labels.loc[labels["dataset"] == cfg.COHORT_TRAIN, "wsi"])
    print(f"{len(nuclei):,} nuclei | {len(feature_cols)} features | "
          f"{len(train_wsi)} training slides")

    print("\n--- rung 1: pooled morphology (M) ---")
    rung1 = build_rung1(nuclei, feature_cols)
    print(f"{rung1.shape[0]} slides x {rung1.shape[1]} features")
    rung1.to_parquet(cfg.FEATURES_DIR / "rung1_morphology.parquet")

    print("\n--- rung 2: subtype ratios + per-subtype morphology (C+M) ---")
    rung2, diag = build_rung2(nuclei, feature_cols, train_wsi)
    print(f"{rung2.shape[0]} slides x {rung2.shape[1]} features")
    print(f"slides missing at least one subtype: {diag['slides_missing_any_subtype']}")
    print("\nnuclei per subtype:")
    print(diag["cluster_sizes"].to_string())
    rung2.to_parquet(cfg.FEATURES_DIR / f"rung2_subtypes_{cfg.CLUSTER_SPACE}.parquet")
    diag["cluster_sizes"].to_csv(cfg.RESULTS_DIR / f"cluster_sizes_{cfg.CLUSTER_SPACE}.csv")

    # Per-nucleus cluster ids, aligned row-for-row with nuclei_filtered.parquet.
    pd.DataFrame(
        {
            "wsi": nuclei["wsi"].astype(str).to_numpy(),
            "cluster": diag["assignments"],
        }
    ).to_parquet(cfg.CLUSTERS_PARQUET, index=False)
    print(f"wrote per-nucleus cluster assignments to {cfg.CLUSTERS_PARQUET}")

    print("\nwrote feature matrices to", cfg.FEATURES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
