"""
Task 6, step 1 -- build the filtered per-nucleus table.

Reads the 781 per-slide QuPath TSV exports, keeps only nuclei that are
epithelial AND inside segmented Barrett epithelium, drops every DACOR-derived
column, and writes a single parquet.

Alongside the 61 features it carries the slide id, the ROI image filename and
the centroid in microns. None of these are features or DACOR-derived; the
filename and centroid are what rung 3 uses to locate and cut nucleus crops.

Expected scale: ~2.86M nuclei x 61 features (~700 MB as float32).

Run:  python 01_build_nucleus_table.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import config as cfg
import lib


def _read_one(path) -> pd.DataFrame | None:
    """Read one slide, apply the epithelium filter, return canonical features."""
    wsi = path.stem

    df = pd.read_csv(path, sep="\t", dtype={"Class": "string", "Parent": "string"})

    keep = (df["Class"] == cfg.CLASS_KEEP) & (df["Parent"] == cfg.PARENT_KEEP)
    df = df.loc[keep]
    if df.empty:
        return None

    feat_cols = lib.feature_columns(df.columns)
    out = df[feat_cols].astype("float32")
    out.columns = [lib.canonical_name(c) for c in feat_cols]

    # Metadata for rung 3: which image to open, and where in it.
    out.insert(0, "centroid_y_um", df["Centroid Y µm"].to_numpy(dtype="float32"))
    out.insert(0, "centroid_x_um", df["Centroid X µm"].to_numpy(dtype="float32"))
    out.insert(0, "image_file", df["Image"].astype("string").to_numpy())
    out.insert(0, "wsi", wsi)
    return out


def main() -> int:
    paths = sorted(cfg.NUC_EXPORT_DIR.glob("*.tsv"))
    if not paths:
        print(f"ERROR: no TSV files under {cfg.NUC_EXPORT_DIR}", file=sys.stderr)
        return 1
    print(f"found {len(paths)} slide exports (expected {cfg.EXPECTED_N_SLIDES})")

    frames, empty = [], []
    with ProcessPoolExecutor(max_workers=cfg.N_JOBS) as pool:
        for path, result in zip(paths, pool.map(_read_one, paths, chunksize=4)):
            if result is None:
                empty.append(path.stem)
            else:
                frames.append(result)

    if empty:
        print(f"WARNING: {len(empty)} slides had no nuclei passing the filter: {empty}")
    if not frames:
        print("ERROR: no nuclei survived the filter", file=sys.stderr)
        return 1

    nuclei = pd.concat(frames, ignore_index=True)
    nuclei["wsi"] = nuclei["wsi"].astype("category")
    nuclei["image_file"] = nuclei["image_file"].astype("category")

    feature_cols = [c for c in nuclei.columns if c not in cfg.NUCLEUS_META_COLS]
    if len(feature_cols) != cfg.EXPECTED_N_FEATURES:
        print(
            f"WARNING: expected {cfg.EXPECTED_N_FEATURES} features, got {len(feature_cols)}",
            file=sys.stderr,
        )

    # Guard: no DACOR-derived column may reach the baseline.
    leaked = [c for c in feature_cols if "attention" in c.lower() or c.lower().startswith("pred")]
    if leaked:
        print(f"ERROR: DACOR-derived columns present: {leaked}", file=sys.stderr)
        return 1

    n_bad = int(np.isinf(nuclei[feature_cols].to_numpy()).sum())
    if n_bad:
        print(f"WARNING: {n_bad} non-finite feature values -> NaN")
        nuclei[feature_cols] = nuclei[feature_cols].replace([np.inf, -np.inf], np.nan)

    per_slide = nuclei.groupby("wsi", observed=True).size()
    print(f"\nnuclei retained: {len(nuclei):,} across {per_slide.size} slides")
    print(f"per slide: median {int(per_slide.median()):,}, min {int(per_slide.min()):,}, "
          f"max {int(per_slide.max()):,}")
    print(f"slides below the sensitivity floor "
          f"({cfg.MIN_NUCLEI_SENSITIVITY}): {int((per_slide < cfg.MIN_NUCLEI_SENSITIVITY).sum())}")
    print(f"features: {len(feature_cols)}")

    nuclei.to_parquet(cfg.NUCLEI_PARQUET, index=False)
    per_slide.rename("n_nuclei").to_frame().to_csv(cfg.INTERIM_DIR / "nuclei_per_slide.csv")
    print(f"\nwrote {cfg.NUCLEI_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
