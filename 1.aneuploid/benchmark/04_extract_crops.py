"""
Task 6, step 4 -- cache nucleus image crops for rung 3 (CPU, parallel).

For each slide: open the ROI pyramid TIFF, cut a fixed-physical-size box around
each pooled nucleus, resample to CROP_PX, and save one compressed .npz per slide.

Two details that matter:

* The micron-per-pixel factor is read from each TIFF's own resolution tag, never
  assumed from the cohort. The manuscript Methods has the two scanner factors
  swapped (for MDA, 0.4548 places nuclei outside the image bounds while 0.5013
  fits), so a cohort lookup would silently mis-scale every crop.
* Crops are defined in microns and then resampled, so a crop covers the same
  physical extent on both scanners. Since we train on one cohort and test on the
  other, a fixed pixel box would confound scanner with cohort.

Storing crops for all ~2.86M nuclei would be ~86 GB, so a bounded pool of
POOL_PER_SLIDE nuclei is cached per slide, stratified by subtype so rare subtypes
survive. Step 05 then resamples CELLS_PER_BAG from the pool each epoch, keeping
Yu et al.'s per-epoch resampling.

Run:  python 04_extract_crops.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

import config as cfg


def _pixel_size_um(page) -> float:
    """Microns per pixel, from the TIFF's own resolution tag.

    These files are ImageJ-style pyramids whose XResolution is stored in pixels
    per 100 um; both scanner values (0.5013 and 0.4548) reproduce exactly under
    this reading, and each has been checked against the nucleus coordinate
    extents for the corresponding slide.
    """
    tag = page.tags.get("XResolution")
    if tag is None:
        raise ValueError("TIFF has no XResolution tag")
    value = tag.value
    res = float(Fraction(int(value[0]), int(value[1])))
    if res <= 0:
        raise ValueError(f"non-positive XResolution: {res}")
    return 100.0 / res


def _validate_um_per_px(um_per_px, xs_um, ys_um, width, height) -> tuple[float, bool]:
    """Check the tag against the coordinates it has to be consistent with.

    A slide's own nuclei must land inside its own image. A small number of
    files carry an XResolution that is wrong by a power of ten -- D0994 reads
    0.0501 where every other MDA slide reads 0.5013 (same numerator scale,
    denominator 2048 instead of 32768). At the tagged value every nucleus sits
    far outside the frame and the entire slide yields blank white crops.

    So the tag is trusted only when it places the nuclei inside the image; when
    it does not, it is corrected by the power of ten that does. Returns the
    resolved value and whether a correction was applied.
    """
    if len(xs_um) == 0:
        return um_per_px, False

    def fits(value: float) -> bool:
        return xs_um.max() / value <= width and ys_um.max() / value <= height

    if fits(um_per_px):
        return um_per_px, False

    for exponent in (1, 2, -1, -2):
        candidate = um_per_px * (10.0**exponent)
        if fits(candidate):
            return candidate, True

    # Nothing sensible fits; keep the tag and let the padding counter show it.
    return um_per_px, False


def _extract_slide(args) -> dict:
    wsi, image_path, xs_um, ys_um, clusters = args
    out_path = cfg.CROPS_DIR / f"{wsi}.npz"
    if out_path.exists():
        return {"wsi": wsi, "status": "cached", "n": int(len(xs_um))}

    try:
        with tifffile.TiffFile(image_path) as tf:
            page = tf.pages[0]
            um_per_px = _pixel_size_um(page)
            full = page.asarray()
    except Exception as exc:  # noqa: BLE001 - report and continue
        return {"wsi": wsi, "status": f"read_error: {exc}", "n": 0}

    height, width = full.shape[0], full.shape[1]
    tag_um_per_px = um_per_px
    um_per_px, corrected = _validate_um_per_px(um_per_px, xs_um, ys_um, width, height)

    half_px = int(round((cfg.CROP_UM / um_per_px) / 2))
    box = 2 * half_px

    xs_px = np.rint(xs_um / um_per_px).astype(int)
    ys_px = np.rint(ys_um / um_per_px).astype(int)

    crops = np.zeros((len(xs_px), cfg.CROP_PX, cfg.CROP_PX, 3), dtype=np.uint8)
    n_padded = 0
    for i, (cx, cy) in enumerate(zip(xs_px, ys_px)):
        x0, y0 = cx - half_px, cy - half_px
        x1, y1 = x0 + box, y0 + box

        # Clip to the image, then pad with white so a nucleus at the edge still
        # yields a correctly centred, correctly scaled crop.
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x1, width), min(y1, height)
        if cx1 <= cx0 or cy1 <= cy0:
            crops[i] = 255
            n_padded += 1
            continue

        patch = np.full((box, box, 3), 255, dtype=np.uint8)
        patch[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = full[cy0:cy1, cx0:cx1, :3]
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            n_padded += 1

        if box != cfg.CROP_PX:
            patch = np.asarray(
                Image.fromarray(patch).resize(
                    (cfg.CROP_PX, cfg.CROP_PX), Image.BILINEAR
                )
            )
        crops[i] = patch

    np.savez_compressed(
        out_path, crops=crops, cluster=clusters.astype(np.int16),
        um_per_px=np.float32(um_per_px)
    )
    return {"wsi": wsi, "status": "ok", "n": int(len(xs_px)),
            "um_per_px": um_per_px, "um_per_px_tag": tag_um_per_px,
            "um_per_px_corrected": corrected, "n_padded": n_padded}


def _pool_indices(clusters: np.ndarray, n_pool: int, rng) -> np.ndarray:
    """Sample up to n_pool nuclei, spread across subtypes so rare ones survive."""
    if len(clusters) <= n_pool:
        return np.arange(len(clusters))

    by_cluster = {c: np.flatnonzero(clusters == c) for c in np.unique(clusters)}
    per = max(1, n_pool // len(by_cluster))

    chosen = []
    for idx in by_cluster.values():
        take = min(per, len(idx))
        chosen.append(rng.choice(idx, size=take, replace=False))
    chosen = np.concatenate(chosen)

    # Fill any remaining budget at random from what is left.
    if len(chosen) < n_pool:
        rest = np.setdiff1d(np.arange(len(clusters)), chosen, assume_unique=False)
        extra = rng.choice(rest, size=min(n_pool - len(chosen), len(rest)), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen[:n_pool])


def _preflight() -> None:
    """The ROI pyramids are JPEG-compressed, which tifffile decodes via imagecodecs.

    Without it every slide fails identically at read time, so check once up front
    rather than after an hour of parallel failures.
    """
    try:
        import imagecodecs  # noqa: F401
    except ImportError:
        sys.exit(
            "imagecodecs is required to decode these JPEG-compressed pyramid TIFFs.\n"
            "  pip install imagecodecs   (or conda install -c conda-forge imagecodecs)"
        )


def main() -> int:
    _preflight()
    for path, hint in (
        (cfg.NUCLEI_PARQUET, "01_build_nucleus_table.py"),
        (cfg.CLUSTERS_PARQUET, "02_build_features.py"),
        (cfg.SLIDE_LABELS_CSV, "00_export_labels.R"),
    ):
        if not path.exists():
            sys.exit(f"missing {path} -- run {hint} first")

    labels = pd.read_csv(cfg.SLIDE_LABELS_CSV).set_index("wsi")

    nuclei = pd.read_parquet(
        cfg.NUCLEI_PARQUET,
        columns=["wsi", "image_file", "centroid_x_um", "centroid_y_um"],
    )
    # The study is the 777 slides in slide_labels.csv. The nucleus export also
    # carries a few slides that were excluded from the study, and
    # 02_build_features.py already drops them before clustering -- so
    # nucleus_clusters.parquet is aligned to the labelled subset, not to the
    # full nucleus table. Restrict to the same slides before attaching it.
    nuclei = nuclei[nuclei["wsi"].astype(str).isin(labels.index)].reset_index(drop=True)

    clusters = pd.read_parquet(cfg.CLUSTERS_PARQUET)
    aligned = len(clusters) == len(nuclei) and (
        clusters["wsi"].astype(str).to_numpy() == nuclei["wsi"].astype(str).to_numpy()
    ).all()
    if not aligned:
        sys.exit(
            f"{cfg.CLUSTERS_PARQUET.name} ({len(clusters):,} rows) is not aligned with "
            f"the labelled nuclei ({len(nuclei):,} rows) -- re-run 02_build_features.py"
        )
    nuclei["cluster"] = clusters["cluster"].to_numpy()

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    jobs, missing = [], []
    pool_manifest = []

    for wsi, group in nuclei.groupby("wsi", observed=True):
        wsi = str(wsi)
        if wsi not in labels.index:
            continue
        cohort = labels.loc[wsi, "dataset"]
        roi_dir = cfg.ROI_DIRS.get(cohort)
        if roi_dir is None:
            missing.append((wsi, f"unknown cohort {cohort}"))
            continue

        image_file = str(group["image_file"].iloc[0])
        image_path = roi_dir / image_file
        if not image_path.exists():
            missing.append((wsi, str(image_path)))
            continue

        sel = _pool_indices(group["cluster"].to_numpy(), cfg.POOL_PER_SLIDE, rng)
        sub = group.iloc[sel]
        jobs.append(
            (
                wsi,
                image_path,
                sub["centroid_x_um"].to_numpy(dtype=np.float64),
                sub["centroid_y_um"].to_numpy(dtype=np.float64),
                sub["cluster"].to_numpy(),
            )
        )
        # The MIL sampler needs the slide's TRUE subtype distribution, not the
        # pool's -- the pool is deliberately over-weighted toward rare subtypes.
        full_counts = np.bincount(group["cluster"].to_numpy(), minlength=cfg.N_CLUSTERS)
        pool_manifest.append(
            {"wsi": wsi, "n_pool": len(sel), "n_total": len(group),
             **{f"true_frac_c{c}": full_counts[c] / full_counts.sum()
                for c in range(cfg.N_CLUSTERS)}}
        )

    if missing:
        print(f"WARNING: {len(missing)} slides have no readable ROI image; "
              f"they will be absent from rung 3.")
        for wsi, why in missing[:10]:
            print(f"   {wsi}: {why}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")

    if not jobs:
        sys.exit("no slides with a readable ROI image -- check ROI_DIRS in config.py")

    print(f"extracting crops for {len(jobs)} slides "
          f"({cfg.CROP_UM} um -> {cfg.CROP_PX} px, pool {cfg.POOL_PER_SLIDE}/slide)")

    report = []
    with ProcessPoolExecutor(max_workers=cfg.N_JOBS) as pool:
        for i, res in enumerate(pool.map(_extract_slide, jobs, chunksize=1), start=1):
            report.append(res)
            if res["status"].startswith("read_error"):
                print(f"   {res['wsi']}: {res['status']}")
            if i % 50 == 0:
                print(f"   {i}/{len(jobs)} slides")

    rep = pd.DataFrame(report)
    rep.to_csv(cfg.INTERIM_DIR / "crop_extraction_report.csv", index=False)
    pd.DataFrame(pool_manifest).to_csv(cfg.INTERIM_DIR / "crop_pool_manifest.csv",
                                       index=False)

    ok = rep[rep["status"].isin(["ok", "cached"])]
    print(f"\ncrops written for {len(ok)}/{len(rep)} slides, "
          f"{int(ok['n'].sum()):,} nuclei total")
    if "um_per_px" in rep:
        print("micron-per-pixel values observed:")
        print(rep["um_per_px"].round(4).value_counts().to_string())
    if rep.get("um_per_px_corrected", pd.Series(dtype=bool)).any():
        bad = rep[rep["um_per_px_corrected"].fillna(False)]
        print(f"\nWARNING: {len(bad)} slide(s) had an out-of-bounds XResolution tag "
              f"and were corrected by a power of ten:")
        for _, r in bad.iterrows():
            print(f"   {r['wsi']}: tag {r['um_per_px_tag']:.4f} -> used {r['um_per_px']:.4f}")
    print(f"crops in {cfg.CROPS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
