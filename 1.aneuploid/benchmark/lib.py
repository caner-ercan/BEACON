"""
Shared helpers for the task 6 benchmark: feature-name canonicalisation,
model fitting under a patient-grouped CV, and ROC comparison (DeLong).
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import config as cfg

# --------------------------------------------------------------------------
# Feature names
# --------------------------------------------------------------------------

_ROI_PREFIX = re.compile(r"^ROI:\s*[\d.]+\s*µm per pixel:\s*")


def canonical_name(col: str) -> str:
    """QuPath column header -> a short, file-safe, deterministic name.

    'ROI: 2.00 µm per pixel: OD Sum: Haralick Contrast (F1)' -> 'OD_Sum_Haralick_Contrast_F1'
    'Area µm^2'                                              -> 'Area_um_2'
    """
    name = _ROI_PREFIX.sub("", col)
    name = name.replace("µ", "u").replace("^", "_")
    name = re.sub(r"[^0-9A-Za-z]+", "_", name)
    return name.strip("_")


def feature_columns(all_columns) -> list[str]:
    """The 61 nuclear feature columns: everything that is not metadata.

    Defined as a complement so that no feature is ever missed, and so that
    attention_score is dropped explicitly rather than by omission.
    """
    return [c for c in all_columns if c not in cfg.NON_FEATURE_COLS]


def stable_hash(text: str) -> int:
    """Process-independent hash. Python's built-in hash() is salted per run."""
    return zlib.crc32(text.encode("utf-8"))


# --------------------------------------------------------------------------
# Colour: sRGB <-> CIE-LAB, and Reinhard stain normalisation
# --------------------------------------------------------------------------
# Written out in numpy rather than pulling in scikit-image, which is not a
# declared dependency of this pipeline. D65 white point, standard sRGB transfer.

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])
_EPS = 216.0 / 24389.0        # (6/29)^3
_KAPPA = 24389.0 / 27.0       # (29/3)^3


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 or float RGB in [0,255], shape (..., 3) -> LAB float64."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _RGB_TO_XYZ.T / _WHITE_D65
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16.0) / 116.0)
    return np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """LAB float -> uint8 RGB in [0,255], shape (..., 3)."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    cube = f**3
    xyz = np.where(cube > _EPS, cube, (116.0 * f - 16.0) / _KAPPA) * _WHITE_D65
    lin = xyz @ _XYZ_TO_RGB.T
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def lab_stats(crops: np.ndarray, bg_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel LAB mean and SD over the tissue pixels of a slide's crops.

    Near-white pixels are excluded: crops are white-padded at image edges and
    contain slide background, which would otherwise dominate the statistics.
    """
    flat = np.asarray(crops).reshape(-1, 3)
    tissue = ~(flat >= bg_threshold).all(axis=1)
    if tissue.sum() < 100:          # degenerate slide: fall back to all pixels
        tissue = np.ones(len(flat), dtype=bool)
    lab = rgb_to_lab(flat[tissue])
    sd = lab.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return lab.mean(axis=0), sd


def reinhard(
    crops: np.ndarray,
    src_mean: np.ndarray,
    src_std: np.ndarray,
    tgt_mean: np.ndarray,
    tgt_std: np.ndarray,
) -> np.ndarray:
    """Map a slide's colour distribution onto a target, in LAB (Reinhard 2001)."""
    shape = crops.shape
    lab = rgb_to_lab(np.asarray(crops).reshape(-1, 3))
    lab = (lab - src_mean) / src_std * tgt_std + tgt_mean
    return lab_to_rgb(lab).reshape(shape)


# --------------------------------------------------------------------------
# Slide-level summarisation
# --------------------------------------------------------------------------


def summarise_by_slide(nuclei: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per slide, summarise every feature with the configured statistics.

    Returns one row per slide, columns named '<stat>_<feature>'.
    """
    grouped = nuclei.groupby("wsi", observed=True)[feature_cols]
    parts = []
    for stat in cfg.SUMMARY_STATS:
        if stat == "mean":
            block = grouped.mean()
        elif stat == "std":
            block = grouped.std()
        elif stat == "median":
            block = grouped.median()
        elif stat == "q25":
            block = grouped.quantile(0.25)
        elif stat == "q75":
            block = grouped.quantile(0.75)
        else:
            raise ValueError(f"unknown summary statistic: {stat}")
        block.columns = [f"{stat}_{c}" for c in block.columns]
        parts.append(block)
    return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def fit_select_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int = cfg.RANDOM_SEED,
) -> dict:
    """Tune by patient-grouped CV within the training split only, refit on the
    training split only, then score train / val / test.

    X_train must be the literal training split (e.g. DACOR's 340 MDA slides),
    not the training split plus validation slides -- hyperparameter selection,
    standardisation and the final fit all use X_train alone. X_val (the held-out
    validation slides, e.g. DACOR's 82) and X_test are scored but never fitted
    on, so all three splits are reported under the same protocol used for DACOR
    and the rung-3 deep models.
    """
    scaler = StandardScaler().fit(X_train.values)
    Xtr = scaler.transform(X_train.values)
    Xva = scaler.transform(X_val.values)
    Xte = scaler.transform(X_test.values)

    n_groups = len(np.unique(groups_train))
    n_splits = min(cfg.CV_FOLDS, n_groups)
    cv = GroupKFold(n_splits=n_splits)
    folds = list(cv.split(Xtr, y_train, groups=groups_train))

    best = {"auc": -np.inf, "C": None, "l1_ratio": None}
    cv_records = []
    for C in cfg.C_GRID:
        for l1_ratio in cfg.L1_RATIO_GRID:
            scores = []
            for tr_idx, va_idx in folds:
                if len(np.unique(y_train[tr_idx])) < 2 or len(np.unique(y_train[va_idx])) < 2:
                    continue
                model = _make_model(C, l1_ratio, seed)
                model.fit(Xtr[tr_idx], y_train[tr_idx])
                p = model.predict_proba(Xtr[va_idx])[:, 1]
                scores.append(roc_auc_score(y_train[va_idx], p))
            if not scores:
                continue
            mean_auc = float(np.mean(scores))
            cv_records.append({"C": C, "l1_ratio": l1_ratio, "cv_auc": mean_auc})
            if mean_auc > best["auc"]:
                best = {"auc": mean_auc, "C": C, "l1_ratio": l1_ratio}

    if best["C"] is None:
        raise RuntimeError("hyperparameter search produced no valid fold")

    final = _make_model(best["C"], best["l1_ratio"], seed)
    final.fit(Xtr, y_train)

    coefs = pd.Series(final.coef_.ravel(), index=X_train.columns).sort_values(
        key=np.abs, ascending=False
    )

    return {
        "model": final,
        "scaler": scaler,
        "best_C": best["C"],
        "best_l1_ratio": best["l1_ratio"],
        "cv_auc": best["auc"],
        "cv_table": pd.DataFrame(cv_records),
        "coefficients": coefs,
        "n_nonzero": int((coefs != 0).sum()),
        "prob_train": final.predict_proba(Xtr)[:, 1],
        "prob_val": final.predict_proba(Xva)[:, 1],
        "prob_test": final.predict_proba(Xte)[:, 1],
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def _make_model(C: float, l1_ratio: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=C,
        l1_ratio=l1_ratio,
        max_iter=cfg.MAX_ITER,
        random_state=seed,
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(thr[np.argmax(tpr - fpr)])


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    """AUC plus the threshold-based metrics Yu et al. report (accuracy, F1, kappa)."""
    pred = (prob >= threshold).astype(int)
    auc, lo, hi = auc_with_ci(y_true, prob)
    return {
        "auc": auc,
        "auc_lo": lo,
        "auc_hi": hi,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, pred),
        "threshold": threshold,
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def aggregate_to_patient(
    df: pd.DataFrame, prob_col: str, label_col: str = "flow", how: str = cfg.PATIENT_AGG
) -> pd.DataFrame:
    """Slide scores -> patient scores, mirroring DACOR's aggregation (max by default)."""
    agg = df.groupby("patient").agg(
        prob=(prob_col, how), label=(label_col, "max"), n_slides=(prob_col, "size")
    )
    return agg.reset_index()


def multisplit_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per split (train/val/test present in `preds`), same operating point.

    `preds` needs columns wsi, prob, flow, split. The classification threshold is
    fixed once, by Youden's index on the train split, and applied unchanged to
    val and test -- so accuracy/F1/kappa on held-out splits reflect a threshold
    chosen without seeing them, rather than each split picking its own best cut.
    A split absent from `preds` (e.g. no test rows) is simply skipped.
    """
    train = preds.loc[preds["split"] == "train"]
    if train.empty:
        raise ValueError("no train-split rows in preds -- cannot fix an operating threshold")
    thr = youden_threshold(train["flow"].to_numpy(), train["prob"].to_numpy())

    rows = []
    for split in ("train", "val", "test"):
        sub = preds.loc[preds["split"] == split]
        if sub.empty:
            continue
        rows.append({"split": split,
                     **classification_metrics(sub["flow"].to_numpy(), sub["prob"].to_numpy(), thr)})
    return pd.DataFrame(rows)


def multisplit_patient_metrics(preds: pd.DataFrame, how: str = cfg.PATIENT_AGG) -> pd.DataFrame:
    """Patient-level counterpart of multisplit_metrics.

    Aggregates slides to patients independently within each split (splits are
    patient-constrained, so no patient's slides cross a split boundary), then
    fixes the threshold on the train split's patient-level scores.
    """
    per_split = []
    for split in ("train", "val", "test"):
        sub = preds.loc[preds["split"] == split]
        if sub.empty:
            continue
        pat = aggregate_to_patient(sub, "prob", "flow", how=how)
        pat["split"] = split
        per_split.append(pat)
    if not per_split:
        return pd.DataFrame()
    long = pd.concat(per_split, ignore_index=True)

    train = long.loc[long["split"] == "train"]
    if train.empty:
        raise ValueError("no train-split rows -- cannot fix an operating threshold")
    thr = youden_threshold(train["label"].to_numpy(), train["prob"].to_numpy())

    rows = []
    for split in ("train", "val", "test"):
        sub = long.loc[long["split"] == split]
        if sub.empty:
            continue
        rows.append({"split": split,
                     **classification_metrics(sub["label"].to_numpy(), sub["prob"].to_numpy(), thr)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------
# Fast DeLong (Sun & Xu 2014). Used for AUC confidence intervals and for the
# paired test of one ROC curve against another on the same samples -- which is
# what makes the DACOR comparison valid: identical slides, identical labels.


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _structural_components(predictions: np.ndarray, n_pos: int):
    """predictions: (k, n) with positives first. Returns AUCs and DeLong covariance."""
    m, n = n_pos, predictions.shape[1] - n_pos
    pos = predictions[:, :m]
    neg = predictions[:, m:]
    k = predictions.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(predictions[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    cov = sx / m + sy / n
    return aucs, cov


def auc_with_ci(y_true: np.ndarray, prob: np.ndarray, alpha: float = 0.05):
    """AUC with a DeLong confidence interval."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    order = np.argsort(-y_true, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    if n_pos == 0 or n_pos == len(y_sorted):
        return float("nan"), float("nan"), float("nan")

    aucs, cov = _structural_components(prob[order][None, :], n_pos)
    auc = float(aucs[0])
    se = float(np.sqrt(np.maximum(cov[0, 0], 0)))
    z = stats.norm.ppf(1 - alpha / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """Paired DeLong test: is AUC(a) different from AUC(b) on the same samples?"""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())

    preds = np.vstack([np.asarray(prob_a, float)[order], np.asarray(prob_b, float)[order]])
    aucs, cov = _structural_components(preds, n_pos)

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float("nan"),
                "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": diff,
        "z": float(z),
        "p_value": p,
    }
