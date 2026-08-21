"""
Task 6, step 3 -- fit each rung, evaluate on train/val/test, compare to DACOR.

Protocol, identical for every model (DACOR, rungs 1-2, rung 3):
  train = DACOR's literal 340-slide MDA training split -- fitting uses ONLY this.
  val   = DACOR's 82-slide MDA validation split -- scored, never fitted on.
  test  = the 355-slide NU cohort -- scored, never fitted on, DACOR's own test set.
  tuning = elastic-net logistic regression, patient-grouped 5-fold CV within train
  head   = logistic regression, matching Yu et al.'s single fully-connected layer

Earlier versions of this script fit rungs 1-2 on train+val pooled (422 MDA
slides), which both leaked validation data into hyperparameter selection and
made a val-split comparison to DACOR/rung 3 impossible. Every model is now
fitted on the training split alone and scored on all three splits, so the same
long-format table can report train/val/test consistently across models -- this
is what makes the val numbers meaningful and the comparison protocol-matched.

Because every model is scored on the same slides with the same labels as DACOR,
the DeLong test is paired and the comparison is like-for-like, split by split.

Outputs: long-format train/val/test table (the primary deliverable), a
test-only wide table kept for backward compatibility, a minimal ROC sanity
figure, and per-rung coefficients/predictions.

Run:  python 03_fit_evaluate.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import config as cfg
import lib

RUNGS = {
    "rung1_M": ("rung1_morphology.parquet", "Pooled nuclear morphology (M)"),
    "rung2_CM": (f"rung2_subtypes_{cfg.CLUSTER_SPACE}.parquet",
                 "Subtype ratios + per-subtype morphology (C+M)"),
}

RUNG3_DESCRIPTIONS = {
    "rung3_D": "DenseNet-121 + transformer MIL (D)",
    "rung3_CMD": "Full hybrid, ref 34 architecture (C+M+D)",
    "rung3_D_reinhard": "D, stain-normalised",
    "rung3_CMD_reinhard": "C+M+D, stain-normalised",
}

# "description" is a static label for the model and belongs on every split row.
# The rest describe the fitting procedure (feature count, CV, selected
# hyperparameters) and are meaningless per-split, so they appear once, on the
# train row, rather than being repeated (or blank) elsewhere.
MODEL_STATIC_COLS = ["description"]
MODEL_FIT_COLS = ["n_features", "n_nonzero", "cv_auc", "best_C", "best_l1_ratio"]


def _load_labels() -> pd.DataFrame:
    if not cfg.SLIDE_LABELS_CSV.exists():
        sys.exit(f"missing {cfg.SLIDE_LABELS_CSV} -- run 00_export_labels.R first")
    labels = pd.read_csv(cfg.SLIDE_LABELS_CSV).set_index("wsi")
    if not {"train", "val", "test"} <= set(labels["split"].unique()):
        sys.exit(f"slide_labels.csv split column must contain train/val/test, "
                 f"found: {sorted(labels['split'].unique())}")
    counts_csv = cfg.INTERIM_DIR / "nuclei_per_slide.csv"
    if counts_csv.exists():
        labels = labels.join(pd.read_csv(counts_csv).set_index("wsi"))
    return labels


def _evaluate(name: str, path, description: str, labels: pd.DataFrame) -> dict | None:
    """Fit on the training split only; return unified train/val/test predictions."""
    full = cfg.FEATURES_DIR / path
    if not full.exists():
        print(f"skipping {name}: {full} not found")
        return None

    X = pd.read_parquet(full)
    X.index = X.index.astype(str)
    meta = labels.loc[labels.index.intersection(X.index)]
    X = X.loc[meta.index]

    tr = meta["split"] == "train"
    va = meta["split"] == "val"
    te = meta["split"] == "test"

    # Any feature that is constant or missing across the TRAINING split (not
    # train+val) is dropped -- fitting must not be informed by val or test.
    usable = X.loc[tr].notna().all() & (X.loc[tr].std() > 0)
    dropped = int((~usable).sum())
    X = X.loc[:, usable]
    X = X.fillna(X.loc[tr].mean())

    print(f"\n=== {name}: {description} ===")
    print(f"features {X.shape[1]} (dropped {dropped}) | "
          f"train {int(tr.sum())} | val {int(va.sum())} | test {int(te.sum())}")

    fit = lib.fit_select_evaluate(
        X_train=X.loc[tr], y_train=meta.loc[tr, "flow"].to_numpy(),
        groups_train=meta.loc[tr, "patient"].to_numpy(),
        X_val=X.loc[va], y_val=meta.loc[va, "flow"].to_numpy(),
        X_test=X.loc[te], y_test=meta.loc[te, "flow"].to_numpy(),
    )
    print(f"selected C={fit['best_C']}, l1_ratio={fit['best_l1_ratio']} "
          f"(CV AUC, within train only: {fit['cv_auc']:.3f}); "
          f"{fit['n_nonzero']} non-zero coefficients")

    preds = pd.concat([
        pd.DataFrame({"wsi": X.loc[tr].index, "prob": fit["prob_train"],
                      "flow": fit["y_train"], "patient": meta.loc[tr, "patient"].to_numpy(),
                      "split": "train"}),
        pd.DataFrame({"wsi": X.loc[va].index, "prob": fit["prob_val"],
                      "flow": fit["y_val"], "patient": meta.loc[va, "patient"].to_numpy(),
                      "split": "val"}),
        pd.DataFrame({"wsi": X.loc[te].index, "prob": fit["prob_test"],
                      "flow": fit["y_test"], "patient": meta.loc[te, "patient"].to_numpy(),
                      "split": "test"}),
    ], ignore_index=True)
    preds.to_csv(cfg.RESULTS_DIR / f"predictions_{name}.csv", index=False)
    fit["coefficients"].to_csv(cfg.RESULTS_DIR / f"coefficients_{name}.csv",
                               header=["coefficient"])
    fit["cv_table"].to_csv(cfg.RESULTS_DIR / f"cv_grid_{name}.csv", index=False)

    split_metrics = lib.multisplit_metrics(preds)
    print(split_metrics[["split", "auc", "auc_lo", "auc_hi", "n", "n_positive"]]
          .to_string(index=False))

    return {"name": name, "description": description, "preds": preds,
            "best_C": fit["best_C"], "best_l1_ratio": fit["best_l1_ratio"],
            "cv_auc": fit["cv_auc"], "n_features": int(X.shape[1]),
            "n_nonzero": fit["n_nonzero"]}


def _load_rung3() -> list[dict]:
    """Fold in rung-3 predictions if 05_rung3_deep.py has produced them.

    Rung 3 is trained separately (GPU), so its predictions are read back from
    disk rather than refitted here. Files from before the train/val/test
    protocol fix have no 'split' column (they hold test-only predictions from a
    since-fixed evaluation-sampler bug) and are skipped rather than silently
    folded in as partial or stale rows -- see README "Revisions after the first
    run" for what was wrong with them.
    """
    out = []
    for path in sorted(cfg.RESULTS_DIR.glob("predictions_rung3_*.csv")):
        name = path.stem.replace("predictions_", "")
        preds = pd.read_csv(path)
        if "split" not in preds.columns:
            print(f"SKIPPING {name}: {path.name} has no 'split' column -- this is a "
                  f"pre-fix file (test-only, void numbers per the eval-sampler bug). "
                  f"Re-run 05_rung3_deep.py and 03_fit_evaluate.py to include it.")
            continue
        preds["wsi"] = preds["wsi"].astype(str)
        missing = {"train", "val", "test"} - set(preds["split"].unique())
        if missing:
            print(f"SKIPPING {name}: {path.name} is missing split(s) {sorted(missing)}")
            continue
        out.append({"name": name, "description": RUNG3_DESCRIPTIONS.get(name, name),
                    "preds": preds, "best_C": np.nan, "best_l1_ratio": np.nan,
                    "cv_auc": np.nan, "n_features": np.nan, "n_nonzero": np.nan})
        test_auc = lib.multisplit_metrics(preds).set_index("split").loc["test", "auc"]
        print(f"loaded {name} from {path.name} (test AUC {test_auc:.3f})")
    return out


def _long_table(entries: list[dict], metrics_fn, dacor_preds: pd.DataFrame) -> pd.DataFrame:
    """Model x split rows, with model-level constants on the train row only and
    DeLong-vs-DACOR computed per split on the slides common to both."""
    rows = []
    for e in entries:
        is_dacor = e["name"] == "DACOR"
        split_metrics = metrics_fn(e["preds"])
        for _, m in split_metrics.iterrows():
            split = m["split"]
            row = {"model": e["name"], "split": split, **m.to_dict()}
            row.update({k: e.get(k, np.nan) for k in MODEL_STATIC_COLS})
            if split == "train":
                row.update({k: e.get(k, np.nan) for k in MODEL_FIT_COLS})
            if not is_dacor:
                model_sub = e["preds"].loc[e["preds"]["split"] == split]
                dacor_sub = dacor_preds.loc[dacor_preds["split"] == split]
                common = pd.Index(model_sub["wsi"]).intersection(dacor_sub["wsi"])
                if len(common):
                    m_prob = model_sub.set_index("wsi").loc[common, "prob"]
                    d_prob = dacor_sub.set_index("wsi").loc[common, "prob"]
                    d_flow = dacor_sub.set_index("wsi").loc[common, "flow"]
                    dl = lib.delong_test(d_flow.to_numpy(), m_prob.to_numpy(), d_prob.to_numpy())
                    row["auc_diff_vs_dacor"] = dl["diff"]
                    row["delong_p_vs_dacor"] = dl["p_value"]
            rows.append(row)
    table = pd.DataFrame(rows)
    order = ["model", "split", "description", "n", "n_positive", "auc", "auc_lo", "auc_hi",
            "accuracy", "balanced_accuracy", "f1", "kappa", "threshold",
            "auc_diff_vs_dacor", "delong_p_vs_dacor", "cv_auc", "n_features",
            "n_nonzero", "best_C", "best_l1_ratio"]
    return table[[c for c in order if c in table.columns]]


def main() -> int:
    labels = _load_labels()

    dacor_preds = labels.reset_index()[["wsi", "dacor_prob", "flow", "patient", "split"]]
    dacor_preds = dacor_preds.rename(columns={"dacor_prob": "prob"}).dropna(subset=["prob"])
    dacor_metrics = lib.multisplit_metrics(dacor_preds)
    print("DACOR reference (from its own reported probabilities):")
    print(dacor_metrics[["split", "auc", "auc_lo", "auc_hi", "n", "n_positive"]]
          .to_string(index=False))
    test_auc = dacor_metrics.set_index("split").loc["test", "auc"]
    if abs(test_auc - 0.825) > 0.005:
        print(f"WARNING: DACOR test AUC {test_auc:.3f} does not match the published "
              f"0.825 -- check slide_labels.csv before trusting anything downstream.")

    results = [r for r in
               (_evaluate(n, p, d, labels) for n, (p, d) in RUNGS.items()) if r]
    results.extend(_load_rung3())
    if not results:
        sys.exit("no rung produced results -- run 02_build_features.py first")

    dacor_entry = {"name": "DACOR", "description": "MIL on tiles (this study)",
                   "preds": dacor_preds}
    all_entries = [dacor_entry] + results

    slide_table = _long_table(all_entries, lib.multisplit_metrics, dacor_preds)
    slide_table.to_csv(cfg.RESULTS_DIR / "benchmark_all_splits_slide_level.csv", index=False)

    patient_table = _long_table(all_entries, lib.multisplit_patient_metrics, dacor_preds)
    patient_table.to_csv(cfg.RESULTS_DIR / "benchmark_all_splits_patient_level.csv", index=False)

    # Test-only wide views, kept for filenames already cited in the response
    # letter / REVISION_MASTER.md -- same shape as before this fix, updated
    # values reflecting the corrected (train-only) fitting protocol.
    def _wide_test(long_table: pd.DataFrame) -> pd.DataFrame:
        return long_table.loc[long_table["split"] == "test"].drop(columns="split")

    _wide_test(slide_table).to_csv(cfg.RESULTS_DIR / "benchmark_slide_level.csv", index=False)
    _wide_test(patient_table).to_csv(cfg.RESULTS_DIR / "benchmark_patient_level.csv", index=False)

    # ---- ROC figure (test split only; sanity check, not the response figure) --
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    dacor_test = dacor_preds.loc[dacor_preds["split"] == "test"]
    fpr, tpr, _ = roc_curve(dacor_test["flow"], dacor_test["prob"])
    dacor_test_auc = dacor_metrics.set_index("split").loc["test", "auc"]
    ax.plot(fpr, tpr, lw=2.2, color="#B2182B", label=f"DACOR (AUC {dacor_test_auc:.3f})")
    palette = ["#2166AC", "#4D9221", "#762A83", "#E08214", "#01665E", "#8C510A"]
    for r, colour in zip(results, palette):
        sub = r["preds"].loc[r["preds"]["split"] == "test"]
        fpr, tpr, _ = roc_curve(sub["flow"], sub["prob"])
        auc = lib.multisplit_metrics(r["preds"]).set_index("split").loc["test", "auc"]
        style = "--" if r["name"].endswith("_reinhard") else "-"
        ax.plot(fpr, tpr, lw=1.8, color=colour, ls=style,
                label=f"{r['name']} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.6")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("DNA content abnormality prediction, test cohort")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(cfg.FIG_DIR / f"roc_benchmark.{ext}", dpi=300)

    print("\n=== combined slide-level table (train/val/test) ===")
    print(slide_table[["model", "split", "auc", "auc_lo", "auc_hi",
                       "delong_p_vs_dacor"]].to_string(index=False))
    print(f"\nwrote {cfg.RESULTS_DIR / 'benchmark_all_splits_slide_level.csv'} "
          f"and {cfg.RESULTS_DIR / 'benchmark_all_splits_patient_level.csv'}")
    print(f"wrote test-only compatibility views to {cfg.RESULTS_DIR}")
    print(f"wrote figures to {cfg.FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
