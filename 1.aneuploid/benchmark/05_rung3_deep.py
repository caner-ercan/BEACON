"""
Task 6, step 5 -- rung 3: Yu et al.'s deep block and full hybrid (GPU).

Reproduces the architecture of ref 34 as closely as our data allows:

  DenseNet-121 (ImageNet-pretrained, fine-tuned end to end)
      -> 64-D per-cell embedding
      -> one-layer transformer encoder with a [CLS] pseudo-cell token
      -> 64-D attention-weighted bag embedding                        = D
  concatenated with subtype ratios (C) and per-subtype morphology (M, BatchNormed)
      -> one fully-connected layer -> sigmoid

Arms (config.DEEP_ARMS):
  "D"   -- deep features only, i.e. Yu's D-Model ablation
  "CMD" -- the full hybrid, i.e. their C+M+D

Each arm is trained once per entry in config.STAIN_NORM_VARIANTS, so the
published method (no colour processing) and a stain-normalised variant can be
reported side by side.

DenseNet-121 is deliberate: it is the backbone Yu et al. used, it is off-the-shelf
ImageNet, and it sits outside this manuscript's narrative. Substituting the
CellViT/Virchow encoder would be less faithful to ref 34 and would pull a model
introduced later in the paper into an earlier comparison.

No image augmentation is applied, because Yu et al. apply none -- their only
augmentation is the per-epoch cell resampling, which is implemented here.

Split: trains on DACOR's own train slides, early-stops on DACOR's val slides,
scores DACOR's test cohort -- so the deep baseline sees exactly the data DACOR saw.
Predictions are saved for all three splits (train/val/test, tagged by a `split`
column) so 03_fit_evaluate.py can report every model under the same protocol.

Run:  python 05_rung3_deep.py
"""

from __future__ import annotations

import gc
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import DenseNet121_Weights, densenet121

import config as cfg
import lib

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

LAB_COLS = ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std"]


# --------------------------------------------------------------------------
# Stain normalisation support
# --------------------------------------------------------------------------


def ensure_lab_stats(labels: pd.DataFrame) -> pd.DataFrame:
    """Per-slide LAB mean/SD over tissue pixels, computed once and cached.

    Reading every crop cache is a few minutes of IO, so the result is written to
    disk and reused across arms and variants.
    """
    path = cfg.INTERIM_DIR / "slide_lab_stats.csv"
    if path.exists():
        stats = pd.read_csv(path).set_index("wsi")
        print(f"loaded slide colour statistics for {len(stats)} slides from {path.name}")
        return stats

    print("computing per-slide LAB colour statistics (one pass over the crop cache)")
    rows = []
    paths = sorted(cfg.CROPS_DIR.glob("*.npz"))
    for i, p in enumerate(paths, start=1):
        crops = np.load(p)["crops"]
        mean, sd = lib.lab_stats(crops, cfg.STAIN_BG_THRESHOLD)
        rows.append({"wsi": p.stem, **dict(zip(LAB_COLS, [*mean, *sd]))})
        if i % 100 == 0:
            print(f"   {i}/{len(paths)} slides")
    stats = pd.DataFrame(rows).set_index("wsi")
    stats.to_csv(path)
    print(f"wrote {path}")
    return stats


def stain_target(stats: pd.DataFrame, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Normalisation target: the average colour of the training cohort."""
    train = labels.index[labels["dataset"] == cfg.COHORT_TRAIN]
    ref = stats.loc[stats.index.intersection(train)]
    return ref[LAB_COLS[:3]].mean().to_numpy(), ref[LAB_COLS[3:]].mean().to_numpy()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


class SlideBags(Dataset):
    """One item = one slide = a bag of CELLS_PER_BAG nucleus crops.

    Cells are resampled so the sampled subtype distribution matches the slide's
    true distribution rather than the pool's (the pool is deliberately
    over-weighted toward rare subtypes). This is Yu et al.'s distribution-
    preserving sampler, and the per-epoch resampling is their only augmentation.

    Evaluation uses the same sampler, seeded per (slide, bag index) so that draws
    are deterministic and identical across epochs -- an epoch-to-epoch change in
    the validation score then reflects the model, not resampling.
    """

    def __init__(self, wsis, labels, handcrafted, manifest, train: bool, seed: int,
                 stain_norm=None, lab_stats_df=None, target=None):
        self.wsis = list(wsis)
        self.labels = labels
        self.handcrafted = handcrafted
        self.manifest = manifest.set_index("wsi")
        self.train = train
        # Sampling mode is separate from the split: the training set is also
        # scored deterministically at the end (to fix the operating threshold),
        # so _evaluate flips this rather than relying on `train`.
        self.deterministic = not train
        self.rng = np.random.default_rng(seed)
        self.stain_norm = stain_norm
        self.lab_stats_df = lab_stats_df
        self.target = target
        self.eval_bag = 0
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.wsis)

    def _load(self, wsi: str):
        if wsi not in self._cache:
            data = np.load(cfg.CROPS_DIR / f"{wsi}.npz")
            crops, cluster = data["crops"], data["cluster"]
            if self.stain_norm == "reinhard" and wsi in self.lab_stats_df.index:
                row = self.lab_stats_df.loc[wsi]
                crops = lib.reinhard(
                    crops,
                    row[LAB_COLS[:3]].to_numpy(dtype=float),
                    row[LAB_COLS[3:]].to_numpy(dtype=float),
                    self.target[0],
                    self.target[1],
                )
            self._cache[wsi] = (crops, cluster)
        return self._cache[wsi]

    def _weights(self, wsi: str, clusters: np.ndarray) -> np.ndarray:
        row = self.manifest.loc[wsi]
        true_frac = np.array(
            [row.get(f"true_frac_c{c}", 0.0) for c in range(cfg.N_CLUSTERS)], dtype=float
        )
        pool_counts = np.bincount(clusters, minlength=cfg.N_CLUSTERS).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_cell = np.where(pool_counts > 0, true_frac / pool_counts, 0.0)
        w = per_cell[clusters]
        if w.sum() <= 0:
            w = np.ones(len(clusters))
        return w / w.sum()

    def _sample(self, wsi: str, clusters: np.ndarray) -> np.ndarray:
        take = min(cfg.CELLS_PER_BAG, len(clusters))
        w = self._weights(wsi, clusters)
        if not self.deterministic:
            return self.rng.choice(len(clusters), size=take, replace=False, p=w)
        rng = np.random.default_rng(
            (cfg.RANDOM_SEED, self.eval_bag, lib.stable_hash(wsi))
        )
        return rng.choice(len(clusters), size=take, replace=False, p=w)

    def __getitem__(self, i):
        wsi = self.wsis[i]
        crops, clusters = self._load(wsi)
        idx = self._sample(wsi, clusters)

        x = torch.from_numpy(crops[idx]).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        if x.shape[0] < cfg.CELLS_PER_BAG:  # pad short bags, masked downstream
            pad = cfg.CELLS_PER_BAG - x.shape[0]
            x = torch.cat([x, torch.zeros(pad, *x.shape[1:])], dim=0)
            mask = torch.cat([torch.zeros(len(idx)), torch.ones(pad)]).bool()
        else:
            mask = torch.zeros(cfg.CELLS_PER_BAG).bool()

        hand = (
            torch.from_numpy(self.handcrafted.loc[wsi].to_numpy(dtype=np.float32))
            if self.handcrafted is not None
            else torch.zeros(0)
        )
        return x, mask, hand, torch.tensor(float(self.labels[wsi])), wsi


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class YuHybrid(nn.Module):
    def __init__(self, n_handcrafted: int, use_handcrafted: bool):
        super().__init__()
        backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Yu et al. replace the classification head with an FC mapping 1024 -> 64.
        self.embed = nn.Linear(1024, cfg.DEEP_EMBED_DIM)

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.DEEP_EMBED_DIM))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.DEEP_EMBED_DIM,
            nhead=cfg.TRANSFORMER_HEADS,
            dim_feedforward=cfg.TRANSFORMER_FF_DIM,
            dropout=cfg.TRANSFORMER_DROPOUT,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)

        self.use_handcrafted = use_handcrafted
        head_dim = cfg.DEEP_EMBED_DIM
        if use_handcrafted:
            self.hand_norm = nn.BatchNorm1d(n_handcrafted)
            head_dim += n_handcrafted
        self.head = nn.Linear(head_dim, 1)

    def forward(self, x, mask, hand):
        b, n = x.shape[0], x.shape[1]
        flat = x.reshape(b * n, *x.shape[2:])
        f = self.pool(torch.relu(self.features(flat))).flatten(1)
        e = self.embed(f).reshape(b, n, cfg.DEEP_EMBED_DIM)

        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, e], dim=1)
        pad_mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=mask.device),
                              mask], dim=1)
        out = self.transformer(seq, src_key_padding_mask=pad_mask)
        deep = out[:, 0, :]  # the [CLS] pseudo-cell: the attention-weighted bag

        z = torch.cat([deep, self.hand_norm(hand)], dim=1) if self.use_handcrafted else deep
        return self.head(z).squeeze(1)


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------


def _run_epoch(model, loader, device, criterion, optimiser=None):
    train = optimiser is not None
    model.train(train)
    total, probs, ys, wsis = 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for x, mask, hand, y, wsi in loader:
            x, mask, hand, y = (t.to(device) for t in (x, mask, hand, y))
            logits = model(x, mask, hand)
            loss = criterion(logits, y)
            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.DEEP_GRAD_CLIP)
                optimiser.step()
            total += loss.detach().item() * len(y)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
            wsis.extend(wsi)

    return (total / max(len(loader.dataset), 1),
            np.concatenate(probs), np.concatenate(ys), wsis)


def _evaluate(model, dataset, device, criterion, n_bags: int):
    """Score a split, averaging each slide over n_bags deterministic draws."""
    loader = DataLoader(dataset, batch_size=cfg.DEEP_BATCH_SIZE, shuffle=False,
                        num_workers=0, drop_last=False)
    acc: dict[str, list[float]] = {}
    truth: dict[str, float] = {}
    losses = []
    was_deterministic = dataset.deterministic
    dataset.deterministic = True
    try:
        for bag in range(n_bags):
            dataset.eval_bag = bag
            loss, probs, ys, wsis = _run_epoch(model, loader, device, criterion)
            losses.append(loss)
            for w, p, y in zip(wsis, probs, ys):
                acc.setdefault(w, []).append(float(p))
                truth[w] = float(y)
    finally:
        dataset.deterministic = was_deterministic
    order = list(acc)
    return (float(np.mean(losses)),
            np.array([np.mean(acc[w]) for w in order]),
            np.array([truth[w] for w in order]),
            order)


def train_arm(arm: str, stain_norm, labels, handcrafted, manifest,
              lab_stats_df, target) -> dict:
    device = torch.device(cfg.DEEP_DEVICE if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    available = {p.stem for p in cfg.CROPS_DIR.glob("*.npz")}
    meta = labels.loc[labels.index.intersection(available)]

    split_train = meta[(meta["dataset"] == cfg.COHORT_TRAIN) & (meta["split"] == "train")].index
    split_val = meta[(meta["dataset"] == cfg.COHORT_TRAIN) & (meta["split"] == "val")].index
    split_test = meta[meta["dataset"] == cfg.COHORT_TEST].index

    use_hand = arm == "CMD"
    n_hand = handcrafted.shape[1] if use_hand else 0
    tag = "" if stain_norm is None else f"_{stain_norm}"
    name = f"rung3_{arm}{tag}"

    common = dict(labels=meta["flow"].to_dict(),
                  handcrafted=handcrafted if use_hand else None,
                  manifest=manifest, stain_norm=stain_norm,
                  lab_stats_df=lab_stats_df, target=target)
    ds_train = SlideBags(split_train, train=True, seed=cfg.RANDOM_SEED, **common)
    ds_val = SlideBags(split_val, train=False, seed=cfg.RANDOM_SEED, **common)
    ds_test = SlideBags(split_test, train=False, seed=cfg.RANDOM_SEED, **common)
    tr_loader = DataLoader(ds_train, batch_size=cfg.DEEP_BATCH_SIZE, shuffle=True,
                           num_workers=0, drop_last=False)

    print(f"\n=== {name} === device {device} | train {len(split_train)} "
          f"val {len(split_val)} test {len(split_test)}"
          + (f" | handcrafted {n_hand}" if use_hand else "")
          + f" | stain norm: {stain_norm or 'none (as published)'}")

    model = YuHybrid(n_hand, use_hand).to(device)
    pos = float(np.mean([meta.loc[w, "flow"] for w in split_train]))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=device)
    )
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.DEEP_LR, betas=cfg.DEEP_ADAM_BETAS,
        eps=cfg.DEEP_ADAM_EPS, weight_decay=cfg.DEEP_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.1, patience=cfg.DEEP_LR_PATIENCE
    )

    # Checkpoint selection on a centred moving average of validation AUC: keep a
    # short buffer of recent states and save the middle one of the best window.
    buf: deque = deque(maxlen=cfg.SELECTION_WINDOW)
    best = {"score": -np.inf, "epoch": -1, "state": None, "raw_auc": np.nan}
    fallback = {"auc": -np.inf, "epoch": -1, "state": None}
    history = []

    for epoch in range(cfg.DEEP_EPOCHS):
        tr_loss, _, _, _ = _run_epoch(model, tr_loader, device, criterion, optimiser)
        # One fixed draw per epoch while monitoring: deterministic and identical
        # across epochs, so differences are attributable to the model.
        va_loss, va_p, va_y, _ = _evaluate(model, ds_val, device, criterion, n_bags=1)
        va_auc = (lib.auc_with_ci(va_y, va_p)[0]
                  if len(np.unique(va_y)) > 1 else float("nan"))
        scheduler.step(va_auc if np.isfinite(va_auc) else 0.0)

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        buf.append((epoch, va_auc, state))
        if np.isfinite(va_auc) and va_auc > fallback["auc"]:
            fallback = {"auc": va_auc, "epoch": epoch, "state": state}

        smoothed = np.nan
        if len(buf) == cfg.SELECTION_WINDOW:
            scores = [b[1] for b in buf]
            if all(np.isfinite(scores)):
                smoothed = float(np.mean(scores))
                if smoothed > best["score"]:
                    mid = buf[cfg.SELECTION_WINDOW // 2]
                    best = {"score": smoothed, "epoch": mid[0],
                            "state": mid[2], "raw_auc": mid[1]}

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_auc": va_auc, "val_auc_smoothed": smoothed})
        print(f"  epoch {epoch:2d}  train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"val AUC {va_auc:.3f}  smoothed {smoothed:.3f}")

        reference = best["epoch"] if best["state"] is not None else fallback["epoch"]
        if epoch - reference >= cfg.DEEP_EARLY_STOP_PATIENCE:
            print(f"  early stop (best epoch {reference})")
            break

    if best["state"] is not None:
        chosen = best
        criterion_note = f"smoothed val AUC {best['score']:.3f}"
    elif fallback["state"] is not None:
        # Too few epochs to fill the smoothing window, or non-finite scores.
        chosen = fallback
        criterion_note = f"raw val AUC {fallback['auc']:.3f}, unsmoothed"
    else:
        # No usable validation score at any epoch: keep the final weights.
        chosen = {"epoch": len(history) - 1, "state": None}
        criterion_note = "no usable validation score; kept final epoch"

    if chosen["state"] is not None:
        model.load_state_dict(chosen["state"])
    print(f"  selected epoch {chosen['epoch']} ({criterion_note})")

    _, tr_p, tr_y, tr_w = _evaluate(model, ds_train, device, criterion, cfg.EVAL_BAGS)
    _, va_p, va_y, va_w = _evaluate(model, ds_val, device, criterion, cfg.EVAL_BAGS)
    _, te_p, te_y, te_w = _evaluate(model, ds_test, device, criterion, cfg.EVAL_BAGS)
    val_auc_final = lib.auc_with_ci(va_y, va_p)[0]

    thr = lib.youden_threshold(tr_y, tr_p)
    metrics = lib.classification_metrics(te_y, te_p, thr)
    print(f"  in-cohort (val) AUC {val_auc_final:.3f} | "
          f"cross-cohort (test) AUC {metrics['auc']:.3f} "
          f"({metrics['auc_lo']:.3f}-{metrics['auc_hi']:.3f})")

    pd.DataFrame(history).to_csv(cfg.RESULTS_DIR / f"training_history_{name}.csv",
                                 index=False)
    # Unified train/val/test predictions, matching rungs 1-2 and DACOR, so
    # 03_fit_evaluate.py can build one train/val/test table across all models.
    preds = pd.concat([
        pd.DataFrame({"wsi": tr_w, "prob": tr_p, "flow": tr_y, "split": "train"}),
        pd.DataFrame({"wsi": va_w, "prob": va_p, "flow": va_y, "split": "val"}),
        pd.DataFrame({"wsi": te_w, "prob": te_p, "flow": te_y, "split": "test"}),
    ], ignore_index=True)
    preds = preds.merge(labels[["patient"]], left_on="wsi", right_index=True, how="left")
    preds.to_csv(cfg.RESULTS_DIR / f"predictions_{name}.csv", index=False)
    torch.save(model.state_dict(), cfg.RESULTS_DIR / f"model_{name}.pt")

    result = {"name": name, "arm": arm, "stain_norm": stain_norm or "none",
              "metrics": metrics, "selected_epoch": chosen["epoch"],
              "val_auc_final": val_auc_final,
              "val_auc_smoothed": best["score"] if best["state"] is not None else np.nan}

    del ds_train, ds_val, ds_test, tr_loader, model, buf, best, fallback
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    if not any(cfg.CROPS_DIR.glob("*.npz")):
        sys.exit(f"no crops in {cfg.CROPS_DIR} -- run 04_extract_crops.py first")
    manifest_path = cfg.INTERIM_DIR / "crop_pool_manifest.csv"
    if not manifest_path.exists():
        sys.exit(f"missing {manifest_path} -- run 04_extract_crops.py first")

    labels = pd.read_csv(cfg.SLIDE_LABELS_CSV).set_index("wsi")
    manifest = pd.read_csv(manifest_path)

    hand_path = cfg.FEATURES_DIR / f"rung2_subtypes_{cfg.CLUSTER_SPACE}.parquet"
    if not hand_path.exists():
        sys.exit(f"missing {hand_path} -- run 02_build_features.py first")
    handcrafted = pd.read_parquet(hand_path)
    handcrafted.index = handcrafted.index.astype(str)
    tr = labels.index[labels["dataset"] == cfg.COHORT_TRAIN].intersection(handcrafted.index)
    mu, sd = handcrafted.loc[tr].mean(), handcrafted.loc[tr].std().replace(0, 1.0)
    handcrafted = ((handcrafted - mu) / sd).fillna(0.0)

    needs_norm = any(v is not None for v in cfg.STAIN_NORM_VARIANTS)
    lab_stats_df = ensure_lab_stats(labels) if needs_norm else None
    target = stain_target(lab_stats_df, labels) if needs_norm else None
    if needs_norm:
        print(f"stain target (training cohort): "
              f"L*={target[0][0]:.1f} a*={target[0][1]:.1f} b*={target[0][2]:.1f}")

    rows = []
    for arm in cfg.DEEP_ARMS:
        for stain_norm in cfg.STAIN_NORM_VARIANTS:
            res = train_arm(arm, stain_norm, labels, handcrafted, manifest,
                            lab_stats_df, target)
            rows.append({"model": res["name"], "arm": res["arm"],
                         "stain_norm": res["stain_norm"],
                         "selected_epoch": res["selected_epoch"],
                         "val_auc_in_cohort": res["val_auc_final"],
                         "val_auc_smoothed": res["val_auc_smoothed"],
                         **res["metrics"]})

    out = pd.DataFrame(rows)
    out.to_csv(cfg.RESULTS_DIR / "benchmark_rung3.csv", index=False)
    print("\n" + out[["model", "stain_norm", "val_auc_in_cohort",
                      "auc", "auc_lo", "auc_hi"]].to_string(index=False))
    print(f"\nwrote {cfg.RESULTS_DIR / 'benchmark_rung3.csv'}")
    print("re-run 03_fit_evaluate.py to fold rung 3 into the combined table and ROC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
