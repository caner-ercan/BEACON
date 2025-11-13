# Notice

This directory vendors [CellViT-plus-plus](https://github.com/TIO-IKIM/CellViT-plus-plus)
(commit [`09f90f8`](https://github.com/TIO-IKIM/CellViT-plus-plus/commit/09f90f804549d51a483600e654771e718c0d5f61)),
used for both nucleus instance segmentation and cell classification — the fine-tuning code for
both stages lives here.

## Licensing

This directory is **not** covered by this repository's root `LICENSE` (GPLv3). It retains its own
upstream license — see [`LICENSE`](LICENSE) in this directory: Apache License 2.0 for
SAM-derivative components, Apache 2.0 modified by the Commons Clause (no commercial exploitation)
for HIPT-derivative and original CellViT components.

If you use this code, cite:

- Hörst, F., Rempe, M., Heine, L., Seibold, C., Keyl, J., Baldini, G., Ugurel, S., Siveke, J.,
  Grünwald, B., Egger, J., & Kleesiek, J. (2023). CellViT: Vision Transformers for precise cell
  segmentation and classification. https://doi.org/10.48550/ARXIV.2306.15350
- Hörst, F., Rempe, M., Becker, H., Heine, L., Keyl, J., & Kleesiek, J. (2025). CellViT++:
  Energy-Efficient and Adaptive Cell Segmentation and Classification Using Foundation Models
  (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2501.05269

`cellvit/training/utils/matching_metrics.py` is vendored separately from
[stardist](https://github.com/stardist/stardist) (BSD-3-Clause).

## Local changes on top of upstream

- `cellvit/models/cell_segmentation/cellvit_256.py` — checkpoint key fixed to
  `"model_state_dict"` (was `"teacher"`)
- `cellvit/train_cell_classifier_head.py`, `cellvit/inference/*.py`,
  `cellvit/models/cell_segmentation/cellvit.py`, `cellvit/training/base_ml/*.py`,
  `cellvit/training/datasets/*.py`, `cellvit/training/evaluate/inference_cellvit_experiment_classifier.py`,
  `cellvit/training/experiments/*.py`, `cellvit/training/trainer/*.py` — adapted for BEACON's
  nucleus segmentation and cell classification fine-tuning (path normalization, dataset/training
  configuration for this study's data)
- `cellvit/pretrain_cellvit.py` — path normalization and a hardcoded W&B API key removed
