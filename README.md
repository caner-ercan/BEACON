# BEACON: Histopathology-based Spatial Profiling of Immune and Molecular Features Predicts Cancer Risk in Barrett’s Esophagus

**Authors:** Caner Ercan, Xiaoxi Pan, Thomas G. Paulson, Matthew D. Stachler, Fahire Göknur Akarca, William M. Grady, Carlo C. Maley, Yinyin Yuan

**Preprint:** [medRxiv 2025.11.11.25339952](https://doi.org/10.1101/2025.11.11.25339952)

---

## Overview

BEACON (**B**arrett **E**sophagus DNA content **A**bnormality and immune E**c**ology for **O**utcome) is a spatially aware framework that predicts DNA content abnormalities and characterizes immune spatial ecology from routine histopathology images.

Key components:
- **DACOR**: A multi-instance learning model for predicting DNA content abnormalities (aneuploidy).
- **Spatial Immune Ecology**: Tools for characterizing the spatial distribution of immune cells relative to epithelial structures.
- **Risk Stratification**: An integrated model for predicting cancer progression risk in Barrett's Esophagus patients.

[Overview](/fig1.png)

## Project Structure

```bash
.
├── 1.aneuploid/         # DACOR model and DNA content abnormality analysis
├── 2.intermediate/      # Intermediate processing (cell detection, segmentation)
├── 3.integration/       # Spatial ecology integration and risk modeling
├── 4.plotting/          # Visualization and plotting scripts
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Data Availability

The annotations and paired H&E image tiles used for training and testing the cell nucleus instance segmentation, cell classification, and tissue component segmentation models are available at zenodo upon publication of peer-reviewed paper.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/canerercan/BEACON.git
   cd BEACON
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Some R scripts used for integration analysis may require additional R packages (see scripts in `3.integration/` for details).*

## Usage

### 1. DNA Content Abnormality Prediction (DACOR)
See `1.aneuploid/mil` for training and inference scripts.
Example:
```bash
python 1.aneuploid/mil/create_heatmaps.py --config_file <config_path>
```

### 2. Spatial Analysis
See `3.integration` for R markdown notebooks used to calculate spatial metrics and integrate with clinical data.

## Citation

If you use this code or data, please cite our preprint:

```bibtex
@article{Ercan2025,
  title = {Histopathology-based Spatial Profiling of Immune and Molecular Features Predicts Cancer risk in Barrett’s Esophagus},
  author = {Ercan, Caner and Pan, Xiaoxi and Paulson, Thomas G. and Stachler, Matthew D. and Akarca, Fahire Göknur and Grady, William M. and Maley, Carlo C. and Yuan, Yinyin},
  journal = {medRxiv},
  year = {2025},
  doi = {10.1101/2025.11.11.25339952},
  url = {https://doi.org/10.1101/2025.11.11.25339952}
}
```
