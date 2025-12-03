import os
from pathlib import Path

# Configuration parameters
ALGORITHM = "simclr"
IMG_SIZE = 224
INIT_LR = 1e-3
TEMPERATURE = 0.5
BATCH_SIZE = 64
PRE_TRAIN_EPOCHS = 20

VAL_STEPS_PER_EPOCH = 20
WEIGHT_DECAY = 5e-4
DIM = 1024
WARMUP_LR = 0.0
WARMUP_STEPS = 0

# Paths
CSV_PATH = "/rsrch5/home/trans_mol_path/cercan/BE_master/1.aneuploid/results/training_250131/sample_list_mda_12.csv"
BASE_PATH = "/rsrch5/home/trans_mol_path/cercan/BE_master/0.input/flow/roi/flow/MDA12_ROI_pyramid"
FOUNDATION_MODEL_BASE = '/rsrch5/home/trans_mol_path/cercan/foundationModels/physionet.org/files/medical-ai-research-foundation/1.0.0/'
DATA_PATH = Path('/rsrch5/home/trans_mol_path/cercan/foundationModels/finetuned/')