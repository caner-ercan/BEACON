import os
from pathlib import Path

BE_MASTER = os.environ.get("BE_MASTER")
if not BE_MASTER:
    raise EnvironmentError("BE_MASTER environment variable must be set to the data root")
FOUNDATION_MODEL_ROOT = os.environ.get("FOUNDATION_MODEL_ROOT")
if not FOUNDATION_MODEL_ROOT:
    raise EnvironmentError(
        "FOUNDATION_MODEL_ROOT environment variable must be set to the REMEDIS foundation "
        "model directory (base model from PhysioNet + local fine-tuned checkpoints)"
    )

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
CSV_PATH = os.path.join(BE_MASTER, "1.aneuploid/results/training_250131/sample_list_mda_12.csv")
BASE_PATH = os.path.join(BE_MASTER, "0.input/flow/roi/flow/MDA12_ROI_pyramid")
FOUNDATION_MODEL_BASE = os.path.join(FOUNDATION_MODEL_ROOT, "physionet.org/files/medical-ai-research-foundation/1.0.0/")
DATA_PATH = Path(FOUNDATION_MODEL_ROOT) / "finetuned"