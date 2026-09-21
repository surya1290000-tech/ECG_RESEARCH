"""
Project-wide configuration.
============================

All paths, hyperparameters, and constants are centralized here.
Modify this file when adapting to a new environment.

SCIENTIFIC NOTE
---------------
These values were used at training time and MUST NOT be silently changed.
Any change to CLASS_NAMES, SAMPLING_RATE, SIGNAL_LENGTH, or BATCH_SIZE
invalidates checkpoint compatibility and requires retraining.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_ecg_model.pth"

# PTB-XL dataset must be placed here before dataset-dependent scripts work
DATA_DIR = PROJECT_ROOT / "data" / "ptbxl"
METADATA_CSV = DATA_DIR / "ptbxl_database.csv"
SCP_STATEMENTS_CSV = DATA_DIR / "scp_statements.csv"
RECORDS_DIR = DATA_DIR / "records100"   # 100 Hz WFDB files

RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create output directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset — PTB-XL v1.0.3
# ---------------------------------------------------------------------------

PTBXL_VERSION = "1.0.3"
PTBXL_URL = "https://physionet.org/content/ptb-xl/1.0.3/"

SAMPLING_RATE = 100         # Hz — 100 Hz (lr) files used, NOT 500 Hz
SIGNAL_LENGTH = 1000        # samples = 10 s * 100 Hz
NUM_LEADS = 12

# These 5 are the PTB-XL diagnostic superclasses used during training
# ORDER MATTERS for checkpoint compatibility
CLASS_NAMES = ["NORM", "STTC", "CD", "MI", "HYP"]
CLASS_TO_ID  = {
    "NORM": 0,
    "STTC": 1,
    "CD": 2,
    "MI": 3,
    "HYP": 4,
}
ID_TO_CLASS  = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES  = len(CLASS_NAMES)  # 5


# ---------------------------------------------------------------------------
# Data splitting (mirrors the notebook GroupShuffleSplit methodology)
# ---------------------------------------------------------------------------

SPLIT_RANDOM_STATE = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10   # remainder after train+val
# Patient-level split on 'patient_id' column — no patient leakage


# ---------------------------------------------------------------------------
# Training hyperparameters — DO NOT change without explicit justification
# ---------------------------------------------------------------------------
# These were the ACTIVE settings when best_ecg_model.pth was produced.
# The model was trained for 20 epochs, single-label CrossEntropyLoss.

BATCH_SIZE   = 16
NUM_EPOCHS   = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
OPTIMIZER     = "Adam"      # torch.optim.Adam
LOSS          = "CrossEntropyLoss"  # nn.CrossEntropyLoss — SINGLE-LABEL

# IMPORTANT: The checkpoint was trained in SINGLE-LABEL mode.
# The notebook also contains multi-label/BCE code, but those cells
# were NOT used to produce the saved checkpoint.
# DO NOT assume multi-label results are valid for the existing checkpoint.
TRAINING_MODE = "single_label"  # "single_label" | "multi_label"


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

# See src/models/ecg_resnet.py for the full definition.
# Version B confirmed by strict=True load in Colab execution_count 21.
MODEL_VERSION = "ECGResNet_VersionB"

# Expected parameter count (for sanity checking)
# Approximate: stem + 4 blocks + classifier
# Exact value is computed at runtime by count_parameters()
APPROXIMATE_PARAM_COUNT = 3_800_000  # ~3.8 M params


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

DECISION_THRESHOLD = 0.50   # for future multi-label experiments
NUM_TEST_SAMPLES_FOR_REPR_ANALYSIS = 20   # Step 20 analysis batch size


# ---------------------------------------------------------------------------
# Representation diversity analysis (Step 20)
# ---------------------------------------------------------------------------

REPR_ANALYSIS_LAYERS = [
    "stem",
    "block1",
    "block2",
    "block3",
    "block4",
    "pool",
]
REPR_ANALYSIS_OUTPUT = RESULTS_DIR / "representation_diversity.csv"
