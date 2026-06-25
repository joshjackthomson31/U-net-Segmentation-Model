"""
config.py — Central configuration for HHO-U-Net on FloodNet
All values sourced directly from the InGARSS 2025 paper unless noted.
"""

import os
import torch

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "FloodNet-Supervised_v1.0")

TRAIN_IMG_DIR   = os.path.join(DATASET_DIR, "train", "train-org-img")
TRAIN_MASK_DIR  = os.path.join(DATASET_DIR, "train", "train-label-img")
VAL_IMG_DIR     = os.path.join(DATASET_DIR, "val",   "val-org-img")
VAL_MASK_DIR    = os.path.join(DATASET_DIR, "val",   "val-label-img")
TEST_IMG_DIR    = os.path.join(DATASET_DIR, "test",  "test-org-img")
TEST_MASK_DIR   = os.path.join(DATASET_DIR, "test",  "test-label-img")

RESULTS_DIR         = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR      = os.path.join(RESULTS_DIR, "checkpoints")
METRICS_DIR         = os.path.join(RESULTS_DIR, "metrics")
VISUALIZATION_DIR   = os.path.join(RESULTS_DIR, "visualizations")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
NUM_CLASSES  = 10
IMAGE_SIZE   = (512, 512)   # All images resized to 512x512 (paper, Sec. IV)

# 10 semantic classes in FloodNet (paper, Sec. III)
CLASS_NAMES = [
    "Background",           # 0
    "Building Flooded",     # 1
    "Building Non-Flooded", # 2
    "Road Flooded",         # 3
    "Road Non-Flooded",     # 4
    "Water",                # 5
    "Tree",                 # 6
    "Vehicle",              # 7
    "Pool",                 # 8
    "Grass",                # 9
]

# RGB color -> class index mapping
# Source: ColorPalette-Values.xlsx in ColorMasks-FloodNetv1.0/
#
# NOTE: FloodNet mask files are GRAYSCALE (pixel value = class index directly).
# This COLOR_TO_CLASS dict is used only for VISUALIZATION (overlaying colored
# class labels on images), NOT for mask loading in dataset.py.
COLOR_TO_CLASS = {
    (0,   0,   0):   0,   # Background
    (255, 0,   0):   1,   # Building Flooded
    (180, 120, 120): 2,   # Building Non-Flooded
    (160, 150, 20):  3,   # Road Flooded
    (140, 140, 140): 4,   # Road Non-Flooded
    (61,  230, 250): 5,   # Water
    (0,   82,  255): 6,   # Tree
    (255, 0,   245): 7,   # Vehicle
    (255, 235, 0):   8,   # Pool
    (4,   250, 7):   9,   # Grass
}

# ─────────────────────────────────────────────
# TRAINING (exact from paper, Sec. IV)
# ─────────────────────────────────────────────
FULL_EPOCHS     = 50        # Final training epochs (increased from paper's 20 for better convergence)
WARMUP_EPOCHS   = 5         # Linear LR warmup before cosine decay (stabilizes early training)
PROXY_EPOCHS    = 5         # Fast evaluation during HHO search (standard practice, not in paper)
OPTIMIZER       = "Adam"    # All models use Adam (Table I)
NUM_WORKERS     = 4         # DataLoader workers
PIN_MEMORY      = True      # Faster GPU/MPS transfers

# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH SPACE (exact from paper, Sec. IV)
# ─────────────────────────────────────────────
SEARCH_SPACE = {
    "lr":           (1e-5, 1e-2),       # Learning rate range [10^-5, 10^-2]
    "batch_size":   [2, 3, 4, 8],       # Discrete options {2, 3, 4, 8}
    "dropout":      (0.1, 0.5),         # Dropout range [0.1, 0.5]
    "weight_decay": (1e-6, 1e-1),       # Weight decay range [10^-6, 10^-1]
}

# ─────────────────────────────────────────────
# HHO ALGORITHM (Heidari et al. 2019)
# Population & iterations from paper (Sec. IV); equations from HHO paper
# ─────────────────────────────────────────────
HHO_POPULATION      = 20       # Number of hawks (paper: "population size of 20")
HHO_MAX_ITERATIONS  = 50       # Max search iterations (paper: "up to 50 iterations")
HHO_LEVY_BETA       = 1.5      # Levy flight exponent beta (HHO paper, Eq. 14)

# Convergence: stop early if avg improvement < threshold for N consecutive iterations
HHO_CONVERGENCE_THRESHOLD   = 1e-4     # (paper: "average improvement below 10^-4")
HHO_CONVERGENCE_PATIENCE    = 5        # (paper: "5 consecutive iterations")

# ─────────────────────────────────────────────
# GOA-U-NET BASELINE (Table I, paper) -- for comparison only
# ─────────────────────────────────────────────
GOA_BEST_HP = {
    "lr":           10 ** (-3.02),  # ~9.55e-4
    "batch_size":   4,
    "dropout":      0.3,
    "weight_decay": 10 ** (-3.10),  # ~7.94e-4
}

# ─────────────────────────────────────────────
# MODEL BACKBONE
# ─────────────────────────────────────────────
# 'scratch'  : original from-scratch U-Net (~7M params, base_filters=32)
# 'resnet34' : pre-trained ResNet-34 encoder + U-Net decoder (~24M params, ImageNet weights)
#
# NOTE: 'resnet34' requires ImageNet normalization in dataset.py (already applied).
# GOA_BEST_HP was tuned for the from-scratch model; HHO re-tuning recommended after switching.
BACKBONE = 'resnet34'

# Path to a locally downloaded ResNet-34 weights file.
# Set this to the .pth file path after downloading manually (see instructions below).
# Download command (run on personal hotspot / home WiFi — Walmart network blocks pytorch.org):
#   mkdir -p ~/.cache/torch/hub/checkpoints
#   curl -L https://download.pytorch.org/models/resnet34-b627a593.pth \
#        -o ~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
# Once downloaded, leave RESNET_WEIGHTS_PATH = None — torch.hub caches it automatically.
RESNET_WEIGHTS_PATH = None   # e.g. "/Users/you/.cache/torch/hub/checkpoints/resnet34-b627a593.pth"

# ─────────────────────────────────────────────
# DEVICE (Apple M4 Pro -> MPS; fallback to CPU)
# ─────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
