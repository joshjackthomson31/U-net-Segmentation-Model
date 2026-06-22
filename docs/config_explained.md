# src/config.py — Explained Simply

---

## What is this file for?

`config.py` is the **single source of truth** for every setting in the project. Instead of scattering numbers like `512`, `20`, `0.0001` across dozens of files, everything is defined here once. Every other file imports from config.py.

Think of it like a **settings panel** for the entire codebase. You change one number here and it applies everywhere automatically.

---

## What does it contain?

### 1. File Paths — Where data lives on your disk

```python
BASE_DIR    = root folder of the project
DATASET_DIR = FloodNet-Supervised_v1.0/
TRAIN_IMG_DIR  = FloodNet.../train/train-org-img/    ← 1445 aerial photos
TRAIN_MASK_DIR = FloodNet.../train/train-label-img/  ← 1445 matching masks
VAL_IMG_DIR    = FloodNet.../val/val-org-img/        ← 450 photos
VAL_MASK_DIR   = FloodNet.../val/val-label-img/      ← 450 masks
TEST_IMG_DIR   = FloodNet.../test/test-org-img/      ← 448 photos
TEST_MASK_DIR  = FloodNet.../test/test-label-img/    ← 448 masks

CHECKPOINT_DIR    = results/checkpoints/   ← model weights (.pth files)
METRICS_DIR       = results/metrics/       ← JSON result files
VISUALIZATION_DIR = results/visualizations/← prediction PNG images
```

**Why compute paths this way?**
`BASE_DIR` uses `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` — this finds the project root automatically no matter which folder you run from. Avoids hardcoding `/Users/yourname/...` which breaks on other machines.

---

### 2. Dataset Constants

```python
NUM_CLASSES = 10      # FloodNet has 10 semantic classes (0-9)
IMAGE_SIZE  = (512, 512)  # All images resized to this before training
```

**Why 512×512?** The paper (Sec. IV) specifies this. It's a common size for segmentation models — large enough to preserve detail, small enough to fit in GPU memory.

```python
CLASS_NAMES = [
    "Background",           # 0 — pixels that don't belong to any specific object
    "Building Flooded",     # 1 — buildings under water
    "Building Non-Flooded", # 2 — buildings above water
    "Road Flooded",         # 3 — roads under water
    "Road Non-Flooded",     # 4 — dry roads
    "Water",                # 5 — water bodies (rivers, lakes)
    "Tree",                 # 6 — trees and vegetation
    "Vehicle",              # 7 — cars, trucks
    "Pool",                 # 8 — swimming pools
    "Grass",                # 9 — grass and open land (56% of all pixels!)
]
```

The index (0-9) is exactly what's stored in the mask PNG files as pixel values.

```python
COLOR_TO_CLASS = {
    (0, 0, 0): 0,       # Black = Background
    (255, 0, 0): 1,     # Red = Building Flooded
    ...
}
```

This maps display colors → class indices. Used ONLY for visualization (coloring prediction images), NOT for loading masks. The actual masks store the class index directly as grayscale pixel values.

---

### 3. Training Constants

```python
FULL_EPOCHS  = 20   # How many complete passes through training data in final training
PROXY_EPOCHS = 5    # How many passes during HHO's quick evaluation per hawk
OPTIMIZER    = "Adam"   # Optimizer type (from paper Table I)
NUM_WORKERS  = 4    # Parallel data loading processes
PIN_MEMORY   = True # Speeds up data transfer to GPU (warning: not supported on MPS)
```

---

### 4. HHO Search Space — What HHO is allowed to search

```python
SEARCH_SPACE = {
    "lr":           (1e-5, 1e-2),    # Learning rate: 0.00001 to 0.01
    "batch_size":   [2, 3, 4, 8],   # Only these 4 discrete options
    "dropout":      (0.1, 0.5),      # 10% to 50% dropout
    "weight_decay": (1e-6, 1e-1),   # Weight decay: 0.000001 to 0.1
}
```

These exact ranges come from the paper (Sec. IV). HHO explores within these bounds.

---

### 5. HHO Algorithm Constants

```python
HHO_POPULATION     = 20    # 20 hawks searching simultaneously
HHO_MAX_ITERATIONS = 50    # At most 50 rounds of updates
HHO_LEVY_BETA      = 1.5   # Controls Lévy flight step distribution

HHO_CONVERGENCE_THRESHOLD = 1e-4  # Stop if improvement < 0.0001
HHO_CONVERGENCE_PATIENCE  = 5     # ...for 5 consecutive iterations
```

---

### 6. GOA Baseline — Paper's best found HPs

```python
GOA_BEST_HP = {
    "lr":           10 ** (-3.02),   # = 0.000955
    "batch_size":   4,
    "dropout":      0.3,
    "weight_decay": 10 ** (-3.10),  # = 0.000794
}
```

These are the hyperparameters the paper's GOA algorithm found. Used to verify our training code matches the paper.

---

### 7. Model Backbone

```python
BACKBONE = 'resnet34'
```

Controls which model architecture is built by `build_unet()` in `unet.py`.

| Value | Model | Parameters | Notes |
|---|---|---|---|
| `'scratch'` | Original U-Net | ~7.8M | Trains from zero — needs a lot of data |
| `'resnet34'` | ResNet-34 U-Net | ~24.5M | Pre-trained ImageNet encoder ← **current** |
| `'resnet50'` | ResNet-50 U-Net | ~33M | Larger pre-trained encoder |

Every file that builds a model (`train.py`, `evaluate.py`, `main.py`) imports this constant — change it here once to switch architectures everywhere.

```python
RESNET_WEIGHTS_PATH = None
```

By default (`None`), PyTorch downloads ResNet-34 weights from `download.pytorch.org` and caches them at `~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth`. Once cached, no internet is needed.

**Walmart corporate network blocks `download.pytorch.org`.** If you get a network error, download the file manually on a personal hotspot:
```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L https://download.pytorch.org/models/resnet34-b627a593.pth \
     -o ~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
```
After this, leave `RESNET_WEIGHTS_PATH = None` — PyTorch finds the cached file automatically.

---

### 8. Device Detection

```python
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon (M1/M2/M3/M4)
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    else:
        return torch.device("cpu")   # Fallback

DEVICE = get_device()  # Runs once at import time
```

On your M4 Mac, this returns `mps` (Metal Performance Shaders — Apple's GPU framework).

---

### 9. Reproducibility

```python
SEED = 42
```

Random seed used everywhere. With the same seed, the model initializes the same way, data loads in the same order, and dropout masks are the same — so results are reproducible across runs.

---

## How other files use config.py

```python
# In dataset.py:
from src.config import TRAIN_IMG_DIR, IMAGE_SIZE, NUM_CLASSES

# In train.py:
from src.config import NUM_CLASSES, FULL_EPOCHS, SEED, CHECKPOINT_DIR, BACKBONE

# In unet.py:
from src.config import NUM_CLASSES, RESNET_WEIGHTS_PATH

# In hho_search.py:
from src.config import HHO_POPULATION, HHO_MAX_ITERATIONS, DEVICE
```

Every file reads from this one place. If you need to change the number of epochs, you change `FULL_EPOCHS = 20` here and it applies everywhere without touching any other file.
