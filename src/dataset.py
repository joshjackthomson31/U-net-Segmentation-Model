"""
dataset.py — FloodNet Dataset loader for HHO-U-Net

IMPORTANT DISCOVERY (verified from actual files):
  FloodNet label masks are GRAYSCALE PNGs, NOT RGB.
  Each pixel value IS already the class index (0-9).
  No color-to-index conversion needed.

  e.g. pixel value 0 = Background
       pixel value 1 = Building Flooded
       pixel value 5 = Water
       ...etc.

  Some boundary pixels have artifact values (250-255) from the labeling tool.
  These are clamped to 0 (Background).

  The COLOR_TO_CLASS in config.py describes display colors per class
  (useful for visualization), not what the mask files store.

What this file does:
1. Loads aerial images (JPG) and grayscale masks (PNG)
2. Resizes to 512x512 (paper requirement)
3. Clamps out-of-range mask values to 0
4. Applies data augmentation during training (flip, rotate)
5. Returns PyTorch tensors ready for the model
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from src.config import (
    TRAIN_IMG_DIR, TRAIN_MASK_DIR,
    VAL_IMG_DIR,   VAL_MASK_DIR,
    TEST_IMG_DIR,  TEST_MASK_DIR,
    IMAGE_SIZE, NUM_CLASSES,
    NUM_WORKERS, PIN_MEMORY, SEED
)

# ─────────────────────────────────────────────
# IMAGE NORMALIZATION
# ─────────────────────────────────────────────

# ImageNet mean/std — standard normalization for natural image models.
# Shifts pixel values from [0,1] to be centered around 0, which helps
# neural networks train faster and more stably.
#
# Example: Red channel pixel = 0.8
#   After: (0.8 - 0.485) / 0.229 = 1.37  (centered, small number)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# ─────────────────────────────────────────────
# FLOODNET DATASET CLASS
# ─────────────────────────────────────────────

class FloodNetDataset(Dataset):
    """
    PyTorch Dataset for FloodNet semantic segmentation.

    Like a smart filing cabinet:
    - __len__     : "how many image-mask pairs do we have?"
    - __getitem__ : "give me pair #42, ready for the model"

    Args:
        img_dir  : folder with aerial images (.jpg)
        mask_dir : folder with grayscale label masks (.png)
        split    : "train" | "val" | "test"  (for logging only)
        augment  : apply random flips/rotations (True for training only)
    """

    def __init__(self, img_dir, mask_dir, split="train", augment=False):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.split    = split
        self.augment  = augment

        # Collect all image filenames (sorted for reproducibility)
        all_imgs = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Match each image to its mask.
        # FloodNet naming convention (verified from actual files):
        #   Image: "10165.jpg"  ->  Mask: "10165_lab.png"
        self.pairs = []
        for img_name in all_imgs:
            stem      = os.path.splitext(img_name)[0]      # "10165"
            mask_name = stem + "_lab.png"                   # "10165_lab.png"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.pairs.append((
                    os.path.join(img_dir, img_name),
                    mask_path
                ))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image-mask pairs found.\n"
                f"  Images : {img_dir}\n"
                f"  Masks  : {mask_dir}\n"
                f"  Expected mask format: <stem>_lab.png"
            )

        print(f"[Dataset] {split}: {len(self.pairs)} image-mask pairs loaded.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # --- Load image as RGB ---
        image = Image.open(img_path).convert("RGB")

        # --- Load mask as GRAYSCALE ---
        # Pixel values 0-9 = class indices directly (verified from actual files)
        mask = Image.open(mask_path).convert("L")

        # --- Resize to 512x512 (paper, Sec. IV) ---
        # BILINEAR for image (smooth), NEAREST for mask (preserve exact index values)
        image = image.resize(IMAGE_SIZE, Image.BILINEAR)
        mask  = mask.resize(IMAGE_SIZE,  Image.NEAREST)

        # --- Data Augmentation (training only) ---
        if self.augment:
            image, mask = self._augment(image, mask)

        # --- Convert mask to numpy and clamp artifact values ---
        mask_np = np.array(mask, dtype=np.int64)
        # Boundary artifact pixels (e.g. 250, 255) found in actual files.
        # These are labeling tool edge artifacts — map to Background (0).
        mask_np[mask_np >= NUM_CLASSES] = 0

        # --- Image -> tensor and normalize ---
        image_tensor = TF.to_tensor(image)    # (3, H, W), float32 in [0,1]
        image_tensor = normalize(image_tensor) # centered around 0

        # --- Mask -> tensor ---
        mask_tensor = torch.from_numpy(mask_np)  # (H, W), int64, values 0-9

        return image_tensor, mask_tensor

    def _augment(self, image, mask):
        """
        Apply identical random geometric transforms to both image AND mask.

        CRITICAL: Both image and mask must receive the exact same transform.
        If we flip the image left-right, the mask must flip left-right too.
        Otherwise labels end up on the wrong side of the image.

        Augmentations:
          - Horizontal flip (50% chance): mirrors image left <-> right
          - Vertical flip   (50% chance): mirrors image top <-> bottom
          - 90/180/270 rotation (50% chance): rotates scene

        NOTE: ColorJitter and scale+crop were tested and both hurt mIoU on
        FloodNet because the dataset's train/val/test share consistent color
        and scale characteristics (same drone campaign). Color/scale invariance
        is noise, not signal, for this dataset.
        """
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

        return image, mask

# ─────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────

def get_dataloaders(batch_size: int):
    """
    Create train, val, and test DataLoaders for a given batch size.

    Called by HHO during search (proxy training) and for final training.
    batch_size is one of the hyperparameters HHO tunes.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = FloodNetDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, split="train", augment=True)
    val_ds   = FloodNetDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   split="val",   augment=False)
    test_ds  = FloodNetDataset(TEST_IMG_DIR,  TEST_MASK_DIR,  split="test",  augment=False)

    # Seeded generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=g,
        drop_last=True,   # Avoid incomplete batches causing BatchNorm issues
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# CLASS WEIGHTS (for handling class imbalance)
# ─────────────────────────────────────────────

def get_class_weights(batch_size: int = 4):
    """
    Compute inverse-frequency weights across the training set.

    Why needed:
      FloodNet is severely imbalanced — Background covers ~40% of pixels,
      while Vehicle and Pool cover < 1%. Without weights, the model ignores
      rare classes and cheats by predicting Background everywhere.

    How it works:
      weight[class] = 1 / (pixel_count[class])
      Rare classes get high weight -> larger loss penalty for missing them.

    Example:
      Background: 40% of pixels -> weight = low   (common, less critical)
      Vehicle:     0.1% of pixels -> weight = high  (rare, model must learn it)

    Returns:
        FloatTensor of shape (NUM_CLASSES,)
    """
    from src.config import CLASS_NAMES

    train_ds = FloodNetDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, split="train", augment=False)
    loader   = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=False, num_workers=NUM_WORKERS)

    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    print("[Weights] Counting class pixel frequencies across training set...")

    for _, masks in loader:
        for c in range(NUM_CLASSES):
            counts[c] += (masks == c).sum().item()

    counts  = counts.clamp(min=1.0)                        # avoid div-by-zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES        # normalize: sum = NUM_CLASSES

    print("[Weights] Per-class weights:")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        pct = counts[i].item() / counts.sum().item() * 100
        print(f"  Class {i:2d} ({name:<22s}): {pct:5.2f}% pixels -> weight {w:.4f}")

    return weights.float()
