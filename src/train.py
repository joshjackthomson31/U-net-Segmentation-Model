"""
train.py — Training utilities for HHO-U-Net on FloodNet

Two entry points:
  proxy_train(hps, class_weights, device) -> float
      Fast 5-epoch training used by HHO for each hawk evaluation.
      Returns final validation mIoU (the fitness score HHO maximizes).

  full_train(hps, class_weights, device) -> dict
      Full 20-epoch training with the best hyperparameters found by HHO.
      Saves best checkpoint (by val mIoU) and final checkpoint.
      Returns epoch-by-epoch history dict.

Why confusion-matrix mIoU (not per-batch IoU average)?
  Per-batch averaging gives equal weight to each batch regardless of class presence.
  Rare classes (Vehicle <0.1%, Pool <1%) are often absent from individual batches,
  so their IoU = 0 for many batches and the final average is biased downward.
  Accumulating a global (10×10) confusion matrix across the FULL val set computes
  IoU over all pixels simultaneously — this matches the paper's reported metrics.

Why seed inside proxy_train?
  HHO caches (HP_key -> mIoU) to avoid re-evaluating identical hawk positions.
  Without seeding, the same HPs would return different mIoU each time,
  breaking cache coherence. Seeding ensures determinism.

Why class_weights as argument (not computed here)?
  get_class_weights() scans all 1445 training images — it takes ~30 seconds.
  If called inside proxy_train, this would happen for every single hawk evaluation
  (20 hawks × 50 iterations = up to 1000 calls). Compute once in hho_search.py,
  pass here.
"""

import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn

from src.config import (
    NUM_CLASSES, PROXY_EPOCHS, FULL_EPOCHS, SEED,
    CHECKPOINT_DIR, METRICS_DIR, DEVICE, BACKBONE,
)
from src.unet    import build_unet
from src.dataset import get_dataloaders


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────

def _seed_everything(seed: int = SEED) -> None:
    """
    Fix all sources of randomness so proxy_train is deterministic.

    Why all three?
      - random     : Python's built-in random (used by some augmentations)
      - np.random  : NumPy's random (used by HHO's Lévy flight)
      - torch      : model weight init, dropout masks, DataLoader shuffling

    Without this, two calls with the same hyperparameters return different mIoU,
    which breaks the HHO evaluation cache.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # MPS does not support torch.cuda.manual_seed_all — skip silently
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# CONFUSION MATRIX  (global mIoU accumulation)
# ─────────────────────────────────────────────

def _update_confusion_matrix(
    cm:         torch.Tensor,   # shape (num_classes, num_classes), dtype=torch.int64
    preds:      torch.Tensor,   # shape (B, H, W), predicted class per pixel
    masks:      torch.Tensor,   # shape (B, H, W), ground-truth class per pixel
    num_classes: int,
) -> None:
    """
    Accumulate predictions into a global confusion matrix IN-PLACE.

    cm[true_class, pred_class] += number_of_pixels_with_that_combination

    Why bincount?
      It's much faster than nested for-loops over classes.
      We flatten both preds and masks to 1D, then compute:
        index = true_class * num_classes + pred_class
        bincount gives count of each (true, pred) pair in one pass.

    Example (simplified, 3 classes):
      true  = [0, 1, 2, 0, 1]
      pred  = [0, 1, 1, 2, 1]
      index = [0*3+0, 1*3+1, 2*3+1, 0*3+2, 1*3+1]
            = [0, 4, 7, 2, 4]
      bincount(index, min=9) → [1, 0, 1, 0, 2, 0, 0, 1, 0]
      reshape(3,3) → cm:
        [[1, 0, 1],   ← row 0: true=0, predicted as 0, 0, 2
         [0, 2, 0],   ← row 1: true=1, predicted as 1, 1
         [0, 1, 0]]   ← row 2: true=2, predicted as 1
    """
    # Flatten to 1D for bincount
    preds_flat = preds.view(-1).long()
    masks_flat = masks.view(-1).long()

    # Filter invalid mask values (artifact pixels clamped in dataset.py, but be safe)
    valid = (masks_flat >= 0) & (masks_flat < num_classes)
    preds_flat = preds_flat[valid]
    masks_flat = masks_flat[valid]

    # Encode (true, pred) pair as single integer
    combined = masks_flat * num_classes + preds_flat

    # Count each combination; accumulate into cm
    cm += torch.bincount(combined, minlength=num_classes * num_classes) \
               .reshape(num_classes, num_classes)


def _miou_from_confusion_matrix(
    cm:          torch.Tensor,   # shape (num_classes, num_classes), accumulated
    num_classes: int,
) -> float:
    """
    Compute mean Intersection-over-Union from the full confusion matrix.

    For each class c:
      TP (true positives)  = cm[c, c]                 ← pixels correctly predicted as c
      FP (false positives) = cm[:, c].sum() - cm[c,c] ← pixels wrongly predicted as c
      FN (false negatives) = cm[c, :].sum() - cm[c,c] ← pixels of class c predicted as other

      IoU_c = TP / (TP + FP + FN)
            = cm[c,c] / (cm[c,:].sum() + cm[:,c].sum() - cm[c,c])

      union = cm[c,:].sum() + cm[:,c].sum() - cm[c,c]
            = (total pixels predicted as c) + (total pixels that are c) - (cm[c,c] counted twice)

    Classes with zero union (absent from both predictions AND ground truth) are skipped.
    Mean is taken over present classes only — this matches standard segmentation evaluation.

    Example (3 classes, class 2 absent):
      cm = [[100, 5,  0],
            [10, 80,  0],
            [0,   0,  0]]
      IoU_0 = 100 / (105 + 110 - 100) = 100/115 ≈ 0.870
      IoU_1 =  80 / ( 90 +  85 -  80) =  80/ 95 ≈ 0.842
      IoU_2:  union = 0 → skip
      mIoU = (0.870 + 0.842) / 2 ≈ 0.856
    """
    ious = []
    cm_f = cm.float()

    for c in range(num_classes):
        tp    = cm_f[c, c]
        union = cm_f[c, :].sum() + cm_f[:, c].sum() - tp
        if union > 0:
            ious.append((tp / union).item())

    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────────
# SINGLE-EPOCH TRAINING
# ─────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """
    Train for one epoch. Returns average training loss over all batches.

    Standard forward-backward-step loop:
      1. Zero gradients (reset accumulated gradients from last batch)
      2. Forward pass: model predicts logits from images
      3. Loss: compare logits to ground-truth masks
      4. Backward pass: compute gradients of loss w.r.t. all parameters
      5. Optimizer step: update parameters using gradients

    Note: No AMP (Automatic Mixed Precision) — MPS does not support it.

    Args:
        model     : U-Net in training mode
        loader    : training DataLoader
        optimizer : Adam optimizer
        criterion : CrossEntropyLoss (with class weights)
        device    : MPS / CUDA / CPU

    Returns:
        Average loss across all training batches (float)
    """
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)               # (B, 3, 512, 512)
        masks  = masks.to(device).long()         # (B, 512, 512)  — class indices

        optimizer.zero_grad()

        logits = model(images)                   # (B, 10, 512, 512) raw logits
        loss   = criterion(logits, masks)        # CrossEntropyLoss applies softmax internally

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    num_classes: int = NUM_CLASSES,
) -> tuple:
    """
    Run full validation set through the model. Returns (avg_loss, mIoU).

    Uses accumulated confusion matrix for correct mIoU:
      - Process every batch, accumulate into a single (10×10) cm
      - Compute mIoU once from the full cm at the end
      This avoids per-batch averaging bias (see module docstring).

    Args:
        model       : U-Net
        loader      : validation DataLoader
        criterion   : CrossEntropyLoss
        device      : device
        num_classes : 10 for FloodNet

    Returns:
        (avg_val_loss: float, mIoU: float)
    """
    model.eval()
    total_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device).long()

            logits = model(images)                          # (B, 10, 512, 512)
            loss   = criterion(logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)                    # (B, 512, 512) — class per pixel

            # Move to CPU for accumulation (cm lives on CPU)
            _update_confusion_matrix(cm, preds.cpu(), masks.cpu(), num_classes)

    avg_loss = total_loss / len(loader)
    miou     = _miou_from_confusion_matrix(cm, num_classes)

    return avg_loss, miou


# ─────────────────────────────────────────────
# PROXY TRAINING  (HHO fitness function)
# ─────────────────────────────────────────────

def proxy_train(
    hps:          dict,           # {"lr": float, "batch_size": int, "dropout": float, "weight_decay": float}
    class_weights: torch.Tensor,  # shape (10,) — computed ONCE in hho_search.py
    device:        torch.device,
) -> float:
    """
    5-epoch training to estimate mIoU for a given hyperparameter set.

    Called by HHO for each hawk position evaluation.
    Returns final validation mIoU — this is the fitness value HHO maximizes.

    Why 5 epochs?
      Full training (20 epochs) would make HHO impossibly slow:
        20 hawks × 50 iterations × 20 epochs = 20,000 epoch-equivalents.
      5 proxy epochs (= 5,000 epoch-equivalents) gives enough signal to rank
      HP sets while remaining tractable. Standard practice in HPO literature.

    Why seed here?
      HHO caches: hp_key -> mIoU. Same HPs must always return the same score.
      Without seeding, random dropout masks and augmentations cause variance,
      making cache hits unreliable.

    Args:
        hps           : decoded hyperparameters from HHO._decode()
        class_weights : inverse-frequency class weights for CrossEntropyLoss
        device        : MPS / CUDA / CPU

    Returns:
        Validation mIoU after PROXY_EPOCHS=5 training epochs (float in [0, 1])
    """
    # CRITICAL: seed before EVERYTHING — model init, dataloader shuffle, dropout
    _seed_everything(SEED)

    # Build model with this hawk's dropout (uses BACKBONE from config)
    model = build_unet(dropout=hps["dropout"], backbone=BACKBONE).to(device)

    # DataLoaders with this hawk's batch_size
    train_loader, val_loader, _ = get_dataloaders(batch_size=hps["batch_size"])

    # Loss: weighted CrossEntropyLoss — handles class imbalance
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    # Optimizer: Adam with this hawk's lr and weight_decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hps["lr"],
        weight_decay=hps["weight_decay"],
    )

    # Train for PROXY_EPOCHS (5) — fast fitness estimate
    for epoch in range(PROXY_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    # Return final validation mIoU (HHO maximizes this)
    _, miou = validate(model, val_loader, criterion, device)

    # Free GPU/MPS memory before next hawk evaluation
    del model, train_loader, val_loader, criterion, optimizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return miou


# ─────────────────────────────────────────────
# FULL TRAINING  (final model after HHO search)
# ─────────────────────────────────────────────

def full_train(
    hps:           dict,           # best HPs from HHO search
    class_weights: torch.Tensor,   # shape (10,) — same tensor from hho_search.py
    device:        torch.device,
) -> dict:
    """
    Full 20-epoch training with the best hyperparameters found by HHO.

    Saves:
      CHECKPOINT_DIR/best_model.pth    — weights at epoch with highest val mIoU
      CHECKPOINT_DIR/final_model.pth   — weights at last epoch
      METRICS_DIR/train_metrics.json   — epoch-by-epoch history dict

    Args:
        hps           : best hyperparameters from HHO.run()
        class_weights : inverse-frequency class weights for CrossEntropyLoss
        device        : MPS / CUDA / CPU

    Returns:
        history: dict with keys:
          "train_loss"  : list of avg training loss per epoch   (len=20)
          "val_loss"    : list of avg validation loss per epoch  (len=20)
          "val_miou"    : list of validation mIoU per epoch      (len=20)
          "best_epoch"  : epoch index (0-based) with highest val mIoU
          "best_miou"   : highest validation mIoU achieved
          "hps"         : the hyperparameters used
    """
    _seed_everything(SEED)

    # Ensure output directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR,    exist_ok=True)

    print(f"\n[FullTrain] Starting {FULL_EPOCHS}-epoch training with HPs:")
    for k, v in hps.items():
        print(f"  {k}: {v}")
    print()

    # Build model (uses BACKBONE from config)
    model = build_unet(dropout=hps["dropout"], backbone=BACKBONE).to(device)

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=hps["batch_size"])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hps["lr"],
        weight_decay=hps["weight_decay"],
    )

    # Cosine annealing scheduler: smoothly decays LR from hps["lr"] → 1e-6.
    # Prevents the LR from staying fixed and causing loss oscillation in later epochs.
    # T_max = total epochs → one full cosine cycle over the entire training run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=FULL_EPOCHS,
        eta_min=1e-6,
    )

    # History tracking
    history = {
        "train_loss": [],
        "val_loss":   [],
        "val_miou":   [],
        "lr":         [],
        "best_epoch": 0,
        "best_miou":  0.0,
        "hps":        hps,
        "backbone":   BACKBONE,
    }
    best_miou   = 0.0
    best_epoch  = 0

    for epoch in range(FULL_EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # ── Train ──────────────────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # ── Step scheduler AFTER each epoch ────────────────────
        scheduler.step()

        # ── Validate ───────────────────────────────────────────
        val_loss, val_miou = validate(model, val_loader, criterion, device)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        print(
            f"[Epoch {epoch+1:02d}/{FULL_EPOCHS}] "
            f"lr={current_lr:.2e}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_mIoU={val_miou:.4f}"
        )

        # ── Save best checkpoint ────────────────────────────────
        if val_miou > best_miou:
            best_miou  = val_miou
            best_epoch = epoch
            torch.save(
                {
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_miou":   val_miou,
                    "hps":        hps,
                    "backbone":   BACKBONE,
                },
                os.path.join(CHECKPOINT_DIR, "best_model.pth"),
            )
            print(f"  → New best mIoU: {best_miou:.4f}  (saved best_model.pth)")

    # Save final checkpoint
    torch.save(
        {
            "epoch":      FULL_EPOCHS - 1,
            "state_dict": model.state_dict(),
            "val_miou":   history["val_miou"][-1],
            "hps":        hps,
            "backbone":   BACKBONE,
        },
        os.path.join(CHECKPOINT_DIR, "final_model.pth"),
    )
    print(f"\n[FullTrain] Saved final_model.pth")
    print(f"[FullTrain] Best mIoU: {best_miou:.4f} at epoch {best_epoch + 1}")

    # Update history with best result
    history["best_epoch"] = best_epoch
    history["best_miou"]  = best_miou

    # Save metrics to JSON
    metrics_path = os.path.join(METRICS_DIR, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[FullTrain] Metrics saved to {metrics_path}")

    return history


# ─────────────────────────────────────────────
# QUICK SANITY CHECK (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Smoke-test proxy_train with GOA_BEST_HP (paper baseline settings).
    Requires FloodNet data at the path in config.py.
    """
    from src.config  import GOA_BEST_HP, DEVICE
    from src.dataset import get_class_weights

    print(f"Device: {DEVICE}")

    # Compute class weights once (as done in hho_search.py)
    print("Computing class weights (scans training set — takes ~30s)...")
    class_weights = get_class_weights()

    # Test proxy_train with paper's GOA baseline HPs
    print(f"\nRunning proxy_train with GOA_BEST_HP: {GOA_BEST_HP}")
    miou = proxy_train(GOA_BEST_HP, class_weights, DEVICE)
    print(f"\nProxy mIoU after {PROXY_EPOCHS} epochs: {miou:.4f}")
    print("Sanity check PASSED." if 0.0 <= miou <= 1.0 else "ERROR: mIoU out of range!")
