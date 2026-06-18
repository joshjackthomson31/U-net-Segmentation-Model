"""
evaluate.py — Full evaluation of the trained U-Net on FloodNet TEST set

What this file does:
  1. Loads best_model.pth (saved by full_train)
  2. Runs the model on the TEST set (not val — val was used during training)
  3. Computes all metrics needed to fill paper Table II:
       - mIoU                    (primary metric)
       - Per-class IoU           (all 10 classes, 0.0 if class absent)
       - Per-class Recall        (how well each class is detected)
       - Dice coefficient        (per-class and mean)
       - Pixel accuracy          (% of correctly labeled pixels overall)
  4. Prints a formatted table
  5. Saves metrics to results/metrics/test_metrics.json
  6. Optionally saves side-by-side visualizations (image | truth | prediction)

Why TEST set, not VAL?
  During training, we used val mIoU to pick the best checkpoint.
  If we report val metrics now, those metrics influenced model selection —
  so they are optimistically biased. Test set is held out completely.

Why confusion matrix (not per-batch average)?
  Same reason as train.py: rare classes (Vehicle <0.1%, Pool <1%) are often
  absent from individual batches. Per-batch averaging gives IoU=0 for those
  batches and pulls down the class score unfairly.
  Accumulated confusion matrix computes IoU over ALL test pixels at once.

Per-class Recall vs Precision:
  Recall (producer's accuracy):  cm[c,c] / cm[c,:].sum()
    = "of all pixels that ARE class c, how many did we find?"
  Precision (user's accuracy):   cm[c,c] / cm[:,c].sum()
    = "of all pixels we CALLED class c, how many actually are?"
  We report Recall — standard in segmentation papers.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

from src.config  import (
    NUM_CLASSES, CLASS_NAMES, COLOR_TO_CLASS,
    CHECKPOINT_DIR, METRICS_DIR, VISUALIZATION_DIR,
)
from src.unet    import build_unet
from src.dataset import get_dataloaders

# ─────────────────────────────────────────────
# CLASS COLOR PALETTE  (for visualization)
# ─────────────────────────────────────────────

# Invert COLOR_TO_CLASS: class_index → (R, G, B)
# Used to color-code segmentation masks for display.
CLASS_COLORS = [None] * NUM_CLASSES
for rgb, idx in COLOR_TO_CLASS.items():
    CLASS_COLORS[idx] = rgb
# Fallback: any class without a defined color gets white
CLASS_COLORS = [c if c is not None else (255, 255, 255) for c in CLASS_COLORS]


# ─────────────────────────────────────────────
# CONFUSION MATRIX HELPERS
# ─────────────────────────────────────────────

def _update_cm(cm, preds, masks, num_classes):
    """
    Accumulate (num_classes × num_classes) confusion matrix IN-PLACE.

    cm[true_class, predicted_class] += pixel_count

    Args:
        cm     : torch.Tensor, shape (num_classes, num_classes), dtype int64, on CPU
        preds  : torch.Tensor, shape (B, H, W), predicted class indices, on CPU
        masks  : torch.Tensor, shape (B, H, W), ground-truth class indices, on CPU
    """
    preds_flat = preds.view(-1).long()
    masks_flat = masks.view(-1).long()

    # Ignore pixels outside valid class range (artifact values clamped in dataset.py)
    valid      = (masks_flat >= 0) & (masks_flat < num_classes)
    preds_flat = preds_flat[valid]
    masks_flat = masks_flat[valid]

    combined   = masks_flat * num_classes + preds_flat
    cm        += torch.bincount(combined, minlength=num_classes * num_classes) \
                      .reshape(num_classes, num_classes)


# ─────────────────────────────────────────────
# METRICS FROM CONFUSION MATRIX
# ─────────────────────────────────────────────

def _compute_metrics(cm: torch.Tensor, num_classes: int) -> dict:
    """
    Derive all evaluation metrics from the accumulated confusion matrix.

    Formulas (c = class index):
      TP_c  = cm[c, c]
      FP_c  = cm[:, c].sum() - cm[c, c]    ← predicted as c but aren't
      FN_c  = cm[c, :].sum() - cm[c, c]    ← are c but predicted as something else

      IoU_c     = TP_c / (TP_c + FP_c + FN_c)
                = cm[c,c] / (cm[c,:].sum() + cm[:,c].sum() - cm[c,c])

      Dice_c    = 2*TP_c / (2*TP_c + FP_c + FN_c)
                = 2*cm[c,c] / (cm[c,:].sum() + cm[:,c].sum())

      Recall_c  = TP_c / (TP_c + FN_c)
                = cm[c,c] / cm[c,:].sum()
                (= "of all true class-c pixels, how many did we catch?")

      PixelAcc  = diagonal_sum / total_pixels

    For IoU, Dice, Recall:
      - If denominator = 0 (class absent from BOTH predictions AND ground truth):
          display 0.0, but DO NOT include in the mean calculation.
      - If denominator = 0 in ground truth only (class absent from test set):
          same treatment: display 0.0, exclude from mean.

    Returns dict with:
      "iou_per_class"    : list[float] len=10, IoU per class (0.0 if absent)
      "dice_per_class"   : list[float] len=10
      "recall_per_class" : list[float] len=10
      "miou"             : float, mean over present classes
      "mean_dice"        : float
      "mean_recall"      : float
      "pixel_accuracy"   : float
    """
    cm_f = cm.float()

    iou_list    = []
    dice_list   = []
    recall_list = []

    iou_for_mean    = []
    dice_for_mean   = []
    recall_for_mean = []

    for c in range(num_classes):
        tp    = cm_f[c, c]
        row   = cm_f[c, :].sum()   # all pixels that ARE class c (TP + FN)
        col   = cm_f[:, c].sum()   # all pixels PREDICTED as c (TP + FP)
        union = row + col - tp     # TP + FP + FN

        # ── IoU ──────────────────────────────────────
        if union > 0:
            iou = (tp / union).item()
            iou_for_mean.append(iou)
        else:
            iou = 0.0
        iou_list.append(iou)

        # ── Dice ─────────────────────────────────────
        denom_dice = row + col
        if denom_dice > 0:
            dice = (2.0 * tp / denom_dice).item()
            dice_for_mean.append(dice)
        else:
            dice = 0.0
        dice_list.append(dice)

        # ── Recall ───────────────────────────────────
        if row > 0:
            recall = (tp / row).item()
            recall_for_mean.append(recall)
        else:
            recall = 0.0
        recall_list.append(recall)

    # ── Pixel Accuracy ───────────────────────────────
    total   = cm_f.sum().item()
    correct = cm_f.diagonal().sum().item()
    pixel_acc = correct / total if total > 0 else 0.0

    return {
        "iou_per_class":    iou_list,
        "dice_per_class":   dice_list,
        "recall_per_class": recall_list,
        "miou":             float(np.mean(iou_for_mean))    if iou_for_mean    else 0.0,
        "mean_dice":        float(np.mean(dice_for_mean))   if dice_for_mean   else 0.0,
        "mean_recall":      float(np.mean(recall_for_mean)) if recall_for_mean else 0.0,
        "pixel_accuracy":   pixel_acc,
    }


# ─────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────

def evaluate_model(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    device:      torch.device,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """
    Run the model on the full test set and compute all metrics.

    Args:
        model       : trained U-Net (already moved to device)
        loader      : test DataLoader (from get_dataloaders()[2])
        device      : MPS / CUDA / CPU
        num_classes : 10 for FloodNet

    Returns:
        metrics dict (see _compute_metrics for full schema)
    """
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    print("[Evaluate] Running inference on test set...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)           # (B, 3, 512, 512)
            masks  = masks.to(device).long()     # (B, 512, 512)

            logits = model(images)               # (B, 10, 512, 512)
            preds  = logits.argmax(dim=1)        # (B, 512, 512) — class per pixel

            # Accumulate into CPU confusion matrix
            _update_cm(cm, preds.cpu(), masks.cpu(), num_classes)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")

    print(f"[Evaluate] Inference complete. Computing metrics...")
    metrics = _compute_metrics(cm, num_classes)
    return metrics


# ─────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────

def print_results(metrics: dict) -> None:
    """
    Print a formatted table of all metrics — mirrors paper Table II layout.

    Example output:
      ┌──────────────────────────┬────────┬────────┬────────┐
      │ Class                    │  IoU   │  Dice  │ Recall │
      ├──────────────────────────┼────────┼────────┼────────┤
      │ Background               │ 0.9123 │ 0.9542 │ 0.9301 │
      │ Building Flooded         │ 0.7234 │ 0.8398 │ 0.8012 │
      ...
      ├──────────────────────────┼────────┼────────┼────────┤
      │ MEAN (mIoU)              │ 0.8012 │ 0.8734 │ 0.8501 │
      │ Pixel Accuracy           │        │        │ 0.9234 │
      └──────────────────────────┴────────┴────────┴────────┘
    """
    col_w = 26   # class name column width

    sep   = "─" * (col_w + 2)
    hdr   = f"{'Class':<{col_w}}  {'IoU':>7}  {'Dice':>7}  {'Recall':>7}"

    print()
    print("─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))

    for c in range(NUM_CLASSES):
        name   = CLASS_NAMES[c]
        iou    = metrics["iou_per_class"][c]
        dice   = metrics["dice_per_class"][c]
        recall = metrics["recall_per_class"][c]
        print(f"{name:<{col_w}}  {iou:>7.4f}  {dice:>7.4f}  {recall:>7.4f}")

    print("─" * len(hdr))
    print(
        f"{'MEAN':<{col_w}}  "
        f"{metrics['miou']:>7.4f}  "
        f"{metrics['mean_dice']:>7.4f}  "
        f"{metrics['mean_recall']:>7.4f}"
    )
    print(f"{'Pixel Accuracy':<{col_w}}  {'':>7}  {'':>7}  {metrics['pixel_accuracy']:>7.4f}")
    print("─" * len(hdr))
    print()


# ─────────────────────────────────────────────
# SAVE METRICS
# ─────────────────────────────────────────────

def save_results(metrics: dict, path: str = None) -> str:
    """
    Save metrics dict to a JSON file.

    Adds class names to the per-class lists for readability:
      "iou_per_class": {"Background": 0.912, "Building Flooded": 0.723, ...}

    Args:
        metrics : dict from _compute_metrics
        path    : save path (defaults to METRICS_DIR/test_metrics.json)

    Returns:
        Absolute path of saved file.
    """
    if path is None:
        os.makedirs(METRICS_DIR, exist_ok=True)
        path = os.path.join(METRICS_DIR, "test_metrics.json")

    # Add class names to per-class entries for human readability
    named = {
        "miou":           metrics["miou"],
        "mean_dice":      metrics["mean_dice"],
        "mean_recall":    metrics["mean_recall"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "iou_per_class": {
            CLASS_NAMES[c]: metrics["iou_per_class"][c]
            for c in range(NUM_CLASSES)
        },
        "dice_per_class": {
            CLASS_NAMES[c]: metrics["dice_per_class"][c]
            for c in range(NUM_CLASSES)
        },
        "recall_per_class": {
            CLASS_NAMES[c]: metrics["recall_per_class"][c]
            for c in range(NUM_CLASSES)
        },
    }

    with open(path, "w") as f:
        json.dump(named, f, indent=2)

    print(f"[Evaluate] Metrics saved → {path}")
    return path


# ─────────────────────────────────────────────
# VISUALIZATION  (optional — call separately)
# ─────────────────────────────────────────────

def visualize_predictions(
    model:       nn.Module,
    dataset,                    # FloodNetDataset instance (not loader)
    device:      torch.device,
    num_samples: int = 5,
    save_dir:    str = None,
) -> None:
    """
    Save side-by-side images: [Original Image | Ground Truth | Prediction].

    Each saved PNG lets you visually inspect what the model got right/wrong.
    Colors follow the official FloodNet color palette from COLOR_TO_CLASS.

    Why denormalize?
      Images in the dataset are normalized (mean subtracted, divided by std).
      Raw normalized tensors look gray/garish. We invert the normalization
      before saving so the original colors are visible.

    Args:
        model       : trained U-Net
        dataset     : FloodNetDataset (test split) — NOT a DataLoader
        device      : device
        num_samples : how many images to save (default 5)
        save_dir    : folder to save PNGs (defaults to VISUALIZATION_DIR)
    """
    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF

    if save_dir is None:
        save_dir = VISUALIZATION_DIR
    os.makedirs(save_dir, exist_ok=True)

    # ImageNet stats used in dataset.py normalize step
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Color palette: class index → (R, G, B)
    palette = CLASS_COLORS   # defined at top of this file

    model.eval()
    indices = list(range(min(num_samples, len(dataset))))

    print(f"[Visualize] Saving {len(indices)} prediction images to {save_dir} ...")

    for i in indices:
        img_tensor, mask_tensor = dataset[i]   # img: (3,512,512), mask: (512,512)

        # Forward pass
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))  # (1,10,512,512)
            pred   = logits.argmax(dim=1).squeeze(0).cpu()      # (512,512)

        # Denormalize image for display
        img_disp = img_tensor.cpu() * STD + MEAN          # undo normalize
        img_disp = img_disp.clamp(0.0, 1.0)
        img_disp = (img_disp * 255).byte().permute(1, 2, 0).numpy()  # (512,512,3) uint8

        # Convert ground truth and prediction to color images
        H, W = mask_tensor.shape
        gt_color   = np.zeros((H, W, 3), dtype=np.uint8)
        pred_color = np.zeros((H, W, 3), dtype=np.uint8)

        for c in range(NUM_CLASSES):
            gt_color[mask_tensor.numpy()  == c] = palette[c]
            pred_color[pred.numpy()        == c] = palette[c]

        # Stitch three images side by side
        combined = np.concatenate([img_disp, gt_color, pred_color], axis=1)  # (512, 1536, 3)

        out_path = os.path.join(save_dir, f"pred_{i:04d}.png")
        PILImage.fromarray(combined).save(out_path)

    print(f"[Visualize] Done. Check {save_dir}/pred_XXXX.png")


# ─────────────────────────────────────────────
# ENTRY POINT  (load checkpoint → evaluate → print → save)
# ─────────────────────────────────────────────

def run_evaluation(
    checkpoint_path: str = None,
    batch_size:      int = 4,
    device:          torch.device = None,
    visualize:       bool = False,
    num_vis_samples: int  = 5,
) -> dict:
    """
    Load best_model.pth and evaluate on the FloodNet test set.

    This is the function called by experiments/final_train.py after training,
    or run standalone via `python -m src.evaluate`.

    Args:
        checkpoint_path : path to .pth file (defaults to CHECKPOINT_DIR/best_model.pth)
        batch_size      : for test DataLoader
        device          : defaults to config DEVICE
        visualize       : if True, also save visual predictions
        num_vis_samples : number of images to visualize

    Returns:
        metrics dict
    """
    from src.config import DEVICE as CFG_DEVICE
    if device is None:
        device = CFG_DEVICE

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # ── Load checkpoint ──────────────────────────────────────
    print(f"[Evaluate] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    hps = ckpt.get("hps", {})
    dropout = hps.get("dropout", 0.0)

    # Build model with SAME dropout used during training
    model = build_unet(dropout=dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[Evaluate] Model loaded (epoch={ckpt.get('epoch', '?')}, "
          f"val_mIoU={ckpt.get('val_miou', '?'):.4f})")

    # ── Get test DataLoader ───────────────────────────────────
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    # ── Evaluate ──────────────────────────────────────────────
    metrics = evaluate_model(model, test_loader, device)

    # ── Print and save ────────────────────────────────────────
    print_results(metrics)
    save_results(metrics)

    # ── Optional visualization ────────────────────────────────
    if visualize:
        from src.dataset import FloodNetDataset
        from src.config  import TEST_IMG_DIR, TEST_MASK_DIR
        test_ds = FloodNetDataset(TEST_IMG_DIR, TEST_MASK_DIR, split="test", augment=False)
        visualize_predictions(model, test_ds, device, num_samples=num_vis_samples)

    return metrics


# ─────────────────────────────────────────────
# RUN STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run as: python -m src.evaluate
    Requires best_model.pth to exist in CHECKPOINT_DIR (run experiments/final_train.py first).
    """
    metrics = run_evaluation(visualize=False)
    print(f"Final Test mIoU: {metrics['miou']:.4f}")
