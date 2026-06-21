# src/evaluate.py — Explained Simply

---

## What is this file for?

`evaluate.py` is the **report card** of the project. After training is complete, this file:
1. Loads the best saved model
2. Runs it on the test set (images the model has NEVER seen)
3. Measures how accurate it is using multiple metrics
4. Prints a formatted table (matching the paper's Table II format)
5. Saves results to a JSON file
6. Optionally saves side-by-side visualization images

---

## Why Test Set, Not Validation Set?

During training, we used the **validation set** to pick the best checkpoint (the epoch with highest mIoU). This means the validation metrics were used to make decisions — they're optimistically biased.

The **test set** is completely held out. We never looked at it during training or model selection. Reporting test set metrics gives an honest measure of real-world performance.

FloodNet split:
- Train: 1,445 images (used to update weights)
- Val: 450 images (used to pick best checkpoint)
- Test: 448 images (used only here, for final evaluation)

---

## Color Palette Setup

```python
CLASS_COLORS = [None] * NUM_CLASSES
for rgb, idx in COLOR_TO_CLASS.items():
    CLASS_COLORS[idx] = rgb
```

**What it does:** Inverts the `COLOR_TO_CLASS` mapping from config.py.
- `COLOR_TO_CLASS` = `{(255, 0, 0): 1}` (RGB color → class index)
- `CLASS_COLORS` = `[(0,0,0), (255,0,0), ...]` (class index → RGB color)

Used when drawing colored overlays on prediction images.

---

## Function: `_update_cm(cm, preds, masks, num_classes)`

**What it does:** Accumulates predictions into a confusion matrix (same logic as in train.py but named `_update_cm` here).

See `train_explained.md` for full details. Key points:
- `cm[true_class, pred_class] += pixel_count`
- Uses `bincount` for efficiency
- Filters out invalid mask values

---

## Function: `_compute_metrics(cm, num_classes)`

**What it does:** Takes the accumulated confusion matrix and computes EVERY metric needed for the paper.

### Metrics computed for each of the 10 classes:

**IoU (Intersection over Union):**
```
IoU_c = cm[c,c] / (cm[c,:].sum() + cm[:,c].sum() - cm[c,c])
```
Measures overlap between predicted and actual class regions. 1.0 = perfect, 0.0 = no overlap.

**Dice coefficient (F1 Score):**
```
Dice_c = 2 × cm[c,c] / (cm[c,:].sum() + cm[:,c].sum())
```
Similar to IoU but gives slightly more weight to correct predictions. Common in medical imaging.

**Recall (Producer's Accuracy):**
```
Recall_c = cm[c,c] / cm[c,:].sum()
```
"Of all pixels that ARE class c, what fraction did we correctly identify?"
Example: Recall=0.80 for Water means we found 80% of water pixels.

### Summary metrics:

**mIoU (mean IoU):**
Average IoU across all classes that appear in the test set. The primary metric for comparing models. This is what the paper reports in Table II.

**Pixel Accuracy:**
```
pixel_accuracy = diagonal_sum / total_pixels
```
"What fraction of all pixels did we label correctly?" Counts every pixel equally — favors dominant classes (like Grass at 56%).

**Handling absent classes:**
If a class has zero union (not in ground truth AND not predicted), its IoU is undefined. Display 0.0, but exclude from the mean calculation.

**Returns dict with:**
```python
{
    "iou_per_class":    [0.00, 0.30, 0.45, ...],  # len=10, one per class
    "dice_per_class":   [0.00, 0.46, 0.62, ...],
    "recall_per_class": [0.00, 0.40, 0.63, ...],
    "miou":             0.4171,
    "mean_dice":        0.5405,
    "mean_recall":      0.6538,
    "pixel_accuracy":   0.8170,
}
```

---

## Function: `evaluate_model(model, loader, device, num_classes)`

**What it does:** Runs the trained model through all 448 test images and builds the full confusion matrix.

```python
model.eval()   # disable dropout, use fixed BatchNorm stats
cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

with torch.no_grad():   # no gradient tracking — inference only
    for batch_idx, (images, masks) in enumerate(loader):
        logits = model(images)           # (B, 10, 512, 512)
        preds  = logits.argmax(dim=1)    # (B, 512, 512) — class per pixel
        _update_cm(cm, preds.cpu(), masks.cpu(), num_classes)
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx+1}/{len(loader)} batches...")
```

**Why `model.eval()`?**
- Dropout is disabled (we want deterministic predictions, not random)
- BatchNorm uses its learned running statistics instead of batch statistics

**Why `torch.no_grad()`?**
During evaluation we don't need gradients (we're not updating weights). Disabling gradient tracking saves memory and speeds up inference significantly.

**Returns:** Metrics dict from `_compute_metrics`.

---

## Function: `print_results(metrics)`

**What it does:** Prints a nicely formatted table matching the paper's Table II layout.

```
─────────────────────────────────────────────────────
Class                           IoU     Dice   Recall
─────────────────────────────────────────────────────
Background                   0.0000   0.0000   0.0000
Building Flooded             0.2993   0.4607   0.3957
Building Non-Flooded         0.4527   0.6232   0.6314
Road Flooded                 0.0578   0.1094   0.0601
Road Non-Flooded             0.5766   0.7314   0.7304
Water                        0.5591   0.7172   0.7066
Tree                         0.6759   0.8066   0.7533
Vehicle                      0.0000   0.0000   0.0000
Pool                         0.1538   0.2666   0.1623
Grass                        0.8039   0.8913   0.9512
─────────────────────────────────────────────────────
MEAN                         0.3579   0.4606   0.4391
Pixel Accuracy                                 0.8134
─────────────────────────────────────────────────────
```

Shows all 10 classes plus the MEAN row and Pixel Accuracy at the bottom.

---

## Function: `save_results(metrics, path)`

**What it does:** Saves the metrics dictionary to a JSON file for later analysis.

**Adds class names** to the per-class lists for readability:
```json
{
  "miou": 0.4171,
  "pixel_accuracy": 0.8170,
  "iou_per_class": {
    "Background": 0.0000,
    "Building Flooded": 0.2993,
    "Grass": 0.8039,
    ...
  }
}
```

Default save location: `results/metrics/test_metrics.json`

---

## Function: `visualize_predictions(model, dataset, device, num_samples, save_dir)`

**What it does:** Saves side-by-side comparison images — original photo, ground truth mask, and model prediction — as PNG files.

**Why denormalize first?**
Images in the dataset are normalized (values centered around 0, range roughly -2 to +2). If we save those directly, the image looks gray and garbled. We must reverse the normalization to get back to normal-looking colors:

```python
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
img_disp = img_tensor * STD + MEAN   # undo normalize
img_disp = img_disp.clamp(0.0, 1.0)  # ensure valid range
```

**Creating colored masks:**
```python
for c in range(NUM_CLASSES):
    gt_color[mask.numpy() == c]  = palette[c]  # true class → its color
    pred_color[pred.numpy() == c] = palette[c]  # predicted → its color
```

**Stitching three images side by side:**
```python
combined = np.concatenate([img_disp, gt_color, pred_color], axis=1)
# Result: (512, 1536, 3) — three 512×512 images in a row
```

Saved as: `results/visualizations/pred_0000.png`, `pred_0001.png`, etc.

---

## Function: `run_evaluation(checkpoint_path, batch_size, device, visualize, num_vis_samples)`

**What it does:** The main entry point — orchestrates the complete evaluation pipeline.

**Step by step:**

**1. Load checkpoint:**
```python
ckpt = torch.load("results/checkpoints/best_model.pth", map_location=device)
hps  = ckpt.get("hps", {})
dropout = hps.get("dropout", 0.0)
```
Reads the saved model weights AND the hyperparameters used during training (to rebuild the model identically).

**2. Rebuild model:**
```python
model = build_unet(dropout=dropout).to(device)
model.load_state_dict(ckpt["state_dict"])
```
Must use the same dropout as training — otherwise the model architecture doesn't match the saved weights and loading fails.

**3. Get test DataLoader:**
```python
_, _, test_loader = get_dataloaders(batch_size=batch_size)
```
Uses `_` to discard train and val loaders — only need test here.

**4. Evaluate, print, save:**
```python
metrics = evaluate_model(model, test_loader, device)
print_results(metrics)
save_results(metrics)
```

**5. Optional visualization:**
```python
if visualize:
    test_ds = FloodNetDataset(TEST_IMG_DIR, TEST_MASK_DIR, split="test", augment=False)
    visualize_predictions(model, test_ds, device, num_samples=num_vis_samples)
```
Note: uses the Dataset directly (not the DataLoader) so we can pick individual images by index.

**Returns:** The metrics dict — used by `final_train.py` to print the final summary.

---

## How this file fits in the workflow

```
experiments/final_train.py
    └── calls run_evaluation()
              ├── loads best_model.pth
              ├── runs model on 448 test images
              ├── computes mIoU, Dice, Recall, Pixel Accuracy
              ├── prints table
              ├── saves test_metrics.json
              └── saves pred_XXXX.png (if visualize=True)
```

Can also be run standalone:
```
python -m src.evaluate
```
(Useful if you want to re-evaluate without retraining.)
