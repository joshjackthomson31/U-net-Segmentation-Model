# src/train.py — Explained Simply

---

## What is this file for?

`train.py` contains the **actual training logic** — the functions that feed images to the U-Net, measure how wrong its predictions are, and update its weights to improve.

It has two main entry points:
- `proxy_train` — fast 5-epoch training used by HHO to evaluate each hawk
- `full_train` — full 20-epoch training after HHO finds the best hyperparameters

---

## Key Concepts Before Diving In

### What is a training epoch?
One complete pass through all training images. With 1444 images and batch_size=4, one epoch = 361 batches of 4 images each.

### What is a training step (batch)?
The model sees 4 images, makes predictions, compares to ground truth, computes error, and updates weights. Then moves to the next 4 images.

### What is a loss?
A number measuring how wrong the model's predictions are. The model tries to make this number as small as possible.

### What is CrossEntropyLoss?
The most common loss for classification tasks. For each pixel, it compares the model's predicted class score to the true class. If the model confidently predicted the right class → small loss. If it was wrong or uncertain → large loss.

---

## Function: `_seed_everything(seed)`

**What it does:** Locks ALL sources of randomness to the same starting point.

```python
random.seed(seed)     # Python's random module
np.random.seed(seed)  # NumPy's random module
torch.manual_seed(seed)  # PyTorch's random module
```

**Why needed for proxy_train specifically?**
HHO caches results: `{HP_set → mIoU}`. For caching to work, the same HP set must always produce the same mIoU. Without seeding:
- Same HPs → different random model initialization → different mIoU each time
- Cache hits become unreliable → HHO makes wrong decisions

Called at the very start of `proxy_train` and `full_train` before anything else.

---

## Function: `_update_confusion_matrix(cm, preds, masks, num_classes)`

**What it does:** Accumulates predictions into a global (10×10) confusion matrix IN-PLACE.

**What is a confusion matrix?**
A 10×10 grid where:
- Row = true class (what the pixel actually is)
- Column = predicted class (what the model said)
- Cell value = how many pixels had that (true, predicted) combination

Example (simplified for 3 classes):
```
            Predicted:   BG    Road   Water
True:
Background             [100,    5,     0]   ← 100 correct, 5 wrong
Road                   [ 10,   80,     0]   ← 10 misclassified as BG
Water                  [  0,    0,    50]   ← 50 all correct
```
Diagonal = correct predictions. Off-diagonal = mistakes.

**How it works efficiently:**
```python
combined = masks_flat * num_classes + preds_flat
cm += torch.bincount(combined, minlength=num_classes*num_classes).reshape(...)
```
Instead of nested for-loops (slow), it encodes each (true, pred) pair as a single number and counts them all at once with `bincount`.

**Example:**
- true=1, pred=2, num_classes=3 → combined index = 1×3 + 2 = 5
- bincount counts how many times index 5 appears → that's cm[1,2]

---

## Function: `_miou_from_confusion_matrix(cm, num_classes)`

**What it does:** Computes mean IoU (the primary metric) from the accumulated confusion matrix.

**IoU formula for class c:**
```
IoU_c = True Positives / (True Positives + False Positives + False Negatives)
      = cm[c,c] / (cm[c,:].sum() + cm[:,c].sum() - cm[c,c])
```

Breaking that down:
- `cm[c,c]` = pixels correctly predicted as class c (True Positives)
- `cm[c,:].sum()` = all pixels that ARE class c (TP + FN)
- `cm[:,c].sum()` = all pixels PREDICTED as class c (TP + FP)
- Subtract `cm[c,c]` because it was counted twice

**mIoU** = average IoU across all classes that appear in the data.

**Why skip classes with zero union?**
If a class appears in neither ground truth nor predictions, its union = 0 and IoU is undefined. Averaging in a 0/0 value would be wrong. Such classes are excluded from the mean (they don't affect the score).

**Why accumulate the FULL confusion matrix instead of averaging per-batch IoU?**
With rare classes (Vehicle = 0.18% of pixels), many batches contain NO vehicle pixels. Per-batch IoU would be 0 for those batches and pull down the average unfairly. The full accumulated CM computes IoU over ALL pixels at once — this is the correct approach.

---

## Function: `train_one_epoch(model, loader, optimizer, criterion, device)`

**What it does:** Trains the model for exactly one epoch (one pass through all training data).

**Step by step for each batch:**
```
1. images.to(device), masks.to(device)   ← move data to GPU/MPS
2. optimizer.zero_grad()                  ← reset accumulated gradients
3. logits = model(images)                 ← model predicts (B,10,512,512)
4. loss = criterion(logits, masks)        ← measure how wrong
5. loss.backward()                        ← compute how to improve (gradients)
6. optimizer.step()                       ← apply improvement
```

**Why zero_grad at step 2?**
PyTorch accumulates gradients by default. If you don't reset them, each batch's gradients add to the previous batch's — the update is wrong. Reset = fresh calculation per batch.

**Returns:** Average training loss across all batches (for monitoring).

**Note:** No AMP (Automatic Mixed Precision). MPS doesn't support float16 mixed precision, so we use full float32.

---

## Function: `validate(model, loader, criterion, device, num_classes)`

**What it does:** Runs the model on the validation set WITHOUT updating weights.

**Why not update weights during validation?**
Validation measures how well the model generalizes to data it hasn't trained on. If we updated weights based on val data, we'd be "cheating" — the model would be optimized for the val set specifically, not for unseen data.

**Step by step:**
```python
model.eval()   # turn off dropout, use BatchNorm running stats
with torch.no_grad():   # don't track gradients (faster, less memory)
    for images, masks in loader:
        logits = model(images)
        loss   = criterion(logits, masks)
        preds  = logits.argmax(dim=1)   # pick highest scoring class per pixel
        _update_confusion_matrix(cm, preds.cpu(), masks.cpu(), num_classes)
```

`logits.argmax(dim=1)` converts raw scores to class predictions:
- Input: `(B, 10, 512, 512)` — 10 scores per pixel
- Output: `(B, 512, 512)` — the winning class index per pixel

**Returns:** `(avg_val_loss, mIoU)` — both numbers monitored during training.

---

## Function: `proxy_train(hps, class_weights, device)`

**What it does:** A fast 5-epoch training run that HHO uses to score each hawk's hyperparameters.

**Parameters:**
- `hps`: dict with `{lr, batch_size, dropout, weight_decay}` — the hawk's position
- `class_weights`: pre-computed tensor (passed in — not recomputed each time)
- `device`: MPS / CUDA / CPU

**Step by step:**
```python
_seed_everything(SEED)        # must be first — ensures reproducibility for caching
model = build_unet(dropout=hps["dropout"]).to(device)
train_loader, val_loader, _ = get_dataloaders(batch_size=hps["batch_size"], include_test=False)
criterion = nn.CrossEntropyLoss()  # standard, no class weights
optimizer = torch.optim.Adam(model.parameters(), lr=hps["lr"], weight_decay=hps["weight_decay"])

for epoch in range(PROXY_EPOCHS):    # 5 epochs only
    train_one_epoch(model, train_loader, optimizer, criterion, device)

_, miou = validate(model, val_loader, criterion, device)
```

**Why `include_test=False`?**
The test DataLoader is not needed here — proxy_train only uses train and val. Creating it would print an extra log line and waste ~1 second per evaluation. Since HHO calls this hundreds of times, it adds up.

**After evaluation — free memory:**
```python
del model, train_loader, val_loader, criterion, optimizer
torch.mps.empty_cache()
```
MPS keeps tensors in GPU memory even after `del`. Explicitly clearing the cache prevents memory buildup across 1000 hawk evaluations.

**Returns:** Final validation mIoU (float). HHO maximizes this.

---

## Function: `full_train(hps, class_weights, device)`

**What it does:** Full 20-epoch training with the best hyperparameters HHO found. This produces the final model.

**Differences from proxy_train:**
| Feature | proxy_train | full_train |
|---|---|---|
| Epochs | 5 | 20 |
| Saves checkpoint | No | Yes (best by val mIoU) |
| Saves metrics JSON | No | Yes |
| Logs each epoch | No | Yes |
| Used for | HHO scoring | Final model |

**Step by step:**

**1. Build model and DataLoaders:**
```python
model = build_unet(dropout=hps["dropout"]).to(device)
train_loader, val_loader, _ = get_dataloaders(batch_size=hps["batch_size"])
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=hps["lr"], weight_decay=hps["weight_decay"])
```

**2. Training loop (20 epochs):**
```python
for epoch in range(FULL_EPOCHS):
    current_lr = optimizer.param_groups[0]["lr"]   # log current LR
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_miou = validate(model, val_loader, criterion, device)
    history["train_loss"].append(train_loss)
    history["val_miou"].append(val_miou)
    print(f"[Epoch {epoch+1}] lr={current_lr:.2e}  train_loss={train_loss:.4f}  val_mIoU={val_miou:.4f}")
```

**3. Save best checkpoint:**
```python
if val_miou > best_miou:
    best_miou = val_miou
    torch.save({"state_dict": model.state_dict(), "val_miou": val_miou, "hps": hps},
               "results/checkpoints/best_model.pth")
```
Only saves when val mIoU improves — keeps the BEST model seen during all 20 epochs, not just the last one. The model at epoch 19 might actually be slightly worse than epoch 17.

**4. Save final checkpoint and metrics:**
```python
torch.save(model.state_dict(), "results/checkpoints/final_model.pth")
json.dump(history, open("results/metrics/train_metrics.json", "w"))
```

**Returns:** `history` dict with:
```json
{
  "train_loss": [1.65, 1.34, 1.25, ...],   // one per epoch
  "val_loss":   [1.17, 1.11, 1.10, ...],
  "val_miou":   [0.22, 0.24, 0.27, ...],
  "lr":         [9.55e-4, 9.55e-4, ...],
  "best_epoch": 18,
  "best_miou":  0.3690,
  "hps":        {"lr": ..., "batch_size": 4, ...}
}
```
