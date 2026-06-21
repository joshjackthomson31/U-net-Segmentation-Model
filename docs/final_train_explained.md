# experiments/final_train.py — Explained Simply

---

## What is this file for?

`final_train.py` is the **second and final step** of the project pipeline. After `hho_search.py` has found the best hyperparameters, this file:

1. Reads those best hyperparameters from `best_hps.json`
2. Computes class weights (same as search did)
3. Trains the U-Net for the full 20 epochs with those HPs
4. Evaluates the best saved model on the test set
5. Saves everything — model weights, training history, test metrics, visualization images

Think of it as the **final exam**: HHO studied and found the best strategy (Step 1), now this file executes that strategy for real (Step 2).

---

## What does it produce?

| Output file | What it contains |
|---|---|
| `results/checkpoints/best_model.pth` | Best model weights (epoch with highest val mIoU) |
| `results/checkpoints/final_model.pth` | Model weights from the very last epoch |
| `results/metrics/train_metrics.json` | Loss and mIoU for every epoch |
| `results/metrics/test_metrics.json` | Final test set metrics (IoU, Dice, Recall per class) |
| `results/visualizations/pred_XXXX.png` | 5 side-by-side comparison images |

---

## Function: `_load_best_hps()`

**What it does:** Reads `results/metrics/best_hps.json` and returns the best hyperparameters.

**Expected file format:**
```json
{
  "best_hps": {
    "lr": 0.000223,
    "batch_size": 4,
    "dropout": 0.190,
    "weight_decay": 0.0002
  },
  "best_miou": 0.3048
}
```

**Error handling:**
```python
if not os.path.exists(best_hps_path):
    raise FileNotFoundError(
        "\n\nbest_hps.json not found.\n\nRun HHO search FIRST:\n  python main.py search\n"
    )
```
If you try to run `python main.py train` without running search first, you get a clear error message telling you exactly what to do. Without this check, you'd get a confusing Python KeyError deep inside the code.

**Prints what was loaded:**
```
[FinalTrain] Loaded best HPs from .../best_hps.json
[FinalTrain] HHO proxy mIoU was: 0.3048
[FinalTrain] Best HPs:
    lr              = 0.000223
    batch_size      = 4
    dropout         = 0.190
    weight_decay    = 0.0002
```

**Returns:** The `best_hps` dict (just the HP values, not the file wrapper).

---

## Function: `run(device, visualize, num_vis_samples)`

**What it does:** Runs the complete 4-step final pipeline.

**Parameters:**
- `device`: which hardware to use (defaults to config DEVICE = MPS on your Mac)
- `visualize`: if `True`, save prediction PNG images after evaluation (default: True)
- `num_vis_samples`: how many test images to visualize (default: 5)

---

### Step 1: Load Best HPs

```python
best_hps = _load_best_hps()
```

Reads `results/metrics/best_hps.json`. This file was written by `hho_search.py`. Contains the winner from the overnight search.

---

### Step 2: Compute Class Weights

```python
class_weights = get_class_weights()
```

Scans all 1444 training images and counts pixels per class. Same computation as in `hho_search.py`. Takes ~53 seconds.

**Why compute it again instead of saving from search?**
`train_metrics.json` and `best_hps.json` don't store class weights — only hyperparameters. Class weights depend on the training dataset, which doesn't change. Re-computing takes 53 seconds but avoids the complexity of saving/loading a tensor.

Also: if you modify class weight computation (e.g., change the clamp threshold), the new weights are automatically used without any extra steps.

---

### Step 3: Full 20-Epoch Training

```python
history = full_train(best_hps, class_weights, device)
```

Calls `full_train` from `src/train.py` with the best HPs. This is where the real training happens:
- 1444 training images × 20 epochs = 28,880 gradient updates
- Each epoch: train on all 1444 images, validate on all 450 images
- Best checkpoint saved whenever val mIoU improves
- Prints one line per epoch showing loss, mIoU, and current LR

**Typical epoch output:**
```
[Epoch 01/20] lr=9.55e-04  train_loss=1.6503  val_loss=1.1668  val_mIoU=0.2228
  → New best mIoU: 0.2228  (saved best_model.pth)
[Epoch 02/20] lr=9.55e-04  train_loss=1.3395  val_loss=1.1110  val_mIoU=0.2361
  → New best mIoU: 0.2361  (saved best_model.pth)
...
```

**After training:**
```
[FinalTrain] Total training time: 1.05 hours
[FinalTrain] Best val mIoU during training: 0.4310 (epoch 19)
```

---

### Step 4: Evaluate on Test Set

```python
test_metrics = run_evaluation(
    checkpoint_path = None,           # loads results/checkpoints/best_model.pth
    batch_size      = best_hps["batch_size"],
    device          = device,
    visualize       = visualize,
    num_vis_samples = num_vis_samples,
)
```

Calls `run_evaluation` from `src/evaluate.py`. This:
1. Loads `best_model.pth` (NOT the final epoch — the best epoch during training)
2. Runs through all 448 test images
3. Computes IoU, Dice, Recall per class + mIoU + Pixel Accuracy
4. Prints the full Table II format
5. Saves `test_metrics.json`
6. Saves `pred_XXXX.png` visualization images

**Why use `best_model.pth` and not `final_model.pth`?**
The model at epoch 20 is not necessarily the best model. If val mIoU peaked at epoch 19 and dropped at epoch 20 (overfitting), we want the epoch 19 model. `best_model.pth` is updated whenever val mIoU improves — it always contains the best version seen during training.

---

### Final Summary Print

```
============================================================
PIPELINE COMPLETE
Best val mIoU (during training) : 0.4310
Test mIoU                       : 0.4171
Test Pixel Accuracy             : 0.8170
Test Mean Dice                  : 0.5405

Outputs saved to results/
  checkpoints/best_model.pth     <- best weights
  checkpoints/final_model.pth    <- final-epoch weights
  metrics/train_metrics.json     <- epoch-by-epoch training history
  metrics/test_metrics.json      <- final test set metrics
  visualizations/pred_XXXX.png   <- side-by-side prediction images
============================================================
```

Notice: **val mIoU (0.4310)** ≠ **test mIoU (0.4171)**. The val set was used to select the checkpoint, so it's slightly optimistic. Test mIoU is the honest, unbiased measure.

---

## The `if __name__ == "__main__":` Guard

```python
if __name__ == "__main__":
    run(visualize=True, num_vis_samples=5)
```

Same reason as `hho_search.py` — prevents macOS multiprocessing from spawning worker processes that re-run this file when DataLoaders start. Without this guard, loading data would trigger a fork bomb.

---

## How this file fits in the workflow

```
python main.py train
         │
         └──→ experiments/final_train.py: run()
                    │
                    ├── Step 1: _load_best_hps()
                    │               └── reads results/metrics/best_hps.json
                    │
                    ├── Step 2: get_class_weights()
                    │               └── src/dataset.py (scans 1444 train images)
                    │
                    ├── Step 3: full_train(best_hps, class_weights, device)
                    │               └── src/train.py
                    │                       ├── src/unet.py: build_unet()
                    │                       ├── src/dataset.py: get_dataloaders()
                    │                       ├── 20 epochs of training
                    │                       └── saves:
                    │                             best_model.pth
                    │                             final_model.pth
                    │                             train_metrics.json
                    │
                    └── Step 4: run_evaluation(...)
                                    └── src/evaluate.py
                                            ├── loads best_model.pth
                                            ├── runs on 448 test images
                                            ├── prints metrics table
                                            └── saves:
                                                  test_metrics.json
                                                  pred_XXXX.png (×5)
```

---

## Running this file

Via `main.py` (recommended):
```bash
python main.py train
```

Standalone (also works):
```bash
python -m experiments.final_train
```

**Prerequisite:** `results/metrics/best_hps.json` must exist. Run `python main.py search` first.
