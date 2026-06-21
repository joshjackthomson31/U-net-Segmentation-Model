# Complete Workflow & File Flow — From Start to Finish

---

## The Big Picture

This project implements **HHO-U-Net**: a hybrid deep learning system that automatically finds the best training settings for a U-Net segmentation model using the Harris Hawks Optimization algorithm.

**Goal:** Label every pixel in a post-flood aerial photo with one of 10 classes (Building Flooded, Water, Road, etc.) as accurately as possible.

**Strategy:**
1. Instead of manually guessing hyperparameters (learning rate, batch size, etc.), let HHO search for the best ones automatically
2. Train the final U-Net with those best settings
3. Measure how well the model segments unseen test images

---

## The Full Pipeline — Two Steps

```
Step 1: python main.py search   (overnight, ~29 hours)
Step 2: python main.py train    (~1 hour)
```

That's it. Two commands. Everything else is internal.

---

## File Map — Every File and Its Role

```
U-net-Segmentation-Model/
│
├── main.py                     ← Entry point. Routes commands to the right code.
│
├── src/                        ← Core library — reusable building blocks
│   ├── config.py               ← All settings: paths, constants, HHO parameters
│   ├── unet.py                 ← U-Net model definition (the neural network)
│   ├── dataset.py              ← Data loading, augmentation, class weights
│   ├── train.py                ← Training loop, validation, proxy_train, full_train
│   ├── evaluate.py             ← Test set evaluation, metrics, visualization
│   └── hho.py                  ← HHO algorithm (hawks, escape energy, Lévy flight)
│
├── experiments/                ← Higher-level scripts that orchestrate the pipeline
│   ├── hho_search.py           ← Runs HHO search, saves best_hps.json
│   └── final_train.py          ← Loads best_hps.json, trains 20 epochs, evaluates
│
├── results/                    ← All outputs (generated, not source code)
│   ├── checkpoints/
│   │   ├── best_model.pth      ← Best weights (by val mIoU) — used for evaluation
│   │   └── final_model.pth     ← Weights at last epoch
│   ├── metrics/
│   │   ├── best_hps.json       ← HHO's winning hyperparameters → read by final_train
│   │   ├── hho_history.json    ← mIoU per HHO iteration (convergence curve)
│   │   ├── hho_cache.json      ← All evaluated HP sets + their scores
│   │   ├── train_metrics.json  ← Loss + mIoU for each of the 20 training epochs
│   │   └── test_metrics.json   ← Final IoU, Dice, Recall per class + pixel accuracy
│   └── visualizations/
│       └── pred_XXXX.png       ← Side-by-side: original | ground truth | prediction
│
├── FloodNet-Supervised_v1.0/  ← Dataset (not in git — download separately)
│   ├── train/ (1445 images + masks)
│   ├── val/   (450 images + masks)
│   └── test/  (448 images + masks)
│
├── HHO_EXPLAINED.md            ← Full HHO algorithm explanation
├── docs/                       ← File-by-file explanations (this folder)
│   ├── config_explained.md
│   ├── unet_explained.md
│   ├── dataset_explained.md
│   ├── train_explained.md
│   ├── evaluate_explained.md
│   ├── hho_search_explained.md
│   ├── final_train_explained.md
│   └── workflow_explained.md   ← (this file)
│
├── requirements.txt            ← Python dependencies
└── .gitignore                  ← Files excluded from git (dataset, results, venv)
```

---

## Step-by-Step Workflow: Search Phase

### Command: `python main.py search`

```
main.py
  └── cmd_search()
        └── experiments/hho_search.py: run_search()
```

**What happens inside `run_search()`:**

```
1. src/dataset.py: get_class_weights()
   ├── Loads src/config.py for paths (TRAIN_IMG_DIR, etc.)
   ├── Creates FloodNetDataset (train split, 1444 images)
   ├── Scans ALL training masks pixel by pixel
   ├── Counts pixels per class (e.g., Grass=56%, Vehicle=0.18%)
   ├── Computes inverse-frequency weights with clamp at 0.1
   └── Returns tensor of shape (10,) ← used in every proxy_train call

2. experiments/hho_search.py: _make_eval_fn(class_weights, device)
   └── Creates eval_fn(hps) → float wrapper

3. src/hho.py: HHO(eval_fn, seed).run()
   │
   ├── INITIALIZE: create 20 hawks with random HP positions
   │     Each hawk = random {lr, batch_size, dropout, weight_decay}
   │     in log-space for lr and weight_decay
   │
   ├── INITIAL EVALUATION: evaluate all 20 hawks
   │     For each hawk:
   │       eval_fn(hps)
   │         └── src/train.py: proxy_train(hps, class_weights, device)
   │               ├── _seed_everything(42)
   │               ├── src/unet.py: build_unet(dropout=hps["dropout"])
   │               ├── src/dataset.py: get_dataloaders(batch_size, include_test=False)
   │               ├── 5 epochs: train_one_epoch() × 5
   │               ├── validate() → val_mIoU
   │               └── return mIoU  (HHO scores this hawk)
   │
   ├── FIND RABBIT: hawk with highest mIoU = "the rabbit" (best solution)
   │
   ├── FOR EACH ITERATION (up to 50):
   │     Compute escape energy E = 2×E0×(1 - iter/max_iter)
   │     
   │     For each hawk, based on |E|:
   │       |E| ≥ 1.0 → Exploration: jump randomly
   │       |E| < 0.5, rabbit escaping → Soft Besiege with Lévy flight
   │       |E| < 0.5, rabbit not escaping → Hard Besiege
   │       |E| ≥ 0.5 → Rapid dives with Lévy flight
   │     
   │     Clip positions to search space bounds
   │     Evaluate new positions (check cache first)
   │     Update rabbit if any hawk improves
   │     Check convergence (stop if no improvement for 5 iterations)
   │
   └── Return (best_hps, best_miou, history)

4. experiments/hho_search.py: _save_results(...)
   ├── Writes results/metrics/best_hps.json       ← KEY OUTPUT
   ├── Writes results/metrics/hho_history.json
   └── Writes results/metrics/hho_cache.json
```

**Output:** `results/metrics/best_hps.json` with the winning hyperparameters.

**Duration:** 10–29 hours. Run overnight.

---

## Step-by-Step Workflow: Training Phase

### Command: `python main.py train`

```
main.py
  └── cmd_train()
        └── experiments/final_train.py: run()
```

**What happens inside `run()`:**

```
1. experiments/final_train.py: _load_best_hps()
   └── Reads results/metrics/best_hps.json
       └── Returns: {lr, batch_size, dropout, weight_decay}

2. src/dataset.py: get_class_weights()
   └── Same as search phase — recomputes weights (~53 seconds)

3. src/train.py: full_train(best_hps, class_weights, device)
   │
   ├── _seed_everything(42)
   │
   ├── src/unet.py: build_unet(dropout=best_hps["dropout"])
   │   └── Creates fresh U-Net (7.8M parameters, Kaiming initialized)
   │
   ├── src/dataset.py: get_dataloaders(batch_size=best_hps["batch_size"])
   │   ├── FloodNetDataset train (1444 images, augment=True)
   │   ├── FloodNetDataset val   (450 images,  augment=False)
   │   └── FloodNetDataset test  (448 images,  augment=False)
   │
   ├── nn.CrossEntropyLoss(weight=class_weights)
   ├── Adam optimizer (lr=best_hps["lr"], weight_decay=best_hps["weight_decay"])
   │
   ├── FOR EACH EPOCH (20 total):
   │     train_one_epoch():
   │       for each batch of 4 images:
   │         1. Forward: model(images) → logits (B,10,512,512)
   │         2. Loss: CrossEntropy(logits, masks)
   │         3. Backward: compute gradients
   │         4. Step: update weights
   │     
   │     validate():
   │       for each val batch:
   │         1. Forward: model(images) → logits (no gradients)
   │         2. Predictions: argmax across 10 classes
   │         3. Accumulate confusion matrix
   │       compute mIoU from full confusion matrix
   │     
   │     if val_mIoU > best_so_far:
   │       save results/checkpoints/best_model.pth
   │     
   │     print epoch summary
   │
   ├── save results/checkpoints/final_model.pth
   └── save results/metrics/train_metrics.json

4. src/evaluate.py: run_evaluation(...)
   │
   ├── Load results/checkpoints/best_model.pth
   ├── Rebuild U-Net with same dropout
   │
   ├── src/dataset.py: get_dataloaders() → test_loader
   │
   ├── evaluate_model(model, test_loader, device):
   │     for each batch of 448 test images:
   │       1. Forward: model(images) → logits
   │       2. Predictions: argmax → class per pixel
   │       3. _update_cm(confusion_matrix, preds, masks)
   │     _compute_metrics(confusion_matrix):
   │       IoU, Dice, Recall per class
   │       mIoU, mean_dice, pixel_accuracy
   │
   ├── print_results(metrics)         ← prints Table II format
   ├── save_results(metrics)          ← writes test_metrics.json
   │
   └── visualize_predictions(model, test_dataset):
         for 5 test images:
           denormalize image
           color-code ground truth mask
           color-code prediction mask
           stitch side-by-side
           save pred_XXXX.png
```

**Output:** Model weights, training history, test metrics, visualizations.

**Duration:** ~1 hour on Apple M4 MPS.

---

## Other Commands

### `python main.py sanity`

Quick health check — no dataset needed:
```
main.py → cmd_sanity()
  ├── src/config.py: DEVICE, NUM_CLASSES
  ├── src/unet.py: build_unet(dropout=0.3, base_filters=32)
  ├── torch.randn(2, 3, 512, 512) → model → output
  └── assert output.shape == (2, 10, 512, 512)
```
Run this FIRST to confirm your Python environment is set up correctly. Takes ~10 seconds.

### `python main.py evaluate`

Re-runs evaluation on the test set without retraining:
```
main.py → cmd_evaluate()
  └── src/evaluate.py: run_evaluation(visualize=False)
        ├── loads best_model.pth
        └── prints test mIoU
```
Useful if you want to change visualization settings or check metrics again.

---

## Data Flow Diagram

```
FloodNet Dataset (disk)
    │
    ▼
src/dataset.py
    ├── FloodNetDataset.__getitem__()
    │     ├── PIL.open(image.jpg) → resize → augment → normalize → tensor (3,512,512)
    │     └── PIL.open(mask.png) → resize → clamp → tensor (512,512)
    │
    └── DataLoader batches 4 images together → (4,3,512,512) + (4,512,512)
                                                      │
                                                      ▼
                                           src/unet.py: UNet.forward(x)
                                               ├── Encoder: compress 512→32×32
                                               ├── Bottleneck: understand scene
                                               ├── Decoder: restore 32→512×512
                                               └── Head: 32 features → 10 logits
                                                              │
                                                              ▼
                                           Logits (4, 10, 512, 512)
                                                              │
                              ┌───────────────────────────────┤
                              ▼                               ▼
                   CrossEntropyLoss                   argmax → predictions
                   (training: update weights)         (validation/test: measure accuracy)
                              │                               │
                              ▼                               ▼
                     gradients → optimizer            confusion matrix
                     → better weights                → mIoU, pixel accuracy
```

---

## Dependency Graph — Which File Imports Which

```
main.py
├── experiments/hho_search.py
│   ├── src/config.py
│   ├── src/dataset.py
│   │   └── src/config.py
│   ├── src/train.py
│   │   ├── src/config.py
│   │   ├── src/unet.py
│   │   │   └── src/config.py
│   │   └── src/dataset.py
│   └── src/hho.py
│       └── src/config.py
│
└── experiments/final_train.py
    ├── src/config.py
    ├── src/dataset.py
    ├── src/train.py
    └── src/evaluate.py
        ├── src/config.py
        ├── src/unet.py
        └── src/dataset.py
```

`src/config.py` is imported by EVERYTHING — it is the foundation that all other files build on.

---

## Key Numbers Summary

| Metric | Value | Source |
|---|---|---|
| Training images | 1,444 | FloodNet (1 missing download) |
| Validation images | 450 | FloodNet |
| Test images | 448 | FloodNet |
| Image size | 512×512 | Paper Sec. IV |
| Classes | 10 | FloodNet |
| Model parameters | 7,763,338 | U-Net base_filters=32 |
| HHO hawks | 20 | Paper Sec. IV |
| HHO max iterations | 50 | Paper Sec. IV |
| Proxy epochs (HHO) | 5 | Standard practice |
| Full training epochs | 20 | Paper Sec. IV |
| Search time | ~29 hours | M4 MPS |
| Training time | ~1 hour | M4 MPS |
| Paper best mIoU (GOA) | 67.97% | Paper Table II |
| Our best mIoU | 41.71% | Run 2 (HHO+clamped weights+scheduler) |

---

## Recommended Order of Operations

```
1. python main.py sanity     ← verify setup (30 seconds)
2. python main.py search     ← HHO finds best HPs (run overnight)
3. python main.py train      ← train final model with best HPs (~1 hour)
4. python main.py evaluate   ← re-evaluate anytime without retraining
```
