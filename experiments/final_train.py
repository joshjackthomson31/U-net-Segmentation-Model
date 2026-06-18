"""
final_train.py — Full 20-epoch training with the best HPs found by HHO

What this file does:
  1. Reads results/metrics/best_hps.json (written by hho_search.py)
  2. Computes class weights (full_train needs them — same as proxy_train)
  3. Runs full_train: 20 epochs with the best HPs
     - Saves best_model.pth whenever val mIoU improves
     - Saves final_model.pth at end of training
     - Saves train_metrics.json with epoch-by-epoch history
  4. Runs evaluation on the TEST set
     - Prints metrics table (paper Table II format)
     - Saves test_metrics.json
     - Optionally saves visualizations

Why class weights again?
  full_train() takes class_weights as an argument (same as proxy_train).
  We compute them here, exactly the same way as hho_search.py.
  Computing them is fast (~30s) compared to the 20-epoch training (~hours),
  so re-computation is not a bottleneck.

Pipeline order:
  1. python -m experiments.hho_search    <- search for best HPs (overnight)
  2. python -m experiments.final_train   <- train with best HPs + evaluate
  3. (results written to results/checkpoints/ and results/metrics/)
"""

import os
import json
import time

from src.config   import DEVICE, METRICS_DIR
from src.dataset  import get_class_weights
from src.train    import full_train
from src.evaluate import run_evaluation


# ─────────────────────────────────────────────
# LOAD BEST HPS
# ─────────────────────────────────────────────

def _load_best_hps() -> dict:
    """
    Read the best hyperparameters saved by hho_search.py.

    File contract (set by hho_search._save_results):
      results/metrics/best_hps.json = {
        "best_hps":  {"lr": ..., "batch_size": ..., "dropout": ..., "weight_decay": ...},
        "best_miou": float
      }

    Returns:
        best_hps dict

    Raises:
        FileNotFoundError with a clear message if HHO search hasn't been run yet.
    """
    best_hps_path = os.path.join(METRICS_DIR, "best_hps.json")

    if not os.path.exists(best_hps_path):
        raise FileNotFoundError(
            f"\n\n{'!'*60}\n"
            f"best_hps.json not found at:\n  {best_hps_path}\n\n"
            f"Run HHO search FIRST:\n"
            f"  python -m experiments.hho_search\n"
            f"{'!'*60}\n"
        )

    with open(best_hps_path, "r") as f:
        data = json.load(f)

    best_hps  = data["best_hps"]
    best_miou = data.get("best_miou", None)

    print(f"[FinalTrain] Loaded best HPs from {best_hps_path}")
    if best_miou is not None:
        print(f"[FinalTrain] HHO proxy mIoU was: {best_miou:.4f}")
    print(f"[FinalTrain] Best HPs:")
    for k, v in best_hps.items():
        print(f"    {k:<15s} = {v}")

    return best_hps


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run(device=None, visualize: bool = True, num_vis_samples: int = 5) -> dict:
    """
    Run the full final-training + evaluation pipeline.

    Args:
        device          : torch.device (defaults to config DEVICE)
        visualize       : if True, save side-by-side prediction images after evaluation
        num_vis_samples : number of test images to visualize

    Returns:
        test_metrics : dict from run_evaluation (mIoU, per-class IoU, etc.)
    """
    if device is None:
        device = DEVICE

    print("=" * 60)
    print("Final Training + Evaluation")
    print(f"Device: {device}")
    print("=" * 60)

    # ── Step 1: Load best HPs from HHO search ────────────────────────────────
    print("\n[Step 1] Loading best hyperparameters from HHO search...")
    best_hps = _load_best_hps()

    # ── Step 2: Compute class weights ─────────────────────────────────────────
    print("\n[Step 2] Computing class weights from training set...")
    print("  (Scans 1445 images — takes ~30 seconds)\n")
    t0 = time.perf_counter()
    class_weights = get_class_weights()
    print(f"  Done in {time.perf_counter() - t0:.1f}s\n")

    # ── Step 3: Full 20-epoch training ────────────────────────────────────────
    print("[Step 3] Starting 20-epoch full training...\n")
    t_train_start = time.perf_counter()

    history = full_train(best_hps, class_weights, device)

    t_train_total = time.perf_counter() - t_train_start
    print(f"\n[FinalTrain] Total training time: {t_train_total/3600:.2f} hours")
    print(f"[FinalTrain] Best val mIoU during training: {history['best_miou']:.4f} "
          f"(epoch {history['best_epoch'] + 1})")

    # ── Step 4: Evaluate on TEST set ─────────────────────────────────────────
    # run_evaluation loads best_model.pth (saved by full_train when val mIoU improved)
    print("\n[Step 4] Evaluating best_model.pth on the TEST set...\n")
    test_metrics = run_evaluation(
        checkpoint_path = None,                  # uses CHECKPOINT_DIR/best_model.pth
        batch_size      = best_hps["batch_size"],
        device          = device,
        visualize       = visualize,
        num_vis_samples = num_vis_samples,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Best val mIoU (during training) : {history['best_miou']:.4f}")
    print(f"Test mIoU                       : {test_metrics['miou']:.4f}")
    print(f"Test Pixel Accuracy             : {test_metrics['pixel_accuracy']:.4f}")
    print(f"Test Mean Dice                  : {test_metrics['mean_dice']:.4f}")
    print()
    print("Outputs saved to results/")
    print("  checkpoints/best_model.pth     <- best weights")
    print("  checkpoints/final_model.pth    <- final-epoch weights")
    print("  metrics/train_metrics.json     <- epoch-by-epoch training history")
    print("  metrics/test_metrics.json      <- final test set metrics")
    if visualize:
        print("  visualizations/pred_XXXX.png   <- side-by-side prediction images")
    print("=" * 60)

    return test_metrics


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run as: python -m experiments.final_train

    Prerequisites:
      results/metrics/best_hps.json must exist (written by hho_search.py)

    Output:
      results/checkpoints/best_model.pth
      results/checkpoints/final_model.pth
      results/metrics/train_metrics.json
      results/metrics/test_metrics.json
      results/visualizations/pred_XXXX.png (5 images)

    NOTE: This MUST be under `if __name__ == "__main__":` on macOS.
    DataLoader uses NUM_WORKERS=4 (multiprocessing). Without this guard,
    importing this file would spawn worker processes immediately.
    """
    run(visualize=True, num_vis_samples=5)
