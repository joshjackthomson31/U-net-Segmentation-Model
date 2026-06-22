"""
hho_search.py — Outer HHO search loop for U-Net hyperparameter tuning

What this file does:
  1. Computes class weights ONCE (scans 1445 training images — ~30 seconds)
  2. Wraps proxy_train into a single-argument function HHO can call
  3. Runs HHO: 20 hawks × up to 50 iterations, each hawk = 5-epoch proxy training
  4. Saves the best hyperparameters to results/metrics/best_hps.json
  5. Saves the full search history and cache for later analysis

Why compute class weights here (not inside proxy_train)?
  get_class_weights() scans all training images to count pixels per class.
  It takes ~30 seconds. If it were inside proxy_train, it would run for EVERY
  hawk evaluation: 20 hawks × 50 iterations = up to 1000 calls → ~8 hours wasted.
  Computing it ONCE here and passing it in = ~30 seconds total.

How long will HHO search take?
  One proxy_train (5 epochs, batch=2-8, 512×512) on Apple MPS ≈ 5-15 minutes.
  Initial population: 20 evaluations = 1.7-5 hours.
  Per iteration: up to 20 more (cache hits reduce this significantly).
  Up to 50 iterations, but early stopping at 5 consecutive no-improvement iterations.
  Realistic total: 10-24 hours. Run overnight.

Output files:
  results/metrics/best_hps.json       ← best hyperparameters (read by final_train.py)
  results/metrics/hho_history.json    ← mIoU per iteration (for plotting)
  results/metrics/hho_cache.json      ← all evaluated HP sets + their mIoU
"""

import os
import json
import time

from src.config  import (
    DEVICE, SEED, METRICS_DIR, RESULTS_DIR,
    HHO_POPULATION, HHO_MAX_ITERATIONS,
)
from src.dataset import get_class_weights
from src.train   import proxy_train
from src.hho     import HHO


# ─────────────────────────────────────────────
# EVALUATION WRAPPER
# ─────────────────────────────────────────────

def _make_eval_fn(class_weights, device):
    """
    Wrap proxy_train into a single-argument function for HHO.

    HHO calls eval_fn(hps) where hps = {"lr", "batch_size", "dropout", "weight_decay"}.
    proxy_train needs (hps, class_weights, device).
    This wrapper bakes class_weights and device in so HHO doesn't need to know about them.

    Also:
      - Tracks wall-clock time per evaluation (important: one call = 5-15 min on MPS)
      - Catches out-of-memory errors: returns 0.0 so HHO penalizes large batches
        rather than crashing the entire search

    Args:
        class_weights : torch.Tensor, shape (10,) — computed ONCE in run_search()
        device        : torch.device

    Returns:
        eval_fn: callable(hps: dict) -> float (mIoU in [0, 1])
    """
    call_count = [0]   # mutable container so inner function can update it

    def eval_fn(hps: dict) -> float:
        call_count[0] += 1
        call_num = call_count[0]

        print(f"\n  [Eval #{call_num}] HPs: lr={hps['lr']:.2e}  "
              f"batch={hps['batch_size']}  dropout={hps['dropout']:.3f}  "
              f"wd={hps['weight_decay']:.2e}")

        t_start = time.perf_counter()

        try:
            miou = proxy_train(hps, class_weights, device)
        except Exception as e:
            # Catch OOM (MemoryError on MPS, RuntimeError on CUDA) and any other crash.
            # Return 0.0 so HHO scores this HP set as worst and avoids it.
            # This prevents one bad batch_size=8 run from killing the whole search.
            elapsed = time.perf_counter() - t_start
            print(f"  [Eval #{call_num}] ERROR after {elapsed:.1f}s: {e}")
            print(f"  [Eval #{call_num}] Returning mIoU=0.0 (penalized)")
            return 0.0

        elapsed = time.perf_counter() - t_start
        print(f"  [Eval #{call_num}] mIoU={miou:.4f}  ({elapsed/60:.1f} min)")
        return miou

    return eval_fn


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────

def _save_results(best_hps: dict, best_score: float, history: list, cache: dict) -> None:
    """
    Save all HHO outputs to results/metrics/.

    Files saved:
      best_hps.json     ← what final_train.py reads
      hho_history.json  ← [(iteration, best_mIoU), ...] for plotting
      hho_cache.json    ← all evaluated HP sets and their mIoU (hours of compute)

    The cache is saved because it represents all proxy_train evaluations.
    If you want to re-run analysis or plot the search landscape,
    you don't need to retrain — just load this file.
    """
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── best_hps.json ─────────────────────────────────────────────────────────
    best_hps_path = os.path.join(METRICS_DIR, "best_hps.json")
    with open(best_hps_path, "w") as f:
        json.dump({"best_hps": best_hps, "best_miou": best_score}, f, indent=2)
    print(f"[HHO Search] Saved best HPs → {best_hps_path}")

    # ── hho_history.json ──────────────────────────────────────────────────────
    history_path = os.path.join(METRICS_DIR, "hho_history.json")
    # Convert list of tuples to list of dicts for JSON readability
    history_json = [{"iteration": it, "best_miou": sc} for it, sc in history]
    with open(history_path, "w") as f:
        json.dump(history_json, f, indent=2)
    print(f"[HHO Search] Saved history  → {history_path}")

    # ── hho_cache.json ────────────────────────────────────────────────────────
    # Cache keys are tuples — convert to strings for JSON
    cache_path = os.path.join(METRICS_DIR, "hho_cache.json")
    cache_json = {str(k): v for k, v in cache.items()}
    with open(cache_path, "w") as f:
        json.dump(cache_json, f, indent=2)
    print(f"[HHO Search] Saved cache    → {cache_path}  ({len(cache)} entries)")


# ─────────────────────────────────────────────
# MAIN SEARCH FUNCTION
# ─────────────────────────────────────────────

def run_search(device=None, resume=False) -> dict:
    """
    Run the full HHO hyperparameter search.

    Steps:
      1. Compute class weights (once)
      2. Build HHO with eval_fn wrapper
      3. Run HHO search (20 hawks, up to 50 iterations)
      4. Save best_hps, history, and cache
      5. Return best_hps dict

    Args:
        device : torch.device (defaults to config DEVICE)

    Returns:
        best_hps : dict with keys lr, batch_size, dropout, weight_decay
    """
    if device is None:
        device = DEVICE

    print("=" * 60)
    print("HHO Hyperparameter Search for U-Net (FloodNet)")
    print(f"Device : {device}")
    print(f"Hawks  : {HHO_POPULATION}")
    print(f"Max iter: {HHO_MAX_ITERATIONS}")
    print("=" * 60)

    # ── Step 1: Class weights (computed ONCE) ─────────────────────────────────
    print("\n[Step 1] Computing class weights from training set...")
    print("  (Scans 1445 images — takes ~30 seconds)\n")
    t0 = time.perf_counter()
    class_weights = get_class_weights()
    print(f"  Done in {time.perf_counter() - t0:.1f}s\n")

    # ── Step 2: Build eval function ───────────────────────────────────────────
    eval_fn = _make_eval_fn(class_weights, device)

    # ── Step 3: Run HHO with incremental checkpointing + optional resume ─────
    print("[Step 2] Starting HHO search...\n")

    checkpoint_path = os.path.join(METRICS_DIR, "hho_checkpoint.json")
    os.makedirs(METRICS_DIR, exist_ok=True)

    def _checkpoint(state_dict):
        """
        Write FULL algorithm state to disk after every completed iteration.

        state_dict contains: hawk positions, fitness, rabbit pos/score,
        convergence counters, numpy RNG state, cache, history.

        On resume, this exact dict is passed back to hho.run(resume_state=...)
        to restore the search exactly where it left off.
        """
        # Add human-readable summary at top level for easy inspection
        state_dict["_best_hps_readable"] = state_dict.get("rabbit_pos")
        state_dict["_iterations_done"]   = state_dict.get("t_last", 0)
        with open(checkpoint_path, "w") as f:
            json.dump(state_dict, f, indent=2)

    # ── Resume from checkpoint if requested ──────────────────────────────────
    resume_state = None
    if resume:
        if os.path.exists(checkpoint_path):
            print(f"  [Resume] Loading checkpoint: {checkpoint_path}")
            with open(checkpoint_path, "r") as f:
                resume_state = json.load(f)
            t_done = resume_state.get("t_last", 0)
            best   = resume_state.get("rabbit_score", 0.0)
            cached = len(resume_state.get("cache", {}))
            print(f"  [Resume] Iteration {t_done} completed, best mIoU={best:.4f}, "
                  f"{cached} HP sets cached (no retraining needed).\n")
        else:
            print(f"  [Resume] No checkpoint found at {checkpoint_path} — starting fresh.\n")
    else:
        print(f"  Progress auto-saved after every iteration → {checkpoint_path}")
        print(f"  If interrupted, restart with: python main.py search --resume\n")

    t_search_start = time.perf_counter()

    hho = HHO(eval_fn=eval_fn, seed=SEED)
    best_hps, best_score, history = hho.run(
        checkpoint_fn=_checkpoint,
        resume_state=resume_state,
    )

    t_search_total = time.perf_counter() - t_search_start
    print(f"\n[HHO Search] Total search time: {t_search_total/3600:.2f} hours")

    # ── Step 4: Save final results ────────────────────────────────────────────
    _save_results(best_hps, best_score, history, hho._cache)

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SEARCH COMPLETE")
    print(f"Best mIoU  : {best_score:.4f}")
    print(f"Best HPs   :")
    for k, v in best_hps.items():
        print(f"  {k:<15s} = {v}")
    print("\nNext step: run experiments/final_train.py")
    print("=" * 60)

    return best_hps


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run as: python -m experiments.hho_search

    IMPORTANT: This MUST be under `if __name__ == "__main__":` on macOS.
    DataLoader uses NUM_WORKERS=4 (multiprocessing). Without this guard,
    importing this file would spawn worker processes immediately, causing
    a fork-bomb that crashes your terminal.
    """
    run_search()
