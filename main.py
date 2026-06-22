"""
main.py — Single entry point for HHO-U-Net on FloodNet

Usage:
  python main.py search       <- Step 1: Run HHO to find best hyperparameters
  python main.py train        <- Step 2: Full 20-epoch training + test evaluation
  python main.py evaluate     <- Re-evaluate best_model.pth on test set (anytime)
  python main.py sanity       <- Quick check: model builds and forward pass works

Run them IN ORDER:
  python main.py search       (overnight — finds best learning rate, batch size, etc.)
  python main.py train        (next day — trains the final model and shows results)

What each command does:
  search   : 20 hawks × up to 50 iterations of 5-epoch proxy training.
             Saves best hyperparameters to results/metrics/best_hps.json.

  train    : Reads best_hps.json, trains U-Net for 20 full epochs.
             Saves best_model.pth, train_metrics.json, test_metrics.json,
             and 5 side-by-side prediction images.

  evaluate : Loads best_model.pth and re-runs evaluation on test set.
             Useful if you want to change evaluation settings without retraining.

  sanity   : Builds the U-Net, runs one dummy image through it, checks output shape.
             Run this FIRST to confirm your environment is set up correctly.
             Takes ~10 seconds. No dataset required.
"""

import sys


def cmd_sanity():
    """
    Quick sanity check — no dataset needed.

    Builds the U-Net with paper settings (base_filters=32, dropout=0.3)
    and runs a dummy forward pass to confirm:
      - All imports work
      - Model builds without errors
      - Output shape is correct: (2, 10, 512, 512)
      - Device (MPS/CUDA/CPU) is detected correctly
    """
    import torch
    from src.config import DEVICE, NUM_CLASSES, BACKBONE
    from src.unet   import build_unet

    print("\n" + "=" * 50)
    print("Sanity Check")
    print("=" * 50)
    print(f"Device      : {DEVICE}")
    print(f"Num classes : {NUM_CLASSES}")
    print(f"Backbone    : {BACKBONE}")

    model = build_unet(dropout=0.3, backbone=BACKBONE).to(DEVICE)

    dummy = torch.randn(2, 3, 512, 512).to(DEVICE)
    out   = model(dummy)

    assert out.shape == (2, NUM_CLASSES, 512, 512), \
        f"Wrong output shape: {out.shape}"

    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print("\nSanity check PASSED. Environment is ready.")
    print("=" * 50 + "\n")


def cmd_search():
    """Run HHO hyperparameter search (overnight). Pass --resume to continue from checkpoint."""
    from experiments.hho_search import run_search
    resume = "--resume" in sys.argv
    run_search(resume=resume)


def cmd_train():
    """Run full 20-epoch training with best HPs, then evaluate on test set."""
    from experiments.final_train import run
    run(visualize=True, num_vis_samples=5)


def cmd_evaluate():
    """Re-evaluate best_model.pth on the test set."""
    from src.evaluate import run_evaluation
    metrics = run_evaluation(visualize=False)
    print(f"\nTest mIoU: {metrics['miou']:.4f}")


# ─────────────────────────────────────────────
# COMMAND DISPATCH
# ─────────────────────────────────────────────

COMMANDS = {
    "sanity":   cmd_sanity,
    "search":   cmd_search,
    "train":    cmd_train,
    "evaluate": cmd_evaluate,
}

HELP = """
HHO-U-Net for FloodNet Semantic Segmentation
=============================================

Commands:
  python main.py sanity          Quick check: model builds correctly (run this first)
  python main.py search          Run HHO to find best hyperparameters (overnight)
  python main.py search --resume Resume search from last saved checkpoint
  python main.py train           Full training + test evaluation (run after search)
  python main.py evaluate        Re-evaluate best_model.pth on test set

Typical workflow:
  1. python main.py sanity              <- confirm setup works
  2. caffeinate -is python main.py search   <- run overnight (plugged in!)
     (if interrupted: caffeinate -is python main.py search --resume)
  3. python main.py train               <- final training + results
"""


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(HELP)
        sys.exit(0)

    COMMANDS[sys.argv[1]]()
