# experiments/hho_search.py — Explained Simply

---

## What is this file for?

`hho_search.py` is the **orchestrator of the HHO hyperparameter search**. It connects the HHO algorithm (`src/hho.py`) with the U-Net training function (`proxy_train` in `src/train.py`) to automatically find the best hyperparameters.

Think of it as a **project manager**:
- It computes what's needed (class weights) once upfront
- It translates between what HHO wants (call a function, get a score) and what training needs (HPs, weights, device)
- It saves all results when the search finishes

---

## What does it do?

1. **Computes class weights once** — scanning 1445 training images takes ~53 seconds. If done inside the evaluation function, it would repeat 1000 times (~14 hours wasted).
2. **Wraps `proxy_train` into a single-argument function** — HHO needs to call `eval_fn(hps)`, but `proxy_train` needs three arguments. This wrapper "bakes in" the fixed arguments.
3. **Runs HHO search** — 20 hawks × up to 50 iterations × 5-epoch proxy training each
4. **Saves all outputs** — best HPs, search history, evaluation cache

---

## Function: `_make_eval_fn(class_weights, device)`

**What it does:** Creates and returns the evaluation function that HHO will call repeatedly.

**Why a function that returns a function?**
HHO's interface requires `eval_fn(hps) → float`. But `proxy_train` requires `(hps, class_weights, device)`. Instead of changing HHO's interface, we "pre-fill" the fixed arguments and return a simpler function.

This pattern is called a **closure** or **factory function**.

**Inside `eval_fn(hps)`:**

```python
call_count[0] += 1    # track how many evaluations total
```
Uses a list `[0]` instead of a simple integer because inner functions in Python can't modify outer variables directly — but they CAN modify the contents of a list. This is a standard Python workaround.

```python
print(f"[Eval #{call_num}] HPs: lr={hps['lr']:.2e}  batch={hps['batch_size']}  ...")
```
Prints every evaluation so you can watch progress. Useful for overnight runs — you can check on it and see what's happening.

```python
t_start = time.perf_counter()
try:
    miou = proxy_train(hps, class_weights, device)
except Exception as e:
    print(f"[Eval #{call_num}] ERROR: {e}")
    return 0.0   # penalize this HP set — don't crash the search
elapsed = time.perf_counter() - t_start
print(f"[Eval #{call_num}] mIoU={miou:.4f}  ({elapsed/60:.1f} min)")
return miou
```

**The try/except is critical:**
Large batch sizes (e.g., batch=8 with 512×512 images) can run out of GPU memory (OOM error). Without catching this, one bad hawk configuration would crash the entire overnight search — losing all progress. By catching and returning `0.0`, HHO simply scores that HP set as worst and avoids it in future iterations.

---

## Function: `_save_results(best_hps, best_score, history, cache)`

**What it does:** Saves all HHO outputs to `results/metrics/` after the search completes.

**Three files saved:**

### 1. `best_hps.json` — Used by final_train.py
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
This is the ONLY file that `final_train.py` reads. Contains the winner: the HP set that achieved the highest proxy val mIoU.

### 2. `hho_history.json` — Search convergence over time
```json
[
  {"iteration": 1, "best_miou": 0.2341},
  {"iteration": 2, "best_miou": 0.2567},
  {"iteration": 3, "best_miou": 0.2890},
  ...
]
```
Shows how the best mIoU improved (or didn't) over each iteration. Useful for plotting a convergence curve.

### 3. `hho_cache.json` — All evaluated HP combinations and their scores
```json
{
  "(0.001, 4, 0.3, 0.001)": 0.3045,
  "(0.0001, 2, 0.2, 0.01)": 0.1823,
  ...
}
```
Contains every unique HP combination evaluated during the search and its mIoU. This represents hours of computation. If you need to re-analyze the search landscape or plot a heatmap, you don't need to retrain — just load this file.

**Why convert cache keys to strings?**
Cache keys are Python tuples like `(0.001, 4, 0.3, 0.001)`. JSON doesn't support tuple keys — it only supports strings. So we convert: `str((0.001, 4, 0.3, 0.001))` → `"(0.001, 4, 0.3, 0.001)"`.

---

## Function: `run_search(device)`

**What it does:** The main function that runs the complete HHO search pipeline.

**Step by step:**

**Step 1: Compute class weights once**
```python
class_weights = get_class_weights()
```
Scans 1444 training images, counts pixels per class, computes inverse-frequency weights (with clamp). Takes ~53 seconds. Passed to every `proxy_train` call so it's only computed once.

**Step 2: Create the evaluation wrapper**
```python
eval_fn = _make_eval_fn(class_weights, device)
```
Creates the single-argument function that HHO can call.

**Step 3: Run HHO**
```python
hho = HHO(eval_fn=eval_fn, seed=SEED)
best_hps, best_score, history = hho.run()
```
Instantiates the HHO algorithm and runs it. Internally, HHO:
- Initializes 20 hawks randomly
- Evaluates each hawk (calls `eval_fn(hps)`)
- Updates hawk positions based on escape energy and attack phases
- Repeats for up to 50 iterations (or until convergence)
- Returns the best HP set and the full iteration history

**Step 4: Save everything**
```python
_save_results(best_hps, best_score, history, hho._cache)
```

**Step 5: Print summary**
```
SEARCH COMPLETE
Best mIoU  : 0.3048
Best HPs   :
  lr              = 0.000223
  batch_size      = 4
  dropout         = 0.190
  weight_decay    = 0.0002

Next step: run experiments/final_train.py
```

**Returns:** The best HPs dict.

---

## The `if __name__ == "__main__":` Guard

```python
if __name__ == "__main__":
    run_search()
```

**Why is this critical on macOS?**
DataLoader uses `NUM_WORKERS=4` (4 parallel data loading processes). Python's multiprocessing on macOS uses the "spawn" method — it re-imports the entire module for each worker process.

Without the guard: when a worker process imports `hho_search.py`, it would immediately call `run_search()` again, creating more workers, which create more workers → **fork bomb** (exponential process explosion that crashes your terminal).

With the guard: `run_search()` only runs when the file is the main entry point, not when imported by worker processes.

---

## How this file fits in the project

```
python main.py search
         │
         └──→ experiments/hho_search.py
                    │
                    ├──→ src/dataset.py: get_class_weights()
                    │
                    ├──→ src/hho.py: HHO algorithm
                    │         │
                    │         └──→ calls eval_fn(hps) up to 20×50=1000 times
                    │                      │
                    │                      └──→ src/train.py: proxy_train(hps, ...)
                    │                                │
                    │                                └──→ src/unet.py: build_unet()
                    │                                └──→ src/dataset.py: get_dataloaders()
                    │
                    └──→ saves results/metrics/best_hps.json  ← read by final_train.py
                         saves results/metrics/hho_history.json
                         saves results/metrics/hho_cache.json
```

---

## How long does the search take?

| Component | Time estimate |
|---|---|
| Class weight computation | ~53 seconds (once) |
| One proxy_train evaluation | 5–15 minutes on M4 MPS |
| Initial population (20 evaluations) | 1.7–5 hours |
| Each subsequent iteration (up to 20 evals, minus cache hits) | variable |
| Early stopping (patience=5) | stops if no improvement for 5 iterations |
| **Total (typical)** | **10–29 hours** |

Run with: `python main.py search` — leave it running overnight.
