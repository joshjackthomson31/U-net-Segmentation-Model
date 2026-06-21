# src/hho.py — Explained Simply

---

## What is this file for?

`hho.py` is the **brain of the optimization**. It implements the Harris Hawks Optimization (HHO) algorithm that automatically searches for the best U-Net hyperparameters.

Think of it as a **smart search engine**: instead of blindly trying thousands of random HP combinations, it uses intelligent strategies — inspired by how real Harris Hawks hunt — to focus its search on the most promising areas.

This file does NOT train the model itself. It receives a scoring function (`eval_fn`) from `hho_search.py` and calls it repeatedly to measure how good each HP combination is.

---

## The Core Idea in One Sentence

20 hawks hunt a rabbit (= best HP set found so far) across a 4D search space, using different strategies depending on how "tired" (close to being caught) the rabbit is.

---

## The 4D Search Space

Every hawk is a position in 4 dimensions. Each dimension represents one hyperparameter:

| Dimension | Hyperparameter | Internal Encoding | Range |
|---|---|---|---|
| 0 | Learning rate (`lr`) | log10(lr) | [-5.0, -2.0] → real: [1e-5, 1e-2] |
| 1 | Batch size | float index | [0.0, 3.0] → decoded to {2, 3, 4, 8} |
| 2 | Dropout | direct float | [0.1, 0.5] |
| 3 | Weight decay (`wd`) | log10(wd) | [-6.0, -1.0] → real: [1e-6, 1e-1] |

**Why log-space for lr and weight_decay?**

These parameters span thousands of orders of magnitude (lr goes from 0.00001 to 0.01 — a 1000× range). If we searched linearly, 99% of samples would cluster near the top of the range, and values like lr=1e-5 would almost never be explored. Log-space gives equal coverage at every order of magnitude.

Example:
- `hawk[0] = -3.0` → `lr = 10^(-3.0) = 0.001`
- `hawk[0] = -4.0` → `lr = 10^(-4.0) = 0.0001`
- Moving 1 unit in log-space = moving across one full order of magnitude

---

## Module-Level Constants

```python
BATCH_OPTIONS = [2, 3, 4, 8]   # from config.py SEARCH_SPACE
DIM = 4                          # dimensions of the search space

BOUNDS_LO = [-5.0, 0.0, 0.1, -6.0]   # lower bounds per dimension
BOUNDS_HI = [-2.0, 3.0, 0.5, -1.0]   # upper bounds per dimension
```

These are the "walls" of the search space. Hawks that fly outside get clipped back to the boundary. This prevents the algorithm from exploring impossible HP values (e.g., lr=1.0, which would be wildly too large).

---

## Function: `levy_flight(dim, beta)`

**What it does:** Generates a Lévy flight step — a random jump vector where most steps are small but occasional steps are very large.

**Why needed?**
When hawks cluster around a good solution (local maximum), they might miss an even better solution elsewhere. The Lévy jump can randomly teleport a hawk far away, giving it a chance to discover better regions.

**The math (Mantegna's algorithm, HHO paper Eq. 14):**

```python
sigma = [Gamma(1+beta)*sin(pi*beta/2) / (Gamma((1+beta)/2)*beta*2^((beta-1)/2))]^(1/beta)
u ~ Normal(0, sigma²)
v ~ Normal(0, 1)
LF = 0.01 * u / |v|^(1/beta)
```

**What beta=1.5 means:**
- β controls how "heavy-tailed" the distribution is
- β=1.5 (from HHO paper) = mostly small steps, but ~10% chance of a step that's 10× larger than typical
- Pure random walk would be β=2.0 (Gaussian). β=1.5 is more "adventurous"

**Simple analogy:**
Imagine an ant walking. Most steps are 1cm. But every so often it takes a 30cm leap. The Lévy flight is that occasional big leap — it helps hawks escape from local traps.

**Returns:** Array of shape `(dim,)` — one signed step size per dimension.

---

## Class: `HHO`

The entire algorithm lives inside this class. You create one HHO object, call `.run()`, and it returns the best hyperparameters found.

### Constructor: `__init__(self, eval_fn, seed)`

```python
hho = HHO(eval_fn=proxy_train_and_score, seed=42)
```

**Parameters:**
- `eval_fn`: The scoring function. Takes an HP dict, returns mIoU (float 0–1). Provided by `hho_search.py`.
- `seed`: Random seed for reproducibility.

**Sets up:**
- `self.eval_fn`: the scoring function
- `self._cache`: empty dict `{}` — will store `{HP_key: mIoU}` for every evaluated HP set

**Why a cache?**
- HHO evaluates up to 20 × 50 = 1000 HP sets during the search
- But batch_size is discrete (only 4 options), and hawks converge → many "different" hawk positions actually round to the same HP set
- The cache avoids re-running 5-epoch proxy training on the same HP set twice
- Typical cache hit rate: 15–35% (hundreds of expensive evaluations avoided)

---

### Private Method: `_clip(hawk)`

```python
hawk_safe = self._clip(hawk)  # np.clip(hawk, BOUNDS_LO, BOUNDS_HI)
```

Keeps a hawk inside the valid search space. Called every time a hawk moves to a new position.

**Example:**
```
hawk = [-1.0, 4.5, 0.7, -0.5]
       ↓ clip to bounds
     = [-2.0, 3.0, 0.5, -1.0]   ← each dimension clamped to [lo, hi]
```

---

### Private Method: `_decode(hawk)`

**What it does:** Converts internal hawk position (4 raw floats) → real HP dict that can be passed to `proxy_train`.

```python
hawk = [-3.021, 2.3, 0.300, -4.102]
         ↓
{
  "lr":           0.000952,    # 10^(-3.021)
  "batch_size":   4,           # round(2.3)=2, BATCH_OPTIONS[2]=4
  "dropout":      0.300,       # direct
  "weight_decay": 0.000079,    # 10^(-4.102)
}
```

**Batch size decoding:**
`hawk[1] = 2.3` → `round(2.3) = 2` → `BATCH_OPTIONS[2] = 4`

The batch index is a continuous float internally, but rounded to an integer at decode time. This lets HHO move smoothly through batch-size space and then snap to the nearest valid option.

---

### Private Method: `_cache_key(hps)`

**What it does:** Creates a hashable tuple from an HP dict, used as the cache lookup key.

```python
key = (
    round(log10(lr),           3),   # e.g. -3.021
    batch_size,                       # exact int: 4
    round(dropout,             3),   # e.g. 0.300
    round(log10(weight_decay), 3),   # e.g. -4.102
)
```

**Why round to 3 decimal places in log-space?**
Two hawks at positions `[-3.0209, 2.3, 0.3001, -4.1021]` and `[-3.0213, 2.3, 0.2998, -4.1019]` would decode to nearly identical HPs. Without rounding, they'd be treated as different cache entries, wasting a full proxy training run. With rounding to 3 decimal places, they share a cache hit and avoid the rerun.

---

### Private Method: `_evaluate(hawk)`

**What it does:** The single entry point for scoring any hawk position.

```python
def _evaluate(self, hawk):
    hawk_clipped = self._clip(hawk)          # 1. enforce bounds
    hps          = self._decode(hawk_clipped) # 2. convert to real HPs
    key          = self._cache_key(hps)       # 3. make cache key
    
    if key in self._cache:
        return self._cache[key]               # 4. cache hit: return instantly
    
    score            = self.eval_fn(hps)      # 5. cache miss: run proxy_train (~5-15 min)
    self._cache[key] = score                  # 6. store in cache
    return score
```

This is called for every hawk at every iteration. The cache check at step 4 saves hours of compute when hawks revisit similar HP regions.

---

### Private Method: `_init_population()`

**What it does:** Creates the initial 20 hawk positions, uniformly sampled across the 4D search space.

```python
X = np.random.uniform(low=BOUNDS_LO, high=BOUNDS_HI, size=(20, 4))
```

**Why uniform sampling?**
Because lr and wd are in log-space, uniform sampling here gives equal coverage per order of magnitude. This is the key benefit of log-encoding: the initial population naturally explores all LR scales.

**Returns:** `ndarray` of shape `(20, 4)` — one row per hawk, one column per HP dimension.

---

### Main Method: `run()`

This is the full HHO algorithm. It returns `(best_hps, best_score, history)`.

**Step-by-step walkthrough:**

---

#### Phase 1: Initialize (before any iterations)

```python
X       = self._init_population()                        # 20 hawks, random positions
fitness = [self._evaluate(X[i]) for i in range(N)]      # score all 20 hawks
```

This phase alone takes 1.7–5 hours (20 × 5-epoch proxy trains).

After scoring, the hawk with highest mIoU becomes the "rabbit":
```python
best_idx     = argmax(fitness)
rabbit_pos   = X[best_idx].copy()
rabbit_score = fitness[best_idx]
```

---

#### Phase 2: Main Iteration Loop (up to 50 iterations)

For each iteration `t` from 1 to 50:

**Step A: Compute escape energy (Eq. 3)**

```python
E0 = 2.0 * random() - 1.0        # random in [-1, 1]
E  = 2.0 * E0 * (1.0 - t / T)   # E decreases in magnitude as t increases
```

| Iteration | `|E|` typical value | Meaning |
|---|---|---|
| t=1 | ~2.0 | Very high energy → Exploration |
| t=25 | ~1.0 | Medium energy → Mixed |
| t=50 | ~0.0 | Very low energy → Exploitation |

`E` controls which attack strategy each hawk uses. Early in the search, `|E|` is large and hawks explore widely. Later, `|E|` drops and hawks converge on the best solution found.

**Step B: For each hawk, choose one of 5 strategies based on |E| and q:**

```
q  = random uniform [0, 1]
r1, r2, r3, r4, r5 = five independent random numbers
J  = 2 * (1 - r5)   # random jump strength
```

---

##### Strategy: Exploration (`|E| >= 1`)

Hawks have high energy. They spread out across the search space to find promising regions.

**If q >= 0.5 — Strategy A: Follow a random teammate**
```python
X_rand = X[random hawk]
new_x  = X_rand - r1 * |X_rand - 2*r2*X[i]|
```
Hawk i moves toward a random other hawk's position. Explores a different region of the space.

**If q < 0.5 — Strategy B: Spread relative to rabbit + mean**
```python
new_x = (rabbit_pos - X_mean) - r3 * (BOUNDS_LO + r4 * (BOUNDS_HI - BOUNDS_LO))
```
Hawks spread across the full search space, anchored by the known best position. Ensures global coverage.

---

##### Strategy: Soft Besiege (`|E| < 1`, `q >= 0.5`, `|E| >= 0.5`)

Rabbit is still somewhat energetic. Hawks encircle it slowly and tighten the ring.

```python
new_x = delta_X - E * |J * rabbit_pos - X[i]|
```
`delta_X = rabbit_pos - X[i]` = direction toward rabbit. Hawks move closer but with controlled randomness.

---

##### Strategy: Hard Besiege (`|E| < 1`, `q >= 0.5`, `|E| < 0.5`)

Rabbit is nearly exhausted. Hawks strike directly.

```python
new_x = rabbit_pos - E * |delta_X|
```
More aggressive move: hawk dives almost straight toward the rabbit position.

---

##### Strategy: Soft Besiege + Lévy Dives (`|E| < 1`, `q < 0.5`, `|E| >= 0.5`)

Hawk computes TWO possible attack paths and picks the better one.

```python
Y = rabbit_pos - E * |J * rabbit_pos - X[i]|    # standard dive
Z = Y + S * levy_flight(DIM)                      # Lévy-perturbed alternative

f_Y = evaluate(Y)
f_Z = evaluate(Z)
X[i] = whichever of Y, Z has higher mIoU
```

The Lévy jump in Z allows a surprise attack from an unexpected angle — prevents getting stuck at a local best.

---

##### Strategy: Hard Besiege + Lévy Dives (`|E| < 1`, `q < 0.5`, `|E| < 0.5`)

Same two-path approach but anchored to the mean position of all hawks:

```python
Y = rabbit_pos - E * |J * rabbit_pos - X_mean|   # anchored to hawk mean
Z = Y + S * levy_flight(DIM)                       # Lévy alternative
```

This is the most aggressive attack. Hawks coordinate their positions (via `X_mean`) for a joint strike with a surprise element.

---

**Step C: Update rabbit**
After every hawk moves:
```python
if fitness[i] > rabbit_score:
    rabbit_pos   = X[i].copy()
    rabbit_score = fitness[i]
```
The rabbit (best solution) is always updated. It never gets worse.

**Step D: Convergence check**
```python
improvement = rabbit_score - prev_best
if improvement < HHO_CONVERGENCE_THRESHOLD:
    no_improve_count += 1
else:
    no_improve_count = 0

if no_improve_count >= HHO_CONVERGENCE_PATIENCE:
    # STOP EARLY
```

If the best mIoU hasn't improved by at least `0.001` for 5 consecutive iterations, search stops early. This saves many hours if HHO has already converged.

---

#### Phase 3: Return Results

```python
best_hps = self._decode(rabbit_pos)
return best_hps, rabbit_score, history
```

`history` is a list of `(iteration, best_mIoU)` tuples — one entry per iteration. Used to plot the convergence curve.

---

## Summary: Decision Tree for Each Hawk at Each Iteration

```
Is |E| >= 1?
├── YES → EXPLORATION (spread out, find new regions)
│         q >= 0.5:  follow random hawk
│         q < 0.5:   spread across full space
│
└── NO  → EXPLOITATION (close in on rabbit)
          q >= 0.5:  direct attack (no Lévy)
          │           |E| >= 0.5:  Soft Besiege
          │           |E| < 0.5:   Hard Besiege
          │
          q < 0.5:   two-path attack (with Lévy surprise dive)
                      |E| >= 0.5:  Soft Besiege + Lévy
                      |E| < 0.5:   Hard Besiege + Lévy (most aggressive)
```

---

## Outputs

| Variable | Type | Description |
|---|---|---|
| `best_hps` | dict | `{"lr", "batch_size", "dropout", "weight_decay"}` — the winning HPs |
| `best_score` | float | mIoU (0–1) achieved by proxy training with `best_hps` |
| `history` | list | `[(0, 0.28), (1, 0.30), ..., (27, 0.41)]` — convergence curve |
| `self._cache` | dict | All evaluated HP sets and their scores — saved by `hho_search.py` |

---

## How this file connects to the rest

```
hho_search.py
    │
    ├── creates eval_fn = proxy_train wrapper
    │
    └── hho = HHO(eval_fn, seed=42)
              best_hps, score, history = hho.run()
                    │
                    ├── _init_population() → 20 random hawks
                    ├── _evaluate() → calls eval_fn → proxy_train (5 epochs)
                    │                 or returns from cache
                    │
                    ├── [exploration] hawks spread wide → _evaluate()
                    ├── [exploitation] hawks attack rabbit → _evaluate()
                    ├── levy_flight() → surprise jumps → _evaluate()
                    │
                    └── returns (best_hps, best_score, history)
                              ↓
                    hho_search.py saves best_hps.json
                              ↓
                    final_train.py reads best_hps.json → 20-epoch full training
```

---

## Key Numbers at a Glance

| Parameter | Value | Source |
|---|---|---|
| Hawks (population) | 20 | `HHO_POPULATION` in config.py |
| Max iterations | 50 | `HHO_MAX_ITERATIONS` in config.py |
| Lévy exponent β | 1.5 | `HHO_LEVY_BETA` in config.py (HHO paper) |
| Convergence patience | 5 | `HHO_CONVERGENCE_PATIENCE` in config.py |
| Convergence threshold | 0.001 | `HHO_CONVERGENCE_THRESHOLD` in config.py |
| Max unique evaluations | 1000 | 20 initial + 20×50 per iteration (minus cache hits) |
| Time per evaluation | 5–15 min | 5 proxy epochs on M4 MPS |
| Total search time | 10–29 hours | Depends on cache hits + early stopping |
