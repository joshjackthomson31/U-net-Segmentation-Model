# Harris Hawks Optimization (HHO) — Complete Explanation

---

## 1. What is HHO and why does it exist?

**The problem it solves:**
When you build a deep learning model like U-Net, you have several "settings" (called hyperparameters) that you must choose before training:
- How fast the model learns (learning rate)
- How many images to process at once (batch size)
- How much to randomly switch off neurons during training (dropout)
- How much to penalize large weights (weight decay)

If you pick these settings randomly or manually, you might get bad results. If you try every possible combination (called grid search), it would take months. HHO is a smarter way — it automatically finds good settings in reasonable time.

**Where the name comes from:**
HHO was invented in 2019 by Ali Asghar Heidari and colleagues. They were inspired by watching **Harris Hawks** — a bird species that hunts together in groups. These birds have a unique cooperative hunting strategy that researchers translated into a mathematical algorithm.

---

## 2. How Harris Hawks hunt in nature

Harris Hawks are unique because they hunt in teams (like wolves, but birds). Here's what they do in real life:

1. **Multiple hawks** sit on high perches (trees, poles, rocks) and watch for a rabbit
2. When one hawk spots a rabbit, it signals the others
3. The **strongest hawk** (the one with the best position to attack) leads the hunt
4. All other hawks **chase toward where the lead hawk is**, adjusting their positions
5. The rabbit tries to escape by running in random directions
6. The hawks **surround** the rabbit from multiple angles
7. Different hawks take different roles depending on how much energy the rabbit has left

The entire group cooperates to catch the rabbit. No single hawk makes all decisions.

---

## 3. Translating the hunt into mathematics

In HHO, every concept from the hawk hunt maps to something in optimization:

| Real World | HHO Mathematics |
|---|---|
| A hawk | A candidate set of hyperparameters |
| Where the hawk sits | The specific HP values (lr=0.001, batch=4, etc.) |
| The rabbit | The optimal hyperparameter set |
| Hawk's fitness | How good its HPs are (measured by mIoU) |
| The best hawk | The HP set with highest mIoU so far |
| Catching the rabbit | Finding the best HPs |
| Rabbit's escape energy | A number that decreases over time, controls algorithm phase |

**Population:** You start with 20 hawks. Each hawk is a different random combination of hyperparameters. For example:
- Hawk 1: lr=0.0001, batch=2, dropout=0.3, weight_decay=0.01
- Hawk 2: lr=0.005, batch=8, dropout=0.1, weight_decay=0.001
- Hawk 3: lr=0.0009, batch=4, dropout=0.45, weight_decay=0.0001
- ... and so on for all 20 hawks

---

## 4. The escape energy — the key concept

This is the most important concept in HHO. It is a single number called **E** that starts at a random value near 2 and decreases toward 0 as iterations progress.

Think of it like a rabbit's stamina:
- **When E is high (rabbit has lots of energy):** The rabbit runs far, randomly, unpredictably → Hawks must explore wide areas to find it → **Exploration phase**
- **When E is low (rabbit is tired):** The rabbit can only move small distances → Hawks can surround and attack precisely → **Exploitation phase**

The formula for E:
```
E = 2 × E₀ × (1 - current_iteration / max_iterations)
```
Where E₀ starts between -1 and 1 (random). As iterations pass from 1 to 50, E shrinks from ~2 to ~0.

---

## 5. The four attack phases — step by step

Depending on the value of E, HHO uses different strategies:

### Phase 1: Exploration (|E| ≥ 1) — "Rabbit hasn't been spotted yet"

Hawks move completely randomly to cover large areas. Each hawk either:
- Jumps toward where the rabbit was last seen (moves toward the best hawk's position)
- Sits on a random tall point (picks a completely random position in the search space)

In HP terms: Hawks try completely different combinations of learning rates, batch sizes, etc. This ensures we don't miss good regions of the search space.

### Phase 2: Soft Besiege (|E| < 0.5, rabbit escaping well)

The rabbit still has some energy but the hawks have surrounded it loosely. Hawks start moving toward the best position but with **random jumps** (called Lévy flight — explained below) to prevent being fooled by a fake escape direction.

### Phase 3: Hard Besiege (|E| < 0.5, rabbit not escaping well)

The hawks tighten the circle. Each hawk directly moves toward the average position of all hawks combined, then toward the best hawk's position. Very targeted attack.

### Phase 4: Soft Besiege with Progressive Rapid Dives (|E| ≥ 0.5 but rabbit still somewhat fresh)

Hawks try two things and pick the better one:
1. Move toward the best position + random jump (Lévy)
2. Move toward average position + random jump

The hawk picks whichever gives it better fitness (better mIoU).

---

## 6. What is Lévy Flight?

Lévy flight is a special type of random movement. It comes from physics — it describes how some animals (albatrosses, sharks) search for food.

**Normal random walk:** Small steps in random directions (like a drunk person walking). You never cover much ground.

**Lévy flight:** Mostly small steps, but occasionally one **very large step** in a random direction. This lets the algorithm escape local traps (places that seem good locally but are not globally best).

In our code (β = 1.5, from the paper):
```python
step = (gamma(1+β) × sin(π×β/2)) / (gamma((1+β)/2) × β × 2^((β-1)/2))
step_size = 0.01 × step × (hawk_position - rabbit_position)
new_position = rabbit_position + random_number × step_size
```

This creates occasional large jumps that help explore widely.

---

## 7. One complete iteration — what happens

**Step 1: Evaluate each hawk**
Every hawk flies to its current position (a set of HPs) and gets evaluated. Evaluation = run proxy_train (5-epoch mini training) with those HPs and measure val mIoU. That mIoU is the hawk's "fitness score."

**Step 2: Find the rabbit**
The hawk with the highest mIoU is called the Rabbit — it is the best solution found so far. Save it.

**Step 3: Calculate escape energy**
Compute E based on current iteration number. Determines which attack phase to use.

**Step 4: Update all hawk positions**
Based on which phase E puts us in (exploration, soft besiege, hard besiege, or rapid dives), update every hawk's position using the formulas above. Now each hawk has new HP values to try.

**Step 5: Clip to search space**
Make sure no hawk goes outside the valid ranges:
- lr stays between 1e-5 and 1e-2
- batch_size stays in {2, 3, 4, 8}
- dropout stays between 0.1 and 0.5
- weight_decay stays between 1e-6 and 1e-1

**Step 6: Check convergence**
If the best mIoU has not improved by more than 0.0001 for 5 consecutive iterations → stop early.

**Step 7: Repeat**
Go back to Step 1 for the next iteration. Do this up to 50 times.

---

## 8. The search space and log-space encoding

Our hyperparameters span very different scales:
- Learning rate: 0.00001 to 0.01 (5 orders of magnitude)
- Weight decay: 0.000001 to 0.1 (5 orders of magnitude)

If HHO searched these directly, it would spend 99% of its time near large values and almost never explore small ones.

**Solution: log-space encoding**

We store `log10(lr)` internally, not lr itself:
- log10(0.00001) = -5
- log10(0.01) = -2
- HHO searches the range [-5, -2] uniformly

Then when we need the actual lr to train with:
```python
actual_lr = 10 ** (internal_value)
```

**Batch size** is discrete ({2, 3, 4, 8}), so we store a continuous value and round it to the nearest valid option:
- 0.0–0.5 → batch_size = 2
- 0.5–0.75 → batch_size = 3
- 0.75–0.875 → batch_size = 4
- 0.875–1.0 → batch_size = 8

---

## 9. The fitness function — proxy training

For every hawk position evaluated, HHO needs to know "how good are these HPs?" It does this by:

1. Build a fresh U-Net with the hawk's dropout value
2. Train it for **5 epochs** (not 20 — too slow) on the training set
3. Evaluate on the validation set → get val mIoU
4. Return that mIoU as the fitness score

HHO wants to **maximize** this fitness score. The hawk that produces the highest 5-epoch val mIoU is the "Rabbit."

**Why only 5 epochs?**

| Scenario | Epochs per eval | Total compute |
|---|---|---|
| Full training per eval | 20 | 20 hawks × 50 iters × 20 epochs = 20,000 epoch-equivalents (weeks) |
| Proxy training per eval | 5 | 20 hawks × 50 iters × 5 epochs = 5,000 epoch-equivalents (overnight) |

5 epochs give enough signal to distinguish good HPs from bad ones, even if not perfectly accurate.

---

## 10. Caching — avoiding redundant computation

HHO sometimes moves multiple hawks to very similar positions. Re-training from scratch wastes time.

**Solution:** Cache every HP combination evaluated:
```
cache["lr=0.001,batch=4,dropout=0.3,wd=0.001"] = 0.3456
```

Before evaluating any hawk, check the cache. If we have seen these HPs before (within tolerance), reuse the stored mIoU. This saves 20–40% of computation time.

---

## 11. Total computation for our run

| Setting | Value |
|---|---|
| Hawks (population) | 20 |
| Max iterations | 50 |
| Proxy epochs per eval | 5 |
| Time per eval (approx.) | ~6 minutes on M4 MPS |
| Total search time (actual) | ~29 hours |

HHO's result: `lr=0.000223, batch=4, dropout=0.19, weight_decay=0.0002`

---

## 12. HHO vs GOA, GWO, GEO (paper's comparison)

The paper compared three algorithms. We implemented HHO (not in the paper). Here is how all four compare:

| Algorithm | Inspired by | Search style | Paper mIoU result |
|---|---|---|---|
| **GOA** (Grasshopper Optimization) | Grasshopper swarms | Attraction + repulsion forces | **67.97%** — best |
| **GWO** (Grey Wolf Optimizer) | Wolf pack hunting | Alpha/Beta/Delta leadership | 52.34% |
| **GEO** (Golden Eagle Optimizer) | Eagle swooping | Spiral attack path | 47.16% |
| **HHO** (Harris Hawks Optimization) | Hawk cooperative hunt | Escape energy phases | Not in paper — we implemented it |

GOA won in the paper because its balanced exploration-exploitation mechanism avoids premature convergence. GEO and GWO tend to converge too fast toward local optima.

HHO is considered a strong general-purpose optimizer. Whether it beats GOA on FloodNet specifically is what this project aims to find out.

---

## 13. What our HHO run found (and the current challenge)

**HHO search result:** `lr=0.000223, batch=4, dropout=0.19, weight_decay=0.0002`

**Problem discovered:** Our proxy training used class-weighted loss. The paper likely uses standard (unweighted) loss. HHO optimized for a different objective than what the final training uses → mismatched HPs.

**Corrected plan:**
1. ✅ Verify training code using paper's exact GOA HPs
2. 🔄 Re-run HHO search with corrected conditions (unweighted loss in proxy_train — already done in code)
3. ⬜ Run final training with HHO's new best HPs
4. ⬜ Goal: If HHO beats GOA's 67.97% → HHO is the contribution over the paper

---

## 14. One-line summary for your supervisor

> "HHO is a cooperative hunting algorithm — 20 hawks (each trying different training settings) explore the search space by sharing information about the best position found, gradually narrowing their search from wide exploration to fine-tuned precision, with the goal of automatically finding the hyperparameters that give U-Net its highest segmentation accuracy without any manual trial-and-error."
