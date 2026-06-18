"""
hho.py — Harris Hawks Optimization for U-Net hyperparameter tuning

Implements: Heidari, A.A. et al. (2019). Harris Hawks Optimization: Algorithm and
            Applications. Future Generation Computer Systems, 97, 849–872.
            (Equations 1–8, 14)

Simple analogy:
  20 hawks hunt a rabbit (= best HP set found so far).
  - High energy (|E| ≥ 1):  hawks roam freely          → explore HP landscape
  - Low energy  (|E| < 1):  hawks close in on rabbit   → exploit good HPs
  - Lévy flight:             occasional big random jump → escape local traps
  After up to 50 iterations, the rabbit's location = best hyperparameters.

Hyperparameters tuned (SEARCH_SPACE from config.py):
  lr           : [10^-5, 10^-2]  → encoded as log10 in [-5.0, -2.0]   (continuous)
  batch_size   : {2, 3, 4, 8}    → encoded as float index  [0.0, 3.0]  (discrete)
  dropout      : [0.1, 0.5]      → stored directly                      (continuous)
  weight_decay : [10^-6, 10^-1]  → encoded as log10 in [-6.0, -1.0]   (continuous)

WHY log-space for lr and weight_decay?
  These parameters span 1,000x to 100,000x ranges. If we sample or move linearly,
  ~99% of samples fall above lr=1e-3 and the model never explores lr=1e-5 or 1e-4.
  Log-space makes every order-of-magnitude equally reachable.
  Evidence from paper: GOA_BEST_HP["lr"] = 10**(-3.02) — exponent form confirms log-space.

Each hawk = 4D position vector [log10(lr), batch_idx, dropout, log10(weight_decay)].
Fitness    = mIoU from 5-epoch proxy training (provided by eval_fn from hho_search.py).
"""

import math
import numpy as np

from src.config import (
    HHO_POPULATION, HHO_MAX_ITERATIONS, HHO_LEVY_BETA,
    HHO_CONVERGENCE_THRESHOLD, HHO_CONVERGENCE_PATIENCE,
    SEARCH_SPACE, SEED,
)

# ─────────────────────────────────────────────
# SEARCH SPACE BOUNDS  (log-space for lr & weight_decay)
# ─────────────────────────────────────────────

BATCH_OPTIONS = SEARCH_SPACE["batch_size"]   # [2, 3, 4, 8]
DIM = 4   # [log10(lr), batch_idx, dropout, log10(weight_decay)]

# Internally hawks fly in this 4D space.
# Dimension 0: log10(lr)           [-5.0, -2.0]
# Dimension 1: batch_size index    [ 0.0,  3.0]   → decoded to {2,3,4,8}
# Dimension 2: dropout             [ 0.1,  0.5]
# Dimension 3: log10(weight_decay) [-6.0, -1.0]
BOUNDS_LO = np.array([
    math.log10(SEARCH_SPACE["lr"][0]),            # log10(1e-5) = -5.0
    0.0,                                           # batch_size index min
    SEARCH_SPACE["dropout"][0],                   # 0.1
    math.log10(SEARCH_SPACE["weight_decay"][0]),  # log10(1e-6) = -6.0
], dtype=np.float64)

BOUNDS_HI = np.array([
    math.log10(SEARCH_SPACE["lr"][1]),            # log10(1e-2) = -2.0
    float(len(BATCH_OPTIONS) - 1),                # 3.0
    SEARCH_SPACE["dropout"][1],                   # 0.5
    math.log10(SEARCH_SPACE["weight_decay"][1]),  # log10(1e-1) = -1.0
], dtype=np.float64)


# ─────────────────────────────────────────────
# LÉVY FLIGHT  (Heidari et al. 2019, Eq. 14)
# ─────────────────────────────────────────────

def levy_flight(dim: int, beta: float = HHO_LEVY_BETA) -> np.ndarray:
    """
    Lévy flight step vector — mostly small steps, occasional big jumps.

    Uses Mantegna's algorithm (cited in HHO paper, Eq. 14):
      sigma = [Gamma(1+beta)*sin(pi*beta/2) / (Gamma((1+beta)/2)*beta*2^((beta-1)/2))]^(1/beta)
      u ~ N(0, sigma^2),   v ~ N(0, 1)
      LF = 0.01 * u / |v|^(1/beta)

    Simple analogy: imagine an ant walking. Most steps are tiny.
    But once in a while, it takes a huge leap across the room.
    That leap is the Lévy jump — it prevents the hawks from
    staying stuck searching the same small area forever.

    Args:
        dim  : number of dimensions (4 for our HP space)
        beta : Lévy exponent — HHO_LEVY_BETA = 1.5 (from config, from HHO paper)

    Returns:
        step : ndarray of shape (dim,) — signed step sizes
    """
    num   = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den   = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma = (num / den) ** (1.0 / beta)

    u = np.random.normal(0.0, sigma, size=dim)
    v = np.random.normal(0.0, 1.0,  size=dim)

    return 0.01 * u / (np.abs(v) ** (1.0 / beta))


# ─────────────────────────────────────────────
# HHO CLASS
# ─────────────────────────────────────────────

class HHO:
    """
    Harris Hawks Optimization.

    Maximizes mIoU by searching the 4D hyperparameter space.
    Calls eval_fn for each unique HP combination evaluated.

    Args:
        eval_fn : callable(hps: dict) -> float
                  hps = {"lr": float, "batch_size": int,
                         "dropout": float, "weight_decay": float}
                  Must return mIoU in [0, 1].  Provided by hho_search.py.
        seed    : RNG seed for reproducibility (default: SEED from config).

    Example:
        hho = HHO(eval_fn=proxy_train_and_score)
        best_hps, best_score, history = hho.run()
        # best_hps   = {"lr": 9.55e-4, "batch_size": 4, "dropout": 0.3, "weight_decay": 7.94e-4}
        # best_score = 0.743
        # history    = [(0, 0.61), (1, 0.64), ..., (27, 0.743)]
    """

    def __init__(self, eval_fn, seed: int = SEED):
        self.eval_fn = eval_fn
        np.random.seed(seed)
        # Cache: decoded HPs -> mIoU
        # Prevents redundant retraining when multiple hawks converge to same HP set
        # (especially common with discrete batch_size or when hawks cluster together)
        self._cache: dict = {}

    # ── Private helpers ───────────────────────────────────────────────────

    def _clip(self, hawk: np.ndarray) -> np.ndarray:
        """Clip hawk position to stay within valid search space bounds."""
        return np.clip(hawk, BOUNDS_LO, BOUNDS_HI)

    def _decode(self, hawk: np.ndarray) -> dict:
        """
        Convert internal hawk position -> real hyperparameter dict.

        Encoding:
          hawk[0] = log10(lr)    -> 10**hawk[0]   (e.g. -3.0 -> 0.001)
          hawk[1] = batch index  -> round -> BATCH_OPTIONS[idx]
          hawk[2] = dropout      -> directly (already in [0.1, 0.5])
          hawk[3] = log10(wd)    -> 10**hawk[3]   (e.g. -4.0 -> 0.0001)

        Example:
          hawk = [-3.0, 2.3, 0.3, -4.0]
          -> {"lr": 0.001, "batch_size": 4, "dropout": 0.3, "weight_decay": 0.0001}
        """
        bs_idx = int(round(float(np.clip(hawk[1], 0, len(BATCH_OPTIONS) - 1))))
        return {
            "lr":           float(10.0 ** hawk[0]),
            "batch_size":   BATCH_OPTIONS[bs_idx],
            "dropout":      float(hawk[2]),
            "weight_decay": float(10.0 ** hawk[3]),
        }

    def _cache_key(self, hps: dict) -> tuple:
        """
        Hashable cache key for an HP dict.

        Round continuous values to 3 decimal places in log-space so that
        near-identical hawk positions share a cache entry and avoid
        redundant proxy training runs.
        """
        return (
            round(math.log10(hps["lr"]),           3),  # e.g. -3.021
            hps["batch_size"],                           # exact int
            round(hps["dropout"],                  3),  # e.g. 0.300
            round(math.log10(hps["weight_decay"]), 3),  # e.g. -4.102
        )

    def _evaluate(self, hawk: np.ndarray) -> float:
        """
        Clip, decode, and score one hawk position.
        Returns cached mIoU if this HP set was already evaluated.
        """
        hawk_clipped = self._clip(hawk)
        hps          = self._decode(hawk_clipped)
        key          = self._cache_key(hps)

        if key in self._cache:
            return self._cache[key]

        score            = self.eval_fn(hps)
        self._cache[key] = score
        return score

    def _init_population(self) -> np.ndarray:
        """
        Uniformly sample N hawk positions across the 4D search space.

        Because lr and weight_decay are in log-space, uniform sampling here
        gives equal coverage per order of magnitude (e.g. equal chance of
        landing in [1e-5,1e-4] vs [1e-3,1e-2]).

        Returns:
            population : ndarray of shape (N, DIM)
        """
        return np.random.uniform(
            low=BOUNDS_LO,
            high=BOUNDS_HI,
            size=(HHO_POPULATION, DIM),
        )

    # ── Main algorithm ────────────────────────────────────────────────────

    def run(self):
        """
        Execute HHO search and return the best hyperparameters found.

        Algorithm (Heidari et al. 2019):
          1. Initialize N hawks randomly in log-space search space
          2. Evaluate each hawk -> fitness (mIoU from 5-epoch proxy training)
          3. Rabbit = hawk with highest mIoU
          4. Repeat for up to T iterations:
             a. E0 = rand[-1,1],  E = 2*E0*(1 - t/T)             [Eq. 3]
             b. For each hawk:
                  |E| >= 1 -> Exploration (Eqs. 1–2): random perching
                  |E| <  1 -> Exploitation (Eqs. 4–8): 4 attack strategies
             c. Clip positions to valid bounds
             d. Evaluate; update rabbit if any hawk improved
             e. Convergence: stop if best mIoU improved by less than
                HHO_CONVERGENCE_THRESHOLD for HHO_CONVERGENCE_PATIENCE
                consecutive iterations.
                Note: checking monotone best-score improvement is equivalent to
                checking average-population improvement once hawks converge.
          5. Return best HPs, best score, iteration history

        Returns:
            best_hps    (dict)  : {"lr", "batch_size", "dropout", "weight_decay"}
            best_score  (float) : mIoU of best HPs, range [0, 1]
            history     (list)  : [(iteration, best_mIoU), ...] one entry per iteration
        """
        N = HHO_POPULATION
        T = HHO_MAX_ITERATIONS

        print(f"\n{'='*60}")
        print(f"[HHO] Search: {N} hawks, up to {T} iterations, beta={HHO_LEVY_BETA}")
        print(f"[HHO] Convergence: patience={HHO_CONVERGENCE_PATIENCE}, "
              f"threshold={HHO_CONVERGENCE_THRESHOLD}")
        print(f"{'='*60}")

        # ── 1. Initialize ──────────────────────────────────────────────────
        X       = self._init_population()                               # (N, 4)
        print(f"\n[HHO] Evaluating {N} initial hawks (proxy training)...")
        fitness = np.array([self._evaluate(X[i]) for i in range(N)])   # (N,)

        # ── 2. Rabbit = hawk with highest mIoU ────────────────────────────
        best_idx     = int(np.argmax(fitness))
        rabbit_pos   = X[best_idx].copy()
        rabbit_score = float(fitness[best_idx])

        print(f"[HHO] Initial best mIoU: {rabbit_score:.4f} -> {self._decode(rabbit_pos)}\n")

        history          = [(0, rabbit_score)]
        no_improve_count = 0
        prev_best        = rabbit_score

        # ── 3. Main iterations ─────────────────────────────────────────────
        for t in range(1, T + 1):

            # Escape energy (Eq. 3): magnitude decreases from ~2 to 0 as t -> T
            # High |E| = hawk has energy to explore; low |E| = hawk closes in
            E0 = 2.0 * np.random.random() - 1.0         # E0 in [-1, 1]
            E  = 2.0 * E0 * (1.0 - float(t) / float(T))

            X_mean = X.mean(axis=0)   # mean position of all hawks (used in Eqs. 1, 8)

            for i in range(N):
                q            = np.random.random()   # selects attack strategy [0,1]
                r1, r2, r3, r4, r5 = np.random.random(5)
                J            = 2.0 * (1.0 - r5)     # jump strength in (0, 2)  [Eq. 2]

                # ═══ EXPLORATION PHASE  |E| >= 1 ══════════════════════════════
                # Hawks have lots of energy. They perch on random trees and wait,
                # scouting the HP landscape before committing to an attack.

                if abs(E) >= 1.0:

                    if q >= 0.5:
                        # Strategy A (Eq. 1, row 1): perch near a random hawk's position
                        # Analogy: follow a teammate's lead to a different area
                        X_rand = X[np.random.randint(N)]
                        new_x  = X_rand - r1 * np.abs(X_rand - 2.0 * r2 * X[i])
                    else:
                        # Strategy B (Eq. 1, row 2): perch based on rabbit + mean position
                        # Analogy: spread out across the landscape, anchored to known best
                        new_x = (rabbit_pos - X_mean) - r3 * (
                            BOUNDS_LO + r4 * (BOUNDS_HI - BOUNDS_LO)
                        )

                    X[i]       = self._clip(new_x)
                    fitness[i] = self._evaluate(X[i])

                # ═══ EXPLOITATION PHASE  |E| < 1 ══════════════════════════════
                # Hawks sense a tired rabbit. They choose one of 4 attack strategies
                # based on |E| (how tired the rabbit is) and q (random choice).

                else:
                    delta_X = rabbit_pos - X[i]   # direction vector toward rabbit

                    if q >= 0.5:
                        # ── No Lévy flight ─────────────────────────────────────
                        if abs(E) >= 0.5:
                            # Strategy 1 — Soft besiege (Eq. 4)
                            # Rabbit still has some energy. Hawks encircle slowly,
                            # cutting off escape routes, letting it tire more.
                            new_x = delta_X - E * np.abs(J * rabbit_pos - X[i])
                        else:
                            # Strategy 2 — Hard besiege (Eq. 6)
                            # Rabbit is nearly exhausted. Hawks strike rapidly.
                            new_x = rabbit_pos - E * np.abs(delta_X)

                        X[i]       = self._clip(new_x)
                        fitness[i] = self._evaluate(X[i])

                    else:
                        # ── With Lévy flight (surprise dives) ─────────────────
                        # Hawks compute two dive paths (Y and Z) and pick the better.
                        S  = np.random.random(DIM)   # random scaling vector
                        LF = levy_flight(DIM)         # Lévy step (big jump possible)

                        if abs(E) >= 0.5:
                            # Strategy 3 — Soft besiege + rapid dives (Eq. 7)
                            Y = rabbit_pos - E * np.abs(J * rabbit_pos - X[i])
                        else:
                            # Strategy 4 — Hard besiege + rapid dives (Eq. 8)
                            Y = rabbit_pos - E * np.abs(J * rabbit_pos - X_mean)

                        Z = Y + S * LF   # Lévy-perturbed alternative dive path

                        # Evaluate both candidates; keep whichever scores higher
                        Y_clip = self._clip(Y)
                        Z_clip = self._clip(Z)
                        f_Y    = self._evaluate(Y_clip)
                        f_Z    = self._evaluate(Z_clip)

                        if f_Y >= f_Z:
                            X[i]       = Y_clip
                            fitness[i] = f_Y
                        else:
                            X[i]       = Z_clip
                            fitness[i] = f_Z

                # Update rabbit if this hawk found a better solution
                if fitness[i] > rabbit_score:
                    rabbit_pos   = X[i].copy()
                    rabbit_score = float(fitness[i])

            # ── Log ───────────────────────────────────────────────────────
            history.append((t, rabbit_score))
            print(f"[HHO] Iter {t:3d}/{T}  mIoU={rabbit_score:.4f}  "
                  f"E={E:+.3f}  cache={len(self._cache):4d}  "
                  f"-> {self._decode(rabbit_pos)}")

            # ── Convergence check ──────────────────────────────────────────
            # Stop early if the best mIoU hasn't improved meaningfully for
            # CONVERGENCE_PATIENCE consecutive iterations (paper: 5 iterations).
            improvement = rabbit_score - prev_best
            if improvement < HHO_CONVERGENCE_THRESHOLD:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_best = rabbit_score

            if no_improve_count >= HHO_CONVERGENCE_PATIENCE:
                print(f"\n[HHO] Early stop: no improvement > {HHO_CONVERGENCE_THRESHOLD} "
                      f"for {HHO_CONVERGENCE_PATIENCE} consecutive iterations.\n")
                break

        # ── Final result ───────────────────────────────────────────────────
        best_hps = self._decode(rabbit_pos)

        print(f"\n{'='*60}")
        print(f"[HHO] Search complete!")
        print(f"[HHO] Unique HP sets evaluated : {len(self._cache)}")
        print(f"[HHO] Best mIoU  : {rabbit_score:.4f}")
        print(f"[HHO] Best HPs   :")
        for k, v in best_hps.items():
            print(f"        {k:<15s} = {v}")
        print(f"{'='*60}\n")

        return best_hps, rabbit_score, history
