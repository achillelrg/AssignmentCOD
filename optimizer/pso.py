from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from .base import Optimizer, project, Bounds

@dataclass
class Particle:
    x: np.ndarray
    v: np.ndarray
    pbest_x: np.ndarray
    pbest_f: float

class PSO(Optimizer):
    """
    Particle Swarm Optimisation (continuous)
    - inertia w, cognitive c1, social c2
    - velocity clamping via vmax_frac
    - gbest (default) or lbest topology
    - optional 2D trajectory tracing (trace_every > 0 and D == 2)
    """
    def __init__(self, bounds: Bounds, seed: int = 0, options: Optional[Dict] = None):
        super().__init__(bounds, seed, options)
        opt = self.options

        self.pop: int = int(opt.get("pop", 40))
        self.w: float = float(opt.get("w", 0.72))
        self.c1: float = float(opt.get("c1", 1.6))
        self.c2: float = float(opt.get("c2", 1.6))
        self.vmax_frac: float = float(opt.get("vmax_frac", 0.2))
        self.topology: str = str(opt.get("topology", "gbest"))
        self.k_neighbors: int = int(opt.get("k_neighbors", 3))  # for lbest

        # ---- NEW: trajectory tracing controls ----
        # 0 disables tracing; otherwise record particle positions every N iterations (only if D==2).
        self.trace_every: int = int(opt.get("trace_every", 0))
        self._positions_trace: List[np.ndarray] = []  # each entry: (pop, 2) array

        lo = np.array([b[0] for b in self.bounds], dtype=float)
        hi = np.array([b[1] for b in self.bounds], dtype=float)
        span = hi - lo
        vmax = self.vmax_frac * span

        self.swarm: List[Particle] = []
        for _ in range(self.pop):
            x = self.rng.uniform(lo, hi)
            v = self.rng.uniform(-0.1 * span, 0.1 * span)
            self.swarm.append(Particle(x=x, v=v, pbest_x=x.copy(), pbest_f=np.inf))

        self.gbest_x = self.swarm[0].x.copy()
        self.gbest_f = np.inf
        self._last_idx: List[int] = list(range(self.pop))
        self._iter_best = np.inf
        self._iter_mean = np.inf
        self._iter_std = np.inf

        self._lo = lo
        self._hi = hi
        self._vmax = vmax
        self._evals_total = 0
        self._iters = 0

    def _local_best_position(self, i: int) -> np.ndarray:
        # ring topology with k neighbors on each side
        k = self.k_neighbors
        idxs = [(i + d) % self.pop for d in range(-k, k + 1)]
        best = min((self.swarm[j] for j in idxs), key=lambda p: p.pbest_f)
        return best.pbest_x

    def ask(self) -> List[np.ndarray]:
        # Evaluate current positions
        self._last_idx = list(range(self.pop))
        return [self.swarm[i].x.copy() for i in self._last_idx]

    def tell(self, fitness: List[float], constraints: Optional[List[np.ndarray]] = None):
        # 1) Update personal/global bests
        f_arr = np.asarray(fitness, dtype=float)
        f_arr[np.isnan(f_arr)] = np.inf
        for k, i in enumerate(self._last_idx):
            fx = f_arr[k]
            p = self.swarm[i]
            if fx < p.pbest_f:
                p.pbest_f = fx
                p.pbest_x = p.x.copy()
            if fx < self.gbest_f:
                self.gbest_f = fx
                self.gbest_x = p.x.copy()

        # 2) Velocity & position updates
        for i, p in enumerate(self.swarm):
            if self.topology.startswith("lbest"):
                g = self._local_best_position(i)
            else:
                g = self.gbest_x

            r1 = self.rng.random(self.D)
            r2 = self.rng.random(self.D)

            p.v = self.w * p.v + self.c1 * r1 * (p.pbest_x - p.x) + self.c2 * r2 * (g - p.x)

            # clamp velocity
            p.v = np.clip(p.v, -self._vmax, self._vmax)

            # update position + projection
            p.x = p.x + p.v
            p.x = project(p.x, self.bounds)

        # 3) iteration stats
        valid_mask = np.isfinite(f_arr)
        if np.any(valid_mask):
            valid_f = f_arr[valid_mask]
            self._iter_best = float(np.min(valid_f))
            self._iter_mean = float(np.mean(valid_f))
            self._iter_std = float(np.std(valid_f))
        else:
             # If all failed (very rare/bad start)
            self._iter_best = float(np.min(f_arr)) # likely inf
            self._iter_mean = float(np.inf)
            self._iter_std = 0.0
        self._evals_total += len(f_arr)
        self._iters += 1

        # ---- NEW: record 2D positions every trace_every iterations ----
        if self.trace_every and (self._iters % self.trace_every == 0) and self.D == 2:
            pts = np.stack([p.x.copy() for p in self.swarm], axis=0)  # shape (pop, 2)
            self._positions_trace.append(pts)

    def best(self):
        return {"x": self.gbest_x.copy(), "f": float(self.gbest_f)}

    def state(self) -> Dict:
        # ---- UPDATED: expose trace length for convenience ----
        return {
            "iter": self._iters,
            "evals_total": self._evals_total,
            "f_best": self._iter_best,
            "f_mean": self._iter_mean,
            "f_std": self._iter_std,
            "gbest_f": float(self.gbest_f),
            "trace_len": len(self._positions_trace),
        }

    # ---- NEW: accessor for plotting code ----
    def positions_trace(self) -> List[np.ndarray]:
        """Return the recorded list of (pop, 2) arrays. Empty if tracing disabled or D != 2."""
        return list(self._positions_trace)
