from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import numpy as np

Bounds = List[Tuple[float, float]]

def project(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.minimum(np.maximum(x, lo), hi)

class Optimizer:
    """
    Solver-agnostic ask/tell interface to enable clean separation between
    candidate proposal (ask) and objective evaluation (tell).
    """
    def __init__(self, bounds: Bounds, seed: int = 0, options: Optional[Dict] = None):
        self.bounds: Bounds = bounds
        self.D: int = len(bounds)
        self.rng = np.random.default_rng(int(seed))
        self.options: Dict = options or {}

    def ask(self) -> List[np.ndarray]:
        raise NotImplementedError

    def tell(self, fitness: List[float], constraints: Optional[List[np.ndarray]] = None):
        raise NotImplementedError

    def best(self):
        raise NotImplementedError

    def state(self) -> Dict:
        return {}

    def done(self) -> bool:
        return False
