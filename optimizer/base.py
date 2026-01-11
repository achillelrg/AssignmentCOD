from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np

Bounds = List[Tuple[float, float]]

def project(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.minimum(np.maximum(x, lo), hi)

@dataclass
class Optimizer:
    """
    Solver-agnostic ask/tell interface to enable clean separation between
    candidate proposal (ask) and objective evaluation (tell).

    This pattern allows the optimizer to be paused, serialized, or run 
    asynchronously without blocking the simulation loop.
    
    Attributes:
        bounds (Bounds): List of (min, max) tuples.
        D (int): Dimensionality of the problem.
        rng (np.random.Generator): Random number generator.
        options (Dict): Configuration parameters.
    """
    def __init__(self, bounds: Bounds, seed: int = 0, options: Optional[Dict] = None):
        self.bounds: Bounds = bounds
        self.D: int = len(bounds)
        self.rng = np.random.default_rng(int(seed))
        self.options: Dict = options or {}

    def ask(self) -> List[np.ndarray]:
        """
        Request a list of new candidate solutions to evaluate.
        
        Returns:
            List[np.ndarray]: A list of design vectors (length N).
        """
        raise NotImplementedError

    def tell(self, fitness: List[float], constraints: Optional[List[np.ndarray]] = None):
        """
        Report the fitness values for the candidate solutions proposed by `ask()`.
        
        Args:
            fitness (List[float]): Scalar objective values (minimization).
            constraints: check for specific constraint violations (optional).
        """
        raise NotImplementedError

    def best(self):
        """Return the best solution found so far as a dict {'x': ..., 'f': ...}."""
        raise NotImplementedError

    def state(self) -> Dict:
        """Return a dictionary of internal state metrics (convergence stats)."""
        return {}

    def done(self) -> bool:
        """Return True if the optimization stopping criteria are met."""
        return False
