from typing import List, Optional, Dict, Tuple
import numpy as np
from .base import Optimizer, project

class GA(Optimizer):
    """
    Real-coded Genetic Algorithm.
    
    Options:
    - pop: population size (default 40)
    - mutation_rate: prob of mutating a gene (default 0.1)
    - mutation_scale: std dev of mutation noise fraction of range (default 0.1)
    - crossover_rate: prob of crossover (default 0.9)
    - elite_frac: fraction of best solutions kept (default 0.1)
    """
    def __init__(self, bounds: List[Tuple[float, float]], seed: int = 0, options: Optional[Dict] = None):
        super().__init__(bounds, seed, options)
        self.pop_size = self.options.get("pop", 40)
        self.mut_rate = self.options.get("mutation_rate", 0.1)
        self.mut_scale = self.options.get("mutation_scale", 0.1)
        self.cross_rate = self.options.get("crossover_rate", 0.9)
        self.elite_frac = self.options.get("elite_frac", 0.1)
        
        # Initialize population
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        self.range = hi - lo
        
        self.X = self.rng.uniform(lo, hi, (self.pop_size, self.D))
        self.fitness = np.full(self.pop_size, np.inf)
        
        self.gbest_x = None
        self.gbest_f = np.inf
        
        self.iter = 0
        self.evals_total = 0
        
        # State: 'ask', 'tell'
        self._state_phase = "ask" 

    def ask(self) -> List[np.ndarray]:
        if self._state_phase != "ask":
             raise RuntimeError("Call tell() before ask()")
             
        # On first iter, return initial random population
        if self.iter == 0:
            self._state_phase = "tell"
            return list(self.X)
            
        # Evolution logic: Create new population from old
        new_X = []
        
        # 1. Elitism
        indices = np.argsort(self.fitness)
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        elite_indices = indices[:n_elite]
        for idx in elite_indices:
            new_X.append(self.X[idx].copy())
            
        # 2. Generate rest
        while len(new_X) < self.pop_size:
            # Tournament selection
            p1 = self._tournament(indices)
            p2 = self._tournament(indices)
            
            # Crossover
            if self.rng.random() < self.cross_rate:
                c1, c2 = self._crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Mutation
            self._mutate(c1)
            self._mutate(c2)
            
            new_X.append(c1)
            if len(new_X) < self.pop_size:
                new_X.append(c2)
                
        self.X = np.array(new_X)
        self._state_phase = "tell"
        return list(self.X)
    
    def tell(self, fitness: List[float], constraints=None):
        if self._state_phase != "tell":
             raise RuntimeError("Call ask() before tell()")
             
        self.fitness = np.array(fitness)
        # Handle NaNs
        self.fitness[np.isnan(self.fitness)] = np.inf
        self.evals_total += len(fitness)
        
        # Update global best
        min_idx = np.argmin(self.fitness)
        min_f = self.fitness[min_idx]
        
        if self.gbest_x is None or min_f < self.gbest_f:
            self.gbest_f = min_f
            self.gbest_x = self.X[min_idx].copy()
            
        self.iter += 1
        self._state_phase = "ask"

    def best(self):
        return {"x": self.gbest_x, "f": self.gbest_f}

    def state(self) -> Dict:
        return {
            "iter": self.iter,
            "evals_total": self.evals_total,
            "gbest_f": self.gbest_f,
            "f_best": np.min(self.fitness) if self.iter > 0 else np.nan,
            "f_mean": np.mean(self.fitness) if self.iter > 0 else np.nan,
            "f_std": np.std(self.fitness) if self.iter > 0 else np.nan,
        }

    def _tournament(self, sorted_indices) -> np.ndarray:
        # Tournament of size 3
        competitors = self.rng.choice(self.pop_size, 3, replace=False)
        # Since we have sorted_indices, we can find the best by finding the one with lowest rank (index in sorted)
        # But simply: comparing fitness is easier if we didn't have ranks.
        # Let's just pick best fitness of the 3.
        best_idx = competitors[np.argmin(self.fitness[competitors])]
        return self.X[best_idx].copy()

    def _crossover(self, p1, p2):
        # Arithmetic crossover
        alpha = self.rng.random()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return c1, c2

    def _mutate(self, x):
        # Gaussian mutation
        mask = self.rng.random(self.D) < self.mut_rate
        if np.any(mask):
            noise = self.rng.normal(0, self.mut_scale, np.sum(mask)) * self.range[mask] # scale by range?
            # Or just user defined scale. Usually scale is relative to bounds.
            # Let's use self.mut_scale as fraction of range.
            x[mask] += noise
            x[:] = project(x, self.bounds) # Ensure valid (update in-place)
