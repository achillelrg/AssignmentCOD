import numpy as np
from optimizer.pso import PSO
from benchmarks.griewank import griewank

def test_pso_runs_and_improves():
    D = 2
    bounds = [(-600.0, 600.0)] * D
    opt = PSO(bounds=bounds, seed=0, options={"pop": 20})
    # run a few iterations
    best_before = np.inf
    for _ in range(20):
        X = opt.ask()
        F = [griewank(x) for x in X]
        opt.tell(F)
        best = opt.state()["gbest_f"]
        assert np.isfinite(best)
        best_before = best
    assert best_before < 0.5  # should have made progress
