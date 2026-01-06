import argparse
import csv
import time
import os
import shutil
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

from optimizer.pso import PSO
from benchmarks.airfoil_xfoil import airfoil_fitness
from experiments.run_opt import optimize

@dataclass
class StudyConfig:
    n_points: int
    n_iter: int
    pop: int
    evals: int
    seed: int

def run_study():
    # Define the parameter grid
    # We want to find a balance between speed and quality.
    
    # 1. Geometry Resolution
    points_list = [150, 200, 250]
    
    # 2. XFOIL Iterations
    iters_list = [100, 200]
    
    # 3. Population Size (Exploration power)
    pop_list = [20, 40]
    
    # Fixed budget for fair comparison of efficiency per evaluation
    FIXED_EVALS = 100
    SEED = 42

    # Prepare output directory
    base_dir = os.path.join("data", "PartB", "study")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(base_dir, f"parameter_study_{timestamp}.csv")

    print(f"Starting Parameter Study...")
    print(f"Results will be saved to: {csv_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_points", "n_iter", "pop", "evals", "time_sec", "best_fitness", "cl", "cd", "cm"])

        total_runs = len(points_list) * len(iters_list) * len(pop_list)
        run_idx = 0

        for pts in points_list:
            for it in iters_list:
                for pop in pop_list:
                    run_idx += 1
                    print(f"\n[{run_idx}/{total_runs}] Testing: Points={pts}, Iter={it}, Pop={pop}")

                    # Setup Optimizer
                    bounds = [(-0.2, 0.5)] * 6
                    options = dict(pop=pop, w=0.7, c1=1.6, c2=1.6)
                    opt = PSO(bounds=bounds, seed=SEED, options=options)

                    # Wall-clock timing
                    start_time = time.time()

                    # Optimize
                    # We need to inject the specific geometry parameters into the fitness function
                    # Use a lambda to bind the current study parameters
                    # NOTE: analyze_airfoil defaults are n_points=201, n_iter=200.
                    # We need to override these.
                    # airfoil_fitness calls analyze_airfoil but doesn't expose n_points/n_iter directly in signature?
                    # Wait, look at benchmarks/airfoil_xfoil.py: airfoil_fitness signature is fixed.
                    # We might need to modify airfoil_fitness or monkey-patch it, or pass via kw args if supported.
                    # Looking at airfoil_fitness source... it calls analyze_airfoil.
                    # It accepts *args/**kwargs? No.
                    # We should probably modify airfoil_fitness to accept kwargs or use a wrapper that changes default behaviour.
                    # OR, better: We modify utils/airfoil_analysis.py directly? No, unsafe.
                    
                    # Hack: We will wrap the inner call. 
                    # Actually, let's just update airfoil_fitness signature in benchmarks/airfoil_xfoil.py to accept **kwargs
                    # and pass them to analyze_airfoil. That is the cleanest way. 
                    # But assuming I can't edit that file right now in this loop (I can, but let's see).
                    
                    # Let's assume for now I will Edit benchmarks/airfoil_xfoil.py to support overrides.
                    # ... Wait, I am the agent, I can do that.
                    
                    # Define fitness function with current params
                    def fitness_fn(v):
                        return airfoil_fitness(
                            np.array(v), 
                            Re=1e6, 
                            alpha=3.0, 
                            n_points=pts, 
                            n_iter=it
                        )

                    # Run optimization
                    best = optimize(
                        fitness_fn,
                        opt,
                        eval_budget=FIXED_EVALS,
                        f_target=-1e9,
                        log_path=os.path.join(base_dir, f"run_pts{pts}_it{it}_pop{pop}.csv")
                    )

                    duration = time.time() - start_time
                    
                    # Compute detailed coefficients for best result
                    best_vec = np.array(best["x"], dtype=float)
                    # We must pass same params to verify
                    cl, cd, cm = airfoil_fitness(best_vec, Re=1e6, alpha=3.0, n_points=pts, n_iter=it, return_all=True)[1:]

                    # Log
                    print(f"   -> Time: {duration:.2f}s | Best f: {best['f']:.6f} | Cl/Cd: {cl}/{cd}")
                    writer.writerow([pts, it, pop, FIXED_EVALS, f"{duration:.2f}", f"{best['f']:.6f}", cl, cd, cm])
                    f.flush()

    print(f"\nStudy complete. Results saved to {csv_path}")

def main():
    run_study()

if __name__ == "__main__":
    main()
