"""
Part B Experiment Driver: Airfoil Optimization.

This script orchestrates the aerodynamic optimization process (Part B).
It connects the optimizer (PSO or GA) with the Physics Engine (XFOIL).

Key Responsibilities:
1.  **Configuration:** Parses CLI arguments (pop size, budget, solver).
2.  **Environment:** Cleans temp folders to prevent XFOIL collisions.
3.  **Estimation:** Benchmarks system speed to estimate runtime.
4.  **Optimization:** Runs the loop using `experiments.run_opt.optimize` (shared driver).
5.  **Plotting:** Automatically generates Geometry and Convergence plots upon completion.

Usage:
    python experiments/run_airfoil.py --solver pso --evals 1000 --jobs 4
"""
import os
from datetime import datetime
import json
import numpy as np

import sys
sys.path.append(os.getcwd())

from optimizer.pso import PSO
from optimizer.ga import GA
from benchmarks.airfoil_xfoil import airfoil_fitness
from experiments.run_opt import optimize   # reuse Part A harness


import argparse
from experiments import plot_airfoil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default=None, help="Assignment Part (A, B, or C)")
    parser.add_argument("--evals", type=int, default=200, help="Evaluation budget")
    parser.add_argument("--pop", type=int, default=20, help="Population size")
    parser.add_argument("--points", type=int, default=200, help="Airfoil geometry points")
    parser.add_argument("--iter", type=int, default=100, help="XFOIL max iterations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--clean", action="store_true", help="Delete existing data folder for this Part before running")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--solver", type=str, default="pso", choices=["pso", "ga"], help="Optimisation algorithm")
    args = parser.parse_args()

    # Pre-run Cleanup (Clean temp folder)
    def cleanup_temp():
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Cleaned temp directory: {temp_dir}")

    cleanup_temp() # Always clean temp/ at start

    # Runtime Estimation removed as per user request (was inaccurate).
    # The real-time ETA in run_opt.py provides better feedback.

    # Handle cleaning request
    if args.clean:
        if args.part:
             target_clean = os.path.join("data", f"Part{args.part}")
             print(f"Cleaning {target_clean}...")
             if os.path.exists(target_clean):
                 import shutil
                 shutil.rmtree(target_clean)
             
             # Also clear in-memory cache which was loaded at import time
             import benchmarks.airfoil_xfoil
             benchmarks.airfoil_xfoil.clear_cache()
             print("In-memory cache cleared.")
        else:
             print("Warning: --clean flag ignored because --part was not specified.")

    # 6 CST coefficients: 3 upper, 3 lower
    bounds = [(-0.2, 0.5)] * 6
    
    seed = args.seed
    eval_budget = args.evals

    if args.solver == "pso":
        # Standard PSO Defaults
        options = dict(pop=args.pop, w=0.7, c1=1.6, c2=1.6)
        opt = PSO(bounds=bounds, seed=seed, options=options)
    elif args.solver == "ga":
        # GA Defaults
        options = dict(pop=args.pop, mutation_rate=0.1, crossover_rate=0.9)
        opt = GA(bounds=bounds, seed=seed, options=options)
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.part:
        # Structured: data/PartB/pso/results
        folder = os.path.join("data", f"Part{args.part}", args.solver, "results")
    else:
        # Fallback
        folder = os.path.join("data", "results", "airfoil", args.solver)
        
    os.makedirs(folder, exist_ok=True)
    # detailed naming: airfoil_{solver}_pop{pop}_evals{evals}_seed{seed}_{stamp}.csv
    out_csv = os.path.join(folder, f"airfoil_{args.solver}_pop{args.pop}_evals{args.evals}_seed{seed}_{stamp}.csv")
    
    from functools import partial
    fitness_fn = partial(airfoil_fitness, Re=1e6, alpha=3.0, n_points=args.points, n_iter=args.iter)

    best = optimize(
        fitness_fn,
        opt,
        eval_budget=eval_budget,
        f_target=-1e9, # Do not stop early for negative fitness
        log_path=out_csv,
        n_jobs=args.jobs
    )
    print("Best design:", best)

    # Save best design vector for plotting
    best_json = os.path.splitext(out_csv)[0] + "_best.json"
    with open(best_json, "w") as f:
        json.dump({"x": best["x"].tolist(), "f": best["f"]}, f, indent=2)
    print("Saved best design to", best_json)

    # Automatic Plotting using plot_airfoil modules
    print("--- Generating Plots ---")
    try:
        # Convergence
        plot_airfoil.plot_convergence(out_csv)
        
        # Geometry
        best_vec = np.array(best["x"], dtype=float)
        plot_airfoil.plot_geometry(best_vec, out_csv) 
        plot_airfoil.plot_coeff_bar(best_vec, Re=1e6, alpha=3.0, out_csv=out_csv)
        plot_airfoil.plot_polar(best_vec, Re=1e6, out_csv=out_csv)

        print("Plots generated successfully.")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
