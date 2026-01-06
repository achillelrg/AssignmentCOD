import os
from datetime import datetime
import json
import numpy as np

from optimizer.pso import PSO
from benchmarks.airfoil_xfoil import airfoil_fitness
from .run_opt import optimize   # reuse Part A harness


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
    args = parser.parse_args()

    # Runtime Estimation (Benchmark-based)
    def estimate_runtime(n_points, n_iter, total_evals):
        # Known benchmarks (approximate seconds per evaluation)
        # 150 pts -> 0.1s
        # 200 pts -> 0.75s
        # 250 pts -> 4.0s
        # Piecewise linear interpolation for 'n_points' factor
        if n_points <= 150:
            base_time = 0.1
        elif n_points <= 200:
            # Interpolate 150->200 (Range 50, Time 0.65)
            ratio = (n_points - 150) / 50.0
            base_time = 0.1 + ratio * (0.75 - 0.1)
        elif n_points <= 250:
             # Interpolate 200->250 (Range 50, Time 3.25)
            ratio = (n_points - 200) / 50.0
            base_time = 0.75 + ratio * (4.0 - 0.75)
        else:
            # Extrapolate harshly (N^2 or worse)
            ratio = (n_points - 250) / 50.0
            base_time = 4.0 + ratio * 8.0 # Guess

        # Scale by Iterations (Baseline 100)
        # XFOIL takes longer if it struggles, but let's assume linear scaling for simplicity
        iter_factor = n_iter / 100.0
        
        # Time per eval
        t_per_eval = base_time * iter_factor
        
        # Parallel speedup (ideal linear speedup approximation)
        n_workers = max(1, args.jobs)
        t_per_eval = t_per_eval / n_workers
        
        total_seconds = t_per_eval * total_evals
        return total_seconds

    predicted_seconds = estimate_runtime(args.points, args.iter, args.evals)
    predicted_mins = predicted_seconds / 60.0
    print(f"--- Simulation Estimate ---")
    print(f"Points: {args.points} | Iter: {args.iter} | Evals: {args.evals} | Jobs: {args.jobs}")
    print(f"Estimated Runtime: {predicted_mins:.1f} minutes ({predicted_seconds:.1f}s)")
    print(f"---------------------------")

    # Handle cleaning request
    if args.clean:
        if args.part:
             target_clean = os.path.join("data", f"Part{args.part}")
             print(f"Cleaning {target_clean}...")
             if os.path.exists(target_clean):
                 import shutil
                 shutil.rmtree(target_clean)
        else:
             print("Warning: --clean flag ignored because --part was not specified.")

    # 6 CST coefficients: 3 upper, 3 lower
    bounds = [(-0.2, 0.5)] * 6
    options = dict(pop=args.pop, w=0.7, c1=1.6, c2=1.6)

    seed = args.seed
    eval_budget = args.evals

    opt = PSO(bounds=bounds, seed=seed, options=options)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.part:
        folder = os.path.join("data", f"Part{args.part}", "results")
    else:
        folder = os.path.join("data", "results", "airfoil")
        
    os.makedirs(folder, exist_ok=True)
    out_csv = os.path.join(folder, f"airfoil_opt_seed{seed}_{stamp}.csv")
    
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
