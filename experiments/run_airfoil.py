import os
from datetime import datetime
import json
import numpy as np

import sys
sys.path.append(os.getcwd())

from optimizer.pso import PSO
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

    # Active Calibration Runtime Estimation
    def estimate_runtime_active(n_points, n_iter, total_evals, n_jobs):
        print("--- Benchmarking System Speed (Active) ---")
        
        # 1. Create a "Heavy" valid airfoil (NACA 0012-ish CST)
        # Random valid-ish vector
        test_vec = [0.1, 0.15, 0.1,  -0.1, -0.1, -0.1] 
        
        # 2. Benchmark logic
        import time
        from multiprocessing import Pool
        
        # We want to measure the throughput of the Pool, not just single eval
        n_bench = max(n_jobs, 999) # Not used really
        n_bench = n_jobs * 2 if n_jobs > 1 else 3
        
        # Create a small batch of tasks
        tasks = [np.array(test_vec) for _ in range(n_bench)]
        
        from functools import partial
        # We need a temporary fitness function just for benchmarking
        # Use partial to pre-bind arguments
        bench_fn = partial(airfoil_fitness, Re=1e6, alpha=3.0, n_points=n_points, n_iter=n_iter)

        start = time.time()
        
        if n_jobs > 1:
            # Benchmark Parallel Throughput
            try:
                with Pool(processes=n_jobs) as pool:
                    _ = pool.map(bench_fn, tasks)
            except Exception as e:
                print(f"Benchmark failed (Pool error): {e}. Fallback to guess.")
                return 600.0 # Random guess
        else:
            # Benchmark Serial
            for t in tasks:
                bench_fn(t)
                
        end = time.time()
        duration = end - start
        
        # Speed per individual eval? No, we measured batch time.
        # Throughput = n_bench / duration (evals per second)
        throughput = n_bench / duration 
        
        print(f"Measured Throughput: {throughput:.2f} evals/sec (Benchmarked {n_bench} items in {duration:.2f}s)")
        
        total_seconds = total_evals / throughput
        
        # Correction Factor:
        # Benchmarking uses "Clean" airfoils (fast).
        # Optimization finds "Dirty" airfoils (slow, many iterations).
        # We apply a factor of 5.0 to be safe.
        total_seconds *= 5.0 
        
        return total_seconds

    predicted_seconds = estimate_runtime_active(args.points, args.iter, args.evals, args.jobs)
    predicted_mins = predicted_seconds / 60.0
    
    def format_time_est(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}h {m}m" if h else f"{m}m {s}s"

    print(f"--- Simulation Estimate ---")
    print(f"Points: {args.points} | Iter: {args.iter} | Evals: {args.evals} | Jobs: {args.jobs}")
    print(f"Estimated Max Runtime: {format_time_est(predicted_seconds)} (Conservative)")
    print(f"---------------------------")

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
