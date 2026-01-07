
import os
import argparse
import subprocess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Constants
PART = "B"
POP = 40
EVALS = 2000 # Enough for B3/B4
SEEDS = [1] # Keep it simple for now, can increase for robustness

def run_experiment(solver, pop, seed, evals):
    """Run experiments.run_airfoil via subprocess"""
    cmd = [
        "python3", "experiments/run_airfoil.py",
        "--part", PART,
        "--solver", solver,
        "--pop", str(pop),
        "--evals", str(evals),
        "--seed", str(seed),
        "--jobs", "4" # Parallel for speed
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def get_latest_csv(solver, pop, seed):
    # Find the file we just created
    # pattern: data/PartB/results/airfoil_{solver}_opt_seed{seed}_*.csv
    # But wait, run_airfoil modifies name based on solver now.
    # Structured: data/PartB/{solver}/results
    folder = os.path.join("data", f"Part{PART}", solver, "results")
    # New naming: airfoil_{solver}_pop{pop}_evals{evals}_seed{seed}_{stamp}.csv
    pattern = os.path.join(folder, f"airfoil_{solver}_pop{pop}_evals*_seed{seed}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["b3", "b4", "all"], default="all")
    parser.add_argument("--evals", type=int, default=2000)
    parser.add_argument("--pop", type=int, default=40, help="Default population size (for B3)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs")
    parser.add_argument("--clean", action="store_true", help="Clean data/PartB before running")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulations, only generate summary plots from existing data")
    args = parser.parse_args()
    
    # Update global constants
    EVALS = args.evals
    POP = args.pop
    SEEDS = [args.seed]
    
    # --- CLEANUP ---
    if args.clean:
        target_dir = os.path.join("data", f"Part{PART}")
        if os.path.exists(target_dir):
            print(f"Cleaning {target_dir}...")
            import shutil
            shutil.rmtree(target_dir)
            # Re-create empty structure to be safe, though runners do it too
            os.makedirs(target_dir, exist_ok=True)
            
    # Helper to pass dynamic args
    def run_experiment_dynamic(solver, pop_size, seed_val, evals_limit):
         cmd = [
            "python3", "experiments/run_airfoil.py",
            "--part", PART,
            "--solver", solver,
            "--pop", str(pop_size),
            "--evals", str(evals_limit),
            "--seed", str(seed_val),
            "--jobs", str(args.jobs)
        ]
         print(f"Running: {' '.join(cmd)}")
         subprocess.check_call(cmd)

    # Use our new dynamic function
    run_experiment = run_experiment_dynamic
    
    # Shared data storage for B3 and B.4
    b3_results = {}
    
    if args.task in ["b3", "all"]:
        # --- Task B.3: Comparing PSO vs GA ---
        print("--- Task B.3: Comparing PSO vs GA ---")
        
        # 1. Run PSO (Pop 40)
        if not args.plot_only:
             run_experiment("pso", POP, SEEDS[0], EVALS)
        csv_pso = get_latest_csv("pso", POP, SEEDS[0])
        if csv_pso: b3_results["pso"] = pd.read_csv(csv_pso)
        
        # 2. Run GA (Pop 40) - Use this for B.4 too!
        if not args.plot_only:
             run_experiment("ga", POP, SEEDS[0], EVALS)
        csv_ga = get_latest_csv("ga", POP, SEEDS[0])
        if csv_ga: b3_results["ga"] = pd.read_csv(csv_ga)

        # Plot B.3
        plt.figure(figsize=(10, 6))
        for solver, df in b3_results.items():
            if "gbest_f" in df.columns:
                plt.plot(df["iter"], df["gbest_f"], label=f"{solver.upper()}")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (Drag + Penalty)")
        plt.title("Solver Comparison: PSO vs GA (Alpha=3°)")
        plt.legend()
        plt.grid(True, which="both", linestyle=":")
        out_path = os.path.join("data", f"Part{PART}", "figures", "b3_solver_comparison.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        print(f"Saved comparison plot to {out_path}")

    if args.task in ["b4", "all"]:
        # --- Task B.4: GA Parameter Study (Population Size) ---
        print("--- Task B.4: GA Parameter Study (Population Size) ---")
        
        # We need results for pops [20, 40, 80]
        b4_results = {}
        
        # Reuse GA(40) if available from B.3 step
        if "ga" in b3_results:
             print("Reusing GA Pop 40 data from B.3 ...")
             b4_results[40] = b3_results["ga"]
        else:
             # If B.3 wasn't run, run GA(40) here
             # Check if file exists first? No, simple logic: check if key exists.
             # But if task=b4 only, b3_results is empty.
             # So logic:
             if not args.plot_only:
                  run_experiment("ga", 40, SEEDS[0], EVALS)
             csv = get_latest_csv("ga", 40, SEEDS[0])
             if csv: b4_results[40] = pd.read_csv(csv)
             
        # Run other pops
        for p in [20, 80]:
             if not args.plot_only:
                 run_experiment("ga", p, SEEDS[0], EVALS)
             csv = get_latest_csv("ga", p, SEEDS[0])
             if csv: b4_results[p] = pd.read_csv(csv)

        # Plot B.4
        plt.figure(figsize=(10, 6))
        for p in sorted(b4_results.keys()):
            df = b4_results[p]
            if "gbest_f" in df.columns:
                plt.plot(df["iter"], df["gbest_f"], label=f"GA Pop {p}")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.title("GA Population Size Study")
        plt.legend()
        plt.grid(True, which="both", linestyle=":")
        out_path = os.path.join("data", f"Part{PART}", "figures", "b4_ga_pop_study.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        print(f"Saved study plot to {out_path}")
