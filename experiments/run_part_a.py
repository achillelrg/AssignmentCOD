"""
Part A: Algorithm Verification and Robustness Suite.

This script executes the statistical validation of the implemented optimizers.
It runs the optimizer on a standard benchmark function (Griewank, 5D) multiple times
with different random seeds to prove stability.

Functions:
1.  **Batch Execution:** Runs `run_opt.py` 10 times (Seeds 41-50).
2.  **Data Aggregation:** Collects CSV logs.
3.  **Statistical Plotting:** Generates Convergence Overlay and Robustness Boxplots.
"""

import os
import glob
import subprocess
import sys
import argparse

# Ensure updated plotting logic is used
sys.path.append(os.getcwd())
from experiments import plotting

def main():
    print("==================================================")
    print("   PART A: ALGORITHM VERIFICATION (ROBUSTNESS)    ")
    print("==================================================")

    # 1. Configuration
    N_RUNS = 10
    POP = 40
    EVALS = 30000
    DIM = 5
    
    # Clean old data if requested
    out_dir = os.path.join("data", "PartA")
    if os.path.exists(out_dir):
        # We can't easily rm -rf in python without shutil, but run_opt can do it
        pass

    csv_files = []

    # Safe Cleanup: Remove only CSVs (Avoid PermissionError on locked figure folders)
    old_csvs = glob.glob(os.path.join("data", "PartA", "results", "*.csv"))
    for f in old_csvs:
        try:
            os.remove(f)
        except OSError:
            pass

    # 2. Run 10 Seeds
    for i in range(1, N_RUNS + 1):
        seed = 40 + i # 41, 42, ...
        # print(f"\n---> Running Seed {seed} ({i}/{N_RUNS})...") # Reduced verbosity
        sys.stdout.write(f"\r---> Running Seed {seed} ({i}/{N_RUNS})...")
        sys.stdout.flush()
        
        cmd = [
            sys.executable, "experiments/run_opt.py",
            "--part", "A",
            "--D", str(DIM),
            "--pop", str(POP),
            "--evals", str(EVALS),
            "--seed", str(seed)
        ]
        
        # subprocess.check_call(cmd, stdout=subprocess.DEVNULL) # Silence detailed output
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
        
    # 3. Collect CSVs
    # Expect data/PartA/results/*.csv
    search_path = os.path.join("data", "PartA", "results", "*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print("ERROR: No CSV files found in", search_path)
        sys.exit(1)
        
    print(f"\nFound {len(csv_files)} runs. Generating Summary Plots...")
    
    # 4. Generate Statistical Plots
    # Overlay -> figures/convergence
    ov_path = os.path.join("data", "PartA", "figures", "convergence", "convergence_overlay.png")
    os.makedirs(os.path.dirname(ov_path), exist_ok=True)
    if os.path.exists(ov_path):
        os.remove(ov_path) # Force delete old
        
    plotting.plot_convergence_overlay(csv_files, outpath=ov_path)
    print(f" - Generated: {ov_path}")
    
    # Boxplot -> figures/analysis
    box_path = os.path.join("data", "PartA", "figures", "analysis", "robustness_boxplot.png")
    os.makedirs(os.path.dirname(box_path), exist_ok=True)
    if os.path.exists(box_path):
        os.remove(box_path) # Force delete old

    plotting.plot_final_boxplot(csv_files, outpath=box_path)
    print(f" - Generated: {box_path}")

    print("\n✅ PART A COMPLETE")

if __name__ == "__main__":
    main()
