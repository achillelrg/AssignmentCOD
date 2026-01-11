"""
Part C Data Generation: Latin Hypercube Sampling (LHS).

This script generates a training dataset for the Surrogate Model (Part C).
It explores the Design Space using LHS to ensure good coverage.

Workflow:
1.  **Sampling:** Generates candidate CST vectors using `scipy.stats.qmc`.
2.  **Filtering:** Performs geometric checks (thickness) to discard invalid shapes quickly.
3.  **Evaluation:** Runs XFOIL in parallel (`multiprocessing`) to obtain aerodynamic coefficients.
4.  **Logging:** Saves valid (and optionally penalized) samples to CSV.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from scipy.stats import qmc
from multiprocessing import Pool
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from utils.airfoil_problem import evaluate_airfoil_theta, AirfoilConfig
from utils.xfoil_runner import run_xfoil_single_alpha

def evaluate_row(args):
    """
    Worker function for parallel XFOIL evaluation.
    
    Args:
        args (Tuple): (index, x_vec, Re, alpha, points, iter).
        
    Returns:
        dict: A dictionary containing design vars (x0..xn) and targets (cl, cd, cm).
              Includes a 'valid' flag if execution failed.
    """
    idx, x, Re, alpha, n_points, n_iter = args
    
    # We need to evaluate aerodynamic coefficients directly
    # 'evaluate_airfoil_theta' returns a scalar fitness (penalized).
    # We want Cl, Cd, Cm.
    # So we replicate the logic but return the raw coeffs.
    
    # 1. Generate Coordinates (handled by xfoil_runner/airfoil_problem logic usually)
    # Actually, evaluate_airfoil_theta calls analyze_airfoil, which calls xfoil.
    # But evaluate_airfoil_theta handles the CST -> Airfoil conversion.
    # Let's use `evaluate_airfoil_theta` but we need to tweak it or copy logic 
    # to return Cl, Cd, Cm.
    
    # To avoid modifying core utils, let's just do the CST -> DAT -> XFOIL chain here.
    # It duplicates code but is safer for established codebase.
    
    import uuid
    from utils.airfoil_analysis import analyze_airfoil
    
    # CST
    n_vars = len(x)
    n_cst = n_vars // 2
    coeffs_u = x[:n_cst]
    coeffs_l = x[n_cst:]
    
    # Delegate to the robust analysis function used in Part B
    # It handles geometry generation, checks, temp file management, and XFOIL runs.
    cl, cd, cm = analyze_airfoil(
        coeffs_upper=coeffs_u,
        coeffs_lower=coeffs_l,
        alpha=alpha,
        Re=Re,
        n_iter=n_iter,
        n_points=n_points
    )
    
    # Penalty Values for Failure
    PENALTY_CL = 0.0   # Loss of lift
    PENALTY_CD = 0.5   # Huge drag wall
    PENALTY_CM = 0.0
    is_valid = True
    
    if cl is None:
        # XFOIL Crashed / Non-convergence
        cl, cd, cm = PENALTY_CL, PENALTY_CD, PENALTY_CM
        is_valid = False
    elif cd > 0.5 or abs(cl) > 2.5:
        # Physical Divergence (garbage values)
        cl, cd, cm = PENALTY_CL, PENALTY_CD, PENALTY_CM
        is_valid = False
        
    return {
        "x0": x[0], "x1": x[1], "x2": x[2], "x3": x[3], "x4": x[4], "x5": x[5],
        "cl": cl, "cd": cd, "cm": cm,
        "valid": is_valid
    }
        
    return {
        "x0": x[0], "x1": x[1], "x2": x[2], "x3": x[3], "x4": x[4], "x5": x[5],
        "cl": cl, "cd": cd, "cm": cm,
        "valid": is_valid
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200, help="Number of VALID samples to generate")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()
    
    print(f"--- Generating Training Data (Target: {args.samples} valid samples) ---")
    
    # Rejection Sampling Loop
    valid_configs = []
    
    # Batch size for LHS
    cnt = 0
    batch_size = args.samples * 10
    
    print("Sampling geometries...")
    from utils.cst import cst_airfoil
    
    while len(valid_configs) < args.samples * 1.5: # Over-sample slightly
        sampler = qmc.LatinHypercube(d=6, seed=42+cnt)
        sample = sampler.random(n=batch_size)
        l_bounds = np.array([-0.2] * 6)
        u_bounds = np.array([0.5] * 6)
        X = qmc.scale(sample, l_bounds, u_bounds)
        
        for x in X:
            n_cst = 3
            coeffs_u = x[:n_cst]
            coeffs_l = x[n_cst:]
            
            # Fast check
            xu, yu, xl, yl = cst_airfoil(100, coeffs_u, coeffs_l, dz_te=0.0)
            
            # Check crossing
            if np.any(yl[1:-1] >= yu[1:-1]):
                continue
                
            valid_configs.append(x)
            
        cnt += 1
        print(f"  Batch {cnt}: Found {len(valid_configs)} potential candidates so far...")
        if cnt > 10: break
        
    print(f"Found {len(valid_configs)} geometrically valid shapes. Selecting {args.samples}...")
    valid_configs = valid_configs[:args.samples]
    
    # 2. Evaluate Parallel
    tasks = []
    # Reduces n_points to 160 for safety
    for i, x in enumerate(valid_configs):
        tasks.append((i, x, 1e6, 3.0, 160, 100))
        
    results = []
    
    import time
    start_time = time.time()
    
    # Conditional Pool for Debugging
    if args.jobs > 1:
        # Parallel
        pool_ctx = Pool(processes=args.jobs)
        iterator = pool_ctx.imap(evaluate_row, tasks)
    else:
        # Serial (No Pool)
        pool_ctx = None
        iterator = map(evaluate_row, tasks)

    print("Running XFOIL evaluations...")
        
    # Use imap (or map) to track progress
    mapped_res = []
    total = len(tasks)
    
    for i, res in enumerate(iterator, 1):
        mapped_res.append(res)
        
        if i % 10 == 0 or i == total:
            elapsed = time.time() - start_time
            avg_t = elapsed / i
            eta = (total - i) * avg_t
            valid_cnt = len([r for r in mapped_res if r is not None and r['valid']])
            sys.stdout.write(f"\r  > Processed {i}/{total} ({i/total*100:.1f}%) | ETA: {eta/60:.1f} min | Found: {valid_cnt} valid")
            sys.stdout.flush()
            
            # Partial Save (Safety) - every 500 samples
            if valid_cnt > 0 and i % 500 == 0:
                temp_df = pd.DataFrame([r for r in mapped_res if r is not None])
                temp_df.to_csv(f"data/PartC/training_data_partial_{i}.csv", index=False)
                
    print() # Newline after loop
        
    # Filter None
    valid_res = [r for r in mapped_res if r is not None]
    
    df = pd.DataFrame(valid_res)
    
    # Save
    out_path = "data/PartC/training_data.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print(f"Generation Complete.")
    print(f"Requested: {args.samples}")
    print(f"Successful XFOIL: {len(df)} ({len(df)/args.samples*100:.1f}%)")
    print(f"Saved to {out_path}")
    
    # Auto-Run Inspection Plot
    print("\n--- Generating Data Inspection Plot ---")
    try:
        import subprocess
        subprocess.run([sys.executable, "experiments/inspect_data.py"], check=True)
        print("Saved data/PartC/figures/cd_dist.png")
    except Exception as e:
        print(f"Failed to plot data: {e}")

if __name__ == "__main__":
    main()
