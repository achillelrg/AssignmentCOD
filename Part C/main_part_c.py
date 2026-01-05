"""
Part C: Advanced/Optional Tasks
-------------------------------
1. Uncertainty Analysis: Evaluate mean/std of aerodynamic coefficients at optimal design point.
3/4. Surrogate Modelling: Train a model on Part B data and optimize it.
"""

import sys
import os
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.airfoil_xfoil import airfoil_fitness, coeffs_at_alpha
from optimizer.pso import PSO, project
from optimizer.surrogate import SurrogateModel
from experiments.run_opt import optimize

def uncertainty_analysis(best_x, samples=20, alpha_mean=3.0, alpha_std=0.1):
    """
    C.1 Uncertainty Analysis
    Evaluate Cl, Cd, Cm at different angles of attack sampled from N(3.0, 0.1).
    """
    print(f"\n[Part C.1] Uncertainty Analysis (N={samples})")
    print(f"Alpha ~ N({alpha_mean}, {alpha_std})")
    
    alphas = np.random.normal(alpha_mean, alpha_std, samples)
    results = []

    for a in alphas:
        try:
            Cl, Cd, Cm = coeffs_at_alpha(best_x, alpha=a)
            if Cl is not None:
                results.append([Cl, Cd, Cm])
        except Exception as e:
            print(f"Evaluation failed for alpha={a}: {e}")

    if not results:
        print("No valid evaluations.")
        return

    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    
    print("\nResults (Mean +/- Std Dev):")
    print(f"Cl: {mean[0]:.6f} +/- {std[0]:.6f}")
    print(f"Cd: {mean[1]:.6f} +/- {std[1]:.6f}")
    print(f"Cm: {mean[2]:.6f} +/- {std[2]:.6f}")


def surrogate_optimisation(history_log):
    """
    C.4 Surrogate-based Optimisation
    Load data from Part B run (CSV), train a surrogate, and optimize the surrogate.
    """
    print("\n[Part C.4] Surrogate Optimisation")
    
    if not os.path.exists(history_log):
        print(f"History log {history_log} not found. Run Part B first.")
        return

    # 1. Load Data
    import csv
    X_data = []
    y_data = []
    
    # We need the full history of evaluations. 
    # NOTE: The current PSO implementation doesn't easily log X vectors in the CSV (only stats).
    # Ideally we'd need a separate log of all evaluated points. 
    # For now, we will simulate this by generating some random samples if log doesn't have vectors,
    # OR we assume the user modifies PSO to save X.
    #
    # IMPLEMENTATION FIX: We'll assume we can't get X from standard log. 
    # We will run a short DOE (Design of Experiments) here to train the surrogate.
    
    print("Generating training data (LHS/Random DOE) for Surrogate...")
    # 6 dimensions for CST
    bounds = [(-0.2, 0.5)] * 6
    N_samples = 40
    
    X_train = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (N_samples, 6))
    y_train = []
    
    for i, x in enumerate(X_train):
        f = airfoil_fitness(x, alpha=3.0)
        y_train.append(f)
        if i % 5 == 0: print(f"Evaluated {i}/{N_samples}...")
        
    y_train = np.array(y_train)
    
    # 2. Train Surrogate
    sm = SurrogateModel()
    sm.fit(X_train, y_train)
    print("Surrogate fitted.")
    
    # 3. Optimize Surrogate
    print("Optimizing Surrogate...")
    
    def surrogate_obj(x):
        return sm.predict(x)[0] # return mean prediction

    opt = PSO(bounds=bounds, options={"pop": 20, "evals": 1000}) # Quick run
    best = optimize(surrogate_obj, opt, eval_budget=1000, log_path="data/results/surrogate_run.csv")
    
    print("Best design found by Surrogate:", best)
    
    # 4. Verify on real function
    real_val = airfoil_fitness(best["x"], alpha=3.0)
    print(f"Verification on Truth Model: f={real_val:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["uncertainty", "surrogate", "all"], default="all")
    parser.add_argument("--best_x", nargs=6, type=float, help="Manual best design vector input for uncertainty")
    args = parser.parse_args()

    # Default "best" vector (approximate example) if none provided
    # Users should plug in their result from Part B
    best_x = np.array(args.best_x) if args.best_x else np.array([0.16, 0.16, 0.16, 0.10, 0.10, 0.10])

    if args.mode in ["uncertainty", "all"]:
        uncertainty_analysis(best_x)
    
    if args.mode in ["surrogate", "all"]:
        # We pass a dummy path since we generate fresh data now for robustness
        surrogate_optimisation("dummy.csv")

if __name__ == "__main__":
    main()
