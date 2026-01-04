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
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

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

    best = optimize(
        lambda v: airfoil_fitness(np.array(v), Re=1e6, alpha=3.0),
        opt,
        eval_budget=eval_budget,
        log_path=out_csv,
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
