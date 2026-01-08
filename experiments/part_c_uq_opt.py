
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.getcwd())

from utils.airfoil_problem import evaluate_airfoil_theta
from utils.xfoil_runner import run_xfoil_single_alpha
from utils.cst import cst_airfoil

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Path to best design JSON from Part B")
    parser.add_argument("--samples", type=int, default=100, help="Number of Monte Carlo samples")
    parser.add_argument("--mean", type=float, default=3.0, help="Mean angle of attack")
    parser.add_argument("--std", type=float, default=0.1, help="Std dev of angle of attack")
    args = parser.parse_args()

    # 1. Load Design
    with open(args.json, 'r') as f:
        data = json.load(f)
    print(f"Loaded design from {args.json}")
    
    if "x" in data:
        x_opt = np.array(data["x"])
    elif "x_opt" in data:
        x_opt = np.array(data["x_opt"])
    else:
        raise KeyError("Could not find design vector ('x' or 'x_opt') in loaded JSON.")

    # 2. Generate Airfoil Coordinates (Standard Re/Mach)
    # We need to run xfoil for varying alpha
    # But evaluating theta (CST) -> .dat file creation -> run xfoil
    # We can reuse evaluate_airfoil_theta logic BUT we just need the .dat file once?
    # Actually, evaluate_airfoil_theta usually writes a temp file.
    # Let's perform the CST generation once, save it to a temp path, then reuse it.
    
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    dat_path = f"temp_best_design_{unique_id}.dat"
    
    from utils.airfoil_problem import write_airfoil_from_theta
    write_airfoil_from_theta(x_opt, dat_path)
    
    # 3. Monte Carlo Simulation
    alphas = np.random.normal(args.mean, args.std, args.samples)
    
    results = [] # (cl, cd, cm)
    
    print(f"Running {args.samples} evaluations for Alpha ~ N({args.mean}, {args.std})...")
    
    valid_alphas = []
    
    print(f"Starting loop...")
    for i, alpha in enumerate(alphas):
        if i % 10 == 0: print(f"Eval {i}/{len(alphas)}...", end='\r')
        cl, cd, cm = run_xfoil_single_alpha(dat_path, alpha=alpha, Re=1e6, mach=0.1, n_iter=100)
        if cl is not None:
            results.append((cl, cd, cm))
            valid_alphas.append(alpha)
            
    # Cleanup
    if os.path.exists(dat_path):
        os.remove(dat_path)
        
    results = np.array(results)
    if len(results) == 0:
        print("Error: No valid evaluations found!")
        return
        
    cls, cds, cms = results[:, 0], results[:, 1], results[:, 2]
    
    # 4. Statistics
    print("\n--- Uncertainty Quantification Results ---")
    print(f"Success Rate: {len(results)}/{args.samples} ({len(results)/args.samples*100:.1f}%)")
    print("-" * 40)
    print(f"{'Metric':<10} | {'Mean':<10} | {'Std Dev':<10} | {'CoV (%)':<10}")
    print("-" * 40)
    print(f"{'Cl':<10} | {np.mean(cls):<10.6f} | {np.std(cls):<10.6f} | {np.std(cls)/np.mean(cls)*100:.2f}%")
    print(f"{'Cd':<10} | {np.mean(cds):<10.6f} | {np.std(cds):<10.6f} | {np.std(cds)/np.mean(cds)*100:.2f}%")
    print(f"{'Cm':<10} | {np.mean(cms):<10.6f} | {np.std(cms):<10.6f} | {abs(np.std(cms)/np.mean(cms))*100:.2f}%")
    print("-" * 40)

    # 5. Plot Histograms
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [cls, cds, cms]
    names = ["Cl", "Cd", "Cm"]
    
    for i, ax in enumerate(axs):
        ax.hist(metrics[i], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f"Distribution of {names[i]}")
        ax.set_xlabel(names[i])
        ax.set_ylabel("Frequency")
        # Add stats box
        mu, sigma = np.mean(metrics[i]), np.std(metrics[i])
        stats = r"$\mu={:.4f}$" "\n" r"$\sigma={:.4f}$".format(mu, sigma)
        ax.annotate(stats, xy=(0.05, 0.95), xycoords='axes fraction', 
                    verticalalignment='top', bbox=dict(boxstyle="round", fc="white"))
                    
    plt.tight_layout()
    out_img = "data/PartC/figures/c1_uncertainty_hist.png"
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img)
    print(f"Saved histogram to {out_img}")

if __name__ == "__main__":
    main()
