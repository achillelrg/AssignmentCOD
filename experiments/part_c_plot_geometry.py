"""
Part C Comparison Plotter: Geometry and Performance.

This script generates a high-quality comparison plot between multiple optimization results:
1.  **Reference (Part B):** Usually the best GA result.
2.  **Surrogate (Part C):** The result from the deterministic surrogate optimization.
3.  **Optional (PSO):** The best PSO result for a 3-way benchmark.

Features:
- **Geometry Override:** Plots all shapes on the same axes.
- **Metric Re-evaluation:** Live XFOIL calls to annotate Cl/Cd/Efficiency directly on the plot.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.append(os.getcwd())
from utils.cst import cst_airfoil

def load_design(json_path):
    """
    Load a design vector from a JSON file.
    Supports both 'x_opt' (Surrogate) and 'x' (Standard) keys.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    if "x_opt" in data: return np.array(data["x_opt"]) # surrogate format
    if "x" in data: return np.array(data["x"]) # best.json format
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Path to Part B reference JSON")
    parser.add_argument("--opt", default="data/PartC/surrogate_comparison.json", help="Path to Surrogate Opt JSON")
    parser.add_argument("--pso", default=None, help="Optional Path to PSO Best JSON for 3-way comparison")
    args = parser.parse_args()
    
    x_ref = load_design(args.ref)
    x_opt = load_design(args.opt)
    x_pso = load_design(args.pso) if args.pso else None
    
    if x_ref is None or x_opt is None:
        print("Error loading designs")
        return

    # Generate Coords
    def get_coords(x):
        n_vars = len(x)
        coeffs_u = x[:n_vars//2]
        coeffs_l = x[n_vars//2:]
        xu, yu, xl, yl = cst_airfoil(200, coeffs_u, coeffs_l)
        return xu, yu, xl, yl
        
    xu1, yu1, xl1, yl1 = get_coords(x_ref)
    xu2, yu2, xl2, yl2 = get_coords(x_opt)
    
    plt.figure(figsize=(12, 7)) # Slightly larger for 3 infos
    
    # 0. Plot PSO (Background)
    if x_pso is not None:
         xu3, yu3, xl3, yl3 = get_coords(x_pso)
         # Blue dotted, low alpha
         plt.plot(xu3, yu3, 'b:', label='Part B Best (PSO)', linewidth=2.5, alpha=0.7, zorder=1)
         plt.plot(xl3, yl3, 'b:', linewidth=2.5, alpha=0.7, zorder=1)
    
    # Determine Label for Ref
    ref_label = 'Part B Best'
    if 'pso' in args.ref.lower(): ref_label = 'Part B Best (PSO)'
    elif 'ga' in args.ref.lower(): ref_label = 'Part B Best (GA)'
    
    # 1. Plot Ref (GA)
    plt.plot(xu1, yu1, 'k--', label=ref_label, linewidth=1.5, zorder=2)
    plt.plot(xl1, yl1, 'k--', linewidth=1.5, zorder=2)
    
    # 2. Plot Opt (Surrogate) - Foreground
    plt.plot(xu2, yu2, 'r-', label='Surrogate Opt (C.4)', linewidth=2, zorder=3)
    plt.plot(xl2, yl2, 'r-', linewidth=2, zorder=3)
    
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.title("Airfoil Shape Comparison: Direct vs Surrogate Optimization")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    
    # --- Metrics Annotation ---
    try:
        from benchmarks.airfoil_xfoil import airfoil_fitness
        
        # 1. Reference Metrics (GA)
        _, cl1, cd1, cm1 = airfoil_fitness(x_ref, return_all=True)
        ld1 = cl1/cd1 if cd1 else 0
        
        # 2. Surrogate Metrics
        _, cl2, cd2, cm2 = airfoil_fitness(x_opt, return_all=True)
        ld2 = cl2/cd2 if cd2 else 0
        
        # 3. PSO Metrics
        pso_str = ""
        if x_pso is not None:
             _, cl3, cd3, cm3 = airfoil_fitness(x_pso, return_all=True)
             if cl3 and cd3:
                ld3 = cl3/cd3
                pso_str = (
                    f"PSO (Part B):\n"
                    f" Cl: {cl3:.3f} | Cd: {cd3:.4f}\n"
                    f" L/D: {ld3:.1f}\n\n"
                )
             else:
                pso_str = "PSO (Part B):\n Failed Evaluation\n\n"
        
        # Text Block
        info = (
            f"{pso_str}"
            f"REF ({'GA' if 'ga' in args.ref.lower() else 'Ref'}):\n"
            f" Cl: {cl1:.3f} | Cd: {cd1:.4f}\n"
            f" L/D: {ld1:.1f}\n\n"
            
            f"OPT (Part C):\n"
            f" Cl: {cl2:.3f} | Cd: {cd2:.4f}\n"
            f" L/D: {ld2:.1f}"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        plt.text(0.02, 0.98, info, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top', bbox=props)
                 
    except Exception as e:
        print(f"Warning: Could not annotate metrics: {e}")
    
    out_path = "data/PartC/figures/c4_geometry_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {out_path}")

if __name__ == "__main__":
    main()
