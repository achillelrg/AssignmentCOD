
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.append(os.getcwd())
from utils.cst import cst_airfoil

def load_design(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if "x_opt" in data: return np.array(data["x_opt"]) # surrogate format
    if "x" in data: return np.array(data["x"]) # best.json format
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Path to Part B reference JSON")
    parser.add_argument("--opt", default="data/PartC/surrogate_comparison.json", help="Path to Surrogate Opt JSON")
    args = parser.parse_args()
    
    x_ref = load_design(args.ref)
    x_opt = load_design(args.opt)
    
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
    
    plt.figure(figsize=(10, 6))
    
    # Plot Ref
    plt.plot(xu1, yu1, 'k--', label='Part B Best (PSO)', linewidth=1.5)
    plt.plot(xl1, yl1, 'k--', linewidth=1.5)
    
    # Plot Opt
    plt.plot(xu2, yu2, 'r-', label='Surrogate Opt (C.4)', linewidth=2)
    plt.plot(xl2, yl2, 'r-', linewidth=2)
    
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.title("Airfoil Shape Comparison: Direct vs Surrogate Optimization")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    
    out_path = "data/PartC/figures/c4_geometry_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved comparison plot to {out_path}")

if __name__ == "__main__":
    main()
