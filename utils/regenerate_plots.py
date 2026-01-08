
import os
import glob
import json
import numpy as np
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from experiments import plot_airfoil

def regenerate_all_plots():
    print("--- Regenerating all plots for Part B ---")
    
    # Pattern: data/PartB/*/results/*_best.json
    results_dir = os.path.join("data", "PartB", "*", "results", "*_best.json")
    files = glob.glob(results_dir)
    
    if not files:
        print("No result files found in data/PartB/*/results/")
        return

    print(f"Found {len(files)} result files.")
    
    for json_path in files:
        print(f"Processing: {os.path.basename(json_path)}")
        
        # Load Data
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        best_vec = np.array(data["x"])
        
        # Corresponding CSV path (swap .json with .csv)
        csv_path = json_path.replace("_best.json", ".csv")
        
        try:
            # Re-run plotting functions
            # 1. Geometry (Updated with Alpha)
            plot_airfoil.plot_geometry(best_vec, out_csv=csv_path, alpha=3.0)
            
            # 2. Coefficients
            plot_airfoil.plot_coeff_bar(best_vec, Re=1e6, alpha=3.0, out_csv=csv_path)
            
            # 3. Polar (Benchmarks full range)
            # plot_airfoil.plot_polar(best_vec, Re=1e6, out_csv=csv_path) 
            # Note: Polar takes time (XFOIL loop). Uncomment if needed, but geometry is instant.
            
            print("  > Plots updated.")
            
        except Exception as e:
            print(f"  > Failed to plot {json_path}: {e}")

if __name__ == "__main__":
    regenerate_all_plots()
