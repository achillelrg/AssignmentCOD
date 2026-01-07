
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from experiments import plot_airfoil

def main():
    print("--- Part C: Plotting Surrogate Design ---")
    
    # 1. Load result
    json_path = os.path.join("data", "PartC", "surrogate", "surrogate_results.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    x = np.array(data["x_opt_surro"])
    
    # 2. Setup output dir
    out_dir = os.path.join("data", "PartC", "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. Plot Geometry
    # We cheat the plot_geometry function which expects a "csv_path" to infer output dir.
    # We will manually set the output path inside the function call or wrapper?
    # Actually plot_geometry saves to "figures/airfoil/geometry_final.png" inferred from CSV.
    # Let's just copy the logic or subclass.
    # Easier: Just use utils.geometry and matplotlib directly here.
    
    from utils.cst import cst_airfoil
    
    # Generate coords
    coords_x, coords_yu, _, coords_yl = cst_airfoil(200, x[:3], x[3:])
    
    plt.figure(figsize=(10, 3))
    plt.plot(coords_x, coords_yu, 'b-', label='Upper')
    plt.plot(coords_x, coords_yl, 'r-', label='Lower')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"Surrogate Optimized Airfoil (Predicted Fitness: {data['f_pred']:.2f})")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    
    # Save
    save_path = os.path.join(out_dir, "surrogate_geometry.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved geometry to {save_path}")

if __name__ == "__main__":
    main()
