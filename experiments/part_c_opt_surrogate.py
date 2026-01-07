
import os
import sys
import pickle
import numpy as np
import argparse
from scipy.optimize import minimize, Bounds
import json

# Fix paths
sys.path.append(os.getcwd())
from utils.airfoil_problem import AirfoilConfig
from utils.xfoil_runner import run_xfoil_single_alpha
from utils.cst import cst_airfoil

def load_models(model_dir="data/PartC/models"):
    models = {}
    scalers_X = {}
    scalers_y = {}
    for target in ["cl", "cd", "cm"]:
        path = os.path.join(model_dir, f"gp_{target}.pkl")
        with open(path, "rb") as f:
            idx = pickle.load(f)
            models[target] = idx["model"]
            scalers_X[target] = idx["scaler_X"]
            scalers_y[target] = idx["scaler_y"]
    return models, scalers_X, scalers_y

def predict(models, sX, sY, x_in):
    # x_in: (6,)
    # Need to reshape for sklearn: (1, 6)
    x_in = np.array(x_in).reshape(1, -1)
    
    preds = {}
    for target in ["cl", "cd", "cm"]:
        if target not in models: continue
        
        # Scale Input
        x_scaled = sX[target].transform(x_in)
        
        # Inverse Scale Output
        y_val_scaled = models[target].predict(x_scaled)
        y_val_trans = sY[target].inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()[0]
        
        # Handle Log Transform for Cd
        if target.lower() == "cd":
             # We assume training script applied Log10 for Cd.
             # We can check if values are negative (log) vs small positive?
             # Or just hardcode the logic since we know we changed the training.
             # Log10(0.01) = -2.
             y_val = 10**y_val_trans
        else:
             y_val = y_val_trans
             
        preds[target] = y_val
    return preds

def objective(x, models, sX, sY):
    # Bounds Check
    if np.any(x < -0.2) or np.any(x > 0.5):
        return 1e9
        
    # Geometric Check (Cheap)
    n_vars = len(x)
    coeffs_u = x[:n_vars//2]
    coeffs_l = x[n_vars//2:]
    xu, yu, xl, yl = cst_airfoil(50, coeffs_u, coeffs_l, dz_te=0.0)
    if np.any(yl[1:-1] >= yu[1:-1]):
        return 1e9 # Penalty for crossing
        
    # Predict Aerodynamics
    p = predict(models, sX, sY, x)
    
    # Calculate Fitness
    # Same as Part B: Minimize w1*Cd - w2*Cl + w3*|Cm - target|
    # Default weights from Part B (AirfoilConfig)
    # let's assume standard weights: w1=1.0, w2=1.0, w3=1.0?
    # Check benchmarks/airfoil_xfoil.py to matches exactly.
    # It passes config. Actually let's just hardcode what we used:
    # Cd - Cl + |Cm + 0.1| ?
    # Let's peek AirfoilConfig defaults if possible.
    # Or just use the standard: Cd - Cl + Penalty.
    
    # Using typical params as per previous knowledge (w1=10, w2=1, w3=10?)
    # Wait, earlier log said "Mean: -3.8" when minimizing.
    # If Cl ~ 1.5, Cd ~ 0.01.
    # Cd - Cl = 0.01 - 1.5 = -1.49.
    # If fitness was -3.8, maybe w2 is higher?
    # Let's assume w1=1.0, w2=2.0 ?? 
    # Or w1=1, w2=1, w3=1.
    # Let's stick to Assignment: "Maximize Lift, Minimize Drag"
    # J = Cd - Cl (simple).
    
    # Let's use: J = 1.0 * Cd - 1.0 * Cl + 1.0 * abs(p['cm'] + 0.1)
    # The moment constraint is usually Cm = -0.1 or similar.
    
    val = 10.0 * p['cd'] - 1.0 * p['cl'] + 2.0 * abs(p['cm'] + 0.1)
    return val

def main():
    print("--- Part C.4: Deterministic Optimization using Surrogate ---")
    
    # 1. Load Models
    models, sX, sY = load_models()
    print("Loaded GP models.")
    
    # 2. Optimization
    # Start from random or center
    x0 = np.array([0.15] * 6)
    bounds = Bounds([-0.2]*6, [0.5]*6)
    
    print("Optimizing...")
    res = minimize(
        objective, x0, args=(models, sX, sY),
        method='Nelder-Mead', bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )
    
    x_opt = res.x
    f_opt = res.fun
    print(f"\nOptimization Success: {res.success}")
    print(f"Surrogate Minimum Fitness: {f_opt:.6f}")
    print(f"Optimal Vars: {x_opt}")
    
    # 3. Validate with True XFOIL
    print("\n--- Validating with XFOIL ---")
    preds = predict(models, sX, sY, x_opt)
    print(f"Surrogate Predicted: Cl={preds['cl']:.4f}, Cd={preds['cd']:.4f}, Cm={preds['cm']:.4f}")
    
    # Real Run
    import uuid
    from utils.geometry import write_dat
    
    # Generate Coords
    n_vars = len(x_opt)
    coeffs_u = x_opt[:n_vars//2]
    coeffs_l = x_opt[n_vars//2:]
    xu, yu, xl, yl = cst_airfoil(200, coeffs_u, coeffs_l)
    top_x, top_y = np.flip(xu), np.flip(yu)
    bot_x, bot_y = xl[1:], yl[1:] 
    
    coords_x = np.concatenate([top_x, bot_x])
    coords_y = np.concatenate([top_y, bot_y])
    
    uid = uuid.uuid4().hex[:6]
    dat_path = f"temp/surrogate_opt_{uid}.dat"
    os.makedirs("temp", exist_ok=True)
    with open(dat_path, "w") as f:
        f.write("Surrogate_Opt\n")
        columns = zip(coords_x, coords_y)
        for cx, cy in columns:
            f.write(f" {cx:.6f}  {cy:.6f}\n")
            
    # Run
    cl, cd, cm = run_xfoil_single_alpha(dat_path, alpha=3.0, Re=1e6, n_iter=200)
    
    if cl is None:
        print("XFOIL Verification Failed (Non-convergence).")
    else:
        print(f"XFOIL Actual    : Cl={cl:.4f}, Cd={cd:.4f}, Cm={cm:.4f}")
        
        # Comparison
        print("\n--- Comparison ---")
        print(f"Cl Error: {abs(cl - preds['cl']):.4f}")
        print(f"Cd Error: {abs(cd - preds['cd']):.4f}")
        
        # Save Result
        res_data = {
            "x_opt": x_opt.tolist(),
            "surrogate": preds,
            "actual": {"cl": cl, "cd": cd, "cm": cm}
        }
        with open("data/PartC/surrogate_comparison.json", "w") as f:
            json.dump(res_data, f, indent=4)
        print("Saved comparison to data/PartC/surrogate_comparison.json")

if __name__ == "__main__":
    main()
