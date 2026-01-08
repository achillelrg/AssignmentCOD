
import os
import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_data(csv_path="data/PartC/training_data.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data not found at {csv_path}. Run part_c_data_gen.py first.")
    
    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter valid
    # We NOW include 'invalid' runs (failed XFOIL) because they have Penalty values.
    # df = df[df["valid"] == True]
    df = df.dropna(subset=["cl", "cd", "cm"])
    
    # Filter physical bounds (approx)
    # XFOIL divergence often gives huge Cd or Cl
    df = df[(df["cl"].abs() < 10.0) & (df["cd"] < 2.0) & (df["cd"] > -0.5) & (df["cm"].abs() < 10.0)]
    
    print(f"Loaded {len(df)} samples after filtering.")
    return df

def train_surrogate(X, y, name="Cl"):
    print(f"\n{'='*40}")
    print(f"--- Training GP for {name.upper()} ---")
    
    # Scale Inputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Transform Output (Log for Cd)
    is_log = (name.lower() == "cd")
    if is_log:
        print(f"Applying Log10 Transform to {name} (Handling ranges 0.001 - 0.1)")
        # Clip to avoid log(0) or negative
        y = np.maximum(y, 1e-6)
        y_trans = np.log10(y)
    else:
        y_trans = y
        
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_trans.reshape(-1, 1)).flatten()
    
    # Kernel: Matern is good
    kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
    
    print("Fitting model...")
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=False)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Train
    gp.fit(X_train, y_train)
    print(f"  > Optimized Kernel: {gp.kernel_}")
    
    # Predict
    y_pred_scaled, y_std_scaled = gp.predict(X_test, return_std=True)
    
    # Inverse Transform
    y_test_trans = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_trans = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    if is_log:
        y_test_orig = 10**y_test_trans
        y_pred_orig = 10**y_pred_trans
    else:
        y_test_orig = y_test_trans
        y_pred_orig = y_pred_trans
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    print(f"RMSE: {rmse:.6f}")
    print(f"R2 Score: {r2:.6f}")
    
    # Plot Goodness of Fit
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, edgecolor='k')
    min_val, max_val = min(y_test_orig), max(y_test_orig)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel(f"Actual {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"{name} Parity Plot (R2={r2:.3f})")
    plt.grid(True)
    out_img = f"data/PartC/figures/c3_parity_{name}.png"
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img)
    plt.close()
    
    return gp, scaler_X, scaler_y, (rmse, r2)
    
def main():
    # 1. Load Data
    df = load_data()
    
    # Inputs: x0..x5
    X = df[[f"x{i}" for i in range(6)]].values
    
    # Outputs: Cl, Cd, Cm
    # We need 3 separate models
    models = {}
    metrics = {}
    
    for target in ["cl", "cd", "cm"]:
        if target in df.columns:
            y = df[target].values
            models[target], scX, scY, met = train_surrogate(X, y, target)
            metrics[target] = met
            # Save Model
            out_pkl = f"data/PartC/models/gp_{target}.pkl"
            os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
            with open(out_pkl, "wb") as f:
                pickle.dump({"model": models[target], "scaler_X": scX, "scaler_y": scY}, f)
            print(f"Saved model to {out_pkl}")
            
    # Summary
    print("\n--- Surrogate Model Performance ---")
    for t, (rmse, r2) in metrics.items():
        print(f"{t.upper()}: RMSE={rmse:.6f}, R2={r2:.6f}")
        
    with open("data/PartC/surrogate_metrics.txt", "w") as f:
        for t, (rmse, r2) in metrics.items():
            f.write(f"{t.upper()}: RMSE={rmse:.6f}, R2={r2:.6f}\n")

if __name__ == "__main__":
    main()
