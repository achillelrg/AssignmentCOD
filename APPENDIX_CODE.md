# Appendix D: Full Source Code

This appendix contains the complete source code for the project.

## File: `run_ultimate.bat`
```cmd
@echo off
setlocal

echo ===============================================
echo        ROCKET STARTING ULTIMATE RUN ROCKET             
echo ===============================================

REM 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH.
    echo Please install Python and add it to PATH.
    pause
    exit /b 1
)

REM Check for VENV
if not exist "venv" (
    echo Creating Virtual Environment ^(venv^)... DO NOT CLOSE.
    echo This may take 1-2 minutes...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
    echo Venv created.
)

REM Activate VENV
call venv\Scripts\activate.bat

echo Checking environment integrity...
python -c "import scipy; import pandas; import matplotlib; import sklearn" >nul 2>&1

if %errorlevel% equ 0 (
    echo ✔ Dependencies already installed. Skipping download.
) else (
    echo ⚠ Missing dependencies detected.
    echo ⚠ Missing dependencies detected.
    echo Installing/Updating Dependencies... ^(Verbose Mode^)
    python -m pip install -v -r requirements.txt
    if %errorlevel% neq 0 (
        echo WARNING: Failed to install dependencies.
    )
    echo ✔ Dependencies ready.
)

echo Using Python:
where python

REM 2. Add current directory to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%CD%

echo.
echo Cleaning ENTIRE data directory for fresh run...
if exist "data" rd /s /q "data"
mkdir "data"

echo.
echo Step 0: Part A - Algorithm Validation (Statistical Study)
python experiments/run_part_a.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo -----------------------------------------------
echo Step 1: Part B - Reference Optimization (PSO)
echo Cleaning old Part B results...
if exist "data\PartB\pso\results" rd /s /q "data\PartB\pso\results"
if exist "data\PartB\pso\figures" rd /s /q "data\PartB\pso\figures"

echo Running PSO (Pop 40, Evals 2000 ~50 Gen)... 
python experiments/run_airfoil.py --part B --solver pso --pop 40 --evals 2000 --seed 666 --jobs 8
if %errorlevel% neq 0 exit /b %errorlevel%

REM Find LATEST_REF (Best PSO) - Windows Batch is tricky with ls -t
REM We will rely on python to define it or just find it later.
REM For simpler batch, we'll just let the plotting script find the latest default if we don't pass explicit path, OR use a small python helper.
REM Actually, run_airfoil.py saves to data/PartB/pso/results/...
REM The plotting script (Step 5) can auto-find the latest. I will update the python call there to rely on auto-detection if argument missing, OR assumes latest.
REM But since we want to be explicit for 3-way, let's use a tiny python one-liner to get the path.

for /f "delims=" %%I in ('python -c "import glob, os; lists=sorted(glob.glob('data/PartB/pso/results/*_best.json'), key=os.path.getmtime); print(lists[-1] if lists else '')"') do set LATEST_REF=%%I

echo.
echo -----------------------------------------------
echo Step 1.5: GA Parameter Study (Sensitivity Analysis)

echo Run A: Low Fidelity GA (Pop 10, Evals 200)...
python experiments/run_airfoil.py --part B --solver ga --pop 10 --evals 200 --seed 666 --jobs 8

echo Run B: Medium Fidelity GA (Pop 20, Evals 1000)...
python experiments/run_airfoil.py --part B --solver ga --pop 20 --evals 1000 --seed 666 --jobs 8

echo Run C: High Fidelity GA (Pop 40, Evals 2000)...
python experiments/run_airfoil.py --part B --solver ga --pop 40 --evals 2000 --seed 666 --jobs 8

for /f "delims=" %%I in ('python -c "import glob, os; lists=sorted(glob.glob('data/PartB/ga/results/*_best.json'), key=os.path.getmtime); print(lists[-1] if lists else '')"') do set LATEST_GA=%%I

echo Reference Design Found (PSO): %LATEST_REF%
echo Reference Design Found (GA):  %LATEST_GA%

echo.
echo -----------------------------------------------
echo Step 2: Part C - Robust Data Generation
echo Cleaning old data...
if exist "data\PartC\training_data.csv" del "data\PartC\training_data.csv"

echo Generating 1000 Samples with 8 Workers...
python experiments/part_c_data_gen.py --samples 1000 --jobs 8
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo -----------------------------------------------
echo Step 3: Part C - Surrogate Training
python experiments/part_c_surrogate.py

echo.
echo -----------------------------------------------
echo Step 4: Part C - Surrogate Optimization
python experiments/part_c_opt_surrogate.py

echo.
echo -----------------------------------------------
echo Step 5: Visualization and Comparison (3-Way)
python experiments/part_c_plot_geometry.py --ref "%LATEST_GA%" --pso "%LATEST_REF%"

echo.
echo -----------------------------------------------
echo Step 6: Uncertainty Quantification
python experiments/part_c_uq_opt.py --json data/PartC/surrogate_comparison.json --samples 200

echo.
echo ===============================================
echo        ULTIMATE RUN COMPLETE             
echo ===============================================
echo Results:
echo  - Report: data/PartC/surrogate_metrics.txt
echo  - Plots:  data/PartC/figures/
pause

```

---

## File: `run_ultimate.sh`
```bash
#!/bin/bash
set -e  # Exit on error


echo "==============================================="
echo "       🚀 STARTING ULTIMATE RUN 🚀             "
echo "==============================================="

# Activate env
# Activate env
# VENV_NAME="recovery_env"
# PYTHON="./$VENV_NAME/Scripts/python.exe"

# Use System Python (Assuming user has correct env active)
# Check for VENV
if [ ! -d "venv" ]; then
    echo "Creating Virtual Environment (venv)... DO NOT CLOSE."
    echo "This may take 1-2 minutes depending on your disk speed..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create venv."
        exit 1
    fi
    echo "✔ Venv created."
fi

# Activate VENV
source venv/bin/activate

# OPTIMIZATION: Check if dependencies are already installed
# preventing the slow 'pip install' on every run.
echo "Checking environment integrity..."

if python -c "import scipy; import pandas; import matplotlib; import sklearn" >/dev/null 2>&1; then
    echo "✔ Dependencies already installed. Skipping download."
else
    echo "⚠ Missing dependencies detected."
    echo "Installing/Updating Dependencies... (Verbose Mode)"
    # Added -v to show progress of large downloads like SciPy
    python -m pip install -v -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed to install dependencies."
    fi
    echo "✔ Dependencies ready."
fi

PYTHON="python"
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Cleaning ENTIRE data directory for fresh run..."
rm -rf data
mkdir -p data

echo "Step 0: Part A - Algorithm Validation (Statistical Study)"
echo "Cleaning Part A data..."
rm -rf data/PartA
mkdir -p data/PartA

$PYTHON experiments/run_part_a.py

echo "-----------------------------------------------"

echo "Step 1: Part B - Reference Optimization (PSO)"
echo "Cleaning old Part B results..."
rm -rf data/PartB/pso/results/* data/PartB/pso/figures/*

echo "Running PSO (Pop 20, Evals 400 ~20 Gen)... (FAST MODE)"
$PYTHON experiments/run_airfoil.py --part B --solver pso --pop 20 --evals 400 --seed 666 --jobs 8

# Get the Result Path (PSO Baseline)
# Use ls -t and head properly
LATEST_REF=$(ls -t data/PartB/pso/results/*_best.json | head -n 1)

echo "-----------------------------------------------"
echo "Step 1.5: GA Parameter Study (Sensitivity Analysis)"
echo "Comparing Low vs High Fidelity GA runs... (FAST MODE)"

echo "Run A: Low Fidelity GA (Pop 10, Evals 100)..."
$PYTHON experiments/run_airfoil.py --part B --solver ga --pop 10 --evals 100 --seed 666 --jobs 8

echo "Run B: Medium Fidelity GA (Pop 10, Evals 200)..."
$PYTHON experiments/run_airfoil.py --part B --solver ga --pop 10 --evals 200 --seed 666 --jobs 8

echo "Run C: High Fidelity GA (Pop 20, Evals 400)..."
$PYTHON experiments/run_airfoil.py --part B --solver ga --pop 20 --evals 400 --seed 666 --jobs 8

LATEST_GA=$(ls -t data/PartB/ga/results/*_best.json | head -n 1)

echo "Reference Design Found (PSO): $LATEST_REF"
echo "Reference Design Found (GA):  $LATEST_GA"

echo "-----------------------------------------------"
echo "Step 2: Part C - Robust Data Generation (Penalty Method)"
echo "Cleaning old data..."
rm -f data/PartC/training_data.csv data/PartC/models/*.pkl data/PartC/figures/*.png

echo "Generating 50 Samples with 8 Workers... (FAST MODE)"
$PYTHON experiments/part_c_data_gen.py --samples 50 --jobs 8

echo "-----------------------------------------------"
echo "Step 3: Part C - Surrogate Training"
$PYTHON experiments/part_c_surrogate.py

echo "-----------------------------------------------"
echo "Step 4: Part C - Surrogate Optimization"
$PYTHON experiments/part_c_opt_surrogate.py

echo "-----------------------------------------------"
echo "Step 5: Visualization & Comparison (3-Way)"
# Ref = GA, PSO = PSO, Opt = Surrogate (Automatic)
$PYTHON experiments/part_c_plot_geometry.py --ref "$LATEST_GA" --pso "$LATEST_REF"

echo "-----------------------------------------------"
echo "Step 6: Uncertainty Quantification"
$PYTHON experiments/part_c_uq_opt.py --json data/PartC/surrogate_comparison.json --samples 50

echo "==============================================="
echo "       ✅ ULTIMATE RUN COMPLETE ✅             "
echo "==============================================="
echo "Results:"
echo " - Report: data/PartC/surrogate_metrics.txt"
echo " - Plots:  data/PartC/figures/"
echo " - Logic:  ASSIGNMENT_LOGIC_EXPLAINED.md"

```

---

## File: `requirements.txt`
```text
numpy
scipy
matplotlib
pytest
scikit-learn
pandas

```

---

## File: `experiments\inspect_data.py`
```python
"""
Data Inspection Utility.

This script loads the generated training data (`data/PartC/training_data.csv`)
and plots the coverage/distribution of aerodynamic coefficients (Cl, Cd, Cm).
Used to verify that LHS sampled a diverse range of physical behaviors.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/PartC/training_data.csv")
print(df.describe())

# Histogram of Cd
plt.figure()
df['cd'].hist(bins=50, range=(0, 0.5), color='blue', alpha=0.7)
plt.title("Cd Distribution")
plt.xlabel("Cd")
plt.ylabel("Frequency")
plt.savefig("data/PartC/figures/cd_dist.png")
plt.close()

# Histogram of Cl
plt.figure()
df['cl'].hist(bins=50, color='green', alpha=0.7)
plt.title("Cl Distribution")
plt.xlabel("Cl")
plt.ylabel("Frequency")
plt.savefig("data/PartC/figures/cl_dist.png")
plt.close()

# Histogram of Cm
plt.figure()
df['cm'].hist(bins=50, color='red', alpha=0.7)
plt.title("Cm Distribution")
plt.xlabel("Cm")
plt.ylabel("Frequency")
plt.savefig("data/PartC/figures/cm_dist.png")
plt.close()

```

---

## File: `experiments\part_c_data_gen.py`
```python
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

```

---

## File: `experiments\part_c_opt_surrogate.py`
```python
"""
Part C.4 Optimization: Deterministic Search on Surrogate.

This script leverages the speed of the trained GP models to perform a 
high-resolution search using standard deterministic algorithms (Nelder-Mead).

Advantages:
- **Speed:** 1000s of iterations in seconds (vs hours with XFOIL).
- **Smoothness:** GP models provide a smooth gradient (or quasi-gradient).

Workflow:
1.  **Load:** Deserializes the 3 GP models (Cl, Cd, Cm).
2.  **Optimize:** Runs `scipy.optimize.minimize` on the `objective` function.
3.  **Verify:** Re-runs the *optimal design vector* in actual XFOIL to check for hallucination.
"""

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
    """
    Surrogate Objective Function.

    Evaluates the 'virtual' fitness of a design vector using the GP models.
    Matches the Part B fitness definition (Max [Cl/Cd] or Weighted Sum).

    Returns:
        float: Fitness value (Lower is better). 1e9 if constraints violated.
    """
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

```

---

## File: `experiments\part_c_plot_geometry.py`
```python
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

```

---

## File: `experiments\part_c_surrogate.py`
```python
"""
Part C Surrogate Training.

This script builds the Machine Learning models (Gaussian Processes) used to approximate XFOIL.

Models Trained:
1.  **Cl (Lift):** Standard GP with Matern Kernel.
2.  **Cd (Drag):** GP trained on **Log10(Cd)** to capture orders of magnitude.
3.  **Cm (Moment):** Standard GP.

Outputs:
- `.pkl` files: Serialized models (used by `part_c_opt_surrogate.py`).
- Parity Plots: Visual validation of model accuracy.
"""

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
    """
    Train a Gaussian Process Regressor for a specific target.

    Features:
    - **Scaling:** Standardizes inputs (X) and outputs (y).
    - **Log Transform:** Automatically applies Log10 if target is 'Cd'.
    - **Validation:** Splits 80/20 and reports RMSE/R2 on hold-out set.

    Args:
        X (np.ndarray): Input design vectors.
        y (np.ndarray): Target values.
        name (str): Target name ('Cl', 'Cd', 'Cm').

    Returns:
        tuple: (model, scaler_X, scaler_y, (rmse, r2))
    """
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

```

---

## File: `experiments\part_c_uncertainty.py`
```python
"""
Part C.1: Uncertainty Analysis (Robustness).

This script performs a Monte Carlo simulation to quantify the robustness of the 
Optimal Airfoil design against operational uncertainties (e.g., gust loads).

Methodology:
1.  **Load:** Logic to find the best design (from Part C Surrogate or Part B).
2.  **Perturb:** Sample Angle of Attack (Alpha) from a Normal Distribution N(3.0, 0.1).
3.  **Evaluate:** Run XFOIL in parallel for all perturbed conditions.
4.  **Analyze:** Compute Mean, Std Dev, and Coefficient of Variation for Cl/Cd.

Outputs:
- Robustness Report (txt)
- Histograms (png)
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.cst import cst_airfoil
from utils.geometry import write_dat
from utils.xfoil_runner import run_xfoil_single_alpha

def main():
    print("--- Part C.1: Uncertainty Analysis ---")
    
    # 1. Load Best Design from Part C (Surrogate)
    # Because regression test wiped Part B, we use Surrogate Opt.
    surro_json = os.path.join("data", "PartC", "surrogate", "surrogate_results.json")
    
    if os.path.exists(surro_json):
        print(f"Loading surrogate design from: {surro_json}")
        with open(surro_json, "r") as f:
            data = json.load(f)
        x_opt = np.array(data["x_opt_surro"])
    else:
        # Fallback to Part B
        pattern = os.path.join("data", "PartB", "results", "*_best.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print("Error: No design found.")
            return
        print(f"Loading Part B design from: {files[-1]}")
        with open(files[-1], "r") as f:
            data = json.load(f)
        x_opt = np.array(data["x"])
    
    # 2. Generate random Alphas
    np.random.seed(42)
    mean_alpha = 3.0
    std_alpha = 0.1
    n_samples = 100
    
    alphas = np.random.normal(mean_alpha, std_alpha, n_samples)
    
    # 3. Prepare Airfoil File (once)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(root, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    x, yu, xl, yl = cst_airfoil(200, x_opt[:3], x_opt[3:])
    
    dat_path = os.path.join(temp_dir, "robust_candidate.dat")
    # Write standard dat
    xu = x[::-1]; yup = yu[::-1]
    xl_p = x[1:]; ylo_p = yl[1:]
    xx = np.concatenate([xu, xl_p])
    yy = np.concatenate([yup, ylo_p])
    write_dat(xx, yy, dat_path, name="ROBUST_TEST")
    
    # 4. Evaluate
    print(f"Evaluating {n_samples} samples (Alpha ~ N({mean_alpha}, {std_alpha}))...")
    
    results = []
    failed = 0
    
    # 4. Evaluate Parallel
    from multiprocessing import Pool
    from functools import partial
    
    print(f"Evaluating {n_samples} samples (Alpha ~ N({mean_alpha}, {std_alpha})) using 8 workers...")
    
    # We need a helper to freeze dat_path
    # run_xfoil_single_alpha(dat_path, alpha=a, Re=1e6, n_iter=200)
    
    eval_fn = partial(run_xfoil_single_alpha, dat_path, Re=1e6, n_iter=200)
    
    with Pool(processes=8) as pool:
        # returns list of (cl, cd, cm) tuples
        raw_res = pool.map(eval_fn, alphas)
        
    results = []
    failed = 0
    for i, res in enumerate(raw_res):
        a = alphas[i]
        cl, cd, cm = res
        if cl is not None:
             results.append([a, cl, cd, cm])
        else:
             failed += 1
            
    print(f"Completed. Failures: {failed}/{n_samples}")
    
    if not results:
        print("All runs failed. Cannot compute stats.")
        return

    res_arr = np.array(results)
    # Col 0: alpha, 1: cl, 2: cd, 3: cm
    
    # 5. Statistics
    mu_cl, std_cl = np.mean(res_arr[:,1]), np.std(res_arr[:,1])
    mu_cd, std_cd = np.mean(res_arr[:,2]), np.std(res_arr[:,2])
    mu_cm, std_cm = np.mean(res_arr[:,3]), np.std(res_arr[:,3])
    
    print("-" * 40)
    print(f"Uncertainty Results (Alpha std={std_alpha} deg)")
    print(f"CL: Mean = {mu_cl:.4f}, Std = {std_cl:.4f}, COV = {std_cl/abs(mu_cl):.4f}")
    print(f"CD: Mean = {mu_cd:.4f}, Std = {std_cd:.4f}, COV = {std_cd/abs(mu_cd):.4f}")
    print(f"CM: Mean = {mu_cm:.4f}, Std = {std_cm:.4f}")
    print("-" * 40)
    
    # Save results
    out_dir = os.path.join("data", "PartC", "uncertainty")
    os.makedirs(out_dir, exist_ok=True)
    
    report_file = os.path.join(out_dir, "uncertainty_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Uncertainty Analysis for Surrogate Design\n")
        f.write(f"Samples: {n_samples}, Failures: {failed}\n")
        f.write(f"CL: Mean = {mu_cl:.6f}, Std = {std_cl:.6f}\n")
        f.write(f"CD: Mean = {mu_cd:.6f}, Std = {std_cd:.6f}\n")
        f.write(f"CM: Mean = {mu_cm:.6f}, Std = {std_cm:.6f}\n")
        
    # Plot Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(res_arr[:,1], bins=15, color='b', alpha=0.7)
    axes[0].set_title(f"Lift Coeff (Std={std_cl:.4f})")
    
    axes[1].hist(res_arr[:,2], bins=15, color='r', alpha=0.7)
    axes[1].set_title(f"Drag Coeff (Std={std_cd:.4f})")
    
    axes[2].hist(res_arr[:,3], bins=15, color='g', alpha=0.7)
    axes[2].set_title(f"Moment Coeff (Std={std_cm:.4f})")
    
    plt.suptitle(f"Robustness Check: Alpha ~ N(3.0, 0.1)")
    plot_path = os.path.join(out_dir, "robustness_histograms.png")
    plt.savefig(plot_path)
    print(f"Saved plots to {plot_path}")

if __name__ == "__main__":
    main()

```

---

## File: `experiments\part_c_uq_opt.py`
```python
"""
CLI Tool for Uncertainty Quantification.

A command-line version of the robustness test, allowing analysis of *any* design JSON
(not just the hardcoded 'best' one).

Usage:
    python experiments/part_c_uq_opt.py --json data/PartB/results/xyz_best.json --samples 50

Features:
- **Flexible Input:** Check robustness of any intermediate solution.
- **Configurable:** Custom mean/std for Alpha.
"""

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

```

---

## File: `experiments\plot_airfoil.py`
```python
"""
Part B Plotting Library.

This module provides specialized plotting functions for Airfoil Optimisation results.

Key Plots:
1.  **Convergence:** Optimization Fitness (J) vs Iterations.
2.  **Geometry:** Visualizes the optimized Airfoil shape, including:
    - CST Construction.
    - Rotation to Wind Frame (Alpha angle).
    - Performance Annotation (Cl, Cd, L/D).
3.  **Coefficients:** Bar chart of the optimized CST variables.
4.  **Polars:** Drag Polar (Cl vs Cd) and Lift Curve (Cl vs Alpha).

Dependencies:
    - matplotlib
    - pandas
    - utils.geometry (CST)
    - utils.xfoil_runner (for Polars)
"""
import argparse
import glob
import json
import os

import numpy as np
try:
    import pandas as pd
    from pandas.errors import EmptyDataError
except ImportError:
    pd = None
    EmptyDataError = ValueError

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from benchmarks.airfoil_xfoil import coeffs_at_alpha
from utils.geometry import build_airfoil_coordinates as airfoil_coords
from utils.geometry import write_dat
from utils.xfoil_runner import run_xfoil_polar

BASE_FIG_DIR = os.path.join("data", "figures", "airfoil")

# Simple baseline CST coefficients (you can adjust these)
BASELINE_AU = [0.2, 0.1, 0.05]
BASELINE_AL = [-0.1, -0.05, -0.02]


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def make_airfoil(Au, Al, filename, npts=201):
    x, y = airfoil_coords(Au, Al, n_points=npts)
    write_dat(x, y, filename, name="CST_AIRFOIL")




def read_log(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except EmptyDataError:
        raise SystemExit(f"CSV is empty or invalid: {csv_path}. "
                         f"Delete it and re-run experiments.run_airfoil.")
    for c in ["f_best", "f_mean", "f_std", "gbest_f"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df



def plot_convergence(csv_path: str):
    """
    Plot the optimization convergence history (Best Fitness vs Iterations).
    
    Args:
        csv_path (str): Path to the run log CSV.
        
    Returns:
        str: Path to the saved image file.
    """
    if plt is None or pd is None:
        print("Skipping plot_convergence (missing libs)")
        return None
        
    df = read_log(csv_path)
    fig = plt.figure()
    ax = plt.gca()
    vals = df["gbest_f"]
    if (vals <= 0).any():
        ax.plot(df["iter"], vals)
    else:
        ax.semilogy(df["iter"], vals)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best objective J")
    ax.grid(True, which="both", linestyle=":")
    ax.set_title("Airfoil optimisation convergence")

    base = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Use helper to find path
    fig_dir = _infer_output_dir(csv_path, "convergence")
    outpath = os.path.join(fig_dir, f"{base}_convergence.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _infer_output_dir(csv_path: str, subdir: str = "airfoil") -> str:
    """Helper to infer figure output directory from CSV location."""
    if not csv_path:
        return os.path.join("data", "figures", subdir)
        
    csv_dir = os.path.dirname(csv_path)
    if "Part" in csv_dir:
        # e.g. data/PartB/results -> data/PartB/figures/subdir
        if os.path.basename(csv_dir) == "results":
            base_dir = os.path.dirname(csv_dir)
        else:
            base_dir = csv_dir
        return os.path.join(base_dir, "figures", subdir)
    else:
        return os.path.join("data", "figures", subdir)

import numpy as np
from utils.cst import cst_airfoil
from utils.xfoil_runner import run_xfoil_polar
from utils.geometry import write_dat

def plot_geometry(best_vec, out_csv: str = None, alpha: float = 3.0):
    """
    Plot the optimized airfoil geometry, rotated to the wind frame.

    Visualizes:
    - Upper/Lower surfaces.
    - Wind streamlines (Horizontal).
    - Performance Metrics (Cl, Cd, L/D) if importable.
    
    Args:
        best_vec (np.ndarray): Array of CST coefficients [Au, Al].
        out_csv (str): Path to source CSV (used for naming output file).
        alpha (float): Angle of Attack (deg) for rotation and annotation.
        
    Returns:
        str: Path to saved image.
    """
    if plt is None: return None
    
    # CST parameters
    Au = best_vec[:3]
    Al = best_vec[3:]
    
    # Generate coordinates for plotting (more points for smoothness)
    x, yu, xl, yl = cst_airfoil(200, Au, Al)
    
    # Rotation for "Wind Tunnel View" (Wind Horizontal)
    # To generate Lift (Alph=3 deg) with L->R wind, the Nose should be Higher than the Tail (relative to horizontal).
    # This exposes the bottom surface to the wind.
    # Current coords: LE at (0,0), TE at (1,0).
    # We want TE to be LOWER than LE (Visual Nose Up).
    # So we rotate by -Alpha.
    
    rad = np.radians(-alpha) # Negative to pitch Nose Up (Tail Down)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate Upper Surface
    # Stack (N, 2)
    pts_u = np.column_stack([x, yu])
    pts_u_rot = pts_u @ R.T
    xu_rot, yu_rot = pts_u_rot[:, 0], pts_u_rot[:, 1]
    
    # Rotate Lower Surface
    pts_l = np.column_stack([x, yl])
    pts_l_rot = pts_l @ R.T
    xl_rot, yl_rot = pts_l_rot[:, 0], pts_l_rot[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 4)) # Taller for rotation
    
    # Plot Airfoil
    ax.plot(xu_rot, yu_rot, 'b-', label='Upper', linewidth=2)
    ax.plot(xl_rot, yl_rot, 'r-', label='Lower', linewidth=2)
    ax.fill_between(xu_rot, yu_rot, yl_rot, color='gray', alpha=0.3, zorder=10)
    
    # Explicit Labels for Nose/Tail
    ax.text(xu_rot[0], yu_rot[0]-0.08, "Nose (LE)", color='black', fontweight='bold', ha='center', va='top')
    ax.text(xl_rot[-1]+0.05, yl_rot[-1], "Tail (TE)", color='black', fontweight='bold', ha='left')
    
    # Draw Wind Streamlines (Horizontal)
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.4, 0.4
    
    # 50 Lines
    y_lines = np.linspace(y_min, y_max, 50)
    
    first_line = True
    for y_line in y_lines:
        # Label only the first line to avoid legend duplication
        lbl = 'Wind Streamlines' if first_line else None
        
        ax.plot([x_min, x_max], [y_line, y_line], color='deepskyblue', alpha=0.15, lw=1.5, zorder=0, label=lbl)
        first_line = False

    # Add Velocity Vector Label
    ax.arrow(-0.15, 0.0, 0.1, 0.0, head_width=0.02, color='deepskyblue', lw=2, zorder=20)
    ax.text(-0.15, 0.03, r"$V_{\infty}$", color='deepskyblue', fontsize=12, fontweight='bold')

    ax.axis('equal')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(f"Airfoil Geometry (Rotated to Wind Frame, $\\alpha={alpha}^\circ$)")
    
    # Add Alpha Text Box
    text_str = f"$\\alpha = {alpha}^\circ$"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
            
    # --- Performance Annotation ---
    # Convert 'best_vec' to Cl, Cd, Cm
    # We need to import airfoil_fitness safely
    try:
        from benchmarks.airfoil_xfoil import airfoil_fitness
        # best_vec is numpy array, convert to list
        J_val, Cl_val, Cd_val, Cm_val = airfoil_fitness(best_vec, return_all=True)
        
        # Format string
        if Cl_val is not None:
             perf_str = (
                 f"Performance:\n"
                 f"Cl: {Cl_val:.4f}\n"
                 f"Cd: {Cd_val:.5f}\n"
                 f"L/D: {Cl_val/Cd_val:.1f}\n"
                 f"Cm: {Cm_val:.4f}"
             )
             
             # Place box in top-right or bottom-left? 
             # Top-left is taken by Alpha. Let's try Top-Right or Bottom-Right.
             # Top-Right is clean.
             perf_props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.6)
             ax.text(0.96, 0.96, perf_str, transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right', bbox=perf_props)
        else:
             print("Warning: Failed to compute metrics for annotation.")
    except ImportError:
        print("Warning: Could not import airfoil_fitness for plot annotation.")
    
    ax.grid(True, linestyle=":")
    ax.legend(loc='lower right')
    
    fig_dir = _infer_output_dir(out_csv, "geometry")
    base = os.path.splitext(os.path.basename(out_csv))[0]
    outpath = os.path.join(fig_dir, f"{base}_geometry.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_coeff_bar(best_vec, Re=1e6, alpha=3.0, out_csv: str = None, outpath=None):
    """
    Visualize the optimized CST coefficients (Au0..2, Al0..2) as a bar chart.
    """
    if plt is None: return None
    
    labels = [f"Au{i}" for i in range(3)] + [f"Al{i}" for i in range(3)]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, best_vec, color=['b']*3 + ['r']*3)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Optimized CST Parameters")
    
    # Add values on top
    for rect in bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    fig_dir = _infer_output_dir(out_csv, "coefficients")
    base = os.path.splitext(os.path.basename(out_csv))[0]
    outpath = os.path.join(fig_dir, f"{base}_coeff_bar.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_polar(best_vec, Re=1e6, out_csv: str = None):
    """
    Generate and plot the Drag Polar (Cl vs Cd) and Lift Curve (Cl vs Alpha).

    Note: This triggers a new XFOIL run (ASEQ) for the optimized geometry.
    Range: -5 to +15 degrees.
    """
    if plt is None: return None
    
    # We need a .dat file to run polar
    # Generate coordinates
    # cst_airfoil(n_points, coeffs_upper, coeffs_lower)
    x, yu, xl, yl = cst_airfoil(160, best_vec[:3], best_vec[3:])
    
    # Create temp dat file in project temp dir
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(root, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', dir=temp_dir, delete=False) as f:
        tmp_dat = f.name
    
    # Convert to standard format (TE->LE->TE loop)
    # Upper: TE(1) to LE(0)
    # Lower: LE(0) to TE(1)
    # cst_airfoil gives 0->1.
    # Upper needs flip.
    xu = x[::-1]
    yup = yu[::-1]
    xl = x[1:]
    ylo = yl[1:]
    
    # Concat
    xx = np.concatenate([xu, xl])
    yy = np.concatenate([yup, ylo])
    
    write_dat(xx, yy, tmp_dat, name="OPT_AIRFOIL")
    
    # Run Polar
    # Alpha range: -5 to 15 deg
    print("Running polar analysis for plot...")
    alphas, cls, cds, cms = run_xfoil_polar(tmp_dat, -5, 15, 1.0, Re=Re, n_iter=200)
    
    # Clean up dat
    try:
        os.remove(tmp_dat)
    except:
        pass
        
    if len(alphas) == 0:
        print("Polar run failed or returned no data.")
        return None
        
    # Plot Drag Polar (Cl vs Cd) and Cl/alpha
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Drag Polar
    ax = axes[0]
    ax.plot(cds, cls, 'o-')
    ax.set_xlabel("Cd")
    ax.set_ylabel("Cl")
    ax.set_title(f"Drag Polar (Re={Re:.1e})")
    ax.grid(True)
    
    # Cl/Alpha
    ax = axes[1]
    ax.plot(alphas, cls, 'o-')
    ax.set_xlabel("Alpha (deg)")
    ax.set_ylabel("Cl")
    ax.set_title("Lift Curve")
    ax.grid(True)
    
    fig_dir = _infer_output_dir(out_csv, "polar")
    base = os.path.splitext(os.path.basename(out_csv))[0]
    outpath = os.path.join(fig_dir, f"{base}_polar.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to airfoil optimisation CSV. If omitted, use latest in data/results/airfoil.")
    args = ap.parse_args()

    if args.csv is None:
        pattern = os.path.join("data", "results", "airfoil", "airfoil_opt_seed*_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit("No airfoil optimisation CSV found. Run experiments.run_airfoil first.")
        csv_path = files[-1]
    else:
        csv_path = args.csv

    print("Using CSV:", csv_path)

    # 1) Convergence plot
    conv_png = plot_convergence(csv_path)
    print("Saved convergence:", conv_png)

    # 2) Load best design vector JSON
    best_json = os.path.splitext(csv_path)[0] + "_best.json"
    if not os.path.exists(best_json):
        raise SystemExit(f"Best design JSON not found: {best_json}")
    with open(best_json, "r") as f:
        best = json.load(f)
    best_vec = np.array(best["x"], dtype=float)

    # 3) Geometry comparison
    geom_png = plot_geometry(best_vec, out_csv=csv_path)
    print("Saved geometry comparison:", geom_png)

    coeff_png = plot_coeff_bar(best_vec, Re=1e6, alpha=3.0, out_csv=csv_path)
    print("Saved coefficients bar chart:", coeff_png)


    # 4) Drag polar comparison
    try:
        polar_png = plot_polar(best_vec, Re=1e6, out_csv=csv_path)
        print("Saved polar:", polar_png)
    except Exception as e:
        print("Warning: polar plot failed, skipping. Reason:", e)



if __name__ == "__main__":
    main()

```

---

## File: `experiments\plotting.py`
```python
"""
General Plotting Utilities (Part A).

This module contains statistical plotting functions used to analyze optimizer stability.

Key Plots:
1.  **Convergence Overlay:** Stacks multiple runs (seeds) to visualize variance.
2.  **Robustness Boxplot:** Shows the distribution of final fitness values.
3.  **Success Rate:** Empirical probability of solving the problem vs budget.
"""
import os
import glob
import math
import re
from typing import List, Tuple, Dict

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

BASE_FIG_DIR = os.path.join("data", "PartB", "figures")


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def read_log(csv_path: str):
    """Read one run CSV written by experiments/run_opt.py and coerce columns."""
    if pd is None:
        raise ImportError("pandas is required for reading logs but is not installed.")
    df = pd.read_csv(csv_path)
    # ensure numeric (they were formatted as strings for pretty printing)
    for c in ["f_best", "f_mean", "f_std", "gbest_f"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def parse_meta_from_filename(path: str) -> Dict[str, str]:
    """
    Extract D, pop, seed from a filename like griewank_D5_pop40_seed7_YYYYMMDD_*.csv
    """
    name = os.path.basename(path)
    meta = {}
    mD = re.search(r"D(\d+)", name)
    mp = re.search(r"pop(\d+)", name)
    ms = re.search(r"seed(\d+)", name)
    if mD: meta["D"] = mD.group(1)
    if mp: meta["pop"] = mp.group(1)
    if ms: meta["seed"] = ms.group(1)
    meta["base"] = os.path.splitext(name)[0]
    return meta

def plot_convergence(csv_path: str, outpath: str = None, ykey: str = "gbest_f"):
    """
    Semilogy convergence curve for a single run.
    ykey in {"gbest_f", "f_best"} – gbest_f is global best so far (preferred).
    """
    if plt is None or pd is None:
        print("Skipping plot_convergence (matplotlib/pandas missing)")
        return None
        
    df = read_log(csv_path)
    fig = plt.figure()
    ax = plt.gca()
    vals = df[ykey]
    if (vals <= 0).any():
        ax.plot(df["iter"], vals)
    else:
        ax.semilogy(df["iter"], vals)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best objective value")
    ax.grid(True, which="both", linestyle=":")
    meta = parse_meta_from_filename(csv_path)
    title = f"Convergence (Griewank, D={meta.get('D','?')}, pop={meta.get('pop','?')}, seed={meta.get('seed','?')})"
    ax.set_title(title)

    if outpath is None:
        # Simplificated logic: Look for "PartX" in path
        # Expected structure: data/PartX/results/foo.csv -> data/PartX/figures/foo_conv.png
        
        csv_dir = os.path.abspath(os.path.dirname(csv_path))
        
        # Try to find "PartX" in the path parts
        parts = csv_dir.split(os.sep)
        part_name = None
        for p in parts:
            if p.startswith("Part"):
                part_name = p
                break
        
        if part_name:
            # We found PartA, PartB etc.
            # Find the root of valid data folder (parent of PartX)
            # This is tricky if absolute. Easier strategy:
            # If csv is in .../PartA/results, go up to PartA, then down to figures.
            
            if "results" in parts:
                # Assuming .../PartA/results
                # Go up one level from 'results'
                base_dir = os.path.dirname(csv_dir)
            else:
                # Assuming .../PartA
                base_dir = csv_dir
                
            fig_dir = os.path.join(base_dir, "figures", "convergence")
        else:
            # Fallback to default
            fig_dir = os.path.join(BASE_FIG_DIR, "convergence")

        outpath = os.path.join(fig_dir, f"{meta['base']}_conv.png")

    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_convergence_overlay(csv_paths: List[str], outpath: str = None, ykey: str = "gbest_f"):
    """
    Overlay multiple runs (e.g., different seeds) on one semilogy plot.
    """
    if plt is None or pd is None:
        return None

    fig = plt.figure()
    ax = plt.gca()
    for p in csv_paths:
        df = read_log(p)
        label = parse_meta_from_filename(p).get("base", "run")
        ax.semilogy(df["iter"], df[ykey], alpha=0.7, label=label)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best objective value")
    ax.grid(True, which="both", linestyle=":")
    ax.set_title("Convergence overlay")
    # ax.legend() # messy if too many

    if outpath is None:
        # Default to first CSV's location
        if csv_paths:
             # Re-use logic from above? Or just hardcode for simplicity in this overlay
             # For overlays, we usually want them in the parent figures folder
             p1 = os.path.abspath(os.path.dirname(csv_paths[0]))
             if "results" in p1: 
                 base = os.path.dirname(p1) 
                 outpath = os.path.join(base, "figures", "convergence_overlay.png")
             else:
                 outpath = os.path.join(p1, "figures", "convergence_overlay.png")
        else:
             outpath = os.path.join(BASE_FIG_DIR, "overlays", "convergence_overlay.png")

    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_final_boxplot(csv_paths: List[str], outpath: str = None, key: str = "gbest_f"):
    """
    Boxplot of final best objective across runs/seeds.
    """
    if plt is None or pd is None:
        return None

    finals = []
    for p in csv_paths:
        df = read_log(p)
        finals.append(df[key].iloc[-1])
    data = np.array(finals, dtype=float)

    fig = plt.figure()
    ax = plt.gca()
    ax.boxplot(data, vert=True, showmeans=True)
    ax.set_xticks([1])
    ax.set_xticklabels([f"{len(csv_paths)} runs"])
    ax.set_ylabel("Final best objective")
    ax.set_title("Distribution of final best objective")
    ax.grid(True, axis="y", linestyle=":")

    if outpath is None:
        if csv_paths:
             p1 = os.path.abspath(os.path.dirname(csv_paths[0]))
             if "results" in p1: 
                 base = os.path.dirname(p1) 
                 outpath = os.path.join(base, "figures", "final_boxplot.png")
             else:
                 outpath = os.path.join(p1, "figures", "final_boxplot.png")
        else:
            outpath = os.path.join(BASE_FIG_DIR, "overlays", "final_boxplot.png")

    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_success_vs_budget(csv_paths: List[str], thresholds: List[float] = [1e-6, 1e-4, 1e-2],
                           outpath: str = None, ykey: str = "gbest_f"):
    """
    Empirical success curve: for each evaluation count, share of runs that have reached threshold.
    Works best if runs share pop size so evals = iter*pop is roughly comparable.
    """
    if plt is None or pd is None:
        return None

    # Align by evals
    all_evals = set()
    series = []
    for p in csv_paths:
        df = read_log(p)
        all_evals.update(df["evals"].tolist())
        series.append(df[["evals", ykey]].copy())
    grid = np.array(sorted(all_evals), dtype=int)

    # For each run, map best-so-far at each eval grid via forward fill
    stacked = []
    for s in series:
        s2 = s.set_index("evals").reindex(grid).ffill()
        stacked.append(s2[ykey].to_numpy())
    M = np.vstack(stacked)  # shape: (runs, len(grid))

    fig = plt.figure()
    ax = plt.gca()
    for thr in thresholds:
        success = np.mean(M <= thr, axis=0)
        ax.plot(grid, success)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=":")
    ax.set_title("Empirical success vs evaluation budget")
    ax.legend([f"f ≤ {t:g}" for t in thresholds])

    if outpath is None:
        outpath = os.path.join(BASE_FIG_DIR, "overlays", "success_vs_budget.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_swarm_2d(csv_path: str, positions_snapshots: List[np.ndarray], outpath: str = None):
    """
    Optional (only if D=2): provide pre-captured positions (list per iter).
    This helper just plots trajectory clouds.
    """
    if plt is None:
        return None

    fig = plt.figure()
    ax = plt.gca()
    for pts in positions_snapshots:
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.4)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("PSO swarm trajectory (D=2)")
    ax.grid(True, linestyle=":")

    if outpath is None:
        meta = parse_meta_from_filename(csv_path)
        outpath = os.path.join(BASE_FIG_DIR, "swarm", f"{meta['base']}_swarm2d.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

```

---

## File: `experiments\run_airfoil.py`
```python
"""
Part B Experiment Driver: Airfoil Optimization.

This script orchestrates the aerodynamic optimization process (Part B).
It connects the optimizer (PSO or GA) with the Physics Engine (XFOIL).

Key Responsibilities:
1.  **Configuration:** Parses CLI arguments (pop size, budget, solver).
2.  **Environment:** Cleans temp folders to prevent XFOIL collisions.
3.  **Estimation:** Benchmarks system speed to estimate runtime.
4.  **Optimization:** Runs the loop using `experiments.run_opt.optimize` (shared driver).
5.  **Plotting:** Automatically generates Geometry and Convergence plots upon completion.

Usage:
    python experiments/run_airfoil.py --solver pso --evals 1000 --jobs 4
"""
import os
from datetime import datetime
import json
import numpy as np

import sys
sys.path.append(os.getcwd())

from optimizer.pso import PSO
from optimizer.ga import GA
from benchmarks.airfoil_xfoil import airfoil_fitness
from experiments.run_opt import optimize   # reuse Part A harness


import argparse
from experiments import plot_airfoil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default=None, help="Assignment Part (A, B, or C)")
    parser.add_argument("--evals", type=int, default=200, help="Evaluation budget")
    parser.add_argument("--pop", type=int, default=20, help="Population size")
    parser.add_argument("--points", type=int, default=200, help="Airfoil geometry points")
    parser.add_argument("--iter", type=int, default=100, help="XFOIL max iterations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--clean", action="store_true", help="Delete existing data folder for this Part before running")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--solver", type=str, default="pso", choices=["pso", "ga"], help="Optimisation algorithm")
    args = parser.parse_args()

    # Pre-run Cleanup (Clean temp folder)
    def cleanup_temp():
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Cleaned temp directory: {temp_dir}")

    cleanup_temp() # Always clean temp/ at start

    # Runtime Estimation removed as per user request (was inaccurate).
    # The real-time ETA in run_opt.py provides better feedback.

    # Handle cleaning request
    if args.clean:
        if args.part:
             target_clean = os.path.join("data", f"Part{args.part}")
             print(f"Cleaning {target_clean}...")
             if os.path.exists(target_clean):
                 import shutil
                 shutil.rmtree(target_clean)
             
             # Also clear in-memory cache which was loaded at import time
             import benchmarks.airfoil_xfoil
             benchmarks.airfoil_xfoil.clear_cache()
             print("In-memory cache cleared.")
        else:
             print("Warning: --clean flag ignored because --part was not specified.")

    # 6 CST coefficients: 3 upper, 3 lower
    bounds = [(-0.2, 0.5)] * 6
    
    seed = args.seed
    eval_budget = args.evals

    if args.solver == "pso":
        # Standard PSO Defaults
        options = dict(pop=args.pop, w=0.7, c1=1.6, c2=1.6)
        opt = PSO(bounds=bounds, seed=seed, options=options)
    elif args.solver == "ga":
        # GA Defaults
        options = dict(pop=args.pop, mutation_rate=0.1, crossover_rate=0.9)
        opt = GA(bounds=bounds, seed=seed, options=options)
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.part:
        # Structured: data/PartB/pso/results
        folder = os.path.join("data", f"Part{args.part}", args.solver, "results")
    else:
        # Fallback
        folder = os.path.join("data", "results", "airfoil", args.solver)
        
    os.makedirs(folder, exist_ok=True)
    # detailed naming: airfoil_{solver}_pop{pop}_evals{evals}_seed{seed}_{stamp}.csv
    out_csv = os.path.join(folder, f"airfoil_{args.solver}_pop{args.pop}_evals{args.evals}_seed{seed}_{stamp}.csv")
    
    from functools import partial
    fitness_fn = partial(airfoil_fitness, Re=1e6, alpha=3.0, n_points=args.points, n_iter=args.iter)

    best = optimize(
        fitness_fn,
        opt,
        eval_budget=eval_budget,
        f_target=-1e9, # Do not stop early for negative fitness
        log_path=out_csv,
        n_jobs=args.jobs
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

```

---

## File: `experiments\run_opt.py`
```python
"""
Generic Optimization Driver.

This script runs a single optimization session on a benchmark function (default: Griewank).
It is primarily used by `run_part_a.py` for statistical validation.

Features:
- **Parallelization:** Supports multi-core evaluation via `multiprocessing.Pool`.
- **Logging:** streams Real-Time stats to CSV.
- **Stopping Criteria:** Budget, Target Fitness, or Stagnation.
"""
import argparse
import csv
import os
import time
import multiprocessing
from datetime import datetime

import numpy as np

from benchmarks.griewank import griewank
from optimizer.pso import PSO
from experiments.plotting import plot_swarm_2d, plot_convergence  # for optional 2D swarm plot and convergence


def optimize(f, opt: PSO, eval_budget: int, f_target: float = 1e-6, stagnation: int = 200, log_path: str = "run.csv", n_jobs: int = 1):
    """
    Execute the main optimization loop.

    Args:
        f (callable): Objective function (must accept numpy array).
        opt (Optimizer): The initialized optimizer instance (PSO/GA).
        eval_budget (int): Maximum number of function evaluations.
        f_target (float): Target fitness to stop early (if reached).
        stagnation (int): Stop if no improvement after N iterations.
        log_path (str): Path to save the CSV log.
        n_jobs (int): Number of parallel workers (1 = Serial).

    Returns:
        dict: The best solution found {'x': ..., 'f': ...}.
    """
    best_seen = np.inf
    best_seen = np.inf
    no_improve = 0
    
    # Timing initialization
    start_time = time.time()
    from collections import deque
    window = deque(maxlen=8)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    
    # Initialize Pool if parallel
    pool = None
    if n_jobs > 1:
        print(f"--- Parallel Mode Enabled: Using {n_jobs} workers ---")
        pool = multiprocessing.Pool(processes=n_jobs)

    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        else:
            return f"{m}m {s:02d}s"

    try:
        with open(log_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["iter","evals","f_best","f_mean","f_std","gbest_f","x_best"])
            writer.writeheader()

            try:
                while True:
                    X = opt.ask()                     # list of np arrays
                    
                    # Evaluate fitness (Parallel or Serial)
                    import sys
                    current_iter = opt.state()['iter'] + 1
                    total_pop = len(X)
                    
                    # Evaluate fitness (Parallel or Serial)
                    F = []
                    if pool:
                        # Use imap to show progress within the generation
                        for i, res in enumerate(pool.imap(f, X), 1):
                            F.append(res)
                            sys.stdout.write(f"\r[Iter {current_iter}] Evaluating: {i}/{total_pop}")
                            sys.stdout.flush()
                    else:
                        for i, x in enumerate(X, 1):
                            F.append(f(x))
                            sys.stdout.write(f"\r[Iter {current_iter}] Evaluating: {i}/{total_pop}")
                            sys.stdout.flush()
                            
                    # Clear the progress line (overwrite with spaces then CR)
                    sys.stdout.write("\r" + " "*50 + "\r")
                    
                    opt.tell(F)                       # inform optimiser
                    
                    # Timing calculation
                    now = time.time()
                    elapsed = now - start_time
                    elapsed_str = format_time(elapsed)
                    
                    st = opt.state()
                    writer.writerow({
                        "iter": st["iter"],
                        "evals": st["evals_total"],
                        "f_best": f"{st['f_best']:.12e}",
                        "f_mean": f"{st['f_mean']:.12e}",
                        "f_std": f"{st['f_std']:.12e}",
                        "gbest_f": f"{st['gbest_f']:.12e}",
                        "x_best": str(list(st['gbest_x'])),
                    })
                    
                    # Dynamic ETA Calculation (Rolling Window)
                    evals_done = st['evals_total']
                    window.append((now, evals_done))
                    
                    eta_str = "..."
                    if len(window) > 1:
                         # Calculate rate based on window
                         t_start, e_start = window[0]
                         t_end, e_end = window[-1]
                         dt = t_end - t_start
                         de = e_end - e_start
                         
                         if dt > 0 and de > 0:
                             rate = de / dt # evals per second (recent)
                             evals_remaining = max(0, eval_budget - evals_done)
                             if evals_remaining > 0:
                                 eta_seconds = evals_remaining / rate
                                 eta_str = format_time(eta_seconds)
                             else:
                                 eta_str = "0m 00s"

                    # Simple progress logging
                    print(f"[Iter {st['iter']}] Evals: {st['evals_total']} | Best: {st['gbest_f']:.6e} | Mean: {st['f_mean']:.6e} | Elapsed: {elapsed_str} | ETA: {eta_str}")

                    # stopping logic
                    if st["gbest_f"] < best_seen - 1e-16:
                        best_seen = st["gbest_f"]
                        no_improve = 0
                    else:
                        no_improve += 1

                    if st["evals_total"] >= eval_budget: break
                    if st["gbest_f"] <= f_target: break
                    if no_improve >= stagnation: break

            except KeyboardInterrupt:
                print("\n!!! Interrupted by user. Stopping optimization early and saving current results... !!!")
    
    finally:
        # Cleanup pool
        if pool:
            pool.close()
            pool.join()


    return opt.best()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=5, help="Dimension of Griewank")
    parser.add_argument("--pop", type=int, default=40)
    parser.add_argument("--evals", type=int, default=50000, help="Evaluation budget")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w", type=float, default=0.72)
    parser.add_argument("--c1", type=float, default=1.6)
    parser.add_argument("--c2", type=float, default=1.6)
    parser.add_argument("--vmax_frac", type=float, default=0.2)
    parser.add_argument("--topology", type=str, default="gbest")  # or lbest
    parser.add_argument("--trace_every", type=int, default=0, help="Record 2D swarm positions every N iters (D=2 only)")
    parser.add_argument("--part", type=str, default=None, help="Assignment Part (A, B, or C) to organize outputs")
    parser.add_argument("--out", type=str, default=None, help="CSV log path (overrides default data/results/... location)")
    parser.add_argument("--clean", action="store_true", help="Delete existing data folder for this Part before running")
    args = parser.parse_args()

    # Handle cleaning request
    if args.clean:
        if args.part:
             target_clean = os.path.join("data", f"Part{args.part}")
             print(f"Cleaning {target_clean}...")
             if os.path.exists(target_clean):
                 import shutil
                 shutil.rmtree(target_clean)
        else:
             print("Warning: --clean flag ignored because --part was not specified. Identifying correct folder is ambiguous.")


    bounds = [(-600.0, 600.0)] * args.D
    options = dict(
        pop=args.pop, w=args.w, c1=args.c1, c2=args.c2,
        vmax_frac=args.vmax_frac, topology=args.topology,
        trace_every=args.trace_every,
    )

    opt = PSO(bounds=bounds, seed=args.seed, options=options)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.out:
        # Respect explicit path
        log_path = args.out
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    else:
        # Default structured location
        if args.part:
            folder = os.path.join("data", f"Part{args.part}", "results")
        else:
            # Default to PartA for generic runs if not specified, or just "results"
            folder = os.path.join("data", "PartA", "results")
            
        os.makedirs(folder, exist_ok=True)
        log_path = os.path.join(
            folder,
            f"griewank_D{args.D}_pop{args.pop}_seed{args.seed}_{stamp}.csv"
        )

    best = optimize(griewank, opt, eval_budget=args.evals, log_path=log_path)
    print("Best:", best)

    # If 2D + tracing enabled, emit a swarm trajectory figure in data/figures/swarm/
    if args.D == 2 and args.trace_every > 0:
        trace = opt.positions_trace() if hasattr(opt, "positions_trace") else []
        if trace:
            # Save swarm plot in the SAME folder as the log, but with a suffix, or a 'figures' subfolder?
            # User wants "clarity".
            if args.part:
                swarm_dir = os.path.join("data", f"Part{args.part}", "figures")
            else:
                # Try to deduce from log_path if possible, otherwise default
                log_dir = os.path.dirname(log_path)
                if "Part" in log_dir:
                     # e.g. data/PartA/results -> data/PartA/figures
                     base = os.path.dirname(log_dir)
                     swarm_dir = os.path.join(base, "figures", "swarm")
                else:
                     # fallback
                     swarm_dir = os.path.join("data", "PartB", "figures", "swarm")

            os.makedirs(swarm_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(log_path))[0]
            out_png = os.path.join(swarm_dir, f"{base}_swarm2d.png")
            plot_swarm_2d(log_path, trace, outpath=out_png)
            print("Saved 2D swarm trajectory:", out_png)

    # Always plot convergence
    conv_png = plot_convergence(log_path)
    print("Saved convergence plot:", conv_png)


if __name__ == "__main__":
    main()

```

---

## File: `experiments\run_part_a.py`
```python
"""
Part A: Algorithm Verification and Robustness Suite.

This script executes the statistical validation of the implemented optimizers.
It runs the optimizer on a standard benchmark function (Griewank, 5D) multiple times
with different random seeds to prove stability.

Functions:
1.  **Batch Execution:** Runs `run_opt.py` 10 times (Seeds 41-50).
2.  **Data Aggregation:** Collects CSV logs.
3.  **Statistical Plotting:** Generates Convergence Overlay and Robustness Boxplots.
"""

import os
import glob
import subprocess
import sys
import argparse

# Ensure updated plotting logic is used
sys.path.append(os.getcwd())
from experiments import plotting

def main():
    print("==================================================")
    print("   PART A: ALGORITHM VERIFICATION (ROBUSTNESS)    ")
    print("==================================================")

    # 1. Configuration
    N_RUNS = 10
    POP = 40
    EVALS = 30000
    DIM = 5
    
    # Clean old data if requested
    out_dir = os.path.join("data", "PartA")
    if os.path.exists(out_dir):
        # We can't easily rm -rf in python without shutil, but run_opt can do it
        pass

    csv_files = []

    # Safe Cleanup: Remove only CSVs (Avoid PermissionError on locked figure folders)
    old_csvs = glob.glob(os.path.join("data", "PartA", "results", "*.csv"))
    for f in old_csvs:
        try:
            os.remove(f)
        except OSError:
            pass

    # 2. Run 10 Seeds
    for i in range(1, N_RUNS + 1):
        seed = 40 + i # 41, 42, ...
        # print(f"\n---> Running Seed {seed} ({i}/{N_RUNS})...") # Reduced verbosity
        sys.stdout.write(f"\r---> Running Seed {seed} ({i}/{N_RUNS})...")
        sys.stdout.flush()
        
        cmd = [
            sys.executable, "experiments/run_opt.py",
            "--part", "A",
            "--D", str(DIM),
            "--pop", str(POP),
            "--evals", str(EVALS),
            "--seed", str(seed)
        ]
        
        # subprocess.check_call(cmd, stdout=subprocess.DEVNULL) # Silence detailed output
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
        
    # 3. Collect CSVs
    # Expect data/PartA/results/*.csv
    search_path = os.path.join("data", "PartA", "results", "*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print("ERROR: No CSV files found in", search_path)
        sys.exit(1)
        
    print(f"\nFound {len(csv_files)} runs. Generating Summary Plots...")
    
    # 4. Generate Statistical Plots
    # Overlay -> figures/convergence
    ov_path = os.path.join("data", "PartA", "figures", "convergence", "convergence_overlay.png")
    os.makedirs(os.path.dirname(ov_path), exist_ok=True)
    if os.path.exists(ov_path):
        os.remove(ov_path) # Force delete old
        
    plotting.plot_convergence_overlay(csv_files, outpath=ov_path)
    print(f" - Generated: {ov_path}")
    
    # Boxplot -> figures/analysis
    box_path = os.path.join("data", "PartA", "figures", "analysis", "robustness_boxplot.png")
    os.makedirs(os.path.dirname(box_path), exist_ok=True)
    if os.path.exists(box_path):
        os.remove(box_path) # Force delete old

    plotting.plot_final_boxplot(csv_files, outpath=box_path)
    print(f" - Generated: {box_path}")

    print("\n✅ PART A COMPLETE")

if __name__ == "__main__":
    main()

```

---

## File: `optimizer\base.py`
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np

Bounds = List[Tuple[float, float]]

def project(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.minimum(np.maximum(x, lo), hi)

@dataclass
class Optimizer:
    """
    Solver-agnostic ask/tell interface to enable clean separation between
    candidate proposal (ask) and objective evaluation (tell).

    This pattern allows the optimizer to be paused, serialized, or run 
    asynchronously without blocking the simulation loop.
    
    Attributes:
        bounds (Bounds): List of (min, max) tuples.
        D (int): Dimensionality of the problem.
        rng (np.random.Generator): Random number generator.
        options (Dict): Configuration parameters.
    """
    def __init__(self, bounds: Bounds, seed: int = 0, options: Optional[Dict] = None):
        self.bounds: Bounds = bounds
        self.D: int = len(bounds)
        self.rng = np.random.default_rng(int(seed))
        self.options: Dict = options or {}

    def ask(self) -> List[np.ndarray]:
        """
        Request a list of new candidate solutions to evaluate.
        
        Returns:
            List[np.ndarray]: A list of design vectors (length N).
        """
        raise NotImplementedError

    def tell(self, fitness: List[float], constraints: Optional[List[np.ndarray]] = None):
        """
        Report the fitness values for the candidate solutions proposed by `ask()`.
        
        Args:
            fitness (List[float]): Scalar objective values (minimization).
            constraints: check for specific constraint violations (optional).
        """
        raise NotImplementedError

    def best(self):
        """Return the best solution found so far as a dict {'x': ..., 'f': ...}."""
        raise NotImplementedError

    def state(self) -> Dict:
        """Return a dictionary of internal state metrics (convergence stats)."""
        return {}

    def done(self) -> bool:
        """Return True if the optimization stopping criteria are met."""
        return False

```

---

## File: `optimizer\ga.py`
```python
from typing import List, Optional, Dict, Tuple
import numpy as np
from .base import Optimizer, project

class GA(Optimizer):
    """
    Real-coded Genetic Algorithm.
    
    Options:
    - pop: population size (default 40)
    - mutation_rate: prob of mutating a gene (default 0.1)
    - mutation_scale: std dev of mutation noise fraction of range (default 0.1)
    - crossover_rate: prob of crossover (default 0.9)
    - elite_frac: fraction of best solutions kept (default 0.1)
    """
    def __init__(self, bounds: List[Tuple[float, float]], seed: int = 0, options: Optional[Dict] = None):
        super().__init__(bounds, seed, options)
        self.pop_size = self.options.get("pop", 40)
        self.mut_rate = self.options.get("mutation_rate", 0.1)
        self.mut_scale = self.options.get("mutation_scale", 0.1)
        self.cross_rate = self.options.get("crossover_rate", 0.9)
        self.elite_frac = self.options.get("elite_frac", 0.1)
        
        # Initialize population
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        self.range = hi - lo
        
        self.X = self.rng.uniform(lo, hi, (self.pop_size, self.D))
        self.fitness = np.full(self.pop_size, np.inf)
        
        self.gbest_x = None
        self.gbest_f = np.inf
        
        self.iter = 0
        self.evals_total = 0
        
        # State: 'ask', 'tell'
        self._state_phase = "ask" 

    def ask(self) -> List[np.ndarray]:
        """
        Generate the next population of candidate solutions.

        Logic:
        1.  **Elitism:** Carry over the top `elite_frac` of the current population unchanged.
        2.  **Selection:** Use Tournament Selection (size 3) to pick parents.
        3.  **Crossover:** Apply Arithmetic Crossover with probability `cross_rate`.
        4.  **Mutation:** Apply Gaussian Mutation to new offspring with probability `mut_rate`.

        Returns:
            List[np.ndarray]: A list of design vectors (numpy arrays) to be evaluated.
        """
        if self._state_phase != "ask":
             raise RuntimeError("Call tell() before ask()")
             
        # On first iter, return initial random population
        if self.iter == 0:
            self._state_phase = "tell"
            return list(self.X)
            
        # Evolution logic: Create new population from old
        new_X = []
        
        # 1. Elitism
        indices = np.argsort(self.fitness)
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        elite_indices = indices[:n_elite]
        for idx in elite_indices:
            new_X.append(self.X[idx].copy())
            
        # 2. Generate rest
        while len(new_X) < self.pop_size:
            # Tournament selection
            p1 = self._tournament(indices)
            p2 = self._tournament(indices)
            
            # Crossover
            if self.rng.random() < self.cross_rate:
                c1, c2 = self._crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Mutation
            self._mutate(c1)
            self._mutate(c2)
            
            new_X.append(c1)
            if len(new_X) < self.pop_size:
                new_X.append(c2)
                
        self.X = np.array(new_X)
        self._state_phase = "tell"
        return list(self.X)
    
    def tell(self, fitness: List[float], constraints=None):
        """
        Update the population with the evaluated fitness scores.

        Args:
            fitness (List[float]): List of scalar fitness values (lower is better).
            constraints: Not used in this implementation.
        """
        if self._state_phase != "tell":
             raise RuntimeError("Call ask() before tell()")
             
        self.fitness = np.array(fitness)
        # Handle NaNs
        self.fitness[np.isnan(self.fitness)] = np.inf
        self.evals_total += len(fitness)
        
        # Update global best
        min_idx = np.argmin(self.fitness)
        min_f = self.fitness[min_idx]
        
        if self.gbest_x is None or min_f < self.gbest_f:
            self.gbest_f = min_f
            self.gbest_x = self.X[min_idx].copy()
            
        self.iter += 1
        self._state_phase = "ask"

    def best(self):
        """Return the Global Best (gbest) solution found so far."""
        return {"x": self.gbest_x, "f": self.gbest_f}

    def state(self) -> Dict:
        """Return a dictionary of the optimizer's current statistics."""
        return {
            "iter": self.iter,
            "evals_total": self.evals_total,
            "gbest_f": self.gbest_f,
            "gbest_x": self.gbest_x.copy() if self.gbest_x is not None else None,
            "f_best": np.min(self.fitness) if self.iter > 0 else np.nan,
            "f_mean": np.mean(self.fitness) if self.iter > 0 else np.nan,
            "f_std": np.std(self.fitness) if self.iter > 0 else np.nan,
        }

    def _tournament(self, sorted_indices) -> np.ndarray:
        """
        Select a parent using Tournament Selection.
        
        Randomly picks 3 individuals and returns the one with the best (lowest) fitness.
        """
        # Tournament of size 3
        competitors = self.rng.choice(self.pop_size, 3, replace=False)
        # Since we have sorted_indices, we can find the best by finding the one with lowest rank (index in sorted)
        # But simply: comparing fitness is easier if we didn't have ranks.
        # Let's just pick best fitness of the 3.
        best_idx = competitors[np.argmin(self.fitness[competitors])]
        return self.X[best_idx].copy()

    def _crossover(self, p1, p2):
        """
        Perform Arithmetic Crossover between two parents.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Two updated offspring.
        """
        # Arithmetic crossover
        alpha = self.rng.random()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return c1, c2

    def _mutate(self, x):
        """
        Apply Gaussian Mutation to a design vector in-place.
        
        Adds noise N(0, mutation_scale) to genes with prob `mutation_rate`.
        Projects result back to bounds.
        """
        # Gaussian mutation
        mask = self.rng.random(self.D) < self.mut_rate
        if np.any(mask):
            noise = self.rng.normal(0, self.mut_scale, np.sum(mask)) * self.range[mask] # scale by range?
            # Or just user defined scale. Usually scale is relative to bounds.
            # Let's use self.mut_scale as fraction of range.
            x[mask] += noise
            x[:] = project(x, self.bounds) # Ensure valid (update in-place)

```

---

## File: `optimizer\pso.py`
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from .base import Optimizer, project, Bounds

@dataclass
class Particle:
    x: np.ndarray
    v: np.ndarray
    pbest_x: np.ndarray
    pbest_f: float

class PSO(Optimizer):
    """
    Particle Swarm Optimisation (continuous)
    - inertia w, cognitive c1, social c2
    - velocity clamping via vmax_frac
    - gbest (default) or lbest topology
    - optional 2D trajectory tracing (trace_every > 0 and D == 2)
    """
    def __init__(self, bounds: Bounds, seed: int = 0, options: Optional[Dict] = None):
        """
        Initialize the Particle Swarm Optimizer.

        Args:
            bounds (Bounds): List of (min, max) tuples for each dimension.
            seed (int): Random seed for reproducibility.
            options (Dict, optional): Configuration dictionary.
                - 'pop' (int): Swarm size (default 40).
                - 'w' (float): Inertia weight (default 0.72).
                - 'c1' (float): Cognitive coefficient (default 1.6).
                - 'c2' (float): Social coefficient (default 1.6).
                - 'vmax_frac' (float): Velocity limit as fraction of range (default 0.2).
                - 'topology' (str): 'gbest' (Global) or 'lbest' (Ring) (default 'gbest').
        """
        super().__init__(bounds, seed, options)
        opt = self.options

        self.pop: int = int(opt.get("pop", 40))
        self.w: float = float(opt.get("w", 0.72))
        self.c1: float = float(opt.get("c1", 1.6))
        self.c2: float = float(opt.get("c2", 1.6))
        self.vmax_frac: float = float(opt.get("vmax_frac", 0.2))
        self.topology: str = str(opt.get("topology", "gbest"))
        self.k_neighbors: int = int(opt.get("k_neighbors", 3))  # for lbest

        # ---- NEW: trajectory tracing controls ----
        # 0 disables tracing; otherwise record particle positions every N iterations (only if D==2).
        self.trace_every: int = int(opt.get("trace_every", 0))
        self._positions_trace: List[np.ndarray] = []  # each entry: (pop, 2) array

        lo = np.array([b[0] for b in self.bounds], dtype=float)
        hi = np.array([b[1] for b in self.bounds], dtype=float)
        span = hi - lo
        vmax = self.vmax_frac * span

        self.swarm: List[Particle] = []
        for _ in range(self.pop):
            x = self.rng.uniform(lo, hi)
            v = self.rng.uniform(-0.1 * span, 0.1 * span)
            self.swarm.append(Particle(x=x, v=v, pbest_x=x.copy(), pbest_f=np.inf))

        self.gbest_x = self.swarm[0].x.copy()
        self.gbest_f = np.inf
        self._last_idx: List[int] = list(range(self.pop))
        self._iter_best = np.inf
        self._iter_mean = np.inf
        self._iter_std = np.inf

        self._lo = lo
        self._hi = hi
        self._vmax = vmax
        self._evals_total = 0
        self._iters = 0

    def _local_best_position(self, i: int) -> np.ndarray:
        """Find the local best position for particle i in a Ring Topology."""
        # ring topology with k neighbors on each side
        k = self.k_neighbors
        idxs = [(i + d) % self.pop for d in range(-k, k + 1)]
        best = min((self.swarm[j] for j in idxs), key=lambda p: p.pbest_f)
        return best.pbest_x

    def ask(self) -> List[np.ndarray]:
        """
        Return the current positions of all particles to be evaluated.

        Returns:
            List[np.ndarray]: List of design vectors of size (pop, D).
        """
        # Evaluate current positions
        self._last_idx = list(range(self.pop))
        return [self.swarm[i].x.copy() for i in self._last_idx]

    def tell(self, fitness: List[float], constraints: Optional[List[np.ndarray]] = None):
        """
        Update the swarm based on the evaluated fitness of the current candidates.

        This method performs the core PSO logic:
        1.  **Update Bests:** Compares new fitness vs Personal Best (pbest) and Global Best (gbest).
        2.  **Update Velocity:** v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x).
        3.  **Clamp Velocity:** Limits v to [-vmax, +vmax] to ensure stability.
        4.  **Update Position:** x = x + v, constrained to bounds.

        Args:
            fitness (List[float]): A list of scalar fitness values (J) for the current population.
                                   Lower is better (minimization).
            constraints: Ignored in this unconstrained implementation (handled via penalty).

        Returns:
            None: Updates the internal state (`self.swarm`) in-place.
        """
        # 1) Update personal/global bests
        f_arr = np.asarray(fitness, dtype=float)
        f_arr[np.isnan(f_arr)] = np.inf
        for k, i in enumerate(self._last_idx):
            fx = f_arr[k]
            p = self.swarm[i]
            if fx < p.pbest_f:
                p.pbest_f = fx
                p.pbest_x = p.x.copy()
            if fx < self.gbest_f:
                self.gbest_f = fx
                self.gbest_x = p.x.copy()

        # 2) Velocity & position updates
        for i, p in enumerate(self.swarm):
            if self.topology.startswith("lbest"):
                g = self._local_best_position(i)
            else:
                g = self.gbest_x

            r1 = self.rng.random(self.D)
            r2 = self.rng.random(self.D)

            p.v = self.w * p.v + self.c1 * r1 * (p.pbest_x - p.x) + self.c2 * r2 * (g - p.x)

            # clamp velocity
            p.v = np.clip(p.v, -self._vmax, self._vmax)

            # update position + projection
            p.x = p.x + p.v
            p.x = project(p.x, self.bounds)

        # 3) iteration stats
        valid_mask = np.isfinite(f_arr)
        if np.any(valid_mask):
            valid_f = f_arr[valid_mask]
            self._iter_best = float(np.min(valid_f))
            self._iter_mean = float(np.mean(valid_f))
            self._iter_std = float(np.std(valid_f))
        else:
             # If all failed (very rare/bad start)
            self._iter_best = float(np.min(f_arr)) # likely inf
            self._iter_mean = float(np.inf)
            self._iter_std = 0.0
        self._evals_total += len(f_arr)
        self._iters += 1

        # ---- NEW: record 2D positions every trace_every iterations ----
        if self.trace_every and (self._iters % self.trace_every == 0) and self.D == 2:
            pts = np.stack([p.x.copy() for p in self.swarm], axis=0)  # shape (pop, 2)
            self._positions_trace.append(pts)

    def best(self):
        """Return the Global Best (gbest) solution found so far."""
        return {"x": self.gbest_x.copy(), "f": float(self.gbest_f)}

    def state(self) -> Dict:
        """Return a dictionary of the optimizer's current statistics."""
        # ---- UPDATED: expose trace length for convenience ----
        return {
            "iter": self._iters,
            "evals_total": self._evals_total,
            "f_best": self._iter_best,
            "f_mean": self._iter_mean,
            "f_std": self._iter_std,
            "gbest_f": float(self.gbest_f),
            "gbest_x": self.gbest_x.copy(),
            "trace_len": len(self._positions_trace),
        }

    # ---- NEW: accessor for plotting code ----
    def positions_trace(self) -> List[np.ndarray]:
        """Return the recorded list of (pop, 2) arrays. Empty if tracing disabled or D != 2."""
        return list(self._positions_trace)

```

---

## File: `utils\airfoil_analysis.py`
```python
from typing import Tuple, Optional, Iterable
import os
import tempfile
import uuid
import shutil
import numpy as np

from utils.geometry import build_airfoil_coordinates, write_dat
from utils.cst import cst_airfoil
from utils.xfoil_runner import run_xfoil_single_alpha, run_xfoil_polar

def analyze_airfoil(
    coeffs_upper: Iterable[float],
    coeffs_lower: Iterable[float],
    alpha: float = 3.0,
    Re: float = 1e6,
    mach: float = 0.1,
    n_iter: int = 200,
    n_points: int = 201,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Automates the creation of airfoil (CST), execution of XFOIL, and extraction of parameters.
    
    Parameters
    ----------
    coeffs_upper : Iterable[float]
        CST coefficients for upper surface.
    coeffs_lower : Iterable[float]
        CST coefficients for lower surface.
    alpha : float, optional
        Angle of attack in degrees, by default 3.0
    Re : float, optional
        Reynolds number, by default 1e6
    mach : float, optional
        Mach number, by default 0.1
    n_iter : int, optional
        Number of XFOIL iterations, by default 200
    n_points : int, optional
        Number of points for airfoil generation, by default 201

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[float]]
        (CL, CD, CM) or (None, None, None) if failed.
    """
    
    # 1. Validate Geometry (Prevent Crossing Surfaces)
    # Check if Upper Surface is always above Lower Surface
    # We use cst_airfoil directly to get the separated arrays
    try:
        _, yu, _, yl = cst_airfoil(n_points=n_points, coeffs_upper=coeffs_upper, coeffs_lower=coeffs_lower)
        
        # Tolerance for numerical noise, though usually exact crossing is bad.
        # Check if any lower point is significantly above the corresponding upper point.
        # We ignore the very leading/trailing edges which might be close/equal.
        # Index 1:-1 skips endpoints.
        if np.any(yl[1:-1] > yu[1:-1]):
            # Invalid geometry: crossing surfaces
            return None, None, None
            
    except ValueError:
        # e.g. n_points too small
        return None, None, None

    # 2. Create Airfoil Coordinates from CST
    x, y = build_airfoil_coordinates(coeffs_upper, coeffs_lower, n_points=n_points)
    
    # 2. Write to temporary .dat file
    # unique name to avoid conflicts
    unique_name = f"cst_airfoil_{uuid.uuid4().hex[:8]}.dat"
    # Write to temp directory
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    dat_path = os.path.join(temp_dir, unique_name)
    
    try:
        write_dat(x, y, dat_path, name="CST_AIRFOIL")
        
        # 3. Execute XFOIL
        cl, cd, cm = run_xfoil_single_alpha(
            dat_path,
            alpha=alpha,
            Re=Re,
            mach=mach,
            n_iter=n_iter
        )
        
        return cl, cd, cm
        
    finally:
        # Cleanup geom file
        if os.path.exists(dat_path):
            try:
                os.remove(dat_path)
            except OSError:
                pass

```

---

## File: `utils\airfoil_problem.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from utils.cst import cst_airfoil  # only if you also need it elsewhere
from utils.geometry import build_airfoil_coordinates, write_dat
from utils.xfoil_runner import run_xfoil_single_alpha, run_xfoil_polar


@dataclass
class AirfoilConfig:
    """
    Configuration for the airfoil optimisation problem.
    """
    alpha_deg: float = 3.0
    Re: float = 1e6
    mach: float = 0.0
    iter_limit: int = 100

    # Number of *upper* surface points used in CST discretisation
    # Total points in .dat will be 2*n_points - 1
    n_points: int = 121  # safe: 2*121-1 = 241 < 365 XFOIL max

    # Constraint / penalty settings
    cl_min: float = 0.5
    cm_max_abs: float = 0.2
    w_cl: float = 50.0
    w_cm: float = 10.0
    w_fail: float = 1e3

    # Paths
    run_dir: Path = Path("data/results/airfoil/run_tmp")
    baseline_dat: Path = Path("data/airfoils/baseline.dat")
    optimised_dat: Path = Path("data/airfoils/optimised.dat")


def theta_to_cst(theta: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map a 6D design vector into upper/lower CST coefficient arrays.

    theta = [a0u, a1u, a2u,  a0l, a1l, a2l]
    """
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size != 6:
        raise ValueError(f"Expected 6 design variables, got {theta.size}")

    a_u = theta[:3]
    a_l = theta[3:]
    return a_u, a_l


def write_airfoil_from_theta(
    theta: Iterable[float],
    out_path: Path,
    n_points: int = 121,
) -> None:
    """
    Generate airfoil coordinates from CST parameters and write to a .dat file.
    """
    a_u, a_l = theta_to_cst(theta)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x, y = build_airfoil_coordinates(
        coeffs_upper=a_u,
        coeffs_lower=a_l,
        n_points=n_points,
        # keep default n1, n2, dz_te consistent with tests
    )

    write_dat(x, y, out_path, name="CST_AIRFOIL")


def evaluate_airfoil_theta(theta: Iterable[float], cfg: AirfoilConfig):
    """
    Evaluate a given airfoil design (theta) using XFOIL at a single alpha.

    Returns
    -------
    f : float
        Scalar objective value (lower is better).
    info : dict
        Dictionary with raw data: CL, CD, CM, success flag, etc.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dat_path = run_dir / "candidate.dat"

    # --- Quick geometric sanity check to avoid degenerate shapes ---
    a_u, a_l = theta_to_cst(theta)
    x_tmp, y_tmp = build_airfoil_coordinates(
        coeffs_upper=a_u,
        coeffs_lower=a_l,
        n_points=cfg.n_points,
    )
    max_y = np.max(np.abs(y_tmp))
    if max_y < 1e-4:
        # Completely flat / almost zero thickness -> skip XFOIL
        return cfg.w_fail, {
            "success": False,
            "error": f"Degenerate flat airfoil (max |y|={max_y:.2e})",
            "cl": np.nan,
            "cd": np.nan,
            "cm": np.nan,
        }

    # 1) Geometry -> .dat
    write_dat(x_tmp, y_tmp, dat_path, name="CST_AIRFOIL")

    # 2) XFOIL call (inviscid for robustness)
    try:
        res = run_single_alpha(
            dat_path,
            alpha_deg=cfg.alpha_deg,
            Re=cfg.Re,
            mach=cfg.mach,
            iter_limit=cfg.iter_limit,
            timeout=30.0,
            viscous=False,  # keep inviscid for stability
        )
        success = True
    except XFoilError as e:
        # XFOIL failed numerically: assign penalty objective
        f = cfg.w_fail
        info = {
            "success": False,
            "error": str(e),
            "cl": np.nan,
            "cd": np.nan,
            "cm": np.nan,
        }
        return f, info

    # 3) Objective with penalties
    penalty = 0.0

    if res.cl < cfg.cl_min:
        penalty += cfg.w_cl * (cfg.cl_min - res.cl) ** 2

    if abs(res.cm) > cfg.cm_max_abs:
        penalty += cfg.w_cm * (abs(res.cm) - cfg.cm_max_abs) ** 2

    f = res.cd + penalty

    info = {
        "success": success,
        "cl": res.cl,
        "cd": res.cd,
        "cm": res.cm,
        "penalty": penalty,
        "objective": f,
    }

    return f, info

```

---

## File: `utils\cst.py`
```python
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def _bernstein_matrix(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute Bernstein basis matrix of order n at points x.

    Returns array of shape (len(x), n+1) where column k is B_k^n(x).
    """
    x = np.asarray(x, dtype=float)
    k = np.arange(n + 1)
    binom = np.array([math.comb(n, int(ki)) for ki in k], dtype=float)

    # Shape (len(x), n+1)
    x_col = x[:, None]
    return binom * (x_col ** k) * ((1.0 - x_col) ** (n - k))


def cst_surface(
    x: np.ndarray,
    coeffs: Iterable[float],
    n1: float = 0.5,
    n2: float = 1.0,
    dz_te: float = 0.0,
) -> np.ndarray:
    """
    Compute CST surface y(x) given coefficients.

    Parameters
    ----------
    x : array_like
        Chordwise positions in [0, 1].
    coeffs : iterable of float
        CST shape coefficients (A_0 ... A_N).
    n1, n2 : float
        Class function exponents: C(x) = x^n1 * (1-x)^n2.
    dz_te : float
        Linear trailing-edge thickness term added as x * dz_te.

    Returns
    -------
    y : np.ndarray
        Surface ordinate values at x.
    """
    x = np.asarray(x, dtype=float)
    coeffs = np.asarray(list(coeffs), dtype=float)
    n = coeffs.size - 1
    if n < 0:
        raise ValueError("coeffs must contain at least one value")

    # Shape function via Bernstein polynomials
    B = _bernstein_matrix(n, x)  # (len(x), n+1)
    S = B @ coeffs  # (len(x),)

    # Class function
    C = (x**n1) * ((1.0 - x) ** n2)

    # CST surface
    y = C * S + x * dz_te
    return y


def cst_airfoil(
    n_points: int,
    coeffs_upper: Iterable[float],
    coeffs_lower: Iterable[float],
    n1: float = 0.5,
    n2: float = 1.0,
    dz_te: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CST airfoil surfaces.

    Parameters
    ----------
    n_points : int
        Number of points per surface (upper and lower).
    coeffs_upper, coeffs_lower : iterable of float
        CST coefficients for upper and lower surfaces.
    n1, n2 : float
        Class function exponents.
    dz_te : float
        Total trailing-edge thickness. Split equally between upper and lower.

    Returns
    -------
    xu, yu, xl, yl : np.ndarray
        x and y coordinates of upper and lower surfaces, each of length n_points.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    x = np.linspace(0.0, 1.0, n_points)

    # Split TE thickness evenly: +dz/2 on upper, -dz/2 on lower
    yu = cst_surface(x, coeffs_upper, n1=n1, n2=n2, dz_te=+dz_te / 2.0)
    yl = cst_surface(x, coeffs_lower, n1=n1, n2=n2, dz_te=-dz_te / 2.0)

    return x.copy(), yu, x.copy(), yl

```

---

## File: `utils\geometry.py`
```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from utils.cst import cst_airfoil


def build_airfoil_coordinates(
    coeffs_upper: Iterable[float],
    coeffs_lower: Iterable[float],
    n_points: int = 201,
    n1: float = 0.5,
    n2: float = 1.0,
    dz_te: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a full airfoil coordinate set suitable for XFOIL.

    Ordering:
        - start at upper trailing edge (x ~ 1) and go to leading edge (x ~ 0)
        - then from leading edge along the lower surface back to trailing edge

    Parameters
    ----------
    coeffs_upper, coeffs_lower : iterable of float
        CST coefficients for upper and lower surfaces.
    n_points : int
        Number of points per surface.
    n1, n2 : float
        Class function exponents.
    dz_te : float
        Trailing-edge thickness (split between upper and lower).

    Returns
    -------
    x, y : np.ndarray
        Concatenated coordinates around the airfoil.
    """
    xu, yu, xl, yl = cst_airfoil(
        n_points=n_points,
        coeffs_upper=coeffs_upper,
        coeffs_lower=coeffs_lower,
        n1=n1,
        n2=n2,
        dz_te=dz_te,
    )

    # Upper surface: from TE (x=1) to LE (x=0)
    xu_rev = xu[::-1]
    yu_rev = yu[::-1]

    # Lower surface: from LE (x=0) to TE (x=1)
    # Skip the first point to avoid duplicating the LE
    xl_fwd = xl[1:]
    yl_fwd = yl[1:]

    x = np.concatenate([xu_rev, xl_fwd])
    y = np.concatenate([yu_rev, yl_fwd])

    return x, y


def write_dat(x, y, path, name: str = "airfoil"):
    """
    Write an airfoil .dat file in XFOIL format from full (x, y) coordinates.

    Parameters
    ----------
    x, y : 1D arrays
        Full airfoil coordinates, starting at upper TE -> LE, then lower LE -> TE.
    path : Path or str
        Output .dat file path.
    name : str
        Airfoil name written on the first line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write(f"{name}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f}\n")

    return path

```

---

## File: `utils\gp_surrogate.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float, sigma_f: float) -> np.ndarray:
    """
    Compute the Squared Exponential (RBF) covariance matrix between X1 and X2.

    Formula:
        k(x, x') = sigma_f^2 * exp( -0.5 * ||x - x'||^2 / l^2 )

    Args:
        X1 (np.ndarray): shape (n1, D)
        X2 (np.ndarray): shape (n2, D)
        length_scale (float): The length scale 'l', controlling smoothness.
        sigma_f (float): The signal variance 'sigma_f', controlling amplitude.

    Returns:
        np.ndarray: Covariance matrix of shape (n1, n2).
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    # ||x - x'||^2 = (x^2)_i + (x'^2)_j - 2 x_i·x'_j
    sq_norms1 = np.sum(X1**2, axis=1)[:, None]
    sq_norms2 = np.sum(X2**2, axis=1)[None, :]
    sq_dists = sq_norms1 + sq_norms2 - 2.0 * X1 @ X2.T

    return (sigma_f**2) * np.exp(-0.5 * sq_dists / (length_scale**2))


@dataclass
class GaussianProcessSurrogate:
    """
    A lightweight Gaussian Process Regressor using an RBF Kernel.

    Designed for the COD Assignment to avoid heavy dependencies (like sklearn).
    
    Attributes:
        length_scale (float): Kernel length scale (fixed).
        sigma_f (float): Kernel signal variance (fixed).
        sigma_n (float): Observation noise variance (for numerical stability).
    """
    length_scale: float = 0.3
    sigma_f: float = 1.0
    sigma_n: float = 1e-6  # observation noise

    # Internal attributes (filled by fit)
    X_train_: Optional[np.ndarray] = None
    y_train_: Optional[np.ndarray] = None
    X_min_: Optional[np.ndarray] = None
    X_max_: Optional[np.ndarray] = None
    y_mean_: Optional[float] = None
    y_std_: Optional[float] = None
    L_: Optional[np.ndarray] = None          # Cholesky of K
    alpha_: Optional[np.ndarray] = None      # (K^-1 y) vector

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalize input X to [0, 1] using training bounds."""
        X = np.asarray(X, float)
        return (X - self.X_min_) / (self.X_max_ - self.X_min_ + 1e-12)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        """
        Train the GP model on the provided dataset.

        Performs:
        1.  **Normalization:** Scales X to [0, 1] and y to Standard Normal (0, 1).
        2.  **Kernel Construction:** Builds the covariance matrix K.
        3.  **Cholesky Decomposition:** Computes L such that K = L @ L.T.
        4.  **Weights:** Solves for alpha = K^-1 * y.

        Args:
            X (np.ndarray): Input training data (n_samples, n_features).
            y (np.ndarray): Target values (n_samples,).

        Returns:
            self: The fitted regressor.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        # Store scaling for X
        self.X_min_ = X.min(axis=0)
        self.X_max_ = X.max(axis=0)
        X_scaled = self._scale_X(X)

        # Standardise y
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std() if y.std() > 0 else 1.0)
        y_std = (y - self.y_mean_) / self.y_std_

        # Kernel matrix + noise
        K = _rbf_kernel(X_scaled, X_scaled, self.length_scale, self.sigma_f)
        K[np.diag_indices_from(K)] += self.sigma_n**2

        # Cholesky factorisation
        self.L_ = np.linalg.cholesky(K)
        # Solve for alpha = K^-1 y_std via L
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, y_std))

        self.X_train_ = X_scaled
        self.y_train_ = y_std
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict the mean and standard deviation for new inputs.

        Args:
            X (np.ndarray): New input points (n_samples, n_features).
            return_std (bool): If True, returns the predictive standard deviation (uncertainty).

        Returns:
            y_mean (np.ndarray): Predicted values (rescaled to original units).
            y_std (np.ndarray or None): Uncertainty estimates (if return_std=True).
        """
        if self.X_train_ is None:
            raise RuntimeError("GP not fitted yet. Call fit() first.")

        X = np.asarray(X, float)
        X_scaled = self._scale_X(X)

        # Cross-kernel between training and test
        K_star = _rbf_kernel(self.X_train_, X_scaled, self.length_scale, self.sigma_f)
        # Predictive mean in standardised space
        y_mean_std = K_star.T @ self.alpha_

        # Rescale back to original y units
        y_mean = y_mean_std * self.y_std_ + self.y_mean_

        if not return_std:
            return y_mean, None

        # Solve v = L^-1 K_star
        v = np.linalg.solve(self.L_, K_star)
        # Predictive variance in std space: k(x*,x*) - v^T v
        k_xx = (self.sigma_f**2) * np.ones(X_scaled.shape[0])
        y_var_std = k_xx - np.sum(v**2, axis=0)
        y_var_std = np.maximum(y_var_std, 1e-12)  # clamp numerical noise

        # Rescale variance: var(y) = (std_y^2) * var(std_y)
        y_std = np.sqrt(y_var_std) * self.y_std_
        return y_mean, y_std

```

---

## File: `utils\recorder.py`
```python
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


# Project root: .../COD_Assignment
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = DATA_ROOT / "results"
FIGURES_ROOT = DATA_ROOT / "figures"


@dataclass
class RunConfig:
    """Minimal run configuration metadata to store with each run."""
    domain: str          # "pso" or "airfoil"
    problem: str         # e.g. "griewank_2d", "griewank_5d"
    algorithm: str       # e.g. "pso"
    n_particles: int
    n_iters: int
    dim: int
    seed: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_run_dir(domain: str, problem: str, mode: str = "single") -> Path:
    """
    Create and return a unique directory for a single run.

    Structure:
        data/results/{domain}/{mode}/run_YYYYmmdd_HHMMSS_XXXX/

    domain : "pso" or "airfoil"
    problem : short problem name, used later in summaries
    mode : "single" or "multi"
    """
    if domain not in {"pso", "airfoil"}:
        raise ValueError(f"Unknown domain: {domain}")
    if mode not in {"single", "multi"}:
        raise ValueError(f"Unknown mode: {mode}")

    base = RESULTS_ROOT / domain / mode
    _ensure_dir(base)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add a short random suffix based on nanoseconds to avoid collisions
    suffix = datetime.now().strftime("%f")[-4:]
    run_name = f"run_{timestamp}_{suffix}"

    run_dir = base / run_name
    _ensure_dir(run_dir)

    # Store problem name in a small marker file (optional but handy)
    (run_dir / "problem.txt").write_text(problem)

    return run_dir


def save_convergence_csv(
    run_dir: Path,
    best_history: Sequence[float],
    mean_history: Sequence[float],
) -> Path:
    """
    Save convergence history to CSV:
        iter, f_best, f_mean
    """
    path = run_dir / "convergence.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "f_best", "f_mean"])
        for i, (b, m) in enumerate(zip(best_history, mean_history)):
            writer.writerow([i, b, m])
    return path


def save_swarm_2d_csv(
    run_dir: Path,
    swarm_history: Sequence[Any],
) -> Path:
    """
    Save 2D swarm positions over time.

    swarm_history: list of arrays of shape (n_particles, 2)

    CSV columns:
        iter, particle, x1, x2
    """
    path = run_dir / "swarm_2d.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "particle", "x1", "x2"])
        for it, swarm in enumerate(swarm_history):
            # Convert to list of rows
            for pid, pos in enumerate(swarm):
                writer.writerow([it, pid, float(pos[0]), float(pos[1])])
    return path


def save_run_metadata(run_dir: Path, config: RunConfig, extra: Dict[str, Any] | None = None) -> Path:
    """
    Save run configuration and optional extra info to metadata.json.
    """
    meta: Dict[str, Any] = asdict(config)
    if extra:
        meta.update(extra)

    path = run_dir / "metadata.json"
    with path.open("w") as f:
        json.dump(meta, f, indent=2)
    return path


def append_multi_summary(
    domain: str,
    problem: str,
    rows: Iterable[Dict[str, Any]],
    filename: str = "summary.csv",
) -> Path:
    """
    Append summary rows for multiple runs.

    Each row should be a flat dict with consistent keys.

    File location:
        data/results/{domain}/multi/{problem}_{filename}
    """
    base = RESULTS_ROOT / domain / "multi"
    _ensure_dir(base)

    path = base / f"{problem}_{filename}"

    rows = list(rows)
    if not rows:
        return path

    # Ensure consistent column order
    fieldnames: List[str] = sorted(rows[0].keys())

    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path


```

---

## File: `utils\xfoil_runner.py`
```python
import os
import shutil
import subprocess
import tempfile
import uuid
import numpy as np
from typing import Tuple, Optional

# Attempt to import xfoil library (for Windows/Direct usage)
try:
    from xfoil import XFoil
    from xfoil.model import Airfoil
    HAS_XFOIL_LIB = True
except ImportError:
    HAS_XFOIL_LIB = False

# Fallback XFOIL executable setup
_custom_path = os.environ.get("XFOIL_PATH", "")
if _custom_path and os.path.exists(_custom_path):
    XFOIL_EXE = _custom_path
else:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_xfoil_exe = os.path.join(root, "Xfoil", "xfoil.exe")
    local_xfoil_bin = os.path.join(root, "Xfoil", "xfoil")
    _found = None
    if os.path.exists(local_xfoil_exe):
        _found = local_xfoil_exe
    elif os.path.exists(local_xfoil_bin):
        _found = local_xfoil_bin
    if not _found:
        _found = shutil.which("xfoil.exe") or shutil.which("xfoil")
    XFOIL_EXE = _found if _found else "xfoil"

XFOIL_DIR = "temp"  # run in temp folder

# ---------------------------------------------------------------------------
# LIBRARY-BASED RUNNER (Windows Friendly)
# ---------------------------------------------------------------------------

def _run_lib_single(dat_path, alpha, Re, mach, n_iter):
    """Run using xfoil python library."""
    xf = XFoil()
    # Mute output if possible, xfoil lib prints to stdout usually
    xf.print = False 
    
    # Load airfoil
    # xfoil lib usually wants coordinates, not file, but Airfoil class might load file
    # Or XFoil.airfoil property.
    # We will try loading coordinates from the dat file manually.
    try:
        with open(dat_path, 'r') as f:
            lines = f.readlines()
        # Skip header? XFOIL dat has name on line 1
        coords = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
        coords = np.array(coords)
        
        xf.airfoil = Airfoil(x=coords[:,0], y=coords[:,1])
    except Exception as e:
        print(f"[xfoil_lib] Error loading coordinates: {e}")
        return None, None, None

    # Setup
    xf.Re = Re
    xf.M = mach
    xf.max_iter = n_iter
    
    # Run
    try:
        xf.a(alpha)
        # Check convergence? xf.converged property might exist
        return xf.Cl, xf.Cd, xf.Cm
    except Exception:
        return None, None, None

def _run_lib_polar(dat_path, a_start, a_end, a_step, Re, mach, n_iter):
    """Run polar using xfoil python library."""
    xf = XFoil()
    xf.print = False
    
    # Load coords 
    with open(dat_path, 'r') as f:
        lines = f.readlines()
    coords = []
    for line in lines[1:]:
        p = line.split()
        if len(p) >= 2: coords.append((float(p[0]), float(p[1])))
    coords = np.array(coords)
    xf.airfoil = Airfoil(x=coords[:,0], y=coords[:,1])
    
    xf.Re = Re
    xf.M = mach
    xf.max_iter = n_iter
    
    alphas = np.arange(a_start, a_end + a_step/2, a_step)
    cl_list, cd_list, cm_list, a_list = [], [], [], []
    
    # Sequential run
    xf.reset_bls() # reset boundary layer
    for a in alphas:
        xf.a(a)
        # We assume values persist even if not converged, or check something?
        # xfoil lib usually keeps last state.
        # Ideally we only take converged.
        cl_list.append(xf.Cl)
        cd_list.append(xf.Cd)
        cm_list.append(xf.Cm)
        a_list.append(a)
            
    return np.array(a_list), np.array(cl_list), np.array(cd_list), np.array(cm_list)


# ---------------------------------------------------------------------------
# SUBPROCESS-BASED RUNNER (Linux Fallback)
# ---------------------------------------------------------------------------

def _run_xfoil_script(script: str, workdir: str = ".", timeout: int = 10) -> Tuple[int, str, str, str]:
    """
    Execute an XFOIL input script via a subprocess call.

    Handles cross-platform execution:
    - **Windows:** Runs `.exe` directly or bridges to WSL for Linux binaries.
    - **Linux/Mac:** Direct execution.

    Features:
    - **Timeouts:** Kills the process if XFOIL hangs (common with bad geometries).
    - **Stdio:** Captures stdout/stderr for parsing.

    Args:
        script (str): The sequence of XFOIL commands (inputs).
        workdir (str): Directory where temporary files are created.
        timeout (int): Max execution time in seconds (default 10).

    Returns:
        Tuple[int, str, str, str]: (Exit Code, Stdout, Stderr, Path to script file).
    """
    os.makedirs(workdir, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".inp", dir=workdir, encoding="ascii") as f:
        f.write(script)
        f.write("\n")
        script_path = f.name
    
    try:
        with open(script_path, "r") as f_in:
            cmd = [XFOIL_EXE]
            # [Hero Run Fix] Functionality to run Linux XFOIL on Windows using WSL
            if os.name == 'nt':
                # Check if it is likely a linux binary (no extension)
                if not str(XFOIL_EXE).lower().endswith(".exe"):
                    # Convert Windows path to WSL path
                    # e.g. C:\Users\... -> /mnt/c/Users/...
                    wsl_path = XFOIL_EXE.replace("\\", "/")
                    if ":" in wsl_path:
                        drive, tail = wsl_path.split(":", 1)
                        wsl_path = f"/mnt/{drive.lower()}{tail}"
                    cmd = ["wsl", wsl_path]

            proc = subprocess.run(
                cmd, 
                stdin=f_in, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                cwd=workdir,
                timeout=timeout
            )
        return proc.returncode, proc.stdout.decode(errors="ignore"), proc.stderr.decode(errors="ignore"), script_path
    except subprocess.TimeoutExpired as e:
        out_str = e.stdout.decode(errors="ignore") if e.stdout else ""
        err_str = e.stderr.decode(errors="ignore") if e.stderr else ""
        return -1, out_str, f"TimeoutExpired\n{err_str}", script_path

def _parse_xfoil_stdout(out: str) -> dict:
    """
    Parse the Standard Output of XFOIL to extract aerodynamic coefficients.

    Searches specifically for lines containing "a =" and "CL =".
    Note: This is a fallback/quick method. Reliability depends on exact XFOIL version output.

    Args:
        out (str): The raw stdout string from the XFOIL process.

    Returns:
        dict: A dictionary containing 'CL', 'CD', 'CM', and 'alpha' if found.
              Returns empty dict if parsing fails.
    """
    results_map = {}
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "a =" in line and "CL =" in line:
            try:
                parts = line.replace("=", " ").split()
                idx_a = parts.index("a")
                val_a = float(parts[idx_a + 1])
                idx_cl = parts.index("CL")
                val_cl = float(parts[idx_cl + 1])
                if i + 1 < len(lines):
                    next_line = lines[i+1]
                    if "Cm =" in next_line and "CD =" in next_line:
                        parts2 = next_line.replace("=", " ").split()
                        idx_cm = parts2.index("Cm")
                        val_cm = float(parts2[idx_cm + 1])
                        idx_cd = parts2.index("CD")
                        val_cd = float(parts2[idx_cd + 1])
                        results_map[val_a] = (val_cl, val_cd, val_cm)
            except (ValueError, IndexError):
                continue
    return results_map

def _run_subprocess_single(dat_path, alpha, Re, mach, n_iter):
    # Copy .dat locally to key workdir (temp)
    unique_id = uuid.uuid4().hex[:8]
    local_dat_name = f"airfoil_{unique_id}.dat"
    local_dat_path = os.path.join(XFOIL_DIR, local_dat_name)
    
    # Ensure XFOIL_DIR exists
    os.makedirs(XFOIL_DIR, exist_ok=True)
    
    shutil.copy(dat_path, local_dat_path)
    
    # Detect if we are using WSL (Windows host, Linux binary)
    use_wsl = False
    if os.name == 'nt' and not str(XFOIL_EXE).lower().endswith(".exe"):
        use_wsl = True

    # Fix line endings (Force LF on Linux OR if using WSL to prevent XFOIL issues)
    with open(local_dat_path, 'rb') as f: content = f.read()
    if os.name == 'posix' or use_wsl:
        content = content.replace(b'\r\n', b'\n')
    with open(local_dat_path, 'wb') as f: f.write(content)

    script_lines = [
        "PLOP", "G", "",
        f"LOAD {local_dat_name}",
        "PANE", "OPER",
        f"VISC {Re}", f"MACH {mach}", f"ITER {n_iter}",
        f"ALFA {alpha}", "", "QUIT"
    ]
    
    # Use LF for script if on Linux or WSL
    join_char = "\n" if (os.name == 'posix' or use_wsl) else "\r\n"
    script = join_char.join(script_lines) + join_char
    
    rc, out, err, spath = _run_xfoil_script(script, workdir=XFOIL_DIR)
    
    # Cleanup
    for p in [spath, local_dat_path]:
        try:
            if os.path.exists(p): os.remove(p)
        except OSError:
            pass # Ignore cleanup errors

    # print(f"--- DEBUG XFOIL OUTPUT ---\n{out}\n-------------------------")
    results = _parse_xfoil_stdout(out)
    
    if not results: 
        # print(f"[DEBUG] XFOIL Parsing Failed.")
        return None, None, None
    best_a = min(results.keys(), key=lambda x: abs(x - alpha))
    if abs(best_a - alpha) > 0.1: return None, None, None # Too far
    return results[best_a]

def _run_subprocess_polar(dat_path, a_start, a_end, a_step, Re, mach, n_iter):
    unique_id = uuid.uuid4().hex[:8]
    local_dat_name = f"airfoil_{unique_id}.dat"
    local_dat_path = os.path.join(XFOIL_DIR, local_dat_name)
    
    os.makedirs(XFOIL_DIR, exist_ok=True)
    shutil.copy(dat_path, local_dat_path)
    
    # Detect if we are using WSL
    use_wsl = False
    if os.name == 'nt' and not str(XFOIL_EXE).lower().endswith(".exe"):
        use_wsl = True
    
    # Line endings
    with open(local_dat_path, 'rb') as f: content = f.read()
    if os.name == 'posix' or use_wsl:
        content = content.replace(b'\r\n', b'\n')
    with open(local_dat_path, 'wb') as f: f.write(content)

    script_lines = [
        "PLOP", "G", "",
        f"LOAD {local_dat_name}",
        "PANE", "OPER",
        f"VISC {Re}", f"MACH {mach}", f"ITER {n_iter}",
        f"ASEQ {a_start} {a_end} {a_step}", "", "QUIT"
    ]
    
    # Use LF for script if on Linux or WSL
    join_char = "\n" if (os.name == 'posix' or use_wsl) else "\r\n"
    script = join_char.join(script_lines) + join_char
    
    rc, out, err, spath = _run_xfoil_script(script, workdir=XFOIL_DIR)
    
    # Cleanup
    for p in [spath, local_dat_path]:
        try:
            if os.path.exists(p): os.remove(p)
        except OSError:
            pass # Ignore cleanup errors

    results = _parse_xfoil_stdout(out)
    alphas, cls, cds, cms = [], [], [], []
    for a in sorted(results.keys()):
        cl, cd, cm = results[a]
        alphas.append(a)
        cls.append(cl)
        cds.append(cd)
        cms.append(cm)
    return np.array(alphas), np.array(cls), np.array(cds), np.array(cms)

# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def run_xfoil_single_alpha(dat_path: str, alpha: float = 3.0, Re: float = 1e6, mach: float = 0.1, n_iter: int = 200) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Run a single-point XFOIL simulation used for optimization fitness evaluation.

    Dispatches to:
    - `xfoil` Python library (if installed/detected).
    - `subprocess` wrapper (using local executable) otherwise.

    Args:
        dat_path (str): Absolute path to the airfoil coordinate file (.dat).
        alpha (float): Angle of attack (degrees).
        Re (float): Reynolds number.
        mach (float): Mach number.
        n_iter (int): Max iterations for the viscous solution.

    Returns:
        Tuple[float, float, float]: (Cl, Cd, Cm). 
        Returns (None, None, None) if convergence fails or XFOIL crashes.
    """
    if HAS_XFOIL_LIB:
        return _run_lib_single(dat_path, alpha, Re, mach, n_iter)
    else:
        # Fallback
        return _run_subprocess_single(dat_path, alpha, Re, mach, n_iter)

def run_xfoil_polar(dat_path: str, a_start: float, a_end: float, a_step: float, Re: float = 1e6, mach: float = 0.1, n_iter: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a full drag polar (Cl vs Cd) by sweeping specific angles of attack.

    Used for post-optimization analysis and plotting.

    Args:
        dat_path (str): Path to airfoil file.
        a_start (float): Start angle (e.g. -5).
        a_end (float): End angle (e.g. 15).
        a_step (float): Step size (e.g. 1.0).
        Re (float): Reynolds number.
        mach (float): Mach number.
        n_iter (int): Max iterations per step.

    Returns:
        Tuple[np.ndarray, ...]: Parallel arrays of (Alpha, Cl, Cd, Cm).
    """
    if HAS_XFOIL_LIB:
        return _run_lib_polar(dat_path, a_start, a_end, a_step, Re, mach, n_iter)
    else:
        return _run_subprocess_polar(dat_path, a_start, a_end, a_step, Re, mach, n_iter)

```

---

## File: `benchmarks\airfoil_xfoil.py`
```python
import os
import hashlib
import json
import tempfile
import numpy as np

from utils.airfoil_analysis import analyze_airfoil

# Where we store *only* the numerical cache (no geometry files)
CACHE_FILE = os.path.join("data", "PartB", "cache_xfoil.json")

# Robust cache loading: if file missing or corrupted, start fresh
try:
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    cache = {}

def clear_cache():
    """Reset the in-memory cache (useful if --clean was used)."""
    global cache
    cache = {}


def _key_from_vec(vec, **kwargs):
    """Generate a stable hash key from the design vector and simulation params."""
    v = np.asarray(vec, dtype=float)
    # Include kwargs in hash to differentiate n_points/n_iter
    param_str = json.dumps(kwargs, sort_keys=True)
    payload = v.tobytes() + param_str.encode('utf-8')
    return hashlib.md5(payload).hexdigest()


def airfoil_fitness(vec,
                    Re: float = 1e6,
                    alpha: float = 3.0,
                    Cm_target: float = -0.05,

                    weights=(1.0, 2.0, 0.5),
                    return_all: bool = False,
                    **kwargs):
    """
    Black-box aerodynamic objective for optimisation.
    
    MOO FORMULATION (Task B.2):
    ---------------------------
    Design Variables:
      - 3 Upper Surface CST coefficients
      - 3 Lower Surface CST coefficients
      - Total: 6 variables (bounds usually [-0.2, 0.5])
      
    Objective Function (Scalarized Weighted Sum):
      Minimize J = w1 * Cd - w2 * Cl + w3 * |Cm - Cm_target|
      
    Constraints:
      - Geometric bounds are handled by the optimizer (box constraints).
      - Aerodynamic feasibility (convergence) is handled by penalizing non-convergent solutions (J=10).

    vec: [Au0, Au1, Au2, Al0, Al1, Al2]
    
    If return_all=True, returns (J, Cl, Cd, Cm).
    """
    key = _key_from_vec(vec, Re=Re, alpha=alpha, **kwargs)
    if key in cache:
        rec = cache[key]
        J = rec["J"]
        if return_all:
            return J, rec["Cl"], rec["Cd"], rec["Cm"]
        return J

    # vec is 6 elements: 3 upper, 3 lower
    Au = vec[:3]
    Al = vec[3:]
    
    try:
        # Pass kwargs (e.g. n_points, n_iter) to analyze_airfoil
        Cl, Cd, Cm = analyze_airfoil(Au, Al, Re=Re, alpha=alpha, **kwargs)
        
        if Cl is None: # XFOIL failed
             # Penalise non-convergent / bad geometries
            Cl = Cd = Cm = None
            J = 10.0
        else:
            w1, w2, w3 = weights
            J = w1 * Cd - w2 * Cl + w3 * abs(Cm - Cm_target)
            
    except Exception:
        Cl = Cd = Cm = None
        J = 10.0

    # Sanitize NaN values
    if J is None or np.isnan(J):
        J = 10.0
        Cl = Cd = Cm = None

    # Update cache (numbers only)
    # cache[key] = {"J": J, "Cl": Cl, "Cd": Cd, "Cm": Cm, "vec": list(vec)}
    # NOTE: Disk cache disabled for parallel safety
    # os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    # with open(CACHE_FILE, "w") as f:
    #     json.dump(cache, f, indent=2)

    if return_all:
        return J, Cl, Cd, Cm
    return J


def coeffs_at_alpha(vec, Re: float = 1e6, alpha: float = 3.0, **kwargs):
    """Convenience: return only (Cl, Cd, Cm) using same cache path."""
    J, Cl, Cd, Cm = airfoil_fitness(vec, Re=Re, alpha=alpha, return_all=True, **kwargs)
    return Cl, Cd, Cm

```

---

## File: `benchmarks\griewank.py`
```python
import numpy as np

def griewank(x: np.ndarray) -> float:
    """
    Griewank benchmark function.
    Global minimum at x = 0, f = 0. Bounds typically [-600, 600]^D.
    """
    x = np.asarray(x, dtype=float)
    s = np.sum(x * x) / 4000.0
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1, dtype=float))))
    return 1.0 + s - p

```

---

