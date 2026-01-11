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
# Activate VENV (Handle Linux/Mac 'bin' vs Windows 'Scripts')
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "ERROR: Cannot find activate script in venv."
    exit 1
fi

# Set PYTHON variable to the venv executable
if [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -f "venv/Scripts/python.exe" ]; then
    PYTHON="venv/Scripts/python.exe"
elif [ -f "venv/Scripts/python" ]; then
    PYTHON="venv/Scripts/python"
else
    echo "ERROR: Cannot find python executable in venv."
    echo "Files found in venv:"
    ls -R venv | head -n 20
    exit 1
fi

# Use explicit python path for pip to avoid 'externally-managed' error
echo "Checking environment integrity..."

if $PYTHON -c "import numpy; import scipy; import pandas; import matplotlib; import sklearn; import pytest" >/dev/null 2>&1; then
    echo "✔ Dependencies already installed. Skipping download."
else
    echo "⚠ Missing dependencies detected."
    echo "Launching Smart Installer..."
    # Use the venv python to run the installer script
    $PYTHON utils/install_requirements.py
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed to install dependencies."
    fi
fi
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
$PYTHON experiments/run_airfoil.py --part B --solver pso --pop 20 --evals 400 --seed 123 --jobs 8

# Get the Result Path (PSO Baseline)
# Use ls -t and head properly
LATEST_REF=$(ls -t data/PartB/pso/results/*_best.json | head -n 1)

echo "-----------------------------------------------"
echo "Step 1.5: GA Parameter Study (Sensitivity Analysis)"
echo "Comparing Low vs High Fidelity GA runs... (FAST MODE)"

echo "Run A: Low Fidelity GA (Pop 10, Evals 100)..."
$PYTHON experiments/run_airfoil.py --part B --solver ga --pop 10 --evals 100 --seed 123 --jobs 8

echo "Run B: Medium Fidelity GA (Pop 10, Evals 200)..."
$PYTHON experiments/run_airfoil.py --part B --solver ga --pop 10 --evals 200 --seed 123 --jobs 8

echo "Run C: High Fidelity GA (Pop 20, Evals 400)..."
$PYTHON experiments/run_airfoil.py --part B --solver ga --pop 20 --evals 400 --seed 123 --jobs 8

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
