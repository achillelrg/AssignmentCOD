#!/bin/bash
set -e  # Exit on error

echo "==============================================="
echo "       🚀 STARTING ULTIMATE RUN 🚀             "
echo "==============================================="

# Activate env
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Step 1: Part B - Reference Optimization (PSO)"
echo "Cleaning old Part B results..."
rm -rf data/PartB/pso/results/* data/PartB/pso/figures/*

echo "Running PSO (Pop 40, Evals 4000 ~100 Gen)... This will take ~20 minutes."
python experiments/run_airfoil.py --part B --solver pso --pop 40 --evals 4000 --seed 999 --jobs 8

# Get the Result Path (PSO Baseline)
LATEST_REF=$(ls -t data/PartB/pso/results/*_best.json | head -n 1)

echo "-----------------------------------------------"
echo "Step 1.5: GA Parameter Study (Sensitivity Analysis)"
echo "Comparing Low vs High Fidelity GA runs..."

echo "Run A: Low Fidelity GA (Pop 10, Evals 500, Gen 50)..."
python experiments/run_airfoil.py --part B --solver ga --pop 10 --evals 500 --seed 999 --jobs 8

echo "Run B: Medium Fidelity GA (Pop 20, Evals 2000, Gen 100)..."
python experiments/run_airfoil.py --part B --solver ga --pop 20 --evals 2000 --seed 999 --jobs 8

echo "Run C: High Fidelity GA (Pop 40, Evals 4000, Gen 100)..."
python experiments/run_airfoil.py --part B --solver ga --pop 40 --evals 4000 --seed 999 --jobs 8
echo "Reference Design Found: $LATEST_REF"

echo "-----------------------------------------------"
echo "Step 2: Part C - Robust Data Generation (Penalty Method)"
echo "Cleaning old data..."
rm -f data/PartC/training_data.csv data/PartC/models/*.pkl data/PartC/figures/*.png

echo "Generating 4000 Samples with 8 Workers..."
python experiments/part_c_data_gen.py --samples 4000 --jobs 8

echo "-----------------------------------------------"
echo "Step 3: Part C - Surrogate Training"
python experiments/part_c_surrogate.py

echo "-----------------------------------------------"
echo "Step 4: Part C - Surrogate Optimization"
python experiments/part_c_opt_surrogate.py

echo "-----------------------------------------------"
echo "Step 5: Visualization & Comparison"
python experiments/part_c_plot_geometry.py --ref "$LATEST_REF"

echo "-----------------------------------------------"
echo "Step 6: Uncertainty Quantification"
python experiments/part_c_uq_opt.py --json data/PartC/surrogate_comparison.json --samples 200

echo "==============================================="
echo "       ✅ ULTIMATE RUN COMPLETE ✅             "
echo "==============================================="
echo "Results:"
echo " - Report: data/PartC/surrogate_metrics.txt"
echo " - Plots:  data/PartC/figures/"
echo " - Logic:  ASSIGNMENT_LOGIC_EXPLAINED.md"
