# Computational Design Assignment (Airfoil Optimization)

This repository contains the implementation for Parts A, B, and C of the Computational Design assignment. It uses Python to automate XFOIL for airfoil analysis and optimization.

## 🛠️ Installation

1.  **System Requirements:**
    -   Linux or WSL (Windows Subsystem for Linux).
    -   `XFOIL` installed (`sudo apt-get install xfoil` or binary in path).
    -   Python 3.8+.

2.  **Setup Environment:**
    ```bash
    # Create Virtual Environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install Dependencies
    pip install -r requirements.txt
    ```

3.  **XFOIL Configuration:**
    Ensure the `xfoil` command is available in your terminal. You can test it by running `xfoil` (type `quit` to exit).

---

## 🚀 How to Run

### Part A: Foundation
Part A is implicitly handled by the `utils/xfoil_runner.py` module used in all experiments.
To run a basic verification test:
```bash
python experiments/run_airfoil.py --cl 1.0 --alpha 4.0
```

### Part B: Direct Optimization
Run the Evolutionary Algorithms (GA or PSO) to find optimal airfoils directly using XFOIL.

**Command:**
```bash
python experiments/run_opt.py --method pso --pop 40 --gen 100 --seed 42
```
**Parameters:**
-   `--method`: `pso` (Particle Swarm) or `ga` (Genetic Algorithm).
-   `--pop`: Population size (e.g., 20, 40, 80). Larger = better search, slower.
-   `--gen`: Number of generations (e.g., 100, 500).
-   `--seed`: Random seed for reproducibility.

**Output:** Results are saved to `data/PartB/[method]/results/`.

### The "Ultimate Run" (Full Pipeline)
To generate the final results for the assignment (Part B + Part C + UQ) in one automated sequence:
```bash
./run_ultimate.sh
```
This script will:
1.  Clean old results.
2.  Run Part B Optimization (PSO).
3.  Run Part C Data Generation (with Penalty Method).
4.  Train Surrogate, Optimize, Plot, and Perform UQ.

*Note: Ensure `run_ultimate.sh` is executable (`chmod +x run_ultimate.sh`).*

### Part C: Surrogate Modeling (Manual Steps)
Part C relies on training a Machine Learning model (Kriging) on a dataset of airfoils. This is a multi-step pipeline.

**Step 1: Generate Training Data**
Generates airfoils and evaluates them in parallel.
```bash
python experiments/part_c_data_gen.py --samples 2000 --jobs 8
```
-   `--samples`: Number of valid samples to generate. Recommended: **1000-2000**.
-   `--jobs`: Number of parallel CPU cores to use.

**Step 2: Train Surrogate Model**
Trains the Gaussian Process on the generated data.
```bash
python experiments/part_c_surrogate.py
```
*Note: This automatically handles the Log-Transform for Drag and saves models to `data/PartC/models/`.*

**Step 3: Optimize using Surrogate**
Uses the trained AI model to find the optimal design instantly.
```bash
python experiments/part_c_opt_surrogate.py
```

**Step 4: Visualize & Compare**
Generates comparison plots between the Surrogate Optimum and Part B Benchmark.
```bash
# Requires a Part B result JSON file as --ref
python experiments/part_c_plot_geometry.py --ref data/PartB/pso/results/YOUR_BEST_RESULT.json
```

**Step 5: Uncertainty Quantification**
Runs Monte Carlo simulation on the optimal design.
```bash
python experiments/part_c_uq_opt.py --json data/PartC/surrogate_comparison.json --samples 200
```

---

## 📂 Key Files & Logic
-   `ASSIGNMENT_LOGIC_EXPLAINED.md`: Detailed explanation of the engineering logic (Why Penalty Method? Why Log-Transform?).
-   `PROJECT_JOURNEY.md`: A log of the development process and challenges solved.
-   `experiments/`: Contains all executable scripts.
-   `utils/`: Core libraries for XFOIL interaction and Geometry generation (CST).

## ⚠️ Note on "Penalty Method"
In Part C, we implement a **Penalty Method** where failed XFOIL runs are assigned $C_d = 0.5$. This appears as a vertical wall in the Parity Plots. This is intentional and crucial for robustness (see `ASSIGNMENT_LOGIC_EXPLAINED.md`).
