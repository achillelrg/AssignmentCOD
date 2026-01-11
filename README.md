# Computational Design Assignment (Airfoil Optimization)

This project implements a comprehensive optimization pipeline for airfoil design using Python and XFOIL. It addresses Parts A, B, and C of the assignment, featuring Particle Swarm Optimization (PSO), Genetic Algorithms (GA), and Surrogate Modeling (Kriging).

## � System Requirements
-   **OS:** Windows 10/11 (with **WSL** installed) OR Linux.
-   **Software:**
    -   Python 3.10+
    -   XFOIL (Linux binary required).
        -   *Note: Windows users must have WSL enabled as the scripts bridge to the Linux XFOIL binary.*

## 🚀 Quick Start: The "Ultimate Run"
To generate all results (Part A, B, C, and Uncertainty) in a single, automated campaign, use the provided "Ultimate Run" scripts.

### ⚠️ Critical First-Run Note
On the very first execution, the script will automatically create a dedicated Python Virtual Environment (`venv`) and install all dependencies.
> **Please wait ~10-15 minutes** for this initialization to complete. Do not close the window.

### Option 1: Windows (Recommended)
This script runs the **full production campaign**, including high-fidelity data generation (1000 samples) and optimization.
```cmd
run_ultimate.bat
```
-   **Runtime:** ~15-20 minutes (depending on CPU).
-   **Output:** Generates a full report and all figures in `data/`.

### Option 2: Linux / Mac / Git Bash
This script defaults to a **"Fast Mode"** (50 samples) for headerless environments or quick verification.
```bash
./run_ultimate.sh
```

## �️ Modular Execution & Parameters
If you wish to run specific parts of the assignment individually or change parameters (e.g., population size, seeds), use the following commands.

### Part A: Algorithm Verification
Validates the PSO algorithm on the 5D Griewank function using 10 independent seeds.
```bash
python experiments/run_part_a.py
```
-   **Output:** `data/PartA/figures/convergence_overlay.png`, `robustness_boxplot.png`

### Part B: Direct Airfoil Optimization
Runs XFOIL-in-the-loop optimization using PSO or GA.
```bash
# Example: Run PSO with 40 particles for 1000 evaluations
python experiments/run_airfoil.py --part B --solver pso --pop 40 --evals 1000 --seed 42 --jobs 8
```
**Key Arguments:**
-   `--solver`: `pso` or `ga`.
-   `--pop`: Population size (default: 20).
-   `--evals`: Total evaluation budget (default: 400).
-   `--jobs`: Number of parallel CPU cores (default: 1).
-   `--seed`: Random seed for reproducibility.

### Part C: Surrogate Modeling Pipeline
Part C requires a multi-step process: Data Generation → Training → Optimization → Uncertainty.

**1. Data Generation (LHS)**
Generates the design space dataset.
```bash
python experiments/part_c_data_gen.py --samples 1000 --jobs 8
```

**2. Train Surrogate & Optimize**
Trains the Kriging model and finds the optimal geometry.
```bash
python experiments/part_c_surrogate.py      # Trains model
python experiments/part_c_opt_surrogate.py  # Optimizes on surface
```

**3. Uncertainty Quantification (UQ)**
Performs Monte Carlo simulation on the optimal design.
```bash
python experiments/part_c_uq_opt.py --json data/PartC/surrogate_comparison.json --samples 200
```
-   `--samples`: Number of Monte Carlo iterations.
-   `--mean`: Mean Angle of Attack (default: 3.0 deg).
-   `--std`: Standard Deviation (default: 0.1 deg).

**4. Visualization**
Compares the Surrogate Optimum against a Reference (Part B) design.
```bash
python experiments/part_c_plot_geometry.py --ref "data/PartB/results/YOUR_BEST_Run.json"
```

---

## 📚 Command Reference
Detailed listing of helper scripts and their roles.

### `experiments/run_opt.py`
The generic optimizer driver used primarily by Part A. It optimizes a mathematical benchmark function.
-   `--D`: Dimension of the problem (default: 5).
-   `--evals`: Evaluation budget.
-   `--out`: Custom output path for the CSV log.

### `experiments/run_airfoil.py`
The main application driver for Part B.
-   `--cl`: Target Lift Coefficient (for constraint handling, if enabled).
-   `--alpha`: Fixed Angle of Attack (default: 3.0).

---

## Appendix C: Source Code Inventory & Description
The project is implemented in Python 3.10+, organized into driver scripts, core algorithms, and support libraries to ensure modularity and fault tolerance. In accordance with the assignment requirements, the following listings describe the custom source code implemented to solve the optimization problems.

### C.1. Experiment Drivers (experiments/)
This directory contains the primary executable scripts responsible for orchestrating optimization campaigns and generating visual evidence for the report.
-   `run_opt.py`: A generic driver used to verify the PSO algorithm on the 5D Griewank function, managing argument parsing and result logging.
-   `run_part_a.py`: The verification suite that executes 10 independent seeds of the Griewank optimization to generate the convergence overlays and boxplots required for statistical validation.
-   `run_airfoil.py`: The main application driver for Part B; it links the population-based optimizers to the XFOIL physics engine and manages lift and moment constraints.
-   `plot_airfoil.py`: Extracts data from optimization result files to generate aerodynamic analysis plots, including geometry and pressure distributions.
-   `part_c_data_gen.py`: Implements Latin Hypercube Sampling (LHS) to generate a space-filling dataset of 2,000 airfoil candidates for surrogate training.
-   `part_c_surrogate.py`: Implements the Log-Space Kriging logic to train Gaussian Process models on the LHS dataset while handling penalized "Wall" values for failed simulations.
-   `part_c_opt_surrogate.py`: Conducts deterministic optimization on the trained Kriging surface to identify the optimal robust geometry.
-   `part_c_uncertainty.py`: Performs the 200-sample Monte Carlo simulation to propagate angle-of-attack uncertainty through the surrogate model.
-   `part_c_plot_geometry.py`: A specialized utility for side-by-side comparison of the Part B (Direct) and Part C (Surrogate) optimized shapes.

### C.2. Core Utilities (utils/)
These modules implement the physical and mathematical kernels that serve as the foundation for the design pipeline.
-   `cst.py`: Implements the Class-Shape Transformation method using Bernstein polynomials and includes geometric guardrails to prevent self-intersecting profiles.
-   `xfoil_runner.py`: The project's "Crown Jewel" of automation; it features the WSL Bridge for cross-platform binary execution, result parsing, and 10-second hard timeouts to prevent legacy code "hanging".
-   `gp_surrogate.py`: A custom wrapper for Gaussian Process Regression, optimized for the high-variance drag data encountered in aerodynamic stall regimes.
-   `airfoil_problem.py`: Centralizes the configuration constants, including the quadratic penalty weights used in the fitness function formulation.

### C.3. Optimization Algorithms (optimizer/)
Custom implementations of the metaheuristics developed to navigate the non-convex aerodynamic landscape.
-   `pso.py`: Particle Swarm Optimization featuring inertia weight management ($w=0.7$) and cognitive/social velocity updates ($c_1=c_2=1.5$).
-   `ga.py`: Genetic Algorithm utilizing tournament selection and arithmetic crossover; used extensively for the parameter sensitivity study in Part B.
-   `base.py`: An abstract base class that defines the ask (request candidate) and tell (update with fitness) interface used to standardize all optimization calls.
