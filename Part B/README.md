# Part B: Airfoil Optimisation using XFOIL and PSO

This module performs aerodynamic optimization of an airfoil using the CST parameterization method and XFOIL.

## Setup

### Requirements
- **XFOIL**: Ensure `xfoil` is installed and available in your PATH or at `Xfoil/xfoil`.
- **Python**: Python 3.8+.
- **Dependencies**: `numpy` is required. `pandas` and `matplotlib` are recommended for plotting but optional.

### Installation (Optional)
If you want to generate plots:
```bash
sudo apt install python3-pandas python3-matplotlib
# OR
pip install pandas matplotlib --break-system-packages
```

## Usage

Run the optimization from the project root:

```bash
# Run with default settings (200 evaluations)
python3 -m experiments.run_airfoil --part B

# Run with custom budget
# Run with custom budget
python3 -m experiments.run_airfoil --part B --evals 500 --pop 20

# Start fresh (delete previous data)
python3 -m experiments.run_airfoil --part B --clean

```

## Output
Results are saved to `data/PartB/results/`.
- `*.csv`: Optimization log (objective values per iteration).
- `*_best.json`: Best design vector found.
- `figures/`: Convergence plots and airfoil geometry comparisons (if plotting libraries are installed).

## Implementation Details
- **Optimization**: Particle Swarm Optimization (PSO).
- **Evaluation**: Calls XFOIL for CL, CD, CM at specified conditions (Re=1e6, Alpha=3.0).
- **Robustness**: Uses an improved XFOIL runner that parses stdout directly to avoid file I/O issues observed with `PACC` on some systems.
