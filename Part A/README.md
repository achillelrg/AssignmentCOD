# Part A: Optimisation Solver (Griewank)

This module satisfies the requirements for **Part A** of the assignment.

## Contents
- `main_part_a.py`: Entry point to run the PSO solver on the 5D Griewank function.

## How to Run
Run the script from the command line:

```bash
python main_part_a.py
```

### Options
You can pass arguments to control the PSO hyperparameters:

```bash
python main_part_a.py --pop 50 --evals 10000 --w 0.7 --c1 1.5 --c2 1.5
```

- `--c2`: Social coefficient

## Outputs
After running the script, results are saved in `data/`:

1.  **CSV Logs**: `data/results/single_runs/` (or `multi_runs` depending on seed). Contains iteration-wise statistics.
2.  **Plots**:
    -   **Convergence**: `data/figures/convergence/` (Semilogy plot of best fitness vs magnitude).
    -   **Swarm Trajectory**: `data/figures/swarm/` (Only if running in 2D with `--D 2 --trace_every 10`).
