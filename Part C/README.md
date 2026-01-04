# Part C: Advanced Tasks

This module implements:
1. **Uncertainty Analysis**: Evaluating the robustness of a design under varying Angle of Attack.
2. **Surrogate Modelling**: Training a Gaussian Process (or RBF) model to predict airfoil performance and optimising it.

## How to Run

### Run All
```bash
python main_part_c.py --mode all
```

### Uncertainty Only
Run uncertainty analysis on a specific design vector (6 floats):
```bash
python main_part_c.py --mode uncertainty --best_x 0.16 0.20 0.20 0.10 0.10 0.10
```

### Surrogate Optimization Only
Train a surrogate model and optimise it:
```bash
python main_part_c.py --mode surrogate
```

## Dependencies
- `scikit-learn` is recommended for Gaussian Process.
- `scipy` is used as a fallback if sklearn is missing.
