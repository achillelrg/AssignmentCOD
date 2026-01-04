# Part B: Airfoil Optimisation

This module satisfies **Part B** of the assignment: Optimising an airfoil for aerodynamic performance using XFOIL.

## Prerequisites
- **XFOIL** must be installed and accessible.
- By default, the system looks for:
  - `xfoil.exe` (on Windows)
  - `xfoil` (on Linux/Mac)
- Ensure the executable is in your system PATH or configure it in `utils/xfoil_runner.py`.

## How to Run
```bash
python main_part_b.py
```

## Structure
- Uses CST (Class-Shape-Transformation) method with 6 variables (3 upper, 3 lower).
- Objective: Minimise Drag/Lift ratio with constraints (handled via penalty in the fitness function).
