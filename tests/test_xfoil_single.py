import numpy as np
from benchmarks.airfoil_xfoil import coeffs_at_alpha

vec = np.array([0.20, 0.15, 0.05,  0.15, 0.10, 0.05], dtype=float)
print("Testing single-alpha at 3° ...")
Cl, Cd, Cm = coeffs_at_alpha(vec, Re=1e6, alpha=3.0)
print("Cl, Cd, Cm =", Cl, Cd, Cm)
