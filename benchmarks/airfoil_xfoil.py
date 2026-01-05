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


def _key_from_vec(vec):
    """Generate a stable hash key from the design vector."""
    v = np.asarray(vec, dtype=float)
    return hashlib.md5(v.tobytes()).hexdigest()


def airfoil_fitness(vec,
                    Re: float = 1e6,
                    alpha: float = 3.0,
                    Cm_target: float = -0.05,
                    weights=(1.0, 2.0, 0.5),
                    return_all: bool = False):
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
    key = _key_from_vec(vec)
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
        Cl, Cd, Cm = analyze_airfoil(Au, Al, Re=Re, alpha=alpha)
        
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
    cache[key] = {"J": J, "Cl": Cl, "Cd": Cd, "Cm": Cm, "vec": list(vec)}
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    if return_all:
        return J, Cl, Cd, Cm
    return J


def coeffs_at_alpha(vec, Re: float = 1e6, alpha: float = 3.0):
    """Convenience: return only (Cl, Cd, Cm) using same cache path."""
    J, Cl, Cd, Cm = airfoil_fitness(vec, Re=Re, alpha=alpha, return_all=True)
    return Cl, Cd, Cm
