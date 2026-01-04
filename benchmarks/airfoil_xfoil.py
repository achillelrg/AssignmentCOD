import os
import hashlib
import json
import tempfile
import numpy as np

from utils.cst_airfoil import make_airfoil
from utils.xfoil_runner import run_xfoil_single_alpha

# Where we store *only* the numerical cache (no geometry files)
CACHE_FILE = os.path.join("data", "cache_xfoil.json")

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


def _eval_coeffs(vec, Re: float, alpha: float):
    """Compute (Cl, Cd, Cm) at given alpha, Re via XFOIL. Uses a temporary .dat."""
    Au = vec[:3]
    Al = vec[3:]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
    dat_path = tmp.name
    tmp.close()
    try:
        make_airfoil(Au, Al, dat_path)
        Cl, Cd, Cm = run_xfoil_single_alpha(dat_path, alpha=alpha, Re=Re)
    finally:
        try:
            os.remove(dat_path)
        except OSError:
            pass
    return Cl, Cd, Cm


def airfoil_fitness(vec,
                    Re: float = 1e6,
                    alpha: float = 3.0,
                    Cm_target: float = -0.05,
                    weights=(1.0, 2.0, 0.5),
                    return_all: bool = False):
    """
    Black-box aerodynamic objective for optimisation.

    vec: [Au0, Au1, Au2, Al0, Al1, Al2]
    J = w1*Cd - w2*Cl + w3*|Cm - Cm_target|.

    If return_all=True, returns (J, Cl, Cd, Cm).
    """
    key = _key_from_vec(vec)
    if key in cache:
        rec = cache[key]
        J = rec["J"]
        if return_all:
            return J, rec["Cl"], rec["Cd"], rec["Cm"]
        return J

    try:
        Cl, Cd, Cm = _eval_coeffs(vec, Re=Re, alpha=alpha)
        w1, w2, w3 = weights
        J = w1 * Cd - w2 * Cl + w3 * abs(Cm - Cm_target)
    except Exception:
        # Penalise non-convergent / bad geometries
        Cl = Cd = Cm = None
        J = 10.0

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
