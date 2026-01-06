from typing import Tuple, Optional, Iterable
import os
import tempfile
import uuid
import shutil
import numpy as np

from utils.geometry import build_airfoil_coordinates, write_dat
from utils.cst import cst_airfoil
from utils.xfoil_runner import run_xfoil_single_alpha, run_xfoil_polar

def analyze_airfoil(
    coeffs_upper: Iterable[float],
    coeffs_lower: Iterable[float],
    alpha: float = 3.0,
    Re: float = 1e6,
    mach: float = 0.1,
    n_iter: int = 200,
    n_points: int = 201,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Automates the creation of airfoil (CST), execution of XFOIL, and extraction of parameters.
    
    Parameters
    ----------
    coeffs_upper : Iterable[float]
        CST coefficients for upper surface.
    coeffs_lower : Iterable[float]
        CST coefficients for lower surface.
    alpha : float, optional
        Angle of attack in degrees, by default 3.0
    Re : float, optional
        Reynolds number, by default 1e6
    mach : float, optional
        Mach number, by default 0.1
    n_iter : int, optional
        Number of XFOIL iterations, by default 200
    n_points : int, optional
        Number of points for airfoil generation, by default 201

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[float]]
        (CL, CD, CM) or (None, None, None) if failed.
    """
    
    # 1. Validate Geometry (Prevent Crossing Surfaces)
    # Check if Upper Surface is always above Lower Surface
    # We use cst_airfoil directly to get the separated arrays
    try:
        _, yu, _, yl = cst_airfoil(n_points=n_points, coeffs_upper=coeffs_upper, coeffs_lower=coeffs_lower)
        
        # Tolerance for numerical noise, though usually exact crossing is bad.
        # Check if any lower point is significantly above the corresponding upper point.
        # We ignore the very leading/trailing edges which might be close/equal.
        # Index 1:-1 skips endpoints.
        if np.any(yl[1:-1] > yu[1:-1]):
            # Invalid geometry: crossing surfaces
            return None, None, None
            
    except ValueError:
        # e.g. n_points too small
        return None, None, None

    # 2. Create Airfoil Coordinates from CST
    x, y = build_airfoil_coordinates(coeffs_upper, coeffs_lower, n_points=n_points)
    
    # 2. Write to temporary .dat file
    # unique name to avoid conflicts
    unique_name = f"cst_airfoil_{uuid.uuid4().hex[:8]}.dat"
    # Write to temp directory
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    dat_path = os.path.join(temp_dir, unique_name)
    
    try:
        write_dat(x, y, dat_path, name="CST_AIRFOIL")
        
        # 3. Execute XFOIL
        cl, cd, cm = run_xfoil_single_alpha(
            dat_path,
            alpha=alpha,
            Re=Re,
            mach=mach,
            n_iter=n_iter
        )
        
        return cl, cd, cm
        
    finally:
        # Cleanup geom file
        if os.path.exists(dat_path):
            try:
                os.remove(dat_path)
            except OSError:
                pass
