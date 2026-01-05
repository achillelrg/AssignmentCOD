from typing import Tuple, Optional, Iterable
import os
import tempfile
import uuid
import shutil
import numpy as np

from utils.geometry import build_airfoil_coordinates, write_dat
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
    
    # 1. Create Airfoil Coordinates from CST
    x, y = build_airfoil_coordinates(coeffs_upper, coeffs_lower, n_points=n_points)
    
    # 2. Write to temporary .dat file
    # unique name to avoid conflicts
    unique_name = f"cst_airfoil_{uuid.uuid4().hex[:8]}.dat"
    # We write it to current dir to ensure XFOIL can load it easily, then clean up
    dat_path = os.path.abspath(unique_name)
    
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
