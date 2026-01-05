from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from utils.cst import cst_airfoil  # only if you also need it elsewhere
from utils.geometry import build_airfoil_coordinates, write_dat
from utils.xfoil_runner import run_single_alpha, XFoilError


@dataclass
class AirfoilConfig:
    """
    Configuration for the airfoil optimisation problem.
    """
    alpha_deg: float = 3.0
    Re: float = 1e6
    mach: float = 0.0
    iter_limit: int = 100

    # Number of *upper* surface points used in CST discretisation
    # Total points in .dat will be 2*n_points - 1
    n_points: int = 121  # safe: 2*121-1 = 241 < 365 XFOIL max

    # Constraint / penalty settings
    cl_min: float = 0.5
    cm_max_abs: float = 0.2
    w_cl: float = 50.0
    w_cm: float = 10.0
    w_fail: float = 1e3

    # Paths
    run_dir: Path = Path("data/results/airfoil/run_tmp")
    baseline_dat: Path = Path("data/airfoils/baseline.dat")
    optimised_dat: Path = Path("data/airfoils/optimised.dat")


def theta_to_cst(theta: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map a 6D design vector into upper/lower CST coefficient arrays.

    theta = [a0u, a1u, a2u,  a0l, a1l, a2l]
    """
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size != 6:
        raise ValueError(f"Expected 6 design variables, got {theta.size}")

    a_u = theta[:3]
    a_l = theta[3:]
    return a_u, a_l


def write_airfoil_from_theta(
    theta: Iterable[float],
    out_path: Path,
    n_points: int = 121,
) -> None:
    """
    Generate airfoil coordinates from CST parameters and write to a .dat file.
    """
    a_u, a_l = theta_to_cst(theta)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x, y = build_airfoil_coordinates(
        coeffs_upper=a_u,
        coeffs_lower=a_l,
        n_points=n_points,
        # keep default n1, n2, dz_te consistent with tests
    )

    write_dat(x, y, out_path, name="CST_AIRFOIL")


def evaluate_airfoil_theta(theta: Iterable[float], cfg: AirfoilConfig):
    """
    Evaluate a given airfoil design (theta) using XFOIL at a single alpha.

    Returns
    -------
    f : float
        Scalar objective value (lower is better).
    info : dict
        Dictionary with raw data: CL, CD, CM, success flag, etc.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dat_path = run_dir / "candidate.dat"

    # --- Quick geometric sanity check to avoid degenerate shapes ---
    a_u, a_l = theta_to_cst(theta)
    x_tmp, y_tmp = build_airfoil_coordinates(
        coeffs_upper=a_u,
        coeffs_lower=a_l,
        n_points=cfg.n_points,
    )
    max_y = np.max(np.abs(y_tmp))
    if max_y < 1e-4:
        # Completely flat / almost zero thickness -> skip XFOIL
        return cfg.w_fail, {
            "success": False,
            "error": f"Degenerate flat airfoil (max |y|={max_y:.2e})",
            "cl": np.nan,
            "cd": np.nan,
            "cm": np.nan,
        }

    # 1) Geometry -> .dat
    write_dat(x_tmp, y_tmp, dat_path, name="CST_AIRFOIL")

    # 2) XFOIL call (inviscid for robustness)
    try:
        res = run_single_alpha(
            dat_path,
            alpha_deg=cfg.alpha_deg,
            Re=cfg.Re,
            mach=cfg.mach,
            iter_limit=cfg.iter_limit,
            timeout=30.0,
            viscous=False,  # keep inviscid for stability
        )
        success = True
    except XFoilError as e:
        # XFOIL failed numerically: assign penalty objective
        f = cfg.w_fail
        info = {
            "success": False,
            "error": str(e),
            "cl": np.nan,
            "cd": np.nan,
            "cm": np.nan,
        }
        return f, info

    # 3) Objective with penalties
    penalty = 0.0

    if res.cl < cfg.cl_min:
        penalty += cfg.w_cl * (cfg.cl_min - res.cl) ** 2

    if abs(res.cm) > cfg.cm_max_abs:
        penalty += cfg.w_cm * (abs(res.cm) - cfg.cm_max_abs) ** 2

    f = res.cd + penalty

    info = {
        "success": success,
        "cl": res.cl,
        "cd": res.cd,
        "cm": res.cm,
        "penalty": penalty,
        "objective": f,
    }

    return f, info
