from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from utils.cst import cst_airfoil


def build_airfoil_coordinates(
    coeffs_upper: Iterable[float],
    coeffs_lower: Iterable[float],
    n_points: int = 201,
    n1: float = 0.5,
    n2: float = 1.0,
    dz_te: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a full airfoil coordinate set suitable for XFOIL.

    Ordering:
        - start at upper trailing edge (x ~ 1) and go to leading edge (x ~ 0)
        - then from leading edge along the lower surface back to trailing edge

    Parameters
    ----------
    coeffs_upper, coeffs_lower : iterable of float
        CST coefficients for upper and lower surfaces.
    n_points : int
        Number of points per surface.
    n1, n2 : float
        Class function exponents.
    dz_te : float
        Trailing-edge thickness (split between upper and lower).

    Returns
    -------
    x, y : np.ndarray
        Concatenated coordinates around the airfoil.
    """
    xu, yu, xl, yl = cst_airfoil(
        n_points=n_points,
        coeffs_upper=coeffs_upper,
        coeffs_lower=coeffs_lower,
        n1=n1,
        n2=n2,
        dz_te=dz_te,
    )

    # Upper surface: from TE (x=1) to LE (x=0)
    xu_rev = xu[::-1]
    yu_rev = yu[::-1]

    # Lower surface: from LE (x=0) to TE (x=1)
    # Skip the first point to avoid duplicating the LE
    xl_fwd = xl[1:]
    yl_fwd = yl[1:]

    x = np.concatenate([xu_rev, xl_fwd])
    y = np.concatenate([yu_rev, yl_fwd])

    return x, y


def write_dat(x, y, path, name: str = "airfoil"):
    """
    Write an airfoil .dat file in XFOIL format from full (x, y) coordinates.

    Parameters
    ----------
    x, y : 1D arrays
        Full airfoil coordinates, starting at upper TE -> LE, then lower LE -> TE.
    path : Path or str
        Output .dat file path.
    name : str
        Airfoil name written on the first line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write(f"{name}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f}\n")

    return path
