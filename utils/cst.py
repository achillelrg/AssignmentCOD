from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def _bernstein_matrix(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute Bernstein basis matrix of order n at points x.

    Returns array of shape (len(x), n+1) where column k is B_k^n(x).
    """
    x = np.asarray(x, dtype=float)
    k = np.arange(n + 1)
    binom = np.array([math.comb(n, int(ki)) for ki in k], dtype=float)

    # Shape (len(x), n+1)
    x_col = x[:, None]
    return binom * (x_col ** k) * ((1.0 - x_col) ** (n - k))


def cst_surface(
    x: np.ndarray,
    coeffs: Iterable[float],
    n1: float = 0.5,
    n2: float = 1.0,
    dz_te: float = 0.0,
) -> np.ndarray:
    """
    Compute CST surface y(x) given coefficients.

    Parameters
    ----------
    x : array_like
        Chordwise positions in [0, 1].
    coeffs : iterable of float
        CST shape coefficients (A_0 ... A_N).
    n1, n2 : float
        Class function exponents: C(x) = x^n1 * (1-x)^n2.
    dz_te : float
        Linear trailing-edge thickness term added as x * dz_te.

    Returns
    -------
    y : np.ndarray
        Surface ordinate values at x.
    """
    x = np.asarray(x, dtype=float)
    coeffs = np.asarray(list(coeffs), dtype=float)
    n = coeffs.size - 1
    if n < 0:
        raise ValueError("coeffs must contain at least one value")

    # Shape function via Bernstein polynomials
    B = _bernstein_matrix(n, x)  # (len(x), n+1)
    S = B @ coeffs  # (len(x),)

    # Class function
    C = (x**n1) * ((1.0 - x) ** n2)

    # CST surface
    y = C * S + x * dz_te
    return y


def cst_airfoil(
    n_points: int,
    coeffs_upper: Iterable[float],
    coeffs_lower: Iterable[float],
    n1: float = 0.5,
    n2: float = 1.0,
    dz_te: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CST airfoil surfaces.

    Parameters
    ----------
    n_points : int
        Number of points per surface (upper and lower).
    coeffs_upper, coeffs_lower : iterable of float
        CST coefficients for upper and lower surfaces.
    n1, n2 : float
        Class function exponents.
    dz_te : float
        Total trailing-edge thickness. Split equally between upper and lower.

    Returns
    -------
    xu, yu, xl, yl : np.ndarray
        x and y coordinates of upper and lower surfaces, each of length n_points.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    x = np.linspace(0.0, 1.0, n_points)

    # Split TE thickness evenly: +dz/2 on upper, -dz/2 on lower
    yu = cst_surface(x, coeffs_upper, n1=n1, n2=n2, dz_te=+dz_te / 2.0)
    yl = cst_surface(x, coeffs_lower, n1=n1, n2=n2, dz_te=-dz_te / 2.0)

    return x.copy(), yu, x.copy(), yl
