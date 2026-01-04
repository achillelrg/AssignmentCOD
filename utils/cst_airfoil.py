# utils/cst_airfoil.py
import math
import numpy as np

def cst_shape(x: np.ndarray, A):
    A = np.asarray(A, dtype=float)
    n = len(A) - 1
    C = np.power(x, 0.5) * (1.0 - x)
    S = np.zeros_like(x)
    for i in range(n + 1):
        S += A[i] * math.comb(n, i) * np.power(x, i) * np.power(1.0 - x, n - i)
    return C * S

def airfoil_coords(Au, Al, npts: int = 201):
    x  = np.linspace(0.0, 1.0, npts)
    Au = np.asarray(Au, dtype=float)
    Al = np.asarray(Al, dtype=float)
    yu = cst_shape(x, Au)
    yl = -cst_shape(x, Al)
    return x, yu, yl

def make_airfoil(Au, Al, filename: str = "airfoil.dat", npts: int = 201):
    x, yu, yl = airfoil_coords(Au, Al, npts=npts)
    with open(filename, "w") as f:
        f.write("CST_airfoil\n")
        f.write("\n")  # blank line helps some builds

        # upper surface: TE -> LE
        for xi, yi in zip(x[::-1], yu[::-1]):
            f.write(f"{xi:.6f} {yi:.6f}\n")

        # lower surface: LE -> TE (skip duplicated LE)
        for xi, yi in zip(x[1:], yl[1:]):
            f.write(f"{xi:.6f} {yi:.6f}\n")
