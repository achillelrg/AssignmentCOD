import os, tempfile
from utils.xfoil_runner import run_xfoil_single_alpha

# NACA 2412 excerpt in proper format with header + blank line
NACA2412 = """NACA2412
"""

# 201 points around the airfoil – to keep it short here, just download a standard
# NACA 2412 .dat (header + blank line + points). For the test now, we generate a very simple symmetric 0012-like
# shape using a cosine x grid and a thin thickness law; but best is to paste a real .dat you trust.

def write_simple_sym_airfoil(path):
    import numpy as np
    # cosine spacing helps XFOIL
    t = np.linspace(0, 1, 201)
    x = 0.5*(1 - np.cos(np.pi*t))
    # simple 12% thickness distribution (not perfect 0012, but valid)
    yt = 0.12/0.2*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    yu = yt
    yl = -yt
    with open(path, "w") as f:
        f.write("TEST0012\n\n")
        for xi, yi in zip(x[::-1], yu[::-1]):   # upper TE->LE
            f.write(f"{xi:.6f} {yi:.6f}\n")
        for xi, yi in zip(x[1:], yl[1:]):       # lower LE->TE
            f.write(f"{xi:.6f} {yi:.6f}\n")

def main():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
    tmp.close()
    write_simple_sym_airfoil(tmp.name)
    Cl, Cd, Cm = run_xfoil_single_alpha(tmp.name, alpha=3.0, Re=1e6)
    print("Known-foil single-α:", Cl, Cd, Cm)
    os.remove(tmp.name)

if __name__ == "__main__":
    main()
