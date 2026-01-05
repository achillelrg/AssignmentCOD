import argparse
import glob
import json
import os

import numpy as np
try:
    import pandas as pd
    from pandas.errors import EmptyDataError
except ImportError:
    pd = None
    EmptyDataError = ValueError

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from benchmarks.airfoil_xfoil import coeffs_at_alpha
from utils.geometry import build_airfoil_coordinates as airfoil_coords
from utils.geometry import write_dat
from utils.xfoil_runner import run_xfoil_polar

BASE_FIG_DIR = os.path.join("data", "figures", "airfoil")

# Simple baseline CST coefficients (you can adjust these)
BASELINE_AU = [0.2, 0.1, 0.05]
BASELINE_AL = [-0.1, -0.05, -0.02]


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def make_airfoil(Au, Al, filename, npts=201):
    x, y = airfoil_coords(Au, Al, n_points=npts)
    write_dat(x, y, filename, name="CST_AIRFOIL")




def read_log(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except EmptyDataError:
        raise SystemExit(f"CSV is empty or invalid: {csv_path}. "
                         f"Delete it and re-run experiments.run_airfoil.")
    for c in ["f_best", "f_mean", "f_std", "gbest_f"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df



def plot_convergence(csv_path: str):
    if plt is None or pd is None:
        print("Skipping plot_convergence (missing libs)")
        return None
        
    df = read_log(csv_path)
    fig = plt.figure()
    ax = plt.gca()
    vals = df["gbest_f"]
    if (vals <= 0).any():
        ax.plot(df["iter"], vals)
    else:
        ax.semilogy(df["iter"], vals)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best objective J")
    ax.grid(True, which="both", linestyle=":")
    ax.set_title("Airfoil optimisation convergence")

    base = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Use helper to find path
    fig_dir = _infer_output_dir(csv_path, "convergence")
    outpath = os.path.join(fig_dir, f"{base}_convergence.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _infer_output_dir(csv_path: str, subdir: str = "airfoil") -> str:
    """Helper to infer figure output directory from CSV location."""
    if not csv_path:
        return os.path.join("data", "figures", subdir)
        
    csv_dir = os.path.dirname(csv_path)
    if "Part" in csv_dir:
        # e.g. data/PartB/results -> data/PartB/figures/subdir
        if os.path.basename(csv_dir) == "results":
            base_dir = os.path.dirname(csv_dir)
        else:
            base_dir = csv_dir
        return os.path.join(base_dir, "figures", subdir)
    else:
        return os.path.join("data", "figures", subdir)

import numpy as np
from utils.cst import cst_airfoil
from utils.xfoil_runner import run_xfoil_polar
from utils.geometry import write_dat

def plot_geometry(best_vec, out_csv: str = None):
    if plt is None: return None
    
    # CST parameters
    Au = best_vec[:3]
    Al = best_vec[3:]
    
    # Generate coordinates for plotting (more points for smoothness)
    # cst_airfoil returns (x, yu, x, yl) -> we need x, yu, yl
    # Signature: cst_airfoil(n_points, coeffs_upper, coeffs_lower)
    x, yu, xl, yl = cst_airfoil(200, Au, Al)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, yu, 'b-', label='Upper')
    ax.plot(x, yl, 'r-', label='Lower')
    ax.fill_between(x, yu, yl, color='gray', alpha=0.1)
    ax.axis('equal')
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Optimized Airfoil Geometry")
    ax.grid(True, linestyle=":")
    ax.legend()
    
    fig_dir = _infer_output_dir(out_csv, "geometry")
    _ensure_dir(os.path.join(fig_dir, "dummy"))
    outpath = os.path.join(fig_dir, "geometry_comparison.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_coeff_bar(best_vec, Re=1e6, alpha=3.0, out_csv: str = None, outpath=None):
    if plt is None: return None
    
    labels = [f"Au{i}" for i in range(3)] + [f"Al{i}" for i in range(3)]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, best_vec, color=['b']*3 + ['r']*3)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Optimized CST Parameters")
    
    # Add values on top
    for rect in bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    fig_dir = _infer_output_dir(out_csv, "coefficients")
    _ensure_dir(os.path.join(fig_dir, "dummy"))
    outpath = os.path.join(fig_dir, "coeff_bar.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_polar(best_vec, Re=1e6, out_csv: str = None):
    if plt is None: return None
    
    # We need a .dat file to run polar
    # Generate coordinates
    # cst_airfoil(n_points, coeffs_upper, coeffs_lower)
    x, yu, xl, yl = cst_airfoil(160, best_vec[:3], best_vec[3:])
    
    # Create temp dat file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
        tmp_dat = f.name
    
    # Convert to standard format (TE->LE->TE loop)
    # Upper: TE(1) to LE(0)
    # Lower: LE(0) to TE(1)
    # cst_airfoil gives 0->1.
    # Upper needs flip.
    xu = x[::-1]
    yup = yu[::-1]
    xl = x[1:]
    ylo = yl[1:]
    
    # Concat
    xx = np.concatenate([xu, xl])
    yy = np.concatenate([yup, ylo])
    
    write_dat(xx, yy, tmp_dat, name="OPT_AIRFOIL")
    
    # Run Polar
    # Alpha range: -5 to 15 deg
    print("Running polar analysis for plot...")
    alphas, cls, cds, cms = run_xfoil_polar(tmp_dat, -5, 15, 1.0, Re=Re, n_iter=200)
    
    # Clean up dat
    try:
        os.remove(tmp_dat)
    except:
        pass
        
    if len(alphas) == 0:
        print("Polar run failed or returned no data.")
        return None
        
    # Plot Drag Polar (Cl vs Cd) and Cl/alpha
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Drag Polar
    ax = axes[0]
    ax.plot(cds, cls, 'o-')
    ax.set_xlabel("Cd")
    ax.set_ylabel("Cl")
    ax.set_title(f"Drag Polar (Re={Re:.1e})")
    ax.grid(True)
    
    # Cl/Alpha
    ax = axes[1]
    ax.plot(alphas, cls, 'o-')
    ax.set_xlabel("Alpha (deg)")
    ax.set_ylabel("Cl")
    ax.set_title("Lift Curve")
    ax.grid(True)
    
    fig_dir = _infer_output_dir(out_csv, "polar")
    _ensure_dir(os.path.join(fig_dir, "dummy"))
    outpath = os.path.join(fig_dir, "drag_polar.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to airfoil optimisation CSV. If omitted, use latest in data/results/airfoil.")
    args = ap.parse_args()

    if args.csv is None:
        pattern = os.path.join("data", "results", "airfoil", "airfoil_opt_seed*_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit("No airfoil optimisation CSV found. Run experiments.run_airfoil first.")
        csv_path = files[-1]
    else:
        csv_path = args.csv

    print("Using CSV:", csv_path)

    # 1) Convergence plot
    conv_png = plot_convergence(csv_path)
    print("Saved convergence:", conv_png)

    # 2) Load best design vector JSON
    best_json = os.path.splitext(csv_path)[0] + "_best.json"
    if not os.path.exists(best_json):
        raise SystemExit(f"Best design JSON not found: {best_json}")
    with open(best_json, "r") as f:
        best = json.load(f)
    best_vec = np.array(best["x"], dtype=float)

    # 3) Geometry comparison
    geom_png = plot_geometry(best_vec, out_csv=csv_path)
    print("Saved geometry comparison:", geom_png)

    coeff_png = plot_coeff_bar(best_vec, Re=1e6, alpha=3.0, out_csv=csv_path)
    print("Saved coefficients bar chart:", coeff_png)


    # 4) Drag polar comparison
    try:
        polar_png = plot_polar(best_vec, Re=1e6, out_csv=csv_path)
        print("Saved polar:", polar_png)
    except Exception as e:
        print("Warning: polar plot failed, skipping. Reason:", e)



if __name__ == "__main__":
    main()
