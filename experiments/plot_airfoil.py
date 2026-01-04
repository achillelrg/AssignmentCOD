import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmarks.airfoil_xfoil import coeffs_at_alpha
from utils.cst_airfoil import airfoil_coords, make_airfoil
from utils.xfoil_runner import run_xfoil_polar
from pandas.errors import EmptyDataError

BASE_FIG_DIR = os.path.join("data", "figures", "airfoil")

# Simple baseline CST coefficients (you can adjust these)
BASELINE_AU = [0.2, 0.1, 0.05]
BASELINE_AL = [-0.1, -0.05, -0.02]


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)




def read_log(csv_path: str) -> pd.DataFrame:
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
    df = read_log(csv_path)
    fig = plt.figure()
    ax = plt.gca()
    ax.semilogy(df["iter"], df["gbest_f"])
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

def plot_geometry(best_vec, out_csv: str = None):
    # Optimised geometry
    Au_opt = best_vec[:3]
    Al_opt = best_vec[3:]
    x_opt, yu_opt, yl_opt = airfoil_coords(Au_opt, Al_opt, npts=201)

    # Baseline geometry
    x0, yu0, yl0 = airfoil_coords(BASELINE_AU, BASELINE_AL, npts=201)

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x0, yu0, label="Baseline upper")
    ax.plot(x0, yl0, label="Baseline lower")
    ax.plot(x_opt, yu_opt, "--", label="Optimised upper")
    ax.plot(x_opt, yl_opt, "--", label="Optimised lower")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Baseline vs optimised airfoil geometry")
    ax.grid(True, linestyle=":")
    ax.legend()

    fig_dir = _infer_output_dir(out_csv, "geometry")
    outpath = os.path.join(fig_dir, "geometry_comparison.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_coeff_bar(best_vec, Re=1e6, alpha=3.0, outpath=None, out_csv: str = None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Baseline = zero CST vector (simple, reproducible baseline)
    base_vec = np.zeros_like(best_vec)

    # Get coefficients (may be None if XFOIL failed)
    Clb, Cdb, Cmb = coeffs_at_alpha(base_vec, Re=Re, alpha=alpha)
    Clo, Cdo, Cmo = coeffs_at_alpha(best_vec, Re=Re, alpha=alpha)

    def _safe(vals):
        out = []
        for v in vals:
            if v is None:
                out.append(np.nan)
            else:
                out.append(float(v))
        return out

    labels    = ["Cl", "Cd", "Cm"]
    baseline  = _safe([Clb, Cdb, Cmb])
    optimised = _safe([Clo, Cdo, Cmo])

    # Warn if anything is NaN
    if any(np.isnan(baseline)) or any(np.isnan(optimised)):
        print("Warning: some coefficients are unavailable (XFOIL failed). "
              "Bars for NaN values will be empty.")

    fig = plt.figure()
    ax = plt.gca()
    x = np.arange(len(labels))
    width = 0.35
    b1 = ax.bar(x - width/2, baseline,  width, label="Baseline")
    b2 = ax.bar(x + width/2, optimised, width, label="Optimised")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coefficient value")
    ax.set_title(f"Aerodynamic coefficients at α={alpha}°")
    ax.grid(True, axis="y", linestyle=":")
    ax.legend()

    # annotate NaNs
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            if np.isnan(h):
                ax.text(rect.get_x() + rect.get_width()/2, 0.02, "n/a",
                        ha="center", va="bottom", fontsize=9, rotation=0)

    if outpath is None:
        fig_dir = _infer_output_dir(out_csv, "coefficients")
        outpath = os.path.join(fig_dir, "coeff_bar.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath



def plot_polar(best_vec, Re=1e6, out_csv: str = None):
    # Write .dat for baseline and optimised
    os.makedirs(os.path.join("data", "airfoils"), exist_ok=True)
    base_dat = os.path.join("data", "airfoils", "baseline.dat")
    opt_dat = os.path.join("data", "airfoils", "optimised.dat")

    make_airfoil(BASELINE_AU, BASELINE_AL, base_dat)
    Au_opt = best_vec[:3]
    Al_opt = best_vec[3:]
    make_airfoil(Au_opt, Al_opt, opt_dat)

    # XFOIL polars
    a0, Cl0, Cd0, Cm0 = run_xfoil_polar(base_dat, a_start=0.0, a_end=6.0, a_step=0.5, Re=Re)
    a1, Cl1, Cd1, Cm1 = run_xfoil_polar(opt_dat, a_start=0.0, a_end=6.0, a_step=0.5, Re=Re)

    # Drag polar Cd vs Cl
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(Cd0, Cl0, "o-", label="Baseline")
    ax.plot(Cd1, Cl1, "s--", label="Optimised")
    ax.set_xlabel("$C_d$")
    ax.set_ylabel("$C_l$")
    ax.set_title("Drag polar comparison")
    ax.grid(True, linestyle=":")
    ax.legend()

    fig_dir = _infer_output_dir(out_csv, "polar")
    outpath = os.path.join(fig_dir, "drag_polar.png")
    _ensure_dir(outpath)
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
    geom_png = plot_geometry(best_vec)
    print("Saved geometry comparison:", geom_png)

    coeff_png = plot_coeff_bar(best_vec, Re=1e6, alpha=3.0)
    print("Saved coefficients bar chart:", coeff_png)


    # 4) Drag polar comparison
    try:
        polar_png = plot_polar(best_vec, Re=1e6)
        print("Saved polar:", polar_png)
    except Exception as e:
        print("Warning: polar plot failed, skipping. Reason:", e)



if __name__ == "__main__":
    main()
