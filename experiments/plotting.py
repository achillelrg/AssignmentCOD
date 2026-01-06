import os
import glob
import math
import re
from typing import List, Tuple, Dict

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

BASE_FIG_DIR = os.path.join("data", "PartB", "figures")


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def read_log(csv_path: str):
    """Read one run CSV written by experiments/run_opt.py and coerce columns."""
    if pd is None:
        raise ImportError("pandas is required for reading logs but is not installed.")
    df = pd.read_csv(csv_path)
    # ensure numeric (they were formatted as strings for pretty printing)
    for c in ["f_best", "f_mean", "f_std", "gbest_f"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def parse_meta_from_filename(path: str) -> Dict[str, str]:
    """
    Extract D, pop, seed from a filename like griewank_D5_pop40_seed7_YYYYMMDD_*.csv
    """
    name = os.path.basename(path)
    meta = {}
    mD = re.search(r"D(\d+)", name)
    mp = re.search(r"pop(\d+)", name)
    ms = re.search(r"seed(\d+)", name)
    if mD: meta["D"] = mD.group(1)
    if mp: meta["pop"] = mp.group(1)
    if ms: meta["seed"] = ms.group(1)
    meta["base"] = os.path.splitext(name)[0]
    return meta

def plot_convergence(csv_path: str, outpath: str = None, ykey: str = "gbest_f"):
    """
    Semilogy convergence curve for a single run.
    ykey in {"gbest_f", "f_best"} – gbest_f is global best so far (preferred).
    """
    if plt is None or pd is None:
        print("Skipping plot_convergence (matplotlib/pandas missing)")
        return None
        
    df = read_log(csv_path)
    fig = plt.figure()
    ax = plt.gca()
    vals = df[ykey]
    if (vals <= 0).any():
        ax.plot(df["iter"], vals)
    else:
        ax.semilogy(df["iter"], vals)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best objective value")
    ax.grid(True, which="both", linestyle=":")
    meta = parse_meta_from_filename(csv_path)
    title = f"Convergence (Griewank, D={meta.get('D','?')}, pop={meta.get('pop','?')}, seed={meta.get('seed','?')})"
    ax.set_title(title)

    if outpath is None:
        # Instead of hardcoded BASE_FIG_DIR, try to save near the CSV if possible, or use a default.
        # Current behavior: hardcoded BASE_FIG_DIR = data/figures.
        # New behavior: if csv_path is in data/PartA/, we might want data/PartA/figures/convergence?
        # Let's check where the CSV is.
        csv_dir = os.path.dirname(csv_path)
        # simplistic heuristic: if "Part" is in the path, use that structure
        if "Part" in csv_dir:
            # e.g. data/PartA/results or data/PartA
            if os.path.basename(csv_dir) == "results":
                base_dir = os.path.dirname(csv_dir) # up one level -> data/PartA
            else:
                base_dir = csv_dir
            fig_dir = os.path.join(base_dir, "figures", "convergence")
        else:
            # fallback to global
            fig_dir = os.path.join(BASE_FIG_DIR, "convergence")
            
        outpath = os.path.join(fig_dir, f"{meta['base']}_{ykey}_conv.png")
        
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_convergence_overlay(csv_paths: List[str], outpath: str = None, ykey: str = "gbest_f"):
    """
    Overlay multiple runs (e.g., different seeds) on one semilogy plot.
    """
    if plt is None or pd is None:
        return None

    fig = plt.figure()
    ax = plt.gca()
    for p in csv_paths:
        df = read_log(p)
        ax.semilogy(df["iter"], df[ykey], alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best objective value")
    ax.grid(True, which="both", linestyle=":")
    ax.set_title("Convergence overlay (multiple seeds)")

    if outpath is None:
        outpath = os.path.join(BASE_FIG_DIR, "overlays", "convergence_overlay.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_final_boxplot(csv_paths: List[str], outpath: str = None, key: str = "gbest_f"):
    """
    Boxplot of final best objective across runs/seeds.
    """
    if plt is None or pd is None:
        return None

    finals = []
    for p in csv_paths:
        df = read_log(p)
        finals.append(df[key].iloc[-1])
    data = np.array(finals, dtype=float)

    fig = plt.figure()
    ax = plt.gca()
    ax.boxplot(data, vert=True, showmeans=True)
    ax.set_xticks([1])
    ax.set_xticklabels([f"{len(csv_paths)} runs"])
    ax.set_ylabel("Final best objective")
    ax.set_title("Distribution of final best objective")
    ax.grid(True, axis="y", linestyle=":")

    if outpath is None:
        outpath = os.path.join(BASE_FIG_DIR, "overlays", "final_boxplot.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_success_vs_budget(csv_paths: List[str], thresholds: List[float] = [1e-6, 1e-4, 1e-2],
                           outpath: str = None, ykey: str = "gbest_f"):
    """
    Empirical success curve: for each evaluation count, share of runs that have reached threshold.
    Works best if runs share pop size so evals = iter*pop is roughly comparable.
    """
    if plt is None or pd is None:
        return None

    # Align by evals
    all_evals = set()
    series = []
    for p in csv_paths:
        df = read_log(p)
        all_evals.update(df["evals"].tolist())
        series.append(df[["evals", ykey]].copy())
    grid = np.array(sorted(all_evals), dtype=int)

    # For each run, map best-so-far at each eval grid via forward fill
    stacked = []
    for s in series:
        s2 = s.set_index("evals").reindex(grid).ffill()
        stacked.append(s2[ykey].to_numpy())
    M = np.vstack(stacked)  # shape: (runs, len(grid))

    fig = plt.figure()
    ax = plt.gca()
    for thr in thresholds:
        success = np.mean(M <= thr, axis=0)
        ax.plot(grid, success)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=":")
    ax.set_title("Empirical success vs evaluation budget")
    ax.legend([f"f ≤ {t:g}" for t in thresholds])

    if outpath is None:
        outpath = os.path.join(BASE_FIG_DIR, "overlays", "success_vs_budget.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_swarm_2d(csv_path: str, positions_snapshots: List[np.ndarray], outpath: str = None):
    """
    Optional (only if D=2): provide pre-captured positions (list per iter).
    This helper just plots trajectory clouds.
    """
    if plt is None:
        return None

    fig = plt.figure()
    ax = plt.gca()
    for pts in positions_snapshots:
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.4)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("PSO swarm trajectory (D=2)")
    ax.grid(True, linestyle=":")

    if outpath is None:
        meta = parse_meta_from_filename(csv_path)
        outpath = os.path.join(BASE_FIG_DIR, "swarm", f"{meta['base']}_swarm2d.png")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath
