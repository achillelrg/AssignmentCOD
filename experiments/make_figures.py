import argparse
import glob
import os
from experiments.plotting import (
    plot_convergence,
    plot_convergence_overlay,
    plot_final_boxplot,
    plot_success_vs_budget,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        type=str,
        default=os.path.join("data", "results", "multi_runs", "griewank_*.csv"),
        help="Glob pattern to pick CSV logs (e.g., data\\results\\multi_runs\\griewank_D5_pop40_*.csv)",
    )
    ap.add_argument("--single", type=str, default=None, help="Path to one CSV for single-run convergence")
    args = ap.parse_args()

    csvs = sorted(glob.glob(args.pattern))
    if args.single:
        out = plot_convergence(args.single)   # saved under data/figures/convergence/
        print("Saved:", out)
    if len(csvs) >= 2:
        out1 = plot_convergence_overlay(csvs) # data/figures/overlays/
        out2 = plot_final_boxplot(csvs)       # data/figures/overlays/
        out3 = plot_success_vs_budget(csvs)   # data/figures/overlays/
        print("Saved:", out1)
        print("Saved:", out2)
        print("Saved:", out3)
    elif not args.single and len(csvs) == 1:
        out = plot_convergence(csvs[0])       # data/figures/convergence/
        print("Saved:", out)
    elif not csvs:
        print("No CSVs matched. Adjust --pattern.")

if __name__ == "__main__":
    main()
