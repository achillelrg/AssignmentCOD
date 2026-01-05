# experiments/run_opt.py
import argparse
import csv
import os
from datetime import datetime

import numpy as np

from benchmarks.griewank import griewank
from optimizer.pso import PSO
from experiments.plotting import plot_swarm_2d, plot_convergence  # for optional 2D swarm plot and convergence


def optimize(f, opt: PSO, eval_budget: int, f_target: float = 1e-6, stagnation: int = 200, log_path: str = "run.csv"):
    best_seen = np.inf
    no_improve = 0

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["iter","evals","f_best","f_mean","f_std","gbest_f"])
        writer.writeheader()

        while True:
            X = opt.ask()                     # list of np arrays
            F = [f(x) for x in X]             # evaluate fitness
            opt.tell(F)                       # inform optimiser

            st = opt.state()
            writer.writerow({
                "iter": st["iter"],
                "evals": st["evals_total"],
                "f_best": f"{st['f_best']:.12e}",
                "f_mean": f"{st['f_mean']:.12e}",
                "f_std": f"{st['f_std']:.12e}",
                "gbest_f": f"{st['gbest_f']:.12e}",
            })

            # stopping logic
            if st["gbest_f"] < best_seen - 1e-16:
                best_seen = st["gbest_f"]
                no_improve = 0
            else:
                no_improve += 1

            if st["evals_total"] >= eval_budget: break
            if st["gbest_f"] <= f_target: break
            if no_improve >= stagnation: break

    return opt.best()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=5, help="Dimension of Griewank")
    parser.add_argument("--pop", type=int, default=40)
    parser.add_argument("--evals", type=int, default=50000, help="Evaluation budget")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w", type=float, default=0.72)
    parser.add_argument("--c1", type=float, default=1.6)
    parser.add_argument("--c2", type=float, default=1.6)
    parser.add_argument("--vmax_frac", type=float, default=0.2)
    parser.add_argument("--topology", type=str, default="gbest")  # or lbest
    parser.add_argument("--trace_every", type=int, default=0, help="Record 2D swarm positions every N iters (D=2 only)")
    parser.add_argument("--part", type=str, default=None, help="Assignment Part (A, B, or C) to organize outputs")
    parser.add_argument("--out", type=str, default=None, help="CSV log path (overrides default data/results/... location)")
    args = parser.parse_args()

    bounds = [(-600.0, 600.0)] * args.D
    options = dict(
        pop=args.pop, w=args.w, c1=args.c1, c2=args.c2,
        vmax_frac=args.vmax_frac, topology=args.topology,
        trace_every=args.trace_every,
    )

    opt = PSO(bounds=bounds, seed=args.seed, options=options)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.out:
        # Respect explicit path
        log_path = args.out
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    else:
        # Default structured location
        if args.part:
            folder = os.path.join("data", f"Part{args.part}", "results")
        else:
            # Default to PartA for generic runs if not specified, or just "results"
            folder = os.path.join("data", "PartA", "results")
            
        os.makedirs(folder, exist_ok=True)
        log_path = os.path.join(
            folder,
            f"griewank_D{args.D}_pop{args.pop}_seed{args.seed}_{stamp}.csv"
        )

    best = optimize(griewank, opt, eval_budget=args.evals, log_path=log_path)
    print("Best:", best)

    # If 2D + tracing enabled, emit a swarm trajectory figure in data/figures/swarm/
    if args.D == 2 and args.trace_every > 0:
        trace = opt.positions_trace() if hasattr(opt, "positions_trace") else []
        if trace:
            # Save swarm plot in the SAME folder as the log, but with a suffix, or a 'figures' subfolder?
            # User wants "clarity". Saving in data/PartA/figures seems good, or just data/PartA/.
            # Let's keep it simple: data/figures/swarm is global.
            # BUT if --part is set, maybe we want data/PartA/figures?
            # For now, let's just stick to the requested structure: data/PartA/
            if args.part:
                swarm_dir = os.path.join("data", f"Part{args.part}", "figures")
            else:
                swarm_dir = os.path.join("data", "figures", "swarm")

            os.makedirs(swarm_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(log_path))[0]
            out_png = os.path.join(swarm_dir, f"{base}_swarm2d.png")
            plot_swarm_2d(log_path, trace, outpath=out_png)
            print("Saved 2D swarm trajectory:", out_png)

    # Always plot convergence
    conv_png = plot_convergence(log_path)
    print("Saved convergence plot:", conv_png)


if __name__ == "__main__":
    main()
