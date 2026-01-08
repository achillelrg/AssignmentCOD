# experiments/run_opt.py
import argparse
import csv
import os
import time
import multiprocessing
from datetime import datetime

import numpy as np

from benchmarks.griewank import griewank
from optimizer.pso import PSO
from experiments.plotting import plot_swarm_2d, plot_convergence  # for optional 2D swarm plot and convergence


def optimize(f, opt: PSO, eval_budget: int, f_target: float = 1e-6, stagnation: int = 200, log_path: str = "run.csv", n_jobs: int = 1):
    best_seen = np.inf
    no_improve = 0
    
    # Timing initialization
    start_time = time.time()
    from collections import deque
    window = deque(maxlen=8)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    
    # Initialize Pool if parallel
    pool = None
    if n_jobs > 1:
        print(f"--- Parallel Mode Enabled: Using {n_jobs} workers ---")
        pool = multiprocessing.Pool(processes=n_jobs)

    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        else:
            return f"{m}m {s:02d}s"

    try:
        with open(log_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["iter","evals","f_best","f_mean","f_std","gbest_f","x_best"])
            writer.writeheader()

            try:
                while True:
                    X = opt.ask()                     # list of np arrays
                    
                    # Evaluate fitness (Parallel or Serial)
                    import sys
                    current_iter = opt.state()['iter'] + 1
                    total_pop = len(X)
                    
                    # Evaluate fitness (Parallel or Serial)
                    F = []
                    if pool:
                        # Use imap to show progress within the generation
                        for i, res in enumerate(pool.imap(f, X), 1):
                            F.append(res)
                            sys.stdout.write(f"\r[Iter {current_iter}] Evaluating: {i}/{total_pop}")
                            sys.stdout.flush()
                    else:
                        for i, x in enumerate(X, 1):
                            F.append(f(x))
                            sys.stdout.write(f"\r[Iter {current_iter}] Evaluating: {i}/{total_pop}")
                            sys.stdout.flush()
                            
                    # Clear the progress line (overwrite with spaces then CR)
                    sys.stdout.write("\r" + " "*50 + "\r")
                    
                    opt.tell(F)                       # inform optimiser
                    
                    # Timing calculation
                    now = time.time()
                    elapsed = now - start_time
                    elapsed_str = format_time(elapsed)
                    
                    st = opt.state()
                    writer.writerow({
                        "iter": st["iter"],
                        "evals": st["evals_total"],
                        "f_best": f"{st['f_best']:.12e}",
                        "f_mean": f"{st['f_mean']:.12e}",
                        "f_std": f"{st['f_std']:.12e}",
                        "gbest_f": f"{st['gbest_f']:.12e}",
                        "x_best": str(list(st['gbest_x'])),
                    })
                    
                    # Dynamic ETA Calculation (Rolling Window)
                    evals_done = st['evals_total']
                    window.append((now, evals_done))
                    
                    eta_str = "..."
                    if len(window) > 1:
                         # Calculate rate based on window
                         t_start, e_start = window[0]
                         t_end, e_end = window[-1]
                         dt = t_end - t_start
                         de = e_end - e_start
                         
                         if dt > 0 and de > 0:
                             rate = de / dt # evals per second (recent)
                             evals_remaining = max(0, eval_budget - evals_done)
                             if evals_remaining > 0:
                                 eta_seconds = evals_remaining / rate
                                 eta_str = format_time(eta_seconds)
                             else:
                                 eta_str = "0m 00s"

                    # Simple progress logging
                    print(f"[Iter {st['iter']}] Evals: {st['evals_total']} | Best: {st['gbest_f']:.6e} | Mean: {st['f_mean']:.6e} | Elapsed: {elapsed_str} | ETA: {eta_str}")

                    # stopping logic
                    if st["gbest_f"] < best_seen - 1e-16:
                        best_seen = st["gbest_f"]
                        no_improve = 0
                    else:
                        no_improve += 1

                    if st["evals_total"] >= eval_budget: break
                    if st["gbest_f"] <= f_target: break
                    if no_improve >= stagnation: break

            except KeyboardInterrupt:
                print("\n!!! Interrupted by user. Stopping optimization early and saving current results... !!!")
    
    finally:
        # Cleanup pool
        if pool:
            pool.close()
            pool.join()


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
    parser.add_argument("--clean", action="store_true", help="Delete existing data folder for this Part before running")
    args = parser.parse_args()

    # Handle cleaning request
    if args.clean:
        if args.part:
             target_clean = os.path.join("data", f"Part{args.part}")
             print(f"Cleaning {target_clean}...")
             if os.path.exists(target_clean):
                 import shutil
                 shutil.rmtree(target_clean)
        else:
             print("Warning: --clean flag ignored because --part was not specified. Identifying correct folder is ambiguous.")


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
            # User wants "clarity".
            if args.part:
                swarm_dir = os.path.join("data", f"Part{args.part}", "figures")
            else:
                # Try to deduce from log_path if possible, otherwise default
                log_dir = os.path.dirname(log_path)
                if "Part" in log_dir:
                     # e.g. data/PartA/results -> data/PartA/figures
                     base = os.path.dirname(log_dir)
                     swarm_dir = os.path.join(base, "figures", "swarm")
                else:
                     # fallback
                     swarm_dir = os.path.join("data", "PartB", "figures", "swarm")

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
