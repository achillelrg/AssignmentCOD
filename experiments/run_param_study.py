import os
import csv
import json
import numpy as np
import argparse
from datetime import datetime

# Import optimizers
from optimizer.pso import PSO
from optimizer.ga import GA as GeneticAlgo

# Import benchmark
from benchmarks.airfoil_xfoil import airfoil_fitness
from experiments.run_opt import optimize

def run_trial(algo_cls, options, seed, evals, label):
    print(f"--- Running {label} (Seed {seed}) ---")
    
    # 6 CST params
    bounds = [(-0.2, 0.5)] * 6
    
    # Instantiate optimizer
    opt = algo_cls(bounds=bounds, seed=seed, options=options)
    
    # Log file
    os.makedirs(os.path.join("data", "PartB", "study"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{label.replace(' ', '_')}_seed{seed}_{stamp}.csv"
    log_path = os.path.join("data", "PartB", "study", log_name)
    
    # Run optimization (using alpha=3.0, Re=1e6 as per requirement B.3)
    best = optimize(
        lambda v: airfoil_fitness(np.array(v), Re=1e6, alpha=3.0),
        opt,
        eval_budget=evals,
        log_path=log_path,
        f_target=-1e9 # No early exit on target, run full budget
    )
    
    return best, log_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals", type=int, default=1000, help="Evaluations per trial")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds/trials per setting")
    parser.add_argument("--test", action="store_true", help="Run shortened smoke test")
    args = parser.parse_args()
    
    evals = 20 if args.test else args.evals
    n_seeds = 1 if args.test else args.seeds
    
    results = []

    # 1. PSO Baseline
    # "Standard" settings from Part A
    pso_opts = {"pop": 20, "w": 0.7, "c1": 1.6, "c2": 1.6}
    
    # 2. GA Settings to Investigate (B.4)
    # Effect of Pop Size: 20 vs 50
    ga_settings = [
        ("GA_Pop20", {"pop": 20, "mutation_rate": 0.1, "crossover_rate": 0.9}),
        ("GA_Pop50", {"pop": 50, "mutation_rate": 0.1, "crossover_rate": 0.9}),
        # Effect of Mutation: High mutation (exploration)
        ("GA_MutHigh", {"pop": 20, "mutation_rate": 0.3, "crossover_rate": 0.7}),
    ]
    
    # List of configs
    configs = [("PSO_Baseline", PSO, pso_opts)] + \
              [(name, GeneticAlgo, opts) for name, opts in ga_settings]
              
    for label, cls, opts in configs:
        for s in range(n_seeds):
            seed = s + 42 # Reproducible seeds
            
            try:
                best, log_path = run_trial(cls, opts, seed, evals, label)
                
                best_x_list = list(best["x"]) if best["x"] is not None else []
                results.append({
                    "config": label,
                    "seed": seed,
                    "best_f": best["f"],
                    "best_x": best_x_list,
                    "log_path": log_path
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Trial failed: {e}")
                
    # Save summary
    summary_path = os.path.join("data", "PartB", "study", f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Study complete. Summary saved to {summary_path}")
    
    # Print comparison
    print("\n=== Summary of Results ===")
    import collections
    grouped = collections.defaultdict(list)
    for r in results:
        grouped[r["config"]].append(r["best_f"])
        
    for cfg, vals in grouped.items():
        print(f"{cfg}: Mean Best J = {np.mean(vals):.6f} (Min: {np.min(vals):.6f})")

if __name__ == "__main__":
    main()
