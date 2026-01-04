import os
from datetime import datetime
import json
import numpy as np

from optimizer.pso import PSO
from benchmarks.airfoil_xfoil import airfoil_fitness
from .run_opt import optimize   # reuse Part A harness


def main():
    # 6 CST coefficients: 3 upper, 3 lower
    bounds = [(-0.2, 0.5)] * 6
    options = dict(pop=10, w=0.7, c1=1.6, c2=1.6)

    seed = 1
    eval_budget = 200

    opt = PSO(bounds=bounds, seed=seed, options=options)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(
        "data", "results", "airfoil",
        f"airfoil_opt_seed{seed}_{stamp}.csv"
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    best = optimize(
        lambda v: airfoil_fitness(np.array(v), Re=1e6, alpha=3.0),
        opt,
        eval_budget=eval_budget,
        log_path=out_csv,
    )
    print("Best design:", best)

    # Save best design vector for plotting
    best_json = os.path.splitext(out_csv)[0] + "_best.json"
    with open(best_json, "w") as f:
        json.dump({"x": best["x"].tolist(), "f": best["f"]}, f, indent=2)
    print("Saved best design to", best_json)


if __name__ == "__main__":
    main()
