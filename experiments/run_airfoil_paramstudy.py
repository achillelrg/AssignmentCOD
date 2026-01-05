from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

from optimizers.pso import pso_optimize
from utils.airfoil_problem import AirfoilConfig, evaluate_airfoil_theta
from experiments.run_airfoil import make_bounds, surrogate_objective, USE_SURROGATE


@dataclass
class PSOConfig:
    name: str
    n_particles: int
    n_iters: int
    inertia: float
    cognitive: float
    social: float


def get_configs() -> List[PSOConfig]:
    """
    A few PSO settings to compare for B.4.
    Tune these if you like.
    """
    return [
        PSOConfig("small_swarm_explore", n_particles=20, n_iters=60,
                  inertia=0.9, cognitive=1.4, social=1.4),
        PSOConfig("baseline", n_particles=40, n_iters=60,
                  inertia=0.72, cognitive=1.4, social=1.4),
        PSOConfig("large_swarm_exploit", n_particles=80, n_iters=60,
                  inertia=0.6, cognitive=1.4, social=1.4),
    ]


def make_eval_fn(cfg: AirfoilConfig):
    """
    Wrapper that chooses between surrogate and XFOIL evaluation
    using the same logic as run_airfoil.py.
    """

    if USE_SURROGATE:
        def eval_fn(theta: np.ndarray) -> Tuple[float, dict]:
            return surrogate_objective(theta)
    else:
        def eval_fn(theta: np.ndarray) -> Tuple[float, dict]:
            return evaluate_airfoil_theta(theta, cfg)

    return eval_fn


def main():
    cfg = AirfoilConfig()
    bounds = make_bounds()
    eval_fn = make_eval_fn(cfg)

    # Where to store multi-run study results
    base_dir = Path("data/results/airfoil/multi")
    base_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_dir / "airfoil_pso_paramstudy.csv"

    # Prepare CSV
    with summary_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "config_name",
                "seed",
                "n_particles",
                "n_iters",
                "inertia",
                "cognitive",
                "social",
                "f_best",
                "success_rate",  # simple boolean flag for surrogate/XFOIL success
            ]
        )

        for cfg_pso in get_configs():
            print(f"\n=== Running config: {cfg_pso.name} ===")
            for seed in [1, 2, 3, 4, 5]:
                print(f"  - seed {seed}")

                # Define objective for this run
                def objective(theta: np.ndarray) -> float:
                    f_val, info = eval_fn(theta)
                    # you could store per-eval logs, but for B.4 a summary is enough
                    return f_val

                result: Dict = pso_optimize(
                    objective,
                    bounds,
                    n_particles=cfg_pso.n_particles,
                    n_iters=cfg_pso.n_iters,
                    inertia=cfg_pso.inertia,
                    cognitive=cfg_pso.cognitive,
                    social=cfg_pso.social,
                    seed=seed,
                )

                f_best = float(result["f_best"])
                # For surrogate we assume success; for XFOIL you could
                # propagate a success flag in eval_fn if you want.
                success_rate = 1.0

                writer.writerow(
                    [
                        cfg_pso.name,
                        seed,
                        cfg_pso.n_particles,
                        cfg_pso.n_iters,
                        cfg_pso.inertia,
                        cfg_pso.cognitive,
                        cfg_pso.social,
                        f_best,
                        success_rate,
                    ]
                )

    print(f"\nParameter study summary written to: {summary_path}")


if __name__ == "__main__":
    main()
