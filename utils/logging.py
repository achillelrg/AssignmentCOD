from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


# Project root: .../COD_Assignment
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = DATA_ROOT / "results"
FIGURES_ROOT = DATA_ROOT / "figures"


@dataclass
class RunConfig:
    """Minimal run configuration metadata to store with each run."""
    domain: str          # "pso" or "airfoil"
    problem: str         # e.g. "griewank_2d", "griewank_5d"
    algorithm: str       # e.g. "pso"
    n_particles: int
    n_iters: int
    dim: int
    seed: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_run_dir(domain: str, problem: str, mode: str = "single") -> Path:
    """
    Create and return a unique directory for a single run.

    Structure:
        data/results/{domain}/{mode}/run_YYYYmmdd_HHMMSS_XXXX/

    domain : "pso" or "airfoil"
    problem : short problem name, used later in summaries
    mode : "single" or "multi"
    """
    if domain not in {"pso", "airfoil"}:
        raise ValueError(f"Unknown domain: {domain}")
    if mode not in {"single", "multi"}:
        raise ValueError(f"Unknown mode: {mode}")

    base = RESULTS_ROOT / domain / mode
    _ensure_dir(base)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add a short random suffix based on nanoseconds to avoid collisions
    suffix = datetime.now().strftime("%f")[-4:]
    run_name = f"run_{timestamp}_{suffix}"

    run_dir = base / run_name
    _ensure_dir(run_dir)

    # Store problem name in a small marker file (optional but handy)
    (run_dir / "problem.txt").write_text(problem)

    return run_dir


def save_convergence_csv(
    run_dir: Path,
    best_history: Sequence[float],
    mean_history: Sequence[float],
) -> Path:
    """
    Save convergence history to CSV:
        iter, f_best, f_mean
    """
    path = run_dir / "convergence.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "f_best", "f_mean"])
        for i, (b, m) in enumerate(zip(best_history, mean_history)):
            writer.writerow([i, b, m])
    return path


def save_swarm_2d_csv(
    run_dir: Path,
    swarm_history: Sequence[Any],
) -> Path:
    """
    Save 2D swarm positions over time.

    swarm_history: list of arrays of shape (n_particles, 2)

    CSV columns:
        iter, particle, x1, x2
    """
    path = run_dir / "swarm_2d.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "particle", "x1", "x2"])
        for it, swarm in enumerate(swarm_history):
            # Convert to list of rows
            for pid, pos in enumerate(swarm):
                writer.writerow([it, pid, float(pos[0]), float(pos[1])])
    return path


def save_run_metadata(run_dir: Path, config: RunConfig, extra: Dict[str, Any] | None = None) -> Path:
    """
    Save run configuration and optional extra info to metadata.json.
    """
    meta: Dict[str, Any] = asdict(config)
    if extra:
        meta.update(extra)

    path = run_dir / "metadata.json"
    with path.open("w") as f:
        json.dump(meta, f, indent=2)
    return path


def append_multi_summary(
    domain: str,
    problem: str,
    rows: Iterable[Dict[str, Any]],
    filename: str = "summary.csv",
) -> Path:
    """
    Append summary rows for multiple runs.

    Each row should be a flat dict with consistent keys.

    File location:
        data/results/{domain}/multi/{problem}_{filename}
    """
    base = RESULTS_ROOT / domain / "multi"
    _ensure_dir(base)

    path = base / f"{problem}_{filename}"

    rows = list(rows)
    if not rows:
        return path

    # Ensure consistent column order
    fieldnames: List[str] = sorted(rows[0].keys())

    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path

