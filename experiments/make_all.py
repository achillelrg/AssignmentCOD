import os
import glob
import subprocess
import sys


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)


def ensure_dirs():
    """
    Make sure the standard data/ directories exist.
    """
    for d in [
        os.path.join("data", "results", "single_runs"),
        os.path.join("data", "results", "multi_runs"),
        os.path.join("data", "figures", "convergence"),
        os.path.join("data", "figures", "overlays"),
        os.path.join("data", "figures", "swarm"),
    ]:
        os.makedirs(d, exist_ok=True)


def main():
    ensure_dirs()

    # -------------------------------------------------------
    # 1) Canonical single run (D=5, seed=42)
    # -------------------------------------------------------
    run([
        sys.executable, "-m", "experiments.run_opt",
        "--D", "5", "--pop", "40", "--evals", "50000", "--seed", "42"
    ])

    # Find the CSV we just created (latest matching file)
    single_glob = os.path.join(
        "data", "results", "single_runs",
        "griewank_D5_pop40_seed42_*.csv",
    )
    single_files = sorted(glob.glob(single_glob))
    if single_files:
        latest_single = single_files[-1]
        run([
            sys.executable, "-m", "experiments.make_figures",
            "--single", latest_single
        ])
    else:
        print("No single-run file found for seed 42 – skipping single-run convergence plot.")

    # -------------------------------------------------------
    # 2) Multi-seed runs (D=5, seeds 1..10)
    # -------------------------------------------------------
    for s in range(1, 11):
        run([
            sys.executable, "-m", "experiments.run_opt",
            "--D", "5", "--pop", "40", "--evals", "50000", "--seed", str(s)
        ])

    multi_pattern = os.path.join(
        "data", "results", "multi_runs",
        "griewank_D5_pop40_*.csv"
    )
    run([
        sys.executable, "-m", "experiments.make_figures",
        "--pattern", multi_pattern
    ])

    # -------------------------------------------------------
    # 3) One 2D swarm trajectory (illustration only)
    # -------------------------------------------------------
    run([
        sys.executable, "-m", "experiments.run_opt",
        "--D", "2", "--pop", "40", "--evals", "20000",
        "--seed", "1", "--trace_every", "5"
    ])


if __name__ == "__main__":
    main()
