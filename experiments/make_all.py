import os, glob, subprocess, sys

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    # 1) multi-seed runs on D=5
    for s in range(1, 11):
        run([sys.executable, "-m", "experiments.run_opt",
             "--D", "5", "--pop", "40", "--evals", "50000", "--seed", str(s)])

    # 2) aggregate figures
    run([sys.executable, "-m", "experiments.make_figures",
         "--pattern", os.path.join("experiments", "griewank_D5_pop40_*.csv")])

    # 3) one 2D swarm trajectory
    run([sys.executable, "-m", "experiments.run_opt",
         "--D", "2", "--pop", "40", "--evals", "20000", "--seed", "1", "--trace_every", "5"])

if __name__ == "__main__":
    main()
