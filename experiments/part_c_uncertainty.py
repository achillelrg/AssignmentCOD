
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.cst import cst_airfoil
from utils.geometry import write_dat
from utils.xfoil_runner import run_xfoil_single_alpha

def main():
    print("--- Part C.1: Uncertainty Analysis ---")
    
    # 1. Load Best Design from Part C (Surrogate)
    # Because regression test wiped Part B, we use Surrogate Opt.
    surro_json = os.path.join("data", "PartC", "surrogate", "surrogate_results.json")
    
    if os.path.exists(surro_json):
        print(f"Loading surrogate design from: {surro_json}")
        with open(surro_json, "r") as f:
            data = json.load(f)
        x_opt = np.array(data["x_opt_surro"])
    else:
        # Fallback to Part B
        pattern = os.path.join("data", "PartB", "results", "*_best.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print("Error: No design found.")
            return
        print(f"Loading Part B design from: {files[-1]}")
        with open(files[-1], "r") as f:
            data = json.load(f)
        x_opt = np.array(data["x"])
    
    # 2. Generate random Alphas
    np.random.seed(42)
    mean_alpha = 3.0
    std_alpha = 0.1
    n_samples = 100
    
    alphas = np.random.normal(mean_alpha, std_alpha, n_samples)
    
    # 3. Prepare Airfoil File (once)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(root, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    x, yu, xl, yl = cst_airfoil(200, x_opt[:3], x_opt[3:])
    
    dat_path = os.path.join(temp_dir, "robust_candidate.dat")
    # Write standard dat
    xu = x[::-1]; yup = yu[::-1]
    xl_p = x[1:]; ylo_p = yl[1:]
    xx = np.concatenate([xu, xl_p])
    yy = np.concatenate([yup, ylo_p])
    write_dat(xx, yy, dat_path, name="ROBUST_TEST")
    
    # 4. Evaluate
    print(f"Evaluating {n_samples} samples (Alpha ~ N({mean_alpha}, {std_alpha}))...")
    
    results = []
    failed = 0
    
    # 4. Evaluate Parallel
    from multiprocessing import Pool
    from functools import partial
    
    print(f"Evaluating {n_samples} samples (Alpha ~ N({mean_alpha}, {std_alpha})) using 8 workers...")
    
    # We need a helper to freeze dat_path
    # run_xfoil_single_alpha(dat_path, alpha=a, Re=1e6, n_iter=200)
    
    eval_fn = partial(run_xfoil_single_alpha, dat_path, Re=1e6, n_iter=200)
    
    with Pool(processes=8) as pool:
        # returns list of (cl, cd, cm) tuples
        raw_res = pool.map(eval_fn, alphas)
        
    results = []
    failed = 0
    for i, res in enumerate(raw_res):
        a = alphas[i]
        cl, cd, cm = res
        if cl is not None:
             results.append([a, cl, cd, cm])
        else:
             failed += 1
            
    print(f"Completed. Failures: {failed}/{n_samples}")
    
    if not results:
        print("All runs failed. Cannot compute stats.")
        return

    res_arr = np.array(results)
    # Col 0: alpha, 1: cl, 2: cd, 3: cm
    
    # 5. Statistics
    mu_cl, std_cl = np.mean(res_arr[:,1]), np.std(res_arr[:,1])
    mu_cd, std_cd = np.mean(res_arr[:,2]), np.std(res_arr[:,2])
    mu_cm, std_cm = np.mean(res_arr[:,3]), np.std(res_arr[:,3])
    
    print("-" * 40)
    print(f"Uncertainty Results (Alpha std={std_alpha} deg)")
    print(f"CL: Mean = {mu_cl:.4f}, Std = {std_cl:.4f}, COV = {std_cl/abs(mu_cl):.4f}")
    print(f"CD: Mean = {mu_cd:.4f}, Std = {std_cd:.4f}, COV = {std_cd/abs(mu_cd):.4f}")
    print(f"CM: Mean = {mu_cm:.4f}, Std = {std_cm:.4f}")
    print("-" * 40)
    
    # Save results
    out_dir = os.path.join("data", "PartC", "uncertainty")
    os.makedirs(out_dir, exist_ok=True)
    
    report_file = os.path.join(out_dir, "uncertainty_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Uncertainty Analysis for Surrogate Design\n")
        f.write(f"Samples: {n_samples}, Failures: {failed}\n")
        f.write(f"CL: Mean = {mu_cl:.6f}, Std = {std_cl:.6f}\n")
        f.write(f"CD: Mean = {mu_cd:.6f}, Std = {std_cd:.6f}\n")
        f.write(f"CM: Mean = {mu_cm:.6f}, Std = {std_cm:.6f}\n")
        
    # Plot Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(res_arr[:,1], bins=15, color='b', alpha=0.7)
    axes[0].set_title(f"Lift Coeff (Std={std_cl:.4f})")
    
    axes[1].hist(res_arr[:,2], bins=15, color='r', alpha=0.7)
    axes[1].set_title(f"Drag Coeff (Std={std_cd:.4f})")
    
    axes[2].hist(res_arr[:,3], bins=15, color='g', alpha=0.7)
    axes[2].set_title(f"Moment Coeff (Std={std_cm:.4f})")
    
    plt.suptitle(f"Robustness Check: Alpha ~ N(3.0, 0.1)")
    plot_path = os.path.join(out_dir, "robustness_histograms.png")
    plt.savefig(plot_path)
    print(f"Saved plots to {plot_path}")

if __name__ == "__main__":
    main()
