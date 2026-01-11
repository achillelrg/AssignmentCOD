
import os
import glob
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
try:
    from benchmarks.airfoil_xfoil import airfoil_fitness
    from utils.cst import cst_airfoil
except ImportError:
    # Fallback if run from wrong dir
    sys.path.append(os.path.join(os.getcwd(), ".."))
    from benchmarks.airfoil_xfoil import airfoil_fitness
    from utils.cst import cst_airfoil

def plot_airfoil_shape(x_vec, metrics, title, out_path):
    """Generate and save airfoil shape plot with metrics annotation."""
    # Reconstruct Geometry
    n_vars = len(x_vec)
    coeffs_u = x_vec[:n_vars//2]
    coeffs_l = x_vec[n_vars//2:]
    
    xu, yu, xl, yl = cst_airfoil(100, coeffs_u, coeffs_l, dz_te=0.0)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xu, yu, 'b-', linewidth=2, label="Upper")
    ax.plot(xl, yl, 'r-', linewidth=2, label="Lower")
    ax.fill_between(xu, yu, yl, color='gray', alpha=0.1)
    
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Annotation Box
    text_str = (
        f"Cl: {metrics['Cl']:.4f}\n"
        f"Cd: {metrics['Cd']:.5f}\n"
        f"Cm: {metrics['Cm']:.4f}\n"
        f"L/D: {metrics['L/D']:.1f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    print("==================================================")
    print("   GENERATING PERFORMANCE GALLERY (Plots)         ")
    print("==================================================")

    # 1. Find all Result JSONs
    # Pattern: data/PartB/*/results/*_best.json
    # Pattern: data/PartB/*/results/*_best.json (nested)
    
    search_patterns = [
        "data/PartB/*/results/*_best.json",
        "data/PartC/*.json" # surrogate_comparison.json
    ]
    
    found_files = []
    for pat in search_patterns:
        found_files.extend(glob.glob(pat))
        
    records = []
    
    for json_path in found_files:
        if "metrics" in json_path: continue # Skip existing metrics files
        if "comparison" in json_path:
             # Part C Comparison File
             with open(json_path, "r") as f:
                 data = json.load(f)
             # "actual": {"cl": ...}
             act = data.get("actual", {})
             rec = {
                 "File": os.path.basename(json_path),
                 "Part": "C (Surrogate)",
                 "Method": "L-BFGS-B (Surrogate)",
                 "Cl": act.get("cl"),
                 "Cd": act.get("cd"),
                 "Cm": act.get("Cm") or act.get("cm"),
                 "L/D": act.get("cl") / act.get("cd") if act.get("cd") else 0,
                 "Fitness": "N/A"
             }
             records.append(rec)
             continue

        # Standard _best.json
        print(f"Processing: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
            
        x = data.get("x")
        f_val = data.get("f")
        
        if not x:
              print(" -> Validation Error: No 'x' vector found.")
              continue
              
        # Re-Evaluate
        try:
             J_new, Cl, Cd, Cm = airfoil_fitness(x, return_all=True)
             
             # Save side-car metrics file
             metrics_path = json_path.replace("_best.json", "_metrics.json")
             metrics_data = {
                 "Cl": Cl, "Cd": Cd, "Cm": Cm, "L/D": Cl/Cd if Cd else 0, "J_recalc": J_new
             }
             with open(metrics_path, "w") as f_out:
                 json.dump(metrics_data, f_out, indent=4)
                 
             # Determine Method from filename
             method = "Unknown"
             if "pso" in json_path.lower(): method = "PSO"
             elif "ga" in json_path.lower(): 
                 # Check pop to distinguish Fidelity
                 if "pop40" in json_path: method = "GA (High-Fidelity)"
                 elif "pop20" in json_path: method = "GA (Med-Fidelity)"
                 elif "pop10" in json_path: method = "GA (Low-Fidelity)"
                 else: method = "GA"
                 
             rec = {
                 "File": os.path.basename(json_path),
                 "Part": "B (Direct)",
                 "Method": method,
                 "Cl": Cl,
                 "Cd": Cd,
                 "Cm": Cm,
                 "L/D": Cl/Cd if Cd else 0,
                 "Fitness": f_val
             }
             records.append(rec)
             
             # --- Plotting ---
             # Determine output path: data/PartB/{method}/figures/performance/
             # json_path is .../data/PartB/ga/results/foo.json
             # We want .../data/PartB/ga/figures/performance/foo_shape.png
             
             base_dir = os.path.dirname(os.path.dirname(json_path)) # data/PartB/ga
             fig_dir = os.path.join(base_dir, "figures", "performance")
             
             base_name = os.path.splitext(os.path.basename(json_path))[0]
             plot_path = os.path.join(fig_dir, f"{base_name}_shape.png")
             
             plot_title = f"{method} Result\nFitness: {f_val:.4f}"
             metric_dict = {"Cl": Cl, "Cd": Cd, "Cm": Cm, "L/D": Cl/Cd if Cd else 0}
             
             plot_airfoil_shape(x, metric_dict, plot_title, plot_path)
             print(f"   -> Plot: {plot_path}")
             
        except Exception as e:
             print(f" -> Evaluation/Plotting Failed: {e}")

    # 2. specific Part C Check
    # Look for surrogate_comparison.json
    c_path = "data/PartC/surrogate_comparison.json"
    if os.path.exists(c_path):
         with open(c_path, "r") as f:
             data = json.load(f)
             
         x_opt = data.get("x_opt")
         act = data.get("actual", {})
         
         if x_opt and act:
             print(f"Processing: {c_path}")
             
             fig_dir = "data/PartC/figures/performance"
             plot_path = os.path.join(fig_dir, "surrogate_opt_shape.png")
             
             cl = act.get("cl") or 0.0
             cd = act.get("cd") or 1.0 # avoid div/0
             cm = act.get("cm") or 0.0
             
             metrics = {
                 "Cl": cl, "Cd": cd, "Cm": cm, 
                 "L/D": cl/cd
             }
             
             plot_airfoil_shape(x_opt, metrics, "Part C: Surrogate Optimization", plot_path)
             print(f"   -> Plot: {plot_path}")
    
    # 3. Create Summary Table
    df = pd.DataFrame(records)
    if df.empty:
        print("No results found.")
        return

    # Sort by Part then Method (Manual Sort)
    records.sort(key=lambda x: (x["Part"], x["Method"]))
    
    # Manual Markdown Table Construction
    headers = ["Part", "Method", "Cl", "Cd", "Cm", "L/D", "Fitness", "File"]
    # width formatting
    row_fmt = "| {:<15} | {:<20} | {:<8.4f} | {:<8.5f} | {:<8.4f} | {:<8.2f} | {:<10} | {:<30} |"
    header_fmt = "| {:<15} | {:<20} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10} | {:<30} |"
    
    lines = []
    lines.append(header_fmt.format(*headers))
    lines.append("|" + "---|"*len(headers))
    
    for r in records:
        f_val_str = f"{r['Fitness']:.4f}" if isinstance(r['Fitness'], float) else r['Fitness']
        lines.append(row_fmt.format(
            r["Part"], r["Method"], 
            r["Cl"] or 0, r["Cd"] or 0, r["Cm"] or 0, r["L/D"] or 0,
            f_val_str, r["File"]
        ))
        
    md_table = "\n".join(lines)
    
    report_path = os.path.join("data", "PartB", "PERFORMANCE_SUMMARY.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Aerodynamic Performance Summary\n\n")
        f.write("Systematic evaluation of best designs found by each optimizer.\n\n")
        f.write(md_table)
        f.write("\n\n*Generated by experiments/generate_performance_report.py*")
        
    print(f"\n✅ Report Generated: {report_path}")
    print(md_table)

if __name__ == "__main__":
    main()
