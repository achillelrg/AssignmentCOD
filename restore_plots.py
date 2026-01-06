
import pandas as pd
import json
import os
import glob
import numpy as np

# Find latest CSV
pattern = os.path.join("data", "PartB", "results", "*.csv")
files = sorted(glob.glob(pattern))
if not files:
    print("No CSV found.")
    exit(1)
csv_path = files[-1]
print(f"Restoring from {csv_path}")

# Read CSV
df = pd.read_csv(csv_path)

# Find best row (min gbest_f)
# Note: The CSV logs 'gbest_f' which is the scalarized fitness
best_row = df.loc[df['gbest_f'].idxmin()]

# We don't have the design vector 'x' in the CSV...
# The CSV only logs metrics, NOT the vector (x).
# This is a flaw in run_opt.py logging! It logs statistics but not the solution itself.
# The solution is ONLY saved in the JSON.
# If the JSON is missing, the solution is LOST.

print("CRITICAL: CSV does not contain design vector.")
print("Cannot restore JSON from CSV.")
