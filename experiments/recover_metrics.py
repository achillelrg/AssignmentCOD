
import sys
import os
import json
sys.path.append(os.getcwd())

from benchmarks.airfoil_xfoil import airfoil_fitness

# 1. PSO Best Vector (from JSON)
x_pso = [0.05146187262584913, -0.2, 0.49739056712437885, 0.02211542497602753, -0.2, 0.14133110548431568]

# 2. GA Best Vector (from JSON)
x_ga = [0.48584578200077044, 0.34707130056130275, 0.38989315219696413, 0.04142898709824025, -0.13309141332581198, 0.3532405469183826]

print("--- Recovering Metrics ---")

# Evaluate PSO
print("\n[PSO Best]")
J, Cl, Cd, Cm = airfoil_fitness(x_pso, return_all=True)
print(f"J: {J}")
print(f"Cl: {Cl}")
print(f"Cd: {Cd}")
print(f"Cm: {Cm}")
print(f"L/D: {Cl/Cd if Cd else 0}")

# Evaluate GA
print("\n[GA Best]")
J, Cl, Cd, Cm = airfoil_fitness(x_ga, return_all=True)
print(f"J: {J}")
print(f"Cl: {Cl}")
print(f"Cd: {Cd}")
print(f"Cm: {Cm}")
print(f"L/D: {Cl/Cd if Cd else 0}")
