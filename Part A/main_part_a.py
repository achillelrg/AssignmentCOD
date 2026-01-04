"""
Part A: Development of Optimisation Solvers
-----------------------------------------
This script runs the Particle Swarm Optimisation (PSO) on the 5-dimensional Griewank function.
"""

import sys
import os

# Add project root to path so we can import 'optimizer', 'benchmarks', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_opt import main

if __name__ == "__main__":
    print("=== Part A: Griewank Function Optimisation ===")
    
    # Inject '--part A' into sys.argv if not present, to ensure outputs go to data/PartA
    if "--part" not in sys.argv:
        sys.argv.extend(["--part", "A"])
        
    main()
