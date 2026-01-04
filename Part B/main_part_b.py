"""
Part B: Airfoil Optimisation using XFOIL
--------------------------------------
This script runs the PSO solver to optimise an airfoil shape using CST parameterisation
and XFOIL for aerodynamic evaluation.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_airfoil import main

if __name__ == "__main__":
    print("=== Part B: Airfoil Optimisation (XFOIL) ===")
    
    # Inject '--part B' into sys.argv if not present
    if "--part" not in sys.argv:
        sys.argv.extend(["--part", "B"])

    main()
