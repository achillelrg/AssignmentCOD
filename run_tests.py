import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from tests.test_griewank import test_griewank_zero
    print("Running test_griewank_zero...", end=" ")
    test_griewank_zero()
    print("PASS")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

try:
    from tests.test_pso_smoke import test_pso_runs_and_improves
    print("Running test_pso_runs_and_improves...", end=" ")
    test_pso_runs_and_improves()
    print("PASS")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("\nAll tests passed!")
