import os
import pytest
from utils.xfoil_runner import run_xfoil_single_alpha

# Use a known dat file or generate one
DAT_FILE = "naca0012.dat"

@pytest.fixture
def sample_dat(tmp_path):
    p = tmp_path / DAT_FILE
    # Simple NACA 0012 coords (truncated for brevity/speed if needed, but better use real one)
    # Actually, let's look for one in the project or create a minimal one.
    # We will use the one validation
    content = """NACA 0012
 1.000000  0.001260
 0.995000  0.002520
 0.000000  0.000000
 0.995000 -0.002520
 1.000000 -0.001260
"""
    # Real naca0012 is better. 
    # But let's assume one exists or we create a dummy one that XFOIL might reject if not valid.
    # Better to find a file in the repo.
    return None

def test_run_single_alpha():
    # Find a .dat file
    dat_files = []
    if os.path.exists("tests/naca0012.dat"):
        dat_files.append("tests/naca0012.dat")
    
    search_dirs = ["data/airfoils", "data", "."]
    
    for d in search_dirs:
        if not os.path.exists(d): continue
        for root, dirs, files in os.walk(d):
            if ".venv" in root: continue
            for f in files:
                if f.endswith(".dat"):
                    dat_files.append(os.path.join(root, f))
                    break
            if dat_files: break
        if dat_files: break
    if not dat_files:
        pytest.skip("No .dat files found for testing")
    
    dat_file = dat_files[0]
    print(f"Using {dat_file}")
    
    # Increase Re and iterations for better convergence
    cl, cd, cm = run_xfoil_single_alpha(dat_file, alpha=0.0, Re=1e6, n_iter=200)
    
    assert cl is not None
    assert cd is not None
    assert cm is not None
    print(f"Result: CL={cl} CD={cd} CM={cm}")
