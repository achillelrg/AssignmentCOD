import os
import shutil
import subprocess
import tempfile
import uuid
import numpy as np
from typing import Tuple, Optional

# Attempt to import xfoil library (for Windows/Direct usage)
try:
    from xfoil import XFoil
    from xfoil.model import Airfoil
    HAS_XFOIL_LIB = True
except ImportError:
    HAS_XFOIL_LIB = False

# Fallback XFOIL executable setup
_custom_path = os.environ.get("XFOIL_PATH", "")
if _custom_path and os.path.exists(_custom_path):
    XFOIL_EXE = _custom_path
else:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_xfoil_exe = os.path.join(root, "Xfoil", "xfoil.exe")
    local_xfoil_bin = os.path.join(root, "Xfoil", "xfoil")
    _found = None
    if os.path.exists(local_xfoil_exe):
        _found = local_xfoil_exe
    elif os.path.exists(local_xfoil_bin):
        _found = local_xfoil_bin
    if not _found:
        _found = shutil.which("xfoil.exe") or shutil.which("xfoil")
    XFOIL_EXE = _found if _found else "xfoil"

XFOIL_DIR = "."  # run in project folder

# ---------------------------------------------------------------------------
# LIBRARY-BASED RUNNER (Windows Friendly)
# ---------------------------------------------------------------------------

def _run_lib_single(dat_path, alpha, Re, mach, n_iter):
    """Run using xfoil python library."""
    xf = XFoil()
    # Mute output if possible, xfoil lib prints to stdout usually
    xf.print = False 
    
    # Load airfoil
    # xfoil lib usually wants coordinates, not file, but Airfoil class might load file
    # Or XFoil.airfoil property.
    # We will try loading coordinates from the dat file manually.
    try:
        with open(dat_path, 'r') as f:
            lines = f.readlines()
        # Skip header? XFOIL dat has name on line 1
        coords = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
        coords = np.array(coords)
        
        xf.airfoil = Airfoil(x=coords[:,0], y=coords[:,1])
    except Exception as e:
        print(f"[xfoil_lib] Error loading coordinates: {e}")
        return None, None, None

    # Setup
    xf.Re = Re
    xf.M = mach
    xf.max_iter = n_iter
    
    # Run
    try:
        xf.a(alpha)
        # Check convergence? xf.converged property might exist
        return xf.Cl, xf.Cd, xf.Cm
    except Exception:
        return None, None, None

def _run_lib_polar(dat_path, a_start, a_end, a_step, Re, mach, n_iter):
    """Run polar using xfoil python library."""
    xf = XFoil()
    xf.print = False
    
    # Load coords 
    with open(dat_path, 'r') as f:
        lines = f.readlines()
    coords = []
    for line in lines[1:]:
        p = line.split()
        if len(p) >= 2: coords.append((float(p[0]), float(p[1])))
    coords = np.array(coords)
    xf.airfoil = Airfoil(x=coords[:,0], y=coords[:,1])
    
    xf.Re = Re
    xf.M = mach
    xf.max_iter = n_iter
    
    alphas = np.arange(a_start, a_end + a_step/2, a_step)
    cl_list, cd_list, cm_list, a_list = [], [], [], []
    
    # Sequential run
    xf.reset_bls() # reset boundary layer
    for a in alphas:
        xf.a(a)
        # We assume values persist even if not converged, or check something?
        # xfoil lib usually keeps last state.
        # Ideally we only take converged.
        cl_list.append(xf.Cl)
        cd_list.append(xf.Cd)
        cm_list.append(xf.Cm)
        a_list.append(a)
            
    return np.array(a_list), np.array(cl_list), np.array(cd_list), np.array(cm_list)


# ---------------------------------------------------------------------------
# SUBPROCESS-BASED RUNNER (Linux Fallback)
# ---------------------------------------------------------------------------

def _run_xfoil_script(script: str, workdir: str = ".") -> Tuple[int, str, str, str]:
    os.makedirs(workdir, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".inp", dir=workdir, encoding="ascii") as f:
        f.write(script)
        f.write("\n")
        script_path = f.name
    with open(script_path, "r") as f_in:
        proc = subprocess.run([XFOIL_EXE], stdin=f_in, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=workdir)
    return proc.returncode, proc.stdout.decode(errors="ignore"), proc.stderr.decode(errors="ignore"), script_path

def _parse_xfoil_stdout(out: str) -> dict:
    results_map = {}
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "a =" in line and "CL =" in line:
            try:
                parts = line.replace("=", " ").split()
                idx_a = parts.index("a")
                val_a = float(parts[idx_a + 1])
                idx_cl = parts.index("CL")
                val_cl = float(parts[idx_cl + 1])
                if i + 1 < len(lines):
                    next_line = lines[i+1]
                    if "Cm =" in next_line and "CD =" in next_line:
                        parts2 = next_line.replace("=", " ").split()
                        idx_cm = parts2.index("Cm")
                        val_cm = float(parts2[idx_cm + 1])
                        idx_cd = parts2.index("CD")
                        val_cd = float(parts2[idx_cd + 1])
                        results_map[val_a] = (val_cl, val_cd, val_cm)
            except (ValueError, IndexError):
                continue
    return results_map

def _run_subprocess_single(dat_path, alpha, Re, mach, n_iter):
    # Copy .dat locally
    unique_id = uuid.uuid4().hex[:8]
    local_dat = f"airfoil_{unique_id}.dat"
    shutil.copy(dat_path, local_dat)
    
    # Fix line endings
    with open(local_dat, 'rb') as f: content = f.read().replace(b'\r\n', b'\n')
    with open(local_dat, 'wb') as f: f.write(content)

    script_lines = [
        "PLOP", "G", "",
        f"LOAD {local_dat}",
        "PANE", "OPER",
        f"VISC {Re}", f"MACH {mach}", f"ITER {n_iter}",
        f"ALFA {alpha}", "", "QUIT"
    ]
    script = ("\r\n" if os.name == 'nt' else "\n").join(script_lines) + "\n"
    
    rc, out, err, spath = _run_xfoil_script(script, workdir=XFOIL_DIR)
    
    # Cleanup
    for p in [spath, local_dat]:
        if os.path.exists(p): os.remove(p)

    results = _parse_xfoil_stdout(out)
    if not results: return None, None, None
    best_a = min(results.keys(), key=lambda x: abs(x - alpha))
    if abs(best_a - alpha) > 0.1: return None, None, None # Too far
    return results[best_a]

def _run_subprocess_polar(dat_path, a_start, a_end, a_step, Re, mach, n_iter):
    unique_id = uuid.uuid4().hex[:8]
    local_dat = f"airfoil_{unique_id}.dat"
    shutil.copy(dat_path, local_dat)
    with open(local_dat, 'rb') as f: content = f.read().replace(b'\r\n', b'\n')
    with open(local_dat, 'wb') as f: f.write(content)

    script_lines = [
        "PLOP", "G", "",
        f"LOAD {local_dat}",
        "PANE", "OPER",
        f"VISC {Re}", f"MACH {mach}", f"ITER {n_iter}",
        f"ASEQ {a_start} {a_end} {a_step}", "", "QUIT"
    ]
    script = ("\r\n" if os.name == 'nt' else "\n").join(script_lines) + "\n"
    rc, out, err, spath = _run_xfoil_script(script, workdir=XFOIL_DIR)
    
    for p in [spath, local_dat]:
        if os.path.exists(p): os.remove(p)

    results = _parse_xfoil_stdout(out)
    alphas, cls, cds, cms = [], [], [], []
    for a in sorted(results.keys()):
        cl, cd, cm = results[a]
        alphas.append(a)
        cls.append(cl)
        cds.append(cd)
        cms.append(cm)
    return np.array(alphas), np.array(cls), np.array(cds), np.array(cms)

# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def run_xfoil_single_alpha(dat_path: str, alpha: float = 3.0, Re: float = 1e6, mach: float = 0.1, n_iter: int = 200) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if HAS_XFOIL_LIB:
        return _run_lib_single(dat_path, alpha, Re, mach, n_iter)
    else:
        # Fallback
        return _run_subprocess_single(dat_path, alpha, Re, mach, n_iter)

def run_xfoil_polar(dat_path: str, a_start: float, a_end: float, a_step: float, Re: float = 1e6, mach: float = 0.1, n_iter: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if HAS_XFOIL_LIB:
        return _run_lib_polar(dat_path, a_start, a_end, a_step, Re, mach, n_iter)
    else:
        return _run_subprocess_polar(dat_path, a_start, a_end, a_step, Re, mach, n_iter)
