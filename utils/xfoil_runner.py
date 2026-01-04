# utils/xfoil_runner.py
import os
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np

# On WSL / Linux we just call the system xfoil (installed with: sudo apt install xfoil)
# On Windows, we look for xfoil.exe in PATH or XFOIL_PATH env var
import shutil
import uuid

# Try to find xfoil in standard locations or PATH
_custom_path = os.environ.get("XFOIL_PATH", "")
if _custom_path and os.path.exists(_custom_path):
    XFOIL_EXE = _custom_path
else:
    # Check for xfoil.exe (Windows) or xfoil (Linux/Mac) in PATH
    _found = shutil.which("xfoil.exe") or shutil.which("xfoil")
    
    # If not in PATH, look in project root "Xfoil" folder
    if not _found:
        # Assuming we are in utils/, project root is ..
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_xfoil = os.path.join(root, "Xfoil", "xfoil.exe")
        if os.path.exists(local_xfoil):
            _found = local_xfoil
            
    # Fallback to "xfoil" if not found, assuming it might be aliased or in a non-standard way
    XFOIL_EXE = _found if _found else "xfoil"

XFOIL_DIR = "."  # run in project folder
print(f"[xfoil_runner] Using XFOIL at: {XFOIL_EXE}")


def _run_xfoil_script(script: str, workdir: str = ".") -> Tuple[int, str, str]:
    """
    Run XFOIL with the given multiline 'script' sent to stdin.
    """
    os.makedirs(workdir, exist_ok=True)
    
    # Keep a copy of the script on disk for debugging
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".inp", dir=workdir, encoding="ascii"
    ) as f:
        f.write(script)
        f.write("\n")
        script_path = f.name

    print(f"[xfoil_runner] cwd={workdir}")
    print(f"[xfoil_runner] Script kept at: {script_path}")

    proc = subprocess.run(
        [XFOIL_EXE],
        input=script.encode("ascii"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workdir,
    )

    out = proc.stdout.decode(errors="ignore")
    err = proc.stderr.decode(errors="ignore")

    print(f"[xfoil_runner] returncode={proc.returncode}")
    print("[xfoil_runner] --- stdout head ---")
    print(out[:400])
    print("[xfoil_runner] --- stdout tail ---")
    print(out[-400:])
    print("[xfoil_runner] --- stderr ---")
    print(err[:400])

    return proc.returncode, out, err, script_path


def run_xfoil_single_alpha(
    dat_path: str,
    alpha: float = 3.0,
    Re: float = 1e6,
    mach: float = 0.1,
    n_iter: int = 200,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Run XFOIL at a single angle of attack.
    Returns (Cl, Cd, Cm).
    """
    # Copy .dat to CWD with a safe name to avoid path issues (spaces, etc.)
    local_dat = f"tmp_{uuid.uuid4().hex[:8]}.dat"
    shutil.copy(dat_path, local_dat)
    dat_full = local_dat  # XFOIL will load this local file

    # polar file will be created in the XFOIL_DIR (same dir as xfoil executable)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".log", dir=XFOIL_DIR
    ) as out:
        polar_name = os.path.basename(out.name)
        polar_path = out.name

    # build script with REAL newlines (no '\n' inside each command)
    script_lines = [
        f"LOAD {dat_full}",
        "PANE",
        "",
        "OPER",
        f"VISC {Re}",
        f"MACH {mach}",
        f"ITER {n_iter}",
        "PACC",
        polar_name,
        "",
        f"ALFA {alpha}",
        "PACC",
        "QUIT",  # Exit OPER
        "QUIT",  # Exit XFOIL
    ]
    sep = "\r\n" if os.name == 'nt' else "\n"
    script = sep.join(script_lines) + sep

    _, _, _, script_path = _run_xfoil_script(script, workdir=XFOIL_DIR)

    # Parse polar file
    if not os.path.exists(polar_path):
        print(f"[xfoil_runner] polar file {polar_path} does not exist.")
        raise RuntimeError("XFOIL did not produce polar file.")

    with open(polar_path, "r") as f:
        lines = f.readlines()

    # find the last data line (skip headers, comments)
    data_line: Optional[str] = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s[0] in ("#", "!"):
            continue
        data_line = line

    if data_line is None:
        print("[xfoil_runner] polar file had no data lines.")
        print("".join(lines[-10:]))
        raise RuntimeError("Could not parse XFOIL polar output for single alpha.")

    tokens = data_line.split()
    # typical format: alpha, Cl, Cd, Cd_visc, Cd_form, Cm, ...
    try:
        alpha_val = float(tokens[0])
        Cl = float(tokens[1])
        Cd = float(tokens[2])
        Cm = float(tokens[5])
    except Exception as e:
        print("[xfoil_runner] Could not parse data line:", data_line)
        raise RuntimeError("Could not parse XFOIL polar output for single alpha.") from e

    # Cleanup temporary files
    try:
        os.remove(polar_path)
    except OSError:
        pass
    try:
        os.remove(script_path)
    except OSError:
        pass
    try:
        if os.path.exists(local_dat):
            os.remove(local_dat)
    except OSError:
        pass

    return Cl, Cd, Cm


def run_xfoil_polar(
    dat_path: str,
    a_start: float,
    a_end: float,
    a_step: float,
    Re: float = 1e6,
    mach: float = 0.1,
    n_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run XFOIL polar over a range of angles.
    Returns arrays: alpha, Cl, Cd, Cm.
    """
    # Copy .dat to CWD with a safe name to avoid path issues (spaces, etc.)
    local_dat = f"tmp_{uuid.uuid4().hex[:8]}.dat"
    shutil.copy(dat_path, local_dat)
    dat_full = local_dat

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".log", dir=XFOIL_DIR
    ) as out:
        polar_name = os.path.basename(out.name)
        polar_path = out.name

    script_lines = [
        f"LOAD {dat_full}",
        "PANE",
        "",
        "OPER",
        f"VISC {Re}",
        f"MACH {mach}",
        f"ITER {n_iter}",
        "PACC",
        polar_name,
        "",
        f"ASEQ {a_start} {a_end} {a_step}",
        "PACC",
        "QUIT",  # Exit OPER
        "QUIT",  # Exit XFOIL
    ]
    sep = "\r\n" if os.name == 'nt' else "\n"
    script = sep.join(script_lines) + sep

    _, _, _, script_path = _run_xfoil_script(script, workdir=XFOIL_DIR)

    if not os.path.exists(polar_path):
        print(f"[xfoil_runner] polar file {polar_path} does not exist.")
        raise RuntimeError("XFOIL did not produce polar file.")

    alphas: list[float] = []
    cls: list[float] = []
    cds: list[float] = []
    cms: list[float] = []

    with open(polar_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in ("#", "!"):
                continue
            tokens = s.split()
            try:
                a = float(tokens[0])
                cl = float(tokens[1])
                cd = float(tokens[2])
                cm = float(tokens[5])
            except Exception:
                continue
            alphas.append(a)
            cls.append(cl)
            cds.append(cd)
            cms.append(cm)

    try:
        os.remove(polar_path)
    except OSError:
        pass
    try:
        os.remove(script_path)
    except OSError:
        pass
    try:
        if os.path.exists(local_dat):
            os.remove(local_dat)
    except OSError:
        pass

    if not alphas:
        raise RuntimeError("No valid polar data parsed from XFOIL.")

    return np.array(alphas), np.array(cls), np.array(cds), np.array(cms)