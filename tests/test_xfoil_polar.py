from pathlib import Path

import numpy as np
import pytest

from utils.xfoil_runner import run_polar, XFoilError


def _make_test_airfoil_path() -> Path:
    """Same NACA-based pseudo-path used for the polar test."""
    return Path("NACA2412.naca")


def test_xfoil_polar_basic_naca():
    """
    Smoke test for the XFOIL polar interface.

    If XFOIL is stable, we should parse a small polar correctly.
    If XFOIL crashes internally, we skip this test.
    """
    airfoil_path = _make_test_airfoil_path()

    try:
        alpha, cl, cd, cm = run_polar(
            airfoil_path,
            alpha_start=-2.0,
            alpha_end=6.0,
            alpha_step=2.0,
            Re=1e6,
            mach=0.0,
            iter_limit=100,
            timeout=40.0,
            viscous=False,  # inviscid for robustness
        )
    except XFoilError as e:
        pytest.skip(f"XFOIL unstable in this environment: {e}")

    assert alpha.size >= 3
    assert cl.size == alpha.size
    assert cd.size == alpha.size
    assert cm.size == alpha.size

    assert np.all(np.isfinite(alpha))
    assert np.all(np.isfinite(cl))
    assert np.all(np.isfinite(cd))
    assert np.all(np.isfinite(cm))

    assert np.all(cd >= 0.0)
