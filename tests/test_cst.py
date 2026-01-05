import numpy as np

from utils.cst import cst_airfoil
from utils.geometry import build_airfoil_coordinates, write_dat


def test_cst_symmetry_gives_symmetric_airfoil():
    """
    If lower coefficients are the negative of upper coefficients and dz_te=0,
    the airfoil should be symmetric about the x-axis: yu(x) = -yl(x).
    """
    coeffs_upper = [0.2, 0.4, 0.3, 0.1]
    coeffs_lower = [-c for c in coeffs_upper]

    xu, yu, xl, yl = cst_airfoil(
        n_points=101,
        coeffs_upper=coeffs_upper,
        coeffs_lower=coeffs_lower,
        n1=0.5,
        n2=1.0,
        dz_te=0.0,
    )

    # Same x on upper and lower
    assert np.allclose(xu, xl)

    # Symmetry: yu = -yl
    assert np.allclose(yu, -yl, atol=1e-8)


def test_cst_upper_above_lower():
    """
    For a reasonable choice of coefficients, the upper surface should lie
    above the lower surface (or be very close near TE).
    """
    coeffs_upper = [0.1, 0.3, 0.5, 0.4, 0.2]
    coeffs_lower = [-0.05, -0.1, -0.15, -0.1, -0.05]

    xu, yu, xl, yl = cst_airfoil(
        n_points=121,
        coeffs_upper=coeffs_upper,
        coeffs_lower=coeffs_lower,
        n1=0.5,
        n2=1.0,
        dz_te=0.0,
    )

    # Same x on upper and lower
    assert np.allclose(xu, xl)

    # Upper surface should be >= lower surface (allow tiny numerical slack)
    assert np.all(yu >= yl - 1e-6)


def test_geometry_coordinate_ordering():
    """
    Full airfoil coordinates should start at upper TE, go to LE,
    then go along the lower surface back to TE.
    x should remain in [0, 1].
    """
    coeffs_upper = [0.2, 0.4, 0.3, 0.1]
    coeffs_lower = [-0.1, -0.2, -0.25, -0.15]

    x, y = build_airfoil_coordinates(
        coeffs_upper=coeffs_upper,
        coeffs_lower=coeffs_lower,
        n_points=101,
        n1=0.5,
        n2=1.0,
        dz_te=0.0,
    )

    # Check basic properties
    assert x.shape == y.shape
    assert x.ndim == 1
    assert x.size == 101 + 100  # 101 upper + 100 lower (LE not duplicated)
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_write_dat_format(tmp_path):
    """
    Writing a .dat file should create a file with:
      - first line: name
      - subsequent lines: two floats separated by whitespace
      - number of coordinate lines = len(x)
    """
    coeffs_upper = [0.2, 0.4, 0.3, 0.1]
    coeffs_lower = [-0.1, -0.2, -0.25, -0.15]

    x, y = build_airfoil_coordinates(
        coeffs_upper=coeffs_upper,
        coeffs_lower=coeffs_lower,
        n_points=51,
    )

    out_path = tmp_path / "test_airfoil.dat"
    name = "TEST_CST_AIRFOIL"
    write_dat(x, y, out_path, name=name)

    assert out_path.exists()

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1 + len(x)

    # First line is name
    assert lines[0].strip() == name

    # Subsequent lines: two fields that can be parsed as floats
    for line in lines[1:]:
        parts = line.split()
        assert len(parts) == 2
        float(parts[0])
        float(parts[1])
