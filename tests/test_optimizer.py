import numpy as np

from optimizers.pso import pso_optimize


def sphere(x: np.ndarray) -> float:
    """Simple convex test function: global minimum at x = 0."""
    return float(np.sum(x**2))


def griewank(x: np.ndarray) -> float:
    """
    Griewank function (d-dimensional).
    Global minimum at x = 0 with f(0) = 0.
    """
    d = x.size
    sum_term = np.sum(x**2) / 4000.0
    i = np.arange(1, d + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(i)))
    return float(sum_term - prod_term + 1.0)


def test_pso_converges_on_sphere():
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    result = pso_optimize(
        sphere,
        bounds,
        n_particles=20,
        n_iters=80,
        inertia=0.7,
        cognitive=1.4,
        social=1.4,
        seed=123,
    )
    assert result["f_best"] < 1e-3, f"Sphere minimum not reached, f_best={result['f_best']}"


def test_pso_reproducible_with_seed():
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    res1 = pso_optimize(
        sphere,
        bounds,
        n_particles=15,
        n_iters=40,
        seed=42,
    )
    res2 = pso_optimize(
        sphere,
        bounds,
        n_particles=15,
        n_iters=40,
        seed=42,
    )

    assert np.allclose(res1["x_best"], res2["x_best"])
    assert np.isclose(res1["f_best"], res2["f_best"])


def test_pso_griewank_5d_smoke():
    # This is a smoke test: we just require that PSO gets reasonably close to the optimum
    dim = 5
    bounds = [(-600.0, 600.0)] * dim

    result = pso_optimize(
        griewank,
        bounds,
        n_particles=40,
        n_iters=150,
        seed=7,
    )

    # Griewank minimum is 0 at x=0. We accept a small residual.
    assert result["f_best"] < 0.1, f"PSO failed to get close on Griewank 5D, f_best={result['f_best']}"
