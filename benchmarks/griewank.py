import numpy as np

def griewank(x: np.ndarray) -> float:
    """
    Griewank benchmark function.
    Global minimum at x = 0, f = 0. Bounds typically [-600, 600]^D.
    """
    x = np.asarray(x, dtype=float)
    s = np.sum(x * x) / 4000.0
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1, dtype=float))))
    return 1.0 + s - p
