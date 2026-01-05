import numpy as np
from optimizer.base import project

def test_project_clips():
    bounds = [(-1.0, 1.0)] * 3
    x = np.array([ -2.0, 0.5,  5.0 ])
    y = project(x, bounds)
    assert np.allclose(y, np.array([-1.0, 0.5, 1.0]))
