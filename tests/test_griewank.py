import numpy as np
from benchmarks.griewank import griewank

def test_griewank_zero():
    assert griewank(np.zeros(5)) == 0.0
