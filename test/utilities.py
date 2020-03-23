import numpy as np


def assert_equal(value, target, precision=1e-3):
    assert np.all(value > target - precision) and np.all(value < target + precision)
