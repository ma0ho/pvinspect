def assert_equal(value, target, precision=1e-3):
    assert value > target-precision and value < target+precision