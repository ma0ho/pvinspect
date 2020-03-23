import random
import numpy
import pytest


@pytest.fixture
def random():
    """Fix random number generation"""
    random.seed(0)
    numpy.random.seed(0)
