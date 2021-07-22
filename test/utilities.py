import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy import random
from pvinspect.data.image.image import EagerImage, Image, LazyImage

RANDOM_IMAGE_SHAPE = (10, 10)


def random_filename(ext: str = ".tif") -> str:
    return uuid.uuid4().hex + ext


def assert_equal(value, target, precision=1e-3):
    assert np.all(value > target - precision) and np.all(
        value < target + precision
    ), "got value={}, target={}".format(value, target)


def random_image(lazy: bool = False, **kwargs) -> Image:
    if lazy:
        data = LazyImage.LazyData(lambda: np.random.rand(10, 10))
        return LazyImage(data, **kwargs)
    else:
        data = np.random.rand(10, 10)
        return EagerImage(data, **kwargs)


def norandom_image(data: np.ndarray, lazy: bool = False, **kwargs) -> Image:
    if lazy:
        x = LazyImage.LazyData(lambda: data)
        return LazyImage(x, **kwargs)
    else:
        return EagerImage(data, **kwargs)
