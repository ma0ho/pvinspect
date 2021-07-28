import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy import random
from pvinspect.data.image.image import EagerImage, Image, LazyImage
from pvinspect.data.image.sequence import (
    EagerImageSequence,
    ImageSequence,
    LazyImageSequence,
)

RANDOM_IMAGE_SHAPE = (10, 10)


def random_filename(ext: str = ".tif") -> str:
    return uuid.uuid4().hex + ext


def assert_equal(value, target, precision=1e-3):
    assert np.all(value > target - precision) and np.all(
        value < target + precision
    ), "got value={}, target={}".format(value, target)


def random_image(lazy: bool = False, seed: int = None, **kwargs) -> Image:
    if seed is not None:
        np.random.seed(seed)
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


def random_sequence(
    seq_lazy: bool = False, imgs_lazy: bool = False, N: int = 3
) -> ImageSequence:
    if seq_lazy:

        def load_fn(meta: pd.Series) -> Image:
            return random_image(imgs_lazy, seed=meta["idx"], meta=meta)

        meta = pd.DataFrame([{"idx": i} for i in range(N)])
        return LazyImageSequence(meta, load_fn)
    else:
        meta = pd.DataFrame([{"idx": i} for i in range(N)])
        imgs = [random_image(imgs_lazy, seed=i, idx=i) for i in range(N)]
        return EagerImageSequence(imgs, meta=meta)
