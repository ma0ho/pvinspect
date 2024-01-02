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
from pvinspect.data.image.type import DType

RANDOM_IMAGE_SHAPE = (10, 10)


def random_filename(ext: str = ".tif") -> str:
    return uuid.uuid4().hex + ext


def assert_equal(value, target, precision=1e-3):
    assert np.all(value > target - precision) and np.all(
        value < target + precision
    ), "got value={}, target={}".format(value, target)


def random_image(
    lazy: bool = False, seed: int = None, dtype: DType = DType.FLOAT, **kwargs
) -> Image:
    def rnd():
        if dtype == DType.FLOAT:
            return np.random.rand(10, 10)
        elif dtype == DType.UNSIGNED_BYTE:
            return np.random.randint(0, 255, size=(10, 10)).astype(np.uint8)
        else:
            raise NotImplementedError()

    if seed is not None:
        np.random.seed(seed)
    if lazy:
        data = LazyImage.LazyData(lambda: rnd())
        return LazyImage(data, **kwargs)
    else:
        data = rnd()
        return EagerImage(data, **kwargs)


def norandom_image(data: np.ndarray, lazy: bool = False, **kwargs) -> Image:
    if lazy:
        x = LazyImage.LazyData(lambda: data)
        return LazyImage(x, **kwargs)
    else:
        return EagerImage(data, **kwargs)


def random_sequence(
    seq_lazy: bool = False,
    imgs_lazy: bool = False,
    N: int = 3,
    dtype: DType = DType.FLOAT,
    meta: Optional[pd.DataFrame] = None,
) -> ImageSequence:

    # create meta if not given
    meta = pd.DataFrame([{"idx": i} for i in range(N)]) if meta is None else meta

    if seq_lazy:

        def load_fn(meta: pd.Series) -> Image:
            return random_image(imgs_lazy, seed=meta.name, meta=meta, dtype=dtype)

        return LazyImageSequence(meta, load_fn)
    else:
        imgs = [random_image(imgs_lazy, seed=i, idx=i, dtype=dtype) for i in range(N)]
        return EagerImageSequence(imgs, meta=meta)
