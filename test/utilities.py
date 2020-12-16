import numpy as np
from pathlib import Path

from numpy import random
from pvinspect.data import (
    Image,
    ModuleImage,
    ImageSequence,
    ModuleImageSequence,
    EL_IMAGE,
)
from typing import List, Dict, Any, Optional
import uuid


def random_filename(ext: str = ".tif") -> str:
    return uuid.uuid4().hex + ext


def assert_equal(value, target, precision=1e-3):
    assert np.all(value > target - precision) and np.all(
        value < target + precision
    ), "got value={}, target={}".format(value, target)


def random_image(lazy: bool = False, **kwargs) -> Image:
    if lazy:
        data = Image.LazyData(lambda: np.random.random((10, 10)))
    else:
        data = np.random.random((10, 10))

    if "modality" not in kwargs.keys():
        kwargs["modality"] = EL_IMAGE
    if "path" not in kwargs.keys():
        kwargs["path"] = Path() / random_filename()

    return Image(data, **kwargs)


def random_uint_image(lazy: bool = False, **kwargs) -> Image:
    if lazy:
        data = Image.LazyData(
            lambda: (np.random.random((10, 10)) * 100).astype(np.uint32)
        )
    else:
        data = (np.random.random((10, 10)) * 100).astype(np.uint32)

    if "modality" not in kwargs.keys():
        kwargs["modality"] = EL_IMAGE
    if "path" not in kwargs.keys():
        kwargs["path"] = Path() / random_filename()

    return Image(data, **kwargs)


def random_ubyte_image(lazy: bool = False, **kwargs) -> Image:
    if lazy:
        data = Image.LazyData(
            lambda: (np.random.random((10, 10)) * 100).astype(np.uint8)
        )
    else:
        data = (np.random.random((10, 10)) * 100).astype(np.uint32)

    if "modality" not in kwargs.keys():
        kwargs["modality"] = EL_IMAGE
    if "path" not in kwargs.keys():
        kwargs["path"] = Path() / random_filename()

    return Image(data, **kwargs)


def random_module_image() -> ModuleImage:
    data = np.random.random((10, 10))
    return ModuleImage(data, modality=EL_IMAGE, path=Path() / random_filename())


def random_image_sequence(
    N: int = 3,
    meta: Optional[List[Dict[str, Any]]] = None,
    random_meta: bool = False,
    **kwargs
) -> ImageSequence:
    if random_meta and meta is None:
        random_keys = [uuid.uuid4().hex for _ in range(10)]
        meta = list()
        for _ in range(N):
            meta.append({k: random.randint(0, 10000) for k in random_keys})

    if meta is None:
        imgs = [random_image(**kwargs) for i in range(N)]
    else:
        assert len(meta) == N
        imgs = [random_image(meta=m, **kwargs) for m in meta]
    return ImageSequence(imgs, False)


def random_uint_image_sequence(
    N: int = 3, meta: Optional[List[Dict[str, Any]]] = None, **kwargs
) -> ImageSequence:
    if meta is None:
        imgs = [random_uint_image(**kwargs) for i in range(N)]
    else:
        assert len(meta) == N
        imgs = [random_uint_image(meta=m, **kwargs) for m in meta]
    return ImageSequence(imgs, False)


def random_ubyte_image_sequence(
    N: int = 3, meta: Optional[List[Dict[str, Any]]] = None, **kwargs
) -> ImageSequence:
    if meta is None:
        imgs = [random_ubyte_image(**kwargs) for i in range(N)]
    else:
        assert len(meta) == N
        imgs = [random_ubyte_image(meta=m, **kwargs) for m in meta]
    return ImageSequence(imgs, False)


def random_module_image_sequence() -> ModuleImageSequence:
    imgs = [random_module_image() for x in range(3)]
    return ModuleImageSequence(imgs, False)
