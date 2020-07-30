import numpy as np
from pathlib import Path
from pvinspect.data import (
    Image,
    ModuleImage,
    ImageSequence,
    ModuleImageSequence,
    EL_IMAGE,
)
from typing import List, Dict, Any, Optional


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
        kwargs["path"] = Path() / "test.tif"

    return Image(data, **kwargs)


def random_uint_image() -> Image:
    data = (np.random.random((10, 10)) * 100).astype(np.uint32)
    return Image(data, modality=EL_IMAGE, path=Path() / "test.png")


def random_module_image() -> ModuleImage:
    data = np.random.random((10, 10))
    return ModuleImage(data, modality=EL_IMAGE, path=Path() / "test.tif")


def random_image_sequence(
    N: int = 3, meta: Optional[List[Dict[str, Any]]] = None, **kwargs
) -> ImageSequence:
    if meta is None:
        imgs = [random_image(**kwargs) for i in range(N)]
    else:
        assert len(meta) == N
        imgs = [random_image(meta=m, **kwargs) for m in meta]
    return ImageSequence(imgs, False)


def random_uint_image_sequence() -> ImageSequence:
    imgs = [random_uint_image() for x in range(3)]
    return ImageSequence(imgs, False)


def random_module_image_sequence() -> ModuleImageSequence:
    imgs = [random_module_image() for x in range(3)]
    return ModuleImageSequence(imgs, False)
