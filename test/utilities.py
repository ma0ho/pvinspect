import numpy as np
from pathlib import Path
from pvinspect.data import (
    Image,
    ModuleImage,
    ImageSequence,
    ModuleImageSequence,
    EL_IMAGE,
)


def assert_equal(value, target, precision=1e-3):
    assert np.all(value > target - precision) and np.all(
        value < target + precision
    ), "got value={}, target={}".format(value, target)


def random_image(**kwargs) -> Image:
    data = np.random.random((10, 10))

    if "modality" not in kwargs.keys():
        kwargs["modality"] = EL_IMAGE
    if "path" not in kwargs.keys():
        kwargs["path"] = Path() / "test.tif"

    return Image(data, **kwargs)


def random_uint_image() -> Image:
    data = (np.random.random((10, 10)) * 100).astype(np.uint32)
    return Image(data, EL_IMAGE, Path() / "test.png")


def random_module_image() -> ModuleImage:
    data = np.random.random((10, 10))
    return ModuleImage(data, EL_IMAGE, Path() / "test.tif")


def random_image_sequence() -> ImageSequence:
    imgs = [random_image() for x in range(2)]
    return ImageSequence(imgs, False)


def random_uint_image_sequence() -> ImageSequence:
    imgs = [random_uint_image() for x in range(2)]
    return ImageSequence(imgs, False)


def random_module_image_sequence() -> ModuleImageSequence:
    imgs = [random_module_image() for x in range(2)]
    return ModuleImageSequence(imgs, False)
