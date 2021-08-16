from pathlib import Path
from test.utilities import *

import pandas as pd
from numpy import arange
from pvinspect import data
from pvinspect.data.image import *

# from pvinspect.data.image import _sequence
from pvinspect.data.image.type import DTYPE_INT


def test_create_lazy_image():
    img = random_image(lazy=True)
    assert isinstance(img, LazyImage)


def test_create_eager_image():
    img = random_image()
    assert isinstance(img, EagerImage)


def test_load_lazy_image():
    img = norandom_image(np.arange(100).reshape(10, 10), lazy=True)
    assert img.data.sum() == np.arange(100).sum()


def test_create_image_with_meta():
    img = random_image(key="value")
    assert img.has_meta("key")


def test_image_has_meta():
    img = random_image(key="value")
    assert img.has_meta("key")
    assert not img.has_meta("key2")


def test_image_get_meta():
    img = random_image(key="value")
    img.get_meta("key") == "value"


def test_image_from_other():
    img = random_image()
    img2 = EagerImage.from_other(img, key="value")
    assert_equal(img.data, img2.data)
    assert not img.has_meta("key")
    assert img2.has_meta("key")


def test_lazy_image_from_other():
    img = random_image(lazy=True)
    img2 = LazyImage.from_other(img, key="value")
    assert_equal(img.data, img2.data)
    assert not img.has_meta("key")
    assert img2.has_meta("key")


def test_image_from_other_preserves_meta():
    img = random_image(key="value")
    img2 = EagerImage.from_other(img)
    assert img2.has_meta("key")


def test_image_from_other_append_meta():
    img = random_image(key="value")
    img2 = EagerImage.from_other(img, key2="value")
    assert not img.has_meta("key2")
    assert img2.has_meta("key2")
    assert img2.has_meta("key")


def test_image_from_other_join_meta():
    img = random_image(key="value")
    img2 = EagerImage.from_other(img, meta=pd.Series({"key2": "value2"}))
    assert img2.has_meta("key")
    assert img2.has_meta("key2")


def test_image_from_other_meta_override():
    img = random_image(key="value")
    img2 = EagerImage.from_other(img, key="value2")
    assert img2.get_meta("key") == "value2"
    assert img.get_meta("key") == "value"


def test_image_from_self():
    img = random_image()
    img2 = img.from_self(key="value")
    assert_equal(img.data, img2.data)
    assert img2.has_meta("key")


def test_image_from_self_meta_override_obj():
    class Test:
        def __init__(self, x):
            self.x = x
            super().__init__()

    t1 = Test("x")
    t2 = Test("y")
    img = random_image(key=t1)
    img2 = img.from_self(key=t2)
    assert img2.get_meta("key").x == "y"
    assert img.get_meta("key").x == "x"


def test_image_as_type():
    img = random_image()
    img_converted = img.as_type(DType.INT)
    assert img.dtype == DType.FLOAT
    assert img_converted.dtype == DType.INT
    assert img_converted.data.dtype == DTYPE_INT


def test_lazy_image_as_type():
    img = random_image(lazy=True)
    img_converted = img.as_type(DType.INT)
    assert isinstance(img_converted, LazyImage)
    assert img_converted.dtype == DType.INT
    assert img_converted.data.dtype == DTYPE_INT


def test_image_data_is_immutable():
    data = random_image().data
    assert data.flags["WRITEABLE"] == False


def test_image_meta_is_immutable():
    img = random_image()
    arraymeta = np.zeros(1)
    othermeta = [1, 2, 3]
    img_with_meta = EagerImage.from_other(img, arraymeta=arraymeta, othermeta=othermeta)

    # array should be immutable
    assert img_with_meta.get_meta("arraymeta").flags["WRITEABLE"] == False

    # other should return a copy
    assert img_with_meta.get_meta("othermeta") is not othermeta


def test_lazy_image_apply_data():
    img = random_image(lazy=True)
    img2 = img.apply_data(lambda x: np.zeros_like(x))
    assert isinstance(img2, LazyImage)
    assert img2.data.sum() == 0.0
