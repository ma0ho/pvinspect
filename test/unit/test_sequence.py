from pathlib import Path
from test.utilities import *

import pandas as pd
from pvinspect import data
from pvinspect.data.image import *
from pvinspect.data.image import _sequence


def test_sequence_element_access():
    seq = data.datasets.poly10x6(2)
    assert seq._images[0].path == seq[0].path
    assert seq._images[1].path == seq[1].path


# def test_sequence_wrapper():
#    @_sequence
#    def _some_fn_mis(seq: ModuleImageSequence):
#        assert type(seq) == ModuleImageSequence
#        return seq
#
#    @_sequence
#    def _some_fn_is(seq: ImageSequence):
#        assert type(seq) == ImageSequence
#        return seq
#
#    img = random_module_image()
#    res = _some_fn_mis(img)
#    assert isinstance(res, ModuleImage)
#
#    img = random_image()
#    res = _some_fn_is(img)
#    assert isinstance(res, Image)
#
#    img_seq = random_module_image_sequence()
#    res = _some_fn_mis(img_seq)
#    assert isinstance(res, ModuleImageSequence)
#
#    img_seq = random_image_sequence()
#    res = _some_fn_is(img_seq)
#    assert isinstance(res, ImageSequence)
#
#
# def test_sequence_wrapper_noarg():
#    @_sequence()
#    def _some_fn(seq: ModuleImageSequence):
#        assert type(seq) == ModuleImageSequence
#        return seq
#
#    img = random_module_image()
#    res = _some_fn(img)
#    assert isinstance(res, ModuleImage)
#
#
# def test_sequence_wrapper_nounwrap():
#    @_sequence(True)
#    def _some_fn(seq: ModuleImageSequence):
#        assert type(seq) == ModuleImageSequence
#        return seq
#
#    img = random_module_image()
#    res = _some_fn(img)
#    assert isinstance(res, ModuleImageSequence)


def test_sequence_apply_image_data():
    seq = random_image_sequence()
    original_data = seq[0].data.copy()

    def fn(x):
        x = x.copy()
        x[:] = 0.0
        return x

    seq2 = seq.apply_image_data(fn)
    assert_equal(seq2[0].data, 0.0)
    assert_equal(seq[0].data, original_data)


def test_sequence_as_type():
    seq = random_image_sequence()
    assert seq.dtype == DType.FLOAT
    seq = seq.as_type(DType.UNSIGNED_INT)
    assert seq.dtype == DType.UNSIGNED_INT


def test_sequence_meta_to_pandas():
    img1 = random_image(meta={"k": 1})
    img2 = random_image(meta={"k": 2})
    seq = ImageSequence([img1, img2], same_camera=False)
    meta = seq.meta_to_pandas()

    assert isinstance(meta, pd.DataFrame)
    assert meta.iloc[0]["k"] == 1
    assert meta.iloc[1]["k"] == 2


def test_sequence_pandas_method():
    img1 = random_image(meta={"k": 1})
    img2 = random_image(meta={"k": 2})
    seq = ImageSequence([img1, img2], same_camera=False)
    seq = seq.pandas.query("k == 1")

    assert len(seq) == 1
    assert seq[0].get_meta("k") == 1


def test_sequence_pandas_array_access_multiple():
    seq = random_image_sequence()
    sub = seq.pandas.iloc[[0, 1]]

    assert len(sub) == 2


def test_sequence_pandas_array_access_single():
    seq = random_image_sequence()
    img = seq.pandas.iloc[0]

    assert img == seq._images[0]
