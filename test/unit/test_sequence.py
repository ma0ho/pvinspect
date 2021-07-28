from pathlib import Path
from test.utilities import *

import pandas as pd
from pvinspect import data
from pvinspect.data.image import *


def test_create_eager_sequence_eager_images():
    seq = random_sequence(seq_lazy=False, imgs_lazy=False, N=3)
    assert len(seq) == 3


def test_create_eager_sequence_lazy_images():
    seq = random_sequence(seq_lazy=False, imgs_lazy=True, N=3)
    assert len(seq) == 3


def test_create_lazy_sequence_lazy_images():
    seq = random_sequence(seq_lazy=True, imgs_lazy=True, N=3)
    assert len(seq) == 3


def test_getitem_eager():
    seq = random_sequence(seq_lazy=False, imgs_lazy=False)
    img = seq[0]
    assert isinstance(img, EagerImage)
    assert img.get_meta("idx") == 0


def test_getitem_lazy():
    seq = random_sequence(seq_lazy=True, imgs_lazy=False)
    img = seq[0]
    assert isinstance(img, EagerImage)
    assert img.get_meta("idx") == 0


def test_getitem_slice():
    seq = random_sequence()
    imgs = seq[0:2]
    assert isinstance(imgs, ImageSequence)
    assert len(imgs) == 2
    assert imgs[1].get_meta("idx") == 1


def test_getitem_list():
    seq = random_sequence()
    imgs = seq[[0, 2]]
    assert isinstance(imgs, ImageSequence)
    assert len(imgs) == 2
    assert imgs[1].get_meta("idx") == 2


def test_pandas_handler_single():
    seq = random_sequence()
    img = seq.pandas.iloc[0]
    assert isinstance(img, Image)
    assert img.get_meta("idx") == 0


def test_pandas_handler_multiple():
    seq = random_sequence()
    imgs = seq.pandas.query("idx > 0")
    assert isinstance(seq, ImageSequence)
    assert imgs[0].get_meta("idx") == 1
    assert imgs[1].get_meta("idx") == 2


def test_eager_get_item_merges_meta():
    imgs = [random_image(idx=0), random_image(idx=1), random_image(idx=2)]
    seq = EagerImageSequence(imgs, meta=[{"idx": "a"}, {"idy": "b"}, {}])

    # idx in seq[0] must be overwritten
    assert seq[0].get_meta("idx") == "a"

    # idx in seq[1] is preserved but idy added
    assert seq[1].get_meta("idx") == 1
    assert seq[1].get_meta("idy") == "b"

    # meta in resulting sequence needs to be set correctly
    seq2 = seq[1:]
    seq2._meta.iloc[0]["idy"] == "b"
    seq2._meta.iloc[0]["idx"] == 1


def test_concat_eager_seqs():
    seq1 = random_sequence(N=3)
    seq2 = random_sequence(N=3)
    seq = seq1 + seq2

    assert seq[3]._data is seq2[0]._data
    assert len(seq) == 6


def test_to_eager():
    seq = random_sequence(seq_lazy=True)
    seqe = seq.to_eager()

    assert isinstance(seqe, EagerImageSequence)
    assert_equal(seq[0].data, seqe[0].data)


def test_eager_apply_image_data():
    seq = random_sequence()
    original_data = seq[0].data.copy()

    def fn(x):
        x = x.copy()
        x[:] = 0.0
        return x

    seq2 = seq.apply_image_data(fn)
    assert_equal(seq2[0].data, 0.0)
    assert_equal(seq[0].data, original_data)


def test_lazy_apply_image_data():
    seq = random_sequence(seq_lazy=True)
    original_data = seq[0].data.copy()

    def fn(x):
        x = x.copy()
        x[:] = 0.0
        return x

    seq2 = seq.apply_image_data(fn)
    assert_equal(seq2[0].data, 0.0)
    assert_equal(seq[0].data, original_data)


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
#
#
#
#