import datetime
import json
from pathlib import Path
from test.utilities import *

import numpy as np
import pvinspect.data as data
from pvinspect.data import datasets
from pvinspect.data.image import *
from pvinspect.data.io import *
from pvinspect.data.io import (
    _get_meta_cache_path,
    _load_json_meta_hook,
    _prepare_json_meta,
)
from pvinspect.preproc.detection import locate_module_and_cells, segment_cells
from shapely.geometry import Point, Polygon
from skimage import io as skio


def _prepare_test_filename(i):
    return "{:03d}.png".format(i)


def _prepare_test_data(i):
    np.random.seed(i)
    return np.random.randint(low=0, high=255, size=(50, 50)).astype(np.uint8)


def _prepare_test_meta(i):
    return pd.Series({"idx": i})


def _prepare_test_imgs(path: Path, N: int = 3):
    for i in range(N):
        skio.imsave(path / _prepare_test_filename(i), _prepare_test_data(i))
    pd.DataFrame([_prepare_test_meta(i) for i in range(N)]).to_pickle(path / "meta.pck")  # type: ignore


def _prepare_test_img_obj(i):
    return EagerImage(_prepare_test_data(i), meta=_prepare_test_meta(i))


def _prepare_test_seq_obj(N=3):
    return EagerImageSequence.from_images([_prepare_test_img_obj(i) for i in range(N)])


def test_read_sequence(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    seq = read_images(tmp_path)
    assert len(seq) == 3
    assert isinstance(seq, EagerImageSequence)
    assert_equal(seq[0].data, _prepare_test_data(0))


def test_read_sequence_lazy(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    seq = read_images(tmp_path, lazy=True)
    assert len(seq) == 3
    assert isinstance(seq, LazyImageSequence)
    assert_equal(seq[0].data, _prepare_test_data(0))


def test_limit(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    seq = read_images(tmp_path, N=2)
    assert len(seq) == 2


def test_dtype(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    seq = read_images(tmp_path)
    seq[0].dtype == DType.UNSIGNED_INT


def test_read_image(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    img = read_image(tmp_path / _prepare_test_filename(0))
    assert isinstance(img, EagerImage)
    assert_equal(img.data, _prepare_test_data(0))


def test_read_image_lazy(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    img = read_image(tmp_path / _prepare_test_filename(0), lazy=True)
    assert isinstance(img, LazyImage)
    assert_equal(img.data, _prepare_test_data(0))


def test_save_image(tmp_path):
    img = _prepare_test_img_obj(0)
    save_image(tmp_path / "img.png", img)
    img_read = read_image(tmp_path / "img.png")
    assert_equal(img_read, img.data)
    assert img_read.dtype == img.data.dtype


def test_save_image_with_filename(tmp_path):
    img = random_image(original_filename="test.png")
    img_read = read_image(tmp_path / "test.png")
    assert_equal(img_read, img.data)


def test_save_sequence(tmp_path):
    seq = _prepare_test_seq_obj(2)
    save_images(tmp_path, seq)
    img_read = skio.imread(tmp_path / "00000.png")
    assert_equal(seq[0].data, img_read)
    img_read = skio.imread(tmp_path / "00001.png")
    assert_equal(seq[1].data, img_read)


def test_read_image_with_meta(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    img = read_image(tmp_path / _prepare_test_filename(0), with_meta=True)
    assert img.meta.equals(_prepare_test_meta(0))


def test_read_sequence_with_meta(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    seq = read_images(tmp_path, with_meta=True)
    assert seq.meta.equals([_prepare_test_meta(i) for i in range(len(seq))])


def test_save_sequence_with_meta(tmp_path: Path):
    seq = _prepare_test_seq_obj()
    save_images(tmp_path, seq, widt_meta=True)
    seq_read = read_images(tmp_path, with_meta=True)
    assert seq_read.meta.equals(eq.meta)


def test_save_image_with_meta():
    img = _prepare_test_img_obj(0)
    save_image(tmp_path / "test.png", img, with_meta=True)
    img_read = read_image(tmp_path / "test.png", with_meta=True)
    assert img.meta.equals(img_read.meta)


# def test_filter():
#    _check_download_demo()
#    seq = read_module_images(EXAMPLES, EL_IMAGE, True, pattern="*.tif", N=1)
#    assert len(seq) == 1
#    seq = read_module_images(EXAMPLES, EL_IMAGE, True, pattern=("*.png", "*.tif"), N=1)
#    assert len(seq) == 1
#    try:
#        seq = read_module_images(EXAMPLES, EL_IMAGE, True, pattern="*.png", N=1)
#        assert False
#    except:
#        assert True
# def test_hierachical_with_keys_save(tmp_path: Path):
#    img1 = random_image(meta={"m1": 1, "m2": 1})
#    img2 = random_image(meta={"m1": 1, "m2": 2})
#    img3 = random_image(meta={"m1": 2, "m2": 1})
#    seq = ImageSequence([img1, img2, img3], same_camera=True)
#
#    save_images(tmp_path, seq, hierarchical=["m1", "m2"])
#
#    for img in seq:
#        p = (
#            tmp_path
#            / ("m1_" + str(img.get_meta("m1")))
#            / ("m2_" + str(img.get_meta("m2")))
#            / img.path.name
#        )
#        ref = read_image(p, lazy=False)
#        assert_equal(img.data, ref.data)
#
#
# def test_hierachical_without_keys_save(tmp_path: Path):
#    img1 = random_image(meta={"m1": 1, "m2": 1})
#    img2 = random_image(meta={"m1": 1, "m2": 2})
#    img3 = random_image(meta={"m1": 2, "m2": 1})
#    seq = ImageSequence([img1, img2, img3], same_camera=True)
#
#    save_images(tmp_path, seq, hierarchical=["m1", "m2"], include_meta_keys=False)
#
#    for img in seq:
#        p = tmp_path / str(img.get_meta("m1")) / str(img.get_meta("m2")) / img.path.name
#        ref = read_image(p, lazy=False)
#        assert_equal(img.data, ref.data)
#
# def test_save_with_prefix(tmp_path: Path):
#    imgs = random_image_sequence()
#    save_images(tmp_path, imgs, filename_prefix="XY_")
#
#    for img in imgs:
#        fn = "XY_" + img.path.name
#        assert (tmp_path / fn).is_file()
#
#
# def test_save_with_suffix(tmp_path: Path):
#    imgs = random_image_sequence()
#    save_images(tmp_path, imgs, filename_suffix="_XY")
#
#    for img in imgs:
#        fn = img.path.stem + "_XY" + img.path.suffix
#        assert (tmp_path / fn).is_file()
