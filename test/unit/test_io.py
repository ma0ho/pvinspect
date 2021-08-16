from pathlib import Path
from test.utilities import *

import numpy as np
from pvinspect.data.image.image import *
from pvinspect.data.io import *
from skimage import io as skio


def _prepare_test_filename(i):
    return "{:03d}.png".format(i)


def _prepare_test_data(i):
    np.random.seed(i)
    return np.random.randint(low=0, high=255, size=(50, 50)).astype(np.uint8)


def _prepare_test_meta(i):
    return pd.Series({"original_filename": "{:03d}.png".format(i), "idx": i})


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
    seq = read_images(tmp_path, limit=2)
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
    assert_equal(img_read.data, img.data)
    assert img_read.dtype == img.dtype


def test_save_images_with_filename(tmp_path):
    imgs = EagerImageSequence.from_images(
        [
            random_image(original_filename="test1.tif"),
            random_image(original_filename="test2.tif"),
        ]
    )
    save_images(tmp_path, imgs)
    img_read1 = read_image(tmp_path / "test1.tif")
    img_read2 = read_image(tmp_path / "test2.tif")
    assert_equal(img_read1.data, imgs[0].data)
    assert_equal(img_read2.data, imgs[1].data)


def test_save_sequence(tmp_path):
    seq = _prepare_test_seq_obj(2)
    save_images(tmp_path, seq)
    img_read = skio.imread(tmp_path / _prepare_test_filename(0))
    assert_equal(seq[0].data, img_read)
    img_read = skio.imread(tmp_path / _prepare_test_filename(1))
    assert_equal(seq[1].data, img_read)


def test_save_sequence_without_given_filename(tmp_path: Path):
    seq = EagerImageSequence.from_images([random_image(), random_image()])
    save_images(tmp_path, seq, default_filetype="tif")
    img_read = skio.imread(tmp_path / "0.tif")
    assert_equal(seq[0].data, img_read)
    img_read = skio.imread(tmp_path / "1.tif")
    assert_equal(seq[1].data, img_read)


def test_read_image_with_meta(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    img = read_image(tmp_path / _prepare_test_filename(0), with_meta=True)
    assert img.meta.equals(_prepare_test_meta(0))


def test_read_sequence_with_meta(tmp_path: Path):
    _prepare_test_imgs(path=tmp_path)
    seq = read_images(tmp_path, with_meta=True)
    assert seq.meta.equals(
        pd.DataFrame([_prepare_test_meta(i) for i in range(len(seq))])
    )


def test_save_sequence_with_meta(tmp_path: Path):
    seq = _prepare_test_seq_obj()
    save_images(tmp_path, seq, with_meta=True)
    seq_read = read_images(tmp_path, with_meta=True)
    assert seq_read.meta.equals(seq.meta)


def test_save_image_with_meta(tmp_path: Path):
    pass  # currently not supported with PandasMetaDriver


def test_save_image_sequence_to_nonempty_dir_raises_error(tmp_path: Path):
    (tmp_path / "x").touch()  # create some file

    try:
        save_images(tmp_path, _prepare_test_seq_obj())
        assert False  # this should never be reached
    except RuntimeError:
        assert True


def test_read_sequence_with_common_meta(tmp_path: Path):
    _prepare_test_imgs(tmp_path)
    common_meta = pd.Series({"a": 1, "b": 2})
    seq = read_images(tmp_path, common_meta=common_meta)
    assert np.all((seq.meta["a"] == 1))
    assert np.all((seq.meta["b"] == 2))


def test_read_sequence_with_common_meta_dict(tmp_path: Path):
    _prepare_test_imgs(tmp_path)
    common_meta = {"a": 1, "b": 2}
    seq = read_images(tmp_path, common_meta=common_meta)
    assert np.all((seq.meta["a"] == 1))
    assert np.all((seq.meta["b"] == 2))


def test_read_sequence_with_str_pattern(tmp_path: Path):
    test_imgs = EagerImageSequence.from_images(
        [
            random_image(original_filename="xabc.tif"),
            random_image(original_filename="yabc.tif"),
        ]
    )
    save_images(tmp_path, test_imgs)
    read_imgs = read_images(tmp_path, pattern="y*")
    assert len(read_imgs) == 1
    assert read_imgs[0].get_meta("original_filename") == "yabc.tif"


def test_read_sequence_with_list_pattern(tmp_path: Path):
    test_imgs = EagerImageSequence.from_images(
        [
            random_image(original_filename="xabc.tif"),
            random_image(original_filename="yabc.tif"),
            random_image(original_filename="ybcd.tif"),
        ]
    )
    save_images(tmp_path, test_imgs)
    read_imgs = read_images(tmp_path, pattern="y*")
    assert len(read_imgs) == 2
    assert read_imgs[0].get_meta("original_filename") == "yabc.tif"
    assert read_imgs[1].get_meta("original_filename") == "ybcd.tif"


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
