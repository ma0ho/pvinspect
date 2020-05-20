from pvinspect import data
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from pathlib import Path
from test.utilities import *
import pandas as pd


def test_sequence_element_access():
    seq = data.datasets.poly10x6(2)
    assert seq._images[0].path == seq[0].path
    assert seq._images[1].path == seq[1].path


def test_sequence_wrapper():
    @_sequence
    def _some_fn_mis(seq: ModuleImageSequence):
        assert type(seq) == ModuleImageSequence
        return seq

    @_sequence
    def _some_fn_is(seq: ImageSequence):
        assert type(seq) == ImageSequence
        return seq

    img = random_module_image()
    res = _some_fn_mis(img)
    assert isinstance(res, ModuleImage)

    img = random_image()
    res = _some_fn_is(img)
    assert isinstance(res, Image)

    img_seq = random_module_image_sequence()
    res = _some_fn_mis(img_seq)
    assert isinstance(res, ModuleImageSequence)

    img_seq = random_image_sequence()
    res = _some_fn_is(img_seq)
    assert isinstance(res, ImageSequence)


def test_sequence_wrapper_noarg():
    @_sequence()
    def _some_fn(seq: ModuleImageSequence):
        assert type(seq) == ModuleImageSequence
        return seq

    img = random_module_image()
    res = _some_fn(img)
    assert isinstance(res, ModuleImage)


def test_sequence_wrapper_nounwrap():
    @_sequence(True)
    def _some_fn(seq: ModuleImageSequence):
        assert type(seq) == ModuleImageSequence
        return seq

    img = random_module_image()
    res = _some_fn(img)
    assert isinstance(res, ModuleImageSequence)


def test_image_from_other():
    p = Path() / "other.png"
    img = random_image()
    img2 = Image.from_other(img, path=p)
    assert img._data is img2._data
    assert img._modality is img2._modality
    assert img2._path is p


def test_module_image_from_other():
    img = random_module_image()
    img2 = ModuleImage.from_other(img, cols=10, rows=6)
    assert img._data is img2._data
    assert img._modality is img2._modality
    assert img._path is img2._path
    assert img2._cols == 10
    assert img2._rows == 6


def test_float_image_is_not_converted():
    data = np.array([[0.1]], dtype=np.float32)
    img = ModuleImage(data, EL_IMAGE, Path() / "test.png")
    assert img.dtype == DType.FLOAT
    data = np.array([[0.1]], dtype=np.float64)
    img = ModuleImage(data, EL_IMAGE, Path() / "test.png")
    assert img.dtype == DType.FLOAT


def test_apply_image_data():
    seq = random_image_sequence()
    original_data = seq[0].data.copy()

    def fn(x):
        x = x.copy()
        x[:] = 0.0
        return x

    seq2 = seq.apply_image_data(fn)
    assert_equal(seq2[0].data, 0.0)
    assert_equal(seq[0].data, original_data)


def test_image_as_type():
    img = random_image()
    assert img.dtype == DType.FLOAT
    img = img.as_type(DType.UNSIGNED_INT)
    assert img.dtype == DType.UNSIGNED_INT


def test_sequence_as_type():
    seq = random_image_sequence()
    assert seq.dtype == DType.FLOAT
    seq = seq.as_type(DType.UNSIGNED_INT)
    assert seq.dtype == DType.UNSIGNED_INT


def test_add_images():
    img1 = random_image()
    img2 = random_image()
    res = img1 + img2
    assert_equal(res.data, img1.data + img2.data)


def test_add_images_differing_dtypes():
    img1 = random_image()
    img2 = random_uint_image()
    try:
        res = img1 + img2
    except RuntimeError:
        return

    # expected an error
    assert False


def test_sub_images():
    img1 = random_image()
    img2 = random_image()
    res = img1 - img2
    assert_equal(res.data, img1.data - img2.data)


def test_sub_uint_images():
    img1 = random_uint_image()
    img2 = random_uint_image()
    res = img1 - img2
    expected = np.clip(
        img1.data.astype(np.int32) - img2.data.astype(np.int32), 0, 2 ** 16
    )
    assert_equal(res.data, expected)


def test_sub_images_differing_dtypes():
    img1 = random_image()
    img2 = random_uint_image()
    try:
        res = img1 - img2
    except RuntimeError:
        return

    # expected an error
    assert False


def test_mul_images():
    img1 = random_image()
    img2 = random_image()
    res = img1 * img2
    assert_equal(res.data, img1.data * img2.data)


def test_mul_images_differing_dtypes():
    img1 = random_image()
    img2 = random_uint_image()
    try:
        res = img1 * img2
    except RuntimeError:
        return

    # expected an error
    assert False


def test_truediv_images():
    img1 = random_image()
    img2 = random_image()
    res = img1 / img2
    assert_equal(res.data, img1.data / img2.data)


def test_truediv_images_nonfloat():
    img1 = random_image()
    img2 = random_uint_image()
    try:
        res = img1 / img2
    except RuntimeError:
        return

    # expected an error
    assert False


def test_floordiv_images():
    img1 = random_uint_image()
    img2 = random_uint_image()
    res = img1 // img2
    assert_equal(res.data, img1.data // img2.data)


def test_floordiv_images_differing_dtypes():
    img1 = random_image()
    img2 = random_uint_image()
    try:
        res = img1 // img2
    except RuntimeError:
        return

    # expected an error
    assert False


def test_mod_images():
    img1 = random_uint_image()
    img2 = random_uint_image()
    res = img1 % img2
    assert_equal(res.data, img1.data % img2.data)


def test_mod_images_differing_dtypes():
    img1 = random_image()
    img2 = random_uint_image()
    try:
        res = img1 % img2
    except RuntimeError:
        return

    # expected an error
    assert False


def test_pow_images():
    img1 = random_image()
    img2 = random_image()
    res = img1 ** img2
    assert_equal(res.data, img1.data ** img2.data)


def test_pow_images_differing_dtypes():
    img1 = random_image()
    img2 = random_uint_image()
    try:
        res = img1 ** img2
    except RuntimeError:
        return

    # expected an error
    assert False


def test_add_image_sequence():
    imgs1 = random_image_sequence()
    imgs2 = random_image_sequence()
    res = imgs1 + imgs2

    for img1, img2, r in zip(imgs1, imgs2, res):
        assert_equal(r.data, img1.data + img2.data)


def test_sub_image_sequence():
    imgs1 = random_image_sequence()
    imgs2 = random_image_sequence()
    res = imgs1 - imgs2

    for img1, img2, r in zip(imgs1, imgs2, res):
        assert_equal(r.data, img1.data - img2.data)


def test_mul_image_sequence():
    imgs1 = random_image_sequence()
    imgs2 = random_image_sequence()
    res = imgs1 * imgs2

    for img1, img2, r in zip(imgs1, imgs2, res):
        assert_equal(r.data, img1.data * img2.data)


def test_truediv_image_sequence():
    imgs1 = random_image_sequence()
    imgs2 = random_image_sequence()
    res = imgs1 / imgs2

    for img1, img2, r in zip(imgs1, imgs2, res):
        assert_equal(r.data, img1.data / img2.data)


def test_floordiv_image_sequence():
    imgs1 = random_image_sequence()
    imgs2 = random_image_sequence()
    res = imgs1 // imgs2

    for img1, img2, r in zip(imgs1, imgs2, res):
        assert_equal(r.data, img1.data // img2.data)


def test_mod_image_sequence():
    imgs1 = random_uint_image_sequence()
    imgs2 = random_uint_image_sequence()
    res = imgs1 % imgs2

    for img1, img2, r in zip(imgs1, imgs2, res):
        assert_equal(r.data, img1.data % img2.data)


def test_pow_image_sequence():
    imgs1 = random_image_sequence()
    imgs2 = random_image_sequence()
    res = imgs1 ** imgs2

    for img1, img2, r in zip(imgs1, imgs2, res):
        assert_equal(r.data, img1.data ** img2.data)


def test_image_meta():
    data = random_image().data
    path = random_image().path
    img_test = Image(data=data, path=path, meta={"key": "value"})

    assert img_test.has_meta("key")
    assert not img_test.has_meta("key2")
    assert img_test.get_meta("key") == "value"
    assert "key" in img_test.list_meta()

    # check from_other preserves meta
    img_other = Image.from_other(img_test, data=random_image().data)
    assert img_other.get_meta("key") == "value"

    # check from_other appends meta
    img_other2 = Image.from_other(img_other, meta={"key2": "value2"})
    assert img_other2.get_meta("key") == "value"
    assert img_other2.get_meta("key2") == "value2"
    assert "key" in img_other2.list_meta()
    assert "key2" in img_other2.list_meta()

    # but img_other not modified
    assert not img_other.has_meta("key2")


def test_image_data_is_immutable():
    data = random_image().data
    assert data.flags["WRITEABLE"] == False


def test_image_meta_is_immutable():
    img = random_image()
    arraymeta = np.zeros(1)
    othermeta = [1, 2, 3]
    img_with_meta = Image.from_other(
        img, meta={"arraymeta": arraymeta, "othermeta": othermeta}
    )

    # array should be immutable
    assert img_with_meta.get_meta("arraymeta").flags["WRITEABLE"] == False

    # other should return a copy
    assert img_with_meta.get_meta("othermeta") is not othermeta


def test_image_meta_to_pandas():
    img = random_image(meta={"k": 1})
    meta = img.meta_to_pandas()

    assert isinstance(meta, pd.Series)
    assert meta["k"] == 1


def test_image_meta_from_path():
    img = random_image(path=Path("img_d12.png"))
    img = img.meta_from_path(pattern=r"img_d(\d+)", key="k", target_type=int, group_n=1)
    meta = img.meta_to_pandas()

    assert meta["k"] == 12
    assert isinstance(meta["k"], int)


def test_image_meta_from_path_transform():
    def t(x):
        assert isinstance(x, str)
        return int(x) * 2

    img = random_image(path=Path("img_d12.png"))
    img = img.meta_from_path(
        pattern=r"img_d(\d+)", key="k", target_type=int, group_n=1, transform=t
    )
    meta = img.meta_to_pandas()

    assert meta["k"] == 24
    assert isinstance(meta["k"], int)


def test_sequence_meta_to_pandas():
    img1 = random_image(meta={"k": 1})
    img2 = random_image(meta={"k": 2})
    seq = ImageSequence([img1, img2], same_camera=False)
    meta = seq.meta_to_pandas()

    assert isinstance(meta, pd.DataFrame)
    assert meta.iloc[0]["k"] == 1
    assert meta.iloc[1]["k"] == 2


def test_sequence_meta_from_path():
    img1 = random_image(path=Path("img_d12.png"))
    img2 = random_image(path=Path("img_d13.png"))
    seq = ImageSequence([img1, img2], same_camera=False)
    seq = seq.meta_from_path(pattern=r"img_d(\d+)", key="k", target_type=int, group_n=1)
    meta = seq.meta_to_pandas()

    assert isinstance(meta, pd.DataFrame)
    assert meta.iloc[0]["k"] == 12
    assert meta.iloc[1]["k"] == 13


def test_sequence_meta_query():
    img1 = random_image(meta={"k": 1})
    img2 = random_image(meta={"k": 2})
    seq = ImageSequence([img1, img2], same_camera=False)
    seq = seq.query("k == 1")

    assert len(seq) == 1
    assert seq[0].get_meta("k") == 1
