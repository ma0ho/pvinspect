from pvinspect.data.io import (
    read_module_images,
    save_image,
    save_images,
    read_module_image,
    read_partial_module_images,
    read_images,
    read_image,
)
from pvinspect.data import datasets
import pvinspect.data as data
from pvinspect.preproc.detection import locate_module_and_cells, segment_cells
from pathlib import Path
import numpy as np
from pvinspect.data.image import *
from skimage.io import imsave, imread
from test.utilities import *

EXAMPLES = (
    Path(__file__).absolute().parent.parent.parent
    / "pvinspect"
    / "data"
    / "datasets"
    / "20191219_poly10x6"
)


def _check_download_demo():
    data.datasets.poly10x6(1)


def test_read_sequence():
    _check_download_demo()
    seq = read_module_images(EXAMPLES, EL_IMAGE, True)
    assert len(seq) == 20
    assert seq.modality == EL_IMAGE


def test_limit():
    _check_download_demo()
    seq = read_module_images(EXAMPLES, EL_IMAGE, False, N=2)
    assert len(seq) == 2


def test_same_camera():
    _check_download_demo()
    seq = read_module_images(EXAMPLES, EL_IMAGE, True, N=1)
    assert seq.same_camera == True


def test_different_dtypes():
    _check_download_demo()
    seq = read_module_images(EXAMPLES, EL_IMAGE, False, allow_different_dtypes=True)
    assert seq.dtype == None


def test_dtype():
    _check_download_demo()
    seq = read_module_images(EXAMPLES, EL_IMAGE, False)
    assert seq.dtype == DType.UNSIGNED_INT


def test_shape():
    _check_download_demo()
    seq = read_module_images(EXAMPLES, EL_IMAGE, True)
    assert seq.shape == (2052, 2046)


def test_shape_not_same_camera():
    _check_download_demo()
    seq = read_module_images(EXAMPLES, EL_IMAGE, False)
    assert seq.shape == None


def test_filter():
    _check_download_demo()
    seq = read_module_images(EXAMPLES, EL_IMAGE, True, pattern="*.tif", N=1)
    assert len(seq) == 1
    seq = read_module_images(EXAMPLES, EL_IMAGE, True, pattern=("*.png", "*.tif"), N=1)
    assert len(seq) == 1
    try:
        seq = read_module_images(EXAMPLES, EL_IMAGE, True, pattern="*.png", N=1)
        assert False
    except:
        assert True


def test_save_and_read_image(tmp_path):
    img = datasets.poly10x6(1)[0]
    save_image(tmp_path / "img.tif", img)
    img_read = read_module_image(tmp_path / "img.tif", EL_IMAGE)
    assert np.linalg.norm(img.data.flatten() - img_read.data.flatten()) == 0


def test_save_image_sequence(tmp_path):
    seq = datasets.poly10x6(5)
    save_images(tmp_path, seq)

    for img in seq:
        img_read = read_module_image(tmp_path / img.path.name, EL_IMAGE)


def test_save_cell_images(tmp_path):
    cells = segment_cells(locate_module_and_cells(datasets.poly10x6(1)[0]))
    save_images(tmp_path, cells)

    for cell in cells:
        p = tmp_path / "{}_row{:02d}_col{:02d}{}".format(
            cell.path.stem, cell.row, cell.col, cell.path.suffix
        )
        img_read = read_module_image(p, EL_IMAGE)


def test_read_images():
    _check_download_demo()
    seq = read_images(EXAMPLES, True, N=2)
    assert isinstance(seq, ImageSequence)
    for img in seq:
        assert isinstance(img, Image)


def test_read_module_images():
    _check_download_demo()
    seq = read_module_images(EXAMPLES, EL_IMAGE, True, N=2)
    assert isinstance(seq, ModuleImageSequence)
    for img in seq:
        assert isinstance(img, ModuleImage)


def test_read_partial_module_images():
    _check_download_demo()
    seq = read_partial_module_images(EXAMPLES, EL_IMAGE, True, N=2)
    assert isinstance(seq, ModuleImageSequence)
    for img in seq:
        assert isinstance(img, PartialModuleImage)


def test_save_image_with_visualization(tmp_path: Path):
    img = datasets.poly10x6(1)[0]
    p = tmp_path / "img.pdf"
    save_image(p, img, with_visusalization=True)
    assert p.is_file()


def test_force_dtype(tmp_path: Path):
    # create samples
    a = np.random.rand(10, 10) * 100
    imsave(tmp_path / "wrong_float.tif", a)

    seq = read_image(tmp_path / "wrong_float.tif", force_dtype=DType.UNSIGNED_INT)
    assert seq.dtype == DType.UNSIGNED_INT


def test_save_float_image_conversion(tmp_path: Path):
    img = random_image()
    img = Image.from_other(img, data=img.data.astype(np.float64))
    save_image(tmp_path / "test.tif", img)

    img = imread(tmp_path / "test.tif")
    assert img.dtype == np.float32


def test_hierachical_with_keys_save(tmp_path: Path):
    img1 = random_image(meta={"m1": 1, "m2": 1})
    img2 = random_image(meta={"m1": 1, "m2": 2})
    img3 = random_image(meta={"m1": 2, "m2": 1})
    seq = ImageSequence([img1, img2, img3], same_camera=True)

    save_images(tmp_path, seq, hierarchical=["m1", "m2"])

    for img in seq:
        p = (
            tmp_path
            / ("m1_" + str(img.get_meta("m1")))
            / ("m2_" + str(img.get_meta("m2")))
            / img.path.name
        )
        ref = read_image(p)
        assert_equal(img.data, ref.data)


def test_hierachical_without_keys_save(tmp_path: Path):
    img1 = random_image(meta={"m1": 1, "m2": 1})
    img2 = random_image(meta={"m1": 1, "m2": 2})
    img3 = random_image(meta={"m1": 2, "m2": 1})
    seq = ImageSequence([img1, img2, img3], same_camera=True)

    save_images(tmp_path, seq, hierarchical=["m1", "m2"], include_meta_keys=False)

    for img in seq:
        p = tmp_path / str(img.get_meta("m1")) / str(img.get_meta("m2")) / img.path.name
        ref = read_image(p)
        assert_equal(img.data, ref.data)
