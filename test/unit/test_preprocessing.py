from pvinspect.data.image import *
from pvinspect.preproc.calibration import *
from pvinspect.preproc.calibration import _calibrate_flatfield, _compensate_flatfield
from pvinspect.data import datasets
import numpy as np
from skimage.exposure import rescale_intensity
from test.utilities import assert_equal
import cv2


def _prepare_ff_data():
    ds = datasets.calibration_ipv40CCD_FF(N=2)
    del ds["1A"]
    imgs_list = []
    ex_list = []
    ex_all = [int(x[0]) for x in ds.keys()]
    ex_max = max(ex_all)
    for ex in ex_all:
        for img in ds["{:d}A".format(ex)]:
            imgs_list.append(img)
            ex_list.append(ex / ex_max)

    return ImageSequence(imgs_list, same_camera=True), ex_list


def _prepare_ff_test_img():
    ds = datasets.calibration_ipv40CCD_FF(N=2)
    del ds["1A"]
    imgs_list = []
    ex_list = []
    ex_all = [int(x[0]) for x in ds.keys()]
    ex_max = max(ex_all)
    for ex in ex_all:
        for img in ds["{:d}A".format(ex)]:
            imgs_list.append(img)
            ex_list.append(ex / ex_max)
    test_img = ModuleImage(
        ds["{:d}A".format(ex_all[1])][0]._data, EL_IMAGE, path=Path() / "test.png"
    )
    return test_img


def _make_image_seq(imgs):
    imgs = [ModuleImage(img, EL_IMAGE, None) for img in imgs]
    return ModuleImageSequence(imgs, True, False)


def test_calibrate_flatfield_linear():
    img0 = np.random.random((10, 10)) / 10
    img1 = (-np.random.random((10, 10)) / 10) + 1.0
    coeff = 1 / (img1 - img0)
    coeff *= np.mean(img1)

    coeff_result = _calibrate_flatfield([img0, img1], [0, 1.0], order=1)
    assert_equal(coeff_result[1], coeff)
    assert_equal(coeff_result[0], -coeff * img0)


def test_calibrate_flatfield_quadratic():
    img0 = np.random.random((10, 10)) / 10
    img1 = np.random.random((10, 10)) / 10 + 0.5 - 0.05
    img2 = (-np.random.random((10, 10)) / 10) + 1.0
    mean = np.mean(img2)

    def compensate(img, coeff):
        return coeff[2] * img ** 2 + coeff[1] * img + coeff[0]

    coeff = _calibrate_flatfield([img0, img1, img2], [0, 0.5, 1.0], order=2)
    assert_equal(np.zeros_like(img0), compensate(img0, coeff))
    assert_equal(np.full_like(img1, 0.5 * mean), compensate(img1, coeff))
    assert_equal(np.full_like(img1, 1.0 * mean), compensate(img2, coeff))


def test_calibrate_flatfield_least_squares_linear():
    img0 = np.random.random((10, 10)) / 10
    img2 = (-np.random.random((10, 10)) / 10) + 1.0
    coeff = 1 / (img2 - img0)
    img1 = (np.full_like(img0, 0.5) + img0) / coeff
    coeff *= np.mean(img2)
    coeff_result = _calibrate_flatfield([img0, img1, img2], [0, 0.5, 1.0], order=1)

    assert_equal(coeff_result[1], coeff, 0.1)
    assert_equal(coeff_result[0], -coeff * img0, 0.1)


def test_compensate_flatfield():
    img = np.random.random((10, 10))
    coeff = np.random.random((4, 10, 10)) / 10
    target = coeff[0] + coeff[1] * img + coeff[2] * img ** 2 + coeff[3] * img ** 3

    res = _compensate_flatfield(_make_image_seq([img]), coeff)[0].data
    assert_equal(res, target, 0.1)


def test_calibrate_and_compensate_flatfield():
    img0 = np.random.random((10, 10)) / 10
    img1 = np.random.random((10, 10)) / 10 + 0.5 - 0.05
    img2 = (-np.random.random((10, 10)) / 10) + 1.0
    mean = np.mean(img2)

    coeff = _calibrate_flatfield([img0, img1, img2], [0, 0.5, 1.0], order=2)
    compensated = _compensate_flatfield(_make_image_seq([img0, img1, img2]), coeff)
    assert_equal(np.zeros_like(img0), compensated[0].data)
    assert_equal(np.full_like(img1, 0.5 * mean), compensated[1].data)
    assert_equal(np.full_like(img1, 1.0 * mean), compensated[2].data)


def test_calibrate_and_compensate_uint_preserves_range():
    img0 = np.array([[0]], dtype=DTYPE_UNSIGNED_INT)
    img1 = np.array([[10000]], dtype=DTYPE_UNSIGNED_INT)
    img_test = np.array([[8000]], dtype=DTYPE_UNSIGNED_INT)

    coeff = calibrate_flatfield(_make_image_seq([img0, img1]), [0, 1.0])
    compensated = compensate_flatfield(_make_image_seq([img_test]), coeff)
    assert_equal(compensated[0].data, 8000)


def test_calibrate_and_compensate_float_preserves_range():
    img0 = np.array([[0.0]], dtype=DTYPE_FLOAT)
    img1 = np.array([[1.0]], dtype=DTYPE_FLOAT)
    img_test = np.array([[0.8]], dtype=DTYPE_FLOAT)

    coeff = calibrate_flatfield(_make_image_seq([img0, img1]), [0, 1.0])
    compensated = compensate_flatfield(_make_image_seq([img_test]), coeff)
    assert_equal(compensated[0].data, 0.8)


def test_calibrate_and_compensate_int_preserves_range():
    img0 = np.array([[-10]], dtype=DTYPE_INT)
    img1 = np.array([[10]], dtype=DTYPE_INT)
    img_test = np.array([[0]], dtype=DTYPE_INT)

    coeff = calibrate_flatfield(_make_image_seq([img0, img1]), [0, 1.0])
    compensated = compensate_flatfield(_make_image_seq([img_test]), coeff)
    assert_equal(compensated[0].data, 0)


def test_calibrate_and_compensate_flatfield_realdata():
    imgs_list, ex_list = _prepare_ff_data()
    test_img = _prepare_ff_test_img()
    coeff = calibrate_flatfield(imgs_list, ex_list)
    compensated = compensate_flatfield(test_img, coeff)
    data = compensated.data
    assert data.std() <= 0.01 * np.mean(test_img.data)


def test_calibrate_distortion():
    ds = datasets.calibration_ipv40CCD_distortion(N=3)
    res = calibrate_distortion(ds, (7, 7))

    assert len(res) == 4
    assert res[0].shape == (3, 3)
    assert res[1].shape == (1, 5)
    assert res[2].shape == (3, 3)
    assert len(res[3]) == 4


def test_compensate_distortion_identity():
    imgs = datasets.calibration_ipv40CCD_distortion(N=2)
    mtx = np.identity(3)
    dist = np.zeros((1, 5), dtype=np.float32)
    roi = (0, 0, imgs.shape[0], imgs.shape[1])
    imgs_comp = compensate_distortion(imgs, mtx, dist, mtx, roi)

    assert (imgs_comp[0].data - imgs[0].data).sum() == 0
    assert (imgs_comp[1].data - imgs[1].data).sum() == 0


def test_calibrate_compensate_distortion_realdata():
    ds = datasets.calibration_ipv40CCD_distortion()
    res = calibrate_distortion(ds, (7, 7))
    ds_comp = compensate_distortion(ds, *res)

    for img, comp in zip(ds, ds_comp):
        assert_equal(comp.data.mean(), img.data.mean(), 0.01 * img.data.max())


def test_calibration_object(tmp_path: Path):
    ff_imgs, ex_list = _prepare_ff_data()
    test_img = _prepare_ff_test_img()
    dist_imgs = datasets.calibration_ipv40CCD_distortion(N=8)

    # perform calibration
    calib = Calibration()
    calib.calibrate_flatfield(images=ff_imgs, targets=ex_list)
    calib.calibrate_distortion(images=dist_imgs, checkerboard_size=(7, 7))

    # save to disk
    calib.save(tmp_path / "calib.pck")

    # load from disk
    calib_loaded = load_calibration(tmp_path / "calib.pck")

    # perform compensation with both
    comp = calib.process(test_img)
    comp_loaded = calib_loaded.process(test_img)

    # assert both result in the same image
    assert_equal(comp.data, comp_loaded.data)


def test_flatfield_sequences_input():
    seqs = list(datasets.calibration_ipv40CCD_FF(N=2).values())
    seqs = [seqs[0], seqs[2]]
    test_img = _prepare_ff_test_img()

    coeff = calibrate_flatfield(seqs, targets=[0.0, 1.0])
    comp = compensate_flatfield(test_img, coeff)

    data = comp.data
    assert data.std() <= 0.01 * np.mean(seqs[1][0].data)
