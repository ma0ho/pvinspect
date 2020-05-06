from pvinspect.data.image import *
from pvinspect.preproc.calibration import *
from pvinspect.preproc.process import *
from pvinspect.preproc.calibration import _calibrate_flatfield
from pvinspect.preproc.process import _compensate_flatfield
from pvinspect.data import datasets
import numpy as np
from skimage.exposure import rescale_intensity
from test.utilities import assert_equal
import cv2


def _make_image_seq(imgs):
    imgs = [ModuleImage(img, EL_IMAGE, None) for img in imgs]
    return ModuleImageSequence(imgs, True, False)


def test_calibrate_flatfield_linear():
    img0 = np.random.random((10, 10)) / 10
    img1 = (-np.random.random((10, 10)) / 10) + 1.0
    coeff = 1 / (img1 - img0)

    coeff_result = _calibrate_flatfield([img0, img1], [0, 1.0])
    assert_equal(coeff_result[1], coeff)
    assert_equal(coeff_result[0], -coeff * img0)


def test_calibrate_flatfield_quadratic():
    img0 = np.random.random((10, 10)) / 10
    img1 = np.random.random((10, 10)) / 10 + 0.5 - 0.05
    img2 = (-np.random.random((10, 10)) / 10) + 1.0

    def compensate(img, coeff):
        return coeff[2] * img ** 2 + coeff[1] * img + coeff[0]

    coeff = _calibrate_flatfield([img0, img1, img2], [0, 0.5, 1.0])
    assert_equal(np.zeros_like(img0), compensate(img0, coeff))
    assert_equal(np.full_like(img1, 0.5), compensate(img1, coeff))
    assert_equal(np.full_like(img1, 1.0), compensate(img2, coeff))


def test_calibrate_flatfield_least_squares_linear():
    img0 = np.random.random((10, 10)) / 10
    img2 = (-np.random.random((10, 10)) / 10) + 1.0
    coeff = 1 / (img2 - img0)
    img1 = (np.full_like(img0, 0.5) + img0) / coeff
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

    coeff = _calibrate_flatfield([img0, img1, img2], [0, 0.5, 1.0])
    compensated = _compensate_flatfield(_make_image_seq([img0, img1, img2]), coeff)
    assert_equal(np.zeros_like(img0), compensated[0].data)
    assert_equal(np.full_like(img1, 0.5), compensated[1].data)
    assert_equal(np.full_like(img1, 1.0), compensated[2].data)


def test_calibrate_and_compensate_preserves_range():
    img0 = np.array([[0]], dtype=np.uint32)
    img1 = np.array([[10000]], dtype=np.uint32)
    img_test = np.array([[8000]], dtype=np.uint32)

    coeff = calibrate_flatfield(_make_image_seq([img0, img1]), [0, 1.0])
    compensated = compensate_flatfield(_make_image_seq([img_test]), coeff)
    assert_equal(compensated[0].data, 0.8)


def test_calibrate_and_compensate_flatfield_realdata():
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
    coeff = calibrate_flatfield(ImageSequence(imgs_list, same_camera=True), ex_list)
    test_img = ModuleImage(
        ds["{:d}A".format(ex_all[1])][0]._data, EL_IMAGE, path=Path() / "test.png"
    )
    compensated = compensate_flatfield(test_img, coeff)
    data = compensated.data
    assert data.min() >= 0.0
    assert data.max() <= 1.0
    assert data.std() <= 0.01


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
