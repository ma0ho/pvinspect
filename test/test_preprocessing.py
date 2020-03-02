from pvinspect.data.image import *
from pvinspect.preproc.calibration import *
from pvinspect.preproc.process import *
from pvinspect.preproc.calibration import _calibrate_flatfield
from pvinspect.preproc.process import _compensate_flatfield
import numpy as np
from skimage.exposure import rescale_intensity
from test.utilities import assert_equal

def _make_image_seq(imgs):
    imgs = [ModuleImage(img, EL_IMAGE, None) for img in imgs]
    return ModuleImageSequence(imgs, True, False)

def test_calibrate_flatfield_linear():
    img0 = np.random.random((10,10))/10
    img1 = (-np.random.random((10,10))/10) + 1.0
    coeff = 1/(img1-img0)

    coeff_result = _calibrate_flatfield(_make_image_seq([img0, img1]), [0, 1.0])
    assert_equal(coeff_result[1], coeff)
    assert_equal(coeff_result[0], -img0)

def test_calibrate_flatfield_quadratic():
    img0 = np.random.random((10,10))/10
    img1 = np.random.random((10,10))/10 + 0.5 - 0.05
    img2 = (-np.random.random((10,10))/10) + 1.0

    def compensate(img, coeff):
        return coeff[2]*img**2 + coeff[1]*img + coeff[0]

    coeff = _calibrate_flatfield(_make_image_seq([img0, img1, img2]), [0, 0.5, 1.0])
    assert_equal(np.zeros_like(img0), compensate(img0, coeff))
    assert_equal(np.full_like(img1, 0.5), compensate(img1, coeff))
    assert_equal(np.full_like(img1, 1.0), compensate(img2, coeff))

def test_calibrate_flatfield_least_squares_linear():
    img0 = np.random.random((10,10))/10
    img2 = (-np.random.random((10,10))/10) + 1.0
    coeff = 1/(img2-img0)
    img1 = (np.full_like(img0, 0.5)+img0) / coeff
    coeff_result = _calibrate_flatfield(_make_image_seq([img0, img1, img2]), [0, 0.5, 1.0], order=1)

    assert_equal(coeff_result[1], coeff, 0.1)
    assert_equal(coeff_result[0], -img0, 0.1)

def test_compensate_flatfield():
    img = np.random.random((10,10))
    coeff = np.random.random((4,10,10))
    target = coeff[0] + coeff[1]*img + coeff[2]*img**2 + coeff[3]*img**3

    res = _compensate_flatfield(_make_image_seq([img]), coeff)[0].data
    assert_equal(res, target, 0.2)

def test_calibrate_and_compensate_flatfield():
    img0 = np.random.random((10,10))/10
    img1 = np.random.random((10,10))/10 + 0.5 - 0.05
    img2 = (-np.random.random((10,10))/10) + 1.0

    coeff = _calibrate_flatfield(_make_image_seq([img0, img1, img2]), [0, 0.5, 1.0])
    compensated = _compensate_flatfield(_make_image_seq([img0, img1, img2]), coeff)
    assert_equal(np.zeros_like(img0), compensated[0].data)
    assert_equal(np.full_like(img1, 0.5), compensated[1].data)
    assert_equal(np.full_like(img1, 1.0), compensated[2].data)

def test_calibrate_and_compensate_preserves_range():
    img0 = np.array([[0]], dtype=np.uint32)
    img1 = np.array([[10000]], dtype=np.uint32)
    img_test = np.array([[8000]])

    coeff = calibrate_flatfield(_make_image_seq([img0, img1]), [0, 1.0])
    compensated = compensate_flatfield(_make_image_seq([img_test]), coeff)
    assert_equal(compensated[0].data, 0.8)