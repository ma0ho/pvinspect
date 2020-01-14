from pvinspect.data import save_images, read_module_images, EL_IMAGE
import pvinspect.data as data
from pathlib import Path
import numpy as np

EXAMPLES = Path(__file__).absolute().parent.parent / 'pvinspect' / 'data' / 'datasets' / '20191219_poly10x6'

def _check_download_demo():
    data.demo.poly10x6(1)

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
    assert seq.dtype == np.uint16

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
    seq = read_module_images(EXAMPLES, EL_IMAGE, True, pattern='*.tif', N=1)
    assert len(seq) == 1
    seq = read_module_images(EXAMPLES, EL_IMAGE, True, pattern=('*.png', '*.tif'), N=1)
    assert len(seq) == 1
    try:
        seq = read_module_images(EXAMPLES, EL_IMAGE, True, pattern='*.png', N=1)
        assert False
    except:
        assert True
