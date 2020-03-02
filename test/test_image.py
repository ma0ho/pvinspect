from pvinspect import data
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from pathlib import Path
from test.utilities import assert_equal

@_sequence
def _some_fn(seq: ModuleImageSequence):
    assert isinstance(seq, ModuleImageSequence)
    return seq

def _random_image() -> Image:
    data = np.random.random((10,10))
    return Image(data, EL_IMAGE, Path() / 'test.png')

def _random_module_image() -> ModuleImage:
    data = np.random.random((10,10))
    return ModuleImage(data, EL_IMAGE, Path() / 'test.png')

def _random_image_sequence() -> ImageSequence:
    imgs = [_random_module_image() for x in range(2)]
    return ImageSequence(imgs, False)

def test_sequence_element_access():
    seq = data.demo.poly10x6(2)
    assert seq._images[0].path == seq[0].path
    assert seq._images[1].path == seq[1].path

def test_sequence_wrapper():
    img = _random_module_image()
    res = _some_fn(img)
    assert isinstance(res, ModuleImage)

def test_image_from_other():
    p = Path() / 'other.png'
    img = _random_image()
    img2 = Image.from_other(img, path=p)
    assert img._data is img2._data
    assert img._modality is img2._modality
    assert img2._path is p

def test_module_image_from_other():
    img = _random_module_image()
    img2 = ModuleImage.from_other(img, cols=10, rows=6)
    assert img._data is img2._data
    assert img._modality is img2._modality
    assert img._path is img2._path
    assert img2._cols == 10
    assert img2._rows == 6

def test_image_dtype_conversion():
    data = np.array([127], dtype=np.uint8)
    img = ModuleImage(data, EL_IMAGE, Path() / 'test.png')
    assert img.dtype == np.uint16
    assert_equal(img._data.max(), int(127/255*65535), 2)

def test_image_dtype_range_high():
    data = np.array([1.3])
    ex_cnt = 0
    try:
        img = ModuleImage(data, EL_IMAGE, Path() / 'test.png')
    except:
        ex_cnt += 1
    assert ex_cnt == 1

def test_image_dtype_range_low():
    data = np.array([-0.1])
    ex_cnt = 0
    try:
        img = ModuleImage(data, EL_IMAGE, Path() / 'test.png')
    except:
        ex_cnt += 1
    assert ex_cnt == 1

def test_float_image_is_not_converted():
    data = np.array([[0.1]], dtype=np.float32)
    img = ModuleImage(data, EL_IMAGE, Path() / 'test.png')
    assert img.dtype == np.float32
    data = np.array([[0.1]], dtype=np.float64)
    img = ModuleImage(data, EL_IMAGE, Path() / 'test.png')
    assert img.dtype == np.float64

def test_apply_does_not_copy():
    seq = _random_image_sequence()
    def fn(x):
        x[:] = 0.0
        return x
    seq.apply_image_data(fn)
    for img in seq:
        assert_equal(img.data, 0.0)
