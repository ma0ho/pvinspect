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
    seq = data.datasets.poly10x6(2)
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

def test_float_image_is_not_converted():
    data = np.array([[0.1]], dtype=np.float32)
    img = ModuleImage(data, EL_IMAGE, Path() / 'test.png')
    assert img.dtype == np.float32
    data = np.array([[0.1]], dtype=np.float64)
    img = ModuleImage(data, EL_IMAGE, Path() / 'test.png')
    assert img.dtype == np.float64

def test_apply_does_copy():
    seq = _random_image_sequence()
    data = seq[0].data
    def fn(x):
        x[:] = 0.0
        return x
    seq.apply_image_data(fn)
    assert_equal(seq[0].data, data)

def test_image_as_type():
    img = _random_image()
    assert img.dtype == np.float32 or img.dtype == np.float64
    img = img.as_type(np.uint16)
    assert img.dtype == np.uint16

def test_sequence_as_type():
    seq = _random_image_sequence()
    assert seq.dtype == np.float32 or seq.dtype == np.float64
    seq = seq.as_type(np.uint16)
    assert seq.dtype == np.uint16

def test_image_data_returns_copy():
    img = _random_image()
    original_data = img._data.copy()
    data = img.data
    data[:] = 0.0
    assert_equal(img.data, original_data)