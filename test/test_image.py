from pvinspect import data
from pvinspect.data.image import _sequence, ModuleImageSequence, ModuleImage

@_sequence
def _some_fn(seq: ModuleImageSequence):
    assert isinstance(seq, ModuleImageSequence)
    return seq

def test_sequence_element_access():
    seq = data.demo.poly10x6(2)
    assert seq._images[0].path == seq[0].path
    assert seq._images[1].path == seq[1].path

def test_sequence_wrapper():
    img = data.demo.poly10x6(1)[0]
    res = _some_fn(img)
    assert isinstance(res, ModuleImage)
