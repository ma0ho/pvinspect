from pvinspect import data

def test_sequence_element_access():
    seq = data.demo.poly10x6(2)
    assert seq._images[0].path == seq[0].path
    assert seq._images[1].path == seq[1].path
