from pvinspect.common.types import ObjectAnnotations
from pvinspect.data.image import *
from pvinspect.datasets import *


def test_poly10x6():
    seq = poly10x6()
    assert len(seq) == 20
    assert seq[0].shape == (2052, 2046)


def test_caip_dataB():
    seq1, seq2, anns = caip_dataB()
    assert len(seq1) == 5
    assert len(seq2) == 3
    assert len(anns) == len(seq1) + len(seq2)


def test_caip_dataC():
    seq, anns = caip_dataC()
    assert len(seq) == 10
    assert len(anns) == len(seq)


def test_caip_dataD():
    seq, anns = caip_dataD()
    assert len(seq) == 9
    assert len(anns) == len(seq)


def test_calibration_ipv40CCD_FF():
    d = calibration_ipv40CCD_FF(N=2)
    assert isinstance(d, dict)
    assert "0A" in d.keys()
    assert len(d) == 3
    assert len(d["0A"]) == 2


def test_multi_module_detection():
    anns, imgs = multi_module_detection()
    assert isinstance(imgs, ImageSequence)
    for img in imgs:
        assert img.get_meta("original_filename") in anns.keys()


def test_elpv():
    imgs = elpv(N=2)
    assert isinstance(imgs, ImageSequence)

    # check meta available
    meta = [
        "defect_probability",
        "wafer",
        "crack",
        "inactive",
        "blob",
        "finger",
        "testset",
    ]

    for m in meta:
        assert imgs[0].has_meta(m)
