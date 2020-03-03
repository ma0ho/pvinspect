from pvinspect import data

def test_poly10x6():
    seq = data.datasets.poly10x6()
    assert len(seq) == 20
    assert seq.modality == data.EL_IMAGE
    assert seq.shape == (2052, 2046)
    assert seq.same_camera == True

def test_caip_dataB():
    seq1, seq2 = data.datasets.caip_dataB()
    assert len(seq1) == 5
    assert seq1.modality == data.EL_IMAGE
    assert seq1.same_camera == False
    assert len(seq2) == 3
    assert seq2.modality == data.EL_IMAGE
    assert seq2.same_camera == False

def test_caip_dataC():
    seq = data.datasets.caip_dataC()
    assert len(seq) == 10
    assert seq.modality == data.EL_IMAGE
    assert seq.same_camera == True

def test_caip_dataD():
    seq = data.datasets.caip_dataD()
    assert len(seq) == 9
    assert seq.modality == data.EL_IMAGE
    assert seq.same_camera == True
