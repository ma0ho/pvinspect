from test.utilities import random_image_sequence, assert_equal
import torch as t
from pvinspect.integration.pytorch import *
from pvinspect.data import EL_IMAGE


def test_dataset():
    seq = random_image_sequence(lazy=True, N=3)
    ds = Dataset(seq)

    assert len(ds) == 3
    for i in range(3):
        assert ds[i].data == seq[i].data


def test_dataset_with_meta():
    meta = [{"k1": v, "k2": -v} for v in range(3)]
    seq = random_image_sequence(lazy=True, N=3, meta=meta)

    # only use a single meta attr
    ds = Dataset(seq, meta_attrs=["k1"])
    for i in range(3):
        x, y = ds[i]
        assert x.data == seq[i].data
        assert y == i

    # or both
    ds = Dataset(seq, meta_attrs=["k2", "k1"])
    for i in range(3):
        x, y, z = ds[i]
        assert x.data == seq[i].data
        assert y == -i
        assert z == i


def test_dataset_with_data_transform():
    seq = random_image_sequence(lazy=True, N=3)
    t = lambda x: x + 1.0
    ds = Dataset(seq, data_transform=t)

    for i in range(3):
        assert_equal(ds[i].data, t(seq[i].data))


def test_dataset_with_meta_transform():
    meta = [{"k1": v, "k2": -v} for v in range(3)]
    seq = random_image_sequence(lazy=True, N=3, meta=meta)
    t = lambda x: -x
    ds = Dataset(seq, meta_attrs=["k2", "k1"], meta_transforms={"k1": t})

    for i in range(3):
        _, y, z = ds[i]
        assert y == -i
        assert z == -i


def test_classification_dataset():
    meta = [{"c1": False, "c2": True}, {"c1": True, "c2": False}]
    seq = random_image_sequence(lazy=True, N=2, meta=meta)
    ds = ClassificationDataset(seq, meta_classes=["c2", "c1"])

    for i in range(2):
        res = ds[i]
        x: t.Tensor = res[0]
        y: t.Tensor = res[1]
        assert isinstance(y, t.Tensor)
        assert len(res) == 2
        assert y.dtype == t.float32
        if i == 0:
            assert y.tolist() == [1.0, 0.0]
        else:
            assert y.tolist() == [0.0, 1.0]


def test_classification_dataset_additional_meta():
    meta = [{"c1": False, "c2": True, "k": 0}, {"c1": True, "c2": False, "k": 1}]
    seq = random_image_sequence(lazy=True, N=2, meta=meta)
    ds = ClassificationDataset(seq, meta_classes=["c2", "c1"], meta_attrs=["k"])

    for i in range(2):
        res = ds[i]
        x: t.Tensor = res[0]
        y: t.Tensor = res[1]
        z: int = res[2]
        assert isinstance(y, t.Tensor)
        assert len(res) == 3
        assert y.dtype == t.float32
        if i == 0:
            assert y.tolist() == [1.0, 0.0]
            assert z == 0
        else:
            assert y.tolist() == [0.0, 1.0]
            assert z == 1


def test_classification_dataset_feed_result():
    meta = [{"c1": False, "c2": True}, {"c1": True, "c2": False}]
    seq = random_image_sequence(lazy=True, N=2, meta=meta)
    ds = ClassificationDataset(seq, meta_classes=["c2", "c1"])

    result = [
        t.tensor([False, True], dtype=t.bool),
        t.tensor([True, False], dtype=t.bool),
    ]
    seq2 = ds.result_sequence(result, prefix="xx_")
    assert seq2[0].get_meta("xx_c2") == False
    assert seq2[0].get_meta("xx_c1") == True
    assert seq2[1].get_meta("xx_c2") == True
    assert seq2[1].get_meta("xx_c1") == False
