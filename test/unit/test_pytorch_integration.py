from test.utilities import assert_equal, random_sequence

import numpy as np
import pandas as pd
import torch as t
from pvinspect.data.image.type import DType
from pvinspect.integration.pytorch import *


def test_dataset():
    seq = random_sequence(seq_lazy=True, imgs_lazy=True, N=3, dtype=DType.UNSIGNED_BYTE)
    ds = Dataset(seq)

    assert len(ds) == 3
    for i in range(3):
        assert np.all(ds[i] == seq[i].data)


def test_dataset_with_meta():
    meta = pd.DataFrame([{"k1": v, "k2": -v} for v in range(3)])
    seq = random_sequence(
        seq_lazy=True, imgs_lazy=True, N=3, dtype=DType.UNSIGNED_BYTE, meta=meta
    )

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
    seq = random_sequence(seq_lazy=True, imgs_lazy=True, N=3, dtype=DType.UNSIGNED_BYTE)
    t = lambda x: x + 1.0
    ds = Dataset(seq, data_transform=t)

    for i in range(3):
        assert_equal(ds[i], t(seq[i].data))


def test_dataset_with_meta_transform():
    meta = pd.DataFrame([{"k1": v, "k2": -v} for v in range(3)])
    seq = random_sequence(
        seq_lazy=True, imgs_lazy=True, N=3, dtype=DType.UNSIGNED_BYTE, meta=meta
    )
    t = lambda x: -x
    ds = Dataset(seq, meta_attrs=["k2", "k1"], meta_transforms={"k1": t})

    for i in range(3):
        _, y, z = ds[i]
        assert y == -i
        assert z == -i


def test_classification_dataset():
    meta = pd.DataFrame([{"c1": False, "c2": True}, {"c1": True, "c2": False}])
    seq = random_sequence(
        seq_lazy=True, imgs_lazy=True, N=3, dtype=DType.UNSIGNED_BYTE, meta=meta
    )
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
    meta = pd.DataFrame(
        [{"c1": False, "c2": True, "k": 0}, {"c1": True, "c2": False, "k": 1}]
    )
    seq = random_sequence(
        seq_lazy=True, imgs_lazy=True, N=3, dtype=DType.UNSIGNED_BYTE, meta=meta
    )
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


def test_classification_dataset_feed_result_bool():
    meta = pd.DataFrame([{"c1": False, "c2": True}, {"c1": True, "c2": False}])
    seq = random_sequence(
        seq_lazy=True, imgs_lazy=True, N=3, dtype=DType.UNSIGNED_BYTE, meta=meta
    )
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


def test_classification_dataset_feed_result_float():
    meta = pd.DataFrame([{"c1": False, "c2": True}, {"c1": True, "c2": False}])
    seq = random_sequence(
        seq_lazy=True, imgs_lazy=True, N=3, dtype=DType.UNSIGNED_BYTE, meta=meta
    )
    ds = ClassificationDataset(seq, meta_classes=["c2", "c1"])

    result = [
        t.tensor([0.0, 1.0], dtype=t.float),
        t.tensor([1.0, 0.0], dtype=t.float),
    ]
    seq2 = ds.result_sequence(result, prefix="xx_")
    assert seq2[0].get_meta("xx_c2") == 0.0
    assert seq2[0].get_meta("xx_c1") == 1.0
    assert seq2[1].get_meta("xx_c2") == 1.0
    assert seq2[1].get_meta("xx_c1") == 0.0
