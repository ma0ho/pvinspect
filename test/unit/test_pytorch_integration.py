from test.utilities import random_image_sequence, assert_equal
import torch as t
from pvinspect.integration.pytorch import Dataset
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
