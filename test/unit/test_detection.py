from pathlib import Path
from test.utilities import assert_equal

import numpy as np
from pvinspect import data
from pvinspect.common.transform import FullTransform, HomographyTransform
from pvinspect.common.util import objdetect_metrics
from pvinspect.data.image import *
from pvinspect.preproc import detection


def test_locate_homography():
    seq = data.datasets.poly10x6(2)
    seq = detection.locate_module_and_cells(
        seq, rows=6, cols=10, estimate_distortion=False
    )

    assert isinstance(seq[0].get_meta("transform"), HomographyTransform)
    assert isinstance(seq[1].get_meta("transform"), HomographyTransform)
    assert seq[0].get_meta("transform").valid
    assert seq[1].get_meta("transform").valid

    # check correct origin
    x = seq[0].get_meta("transform")(np.array([[0.0, 0.0]])).flatten()
    assert x[0] > 1760 and x[0] < 1840
    assert x[1] > 80 and x[1] < 160
    x = seq[1].get_meta("transform")(np.array([[0.0, 0.0]])).flatten()
    assert x[0] > 1760 and x[0] < 1840
    assert x[1] > 80 and x[1] < 160


def test_locate_full():
    seq = data.datasets.poly10x6(2)
    seq = detection.locate_module_and_cells(
        seq, rows=6, cols=10, estimate_distortion=True
    )

    assert isinstance(seq[0].get_meta("transform"), FullTransform)
    assert isinstance(seq[1].get_meta("transform"), FullTransform)
    assert seq[0].get_meta("transform").valid
    assert seq[1].get_meta("transform").valid

    # check correct origin
    x = seq[0].get_meta("transform")(np.array([[0.0, 0.0]])).flatten()
    assert x[0] > 1760 and x[0] < 1840
    assert x[1] > 80 and x[1] < 160
    x = seq[1].get_meta("transform")(np.array([[0.0, 0.0]])).flatten()
    assert x[0] > 1760 and x[0] < 1840
    assert x[1] > 80 and x[1] < 160


def test_segment_cells():
    seq = data.datasets.poly10x6(2)
    seq = detection.locate_module_and_cells(
        seq, rows=6, cols=10, estimate_distortion=True
    )
    cells = detection.segment_cells(seq)

    assert isinstance(cells, ImageSequence)
    assert len(cells) == 120
    assert isinstance(cells[0], Image)
    assert cells[0].get_meta("row") == 0
    assert cells[0].get_meta("col") == 0
    assert cells[1].get_meta("col") == 1
    assert cells[11].get_meta("row") == 1
    assert cells[0].has_meta("segment_module_original")
    assert (
        cells[0]
        .get_meta("segment_module_original")
        .has_meta("segment_module_original_box")
    )


def test_segment_cells_single_image():
    seq = data.datasets.poly10x6(1)
    seq = detection.locate_module_and_cells(
        seq, rows=6, cols=10, estimate_distortion=True
    )
    cells = detection.segment_cells(seq[0])

    assert isinstance(cells, ImageSequence)
    assert len(cells) == 60
    assert isinstance(cells[0], Image)


def test_segment_modules():
    seq = data.datasets.poly10x6(2)
    seq = detection.locate_module_and_cells(
        seq, rows=6, cols=10, estimate_distortion=True
    )
    modules = detection.segment_modules(seq)

    assert isinstance(modules, ImageSequence)
    assert isinstance(modules[0], Image)
    assert len(modules) == 2

    x = modules[0].get_meta("transform")(np.array([[0.0, 0.0]])).flatten()
    assert_equal(x[0], 0.0)
    assert_equal(x[1], 0.0)
    x = modules[0].get_meta("transform")(np.array([[10.0, 0.0]])).flatten()
    assert_equal(x[0], modules[0].shape[1])
    assert_equal(x[1], 0.0)
    x = modules[0].get_meta("transform")(np.array([[10.0, 6.0]])).flatten()
    assert_equal(x[0], modules[0].shape[1])
    assert_equal(x[1], modules[0].shape[0])


def test_segment_module_part():
    mod = data.datasets.poly10x6(1)[0]
    mod = detection.locate_module_and_cells(mod, rows=6, cols=10)
    part = detection.segment_module_part(mod, 1, 2, 2, 3)

    assert_equal(part.shape[1], 2 / 3 * part.shape[0], 0.1)
    assert part.get_meta("first_col") == 1
    assert part.get_meta("first_row") == 2
    assert part.get_meta("cols") == 2
    assert part.get_meta("rows") == 3


def test_segment_module_part_padding():
    mod = data.datasets.poly10x6(1)[0]
    mod = detection.locate_module_and_cells(mod, rows=6, cols=10)
    part = detection.segment_module_part(mod, -1, -2, 2, 3)

    assert_equal(part.shape[1], 2 / 3 * part.shape[0], 0.1)
    assert part.get_meta("first_col") == None
    assert part.get_meta("first_row") == None
    assert part.get_meta("cols") == 1
    assert part.get_meta("rows") == 1


def test_segment_size():
    img = detection.locate_module_and_cells(
        data.datasets.poly10x6(1)[0], rows=6, cols=10
    )
    part = detection.segment_module_part(img, 1, 3, 2, 1, size=20)
    assert_equal(part.shape[1], 2 * 20)
    assert_equal(part.shape[0], 1 * 20)


def test_segment_padding():
    img = detection.locate_module_and_cells(
        data.datasets.poly10x6(1)[0], rows=6, cols=10
    )
    part = detection.segment_module_part(img, 0, 0, 3, 2, padding=0.5)
    assert_equal(part.shape[1] / part.shape[0], 8 / 6)


def test_segment_padding_transform():
    img = detection.locate_module_and_cells(
        data.datasets.poly10x6(1)[0], rows=6, cols=10
    )
    part = detection.segment_module_part(img, 0, 0, 3, 2, padding=0.5, size=20)
    x1 = part.get_meta("transform")(np.array([[0.0, 1.0]])).flatten()
    x2 = part.get_meta("transform")(np.array([[2.0, 3.0]])).flatten()
    assert_equal(x1[0], 10)
    assert_equal(x1[1], 30)
    assert_equal(x2[0], 50)
    assert_equal(x2[1], 70)


def test_locate_partial_module():
    img = detection.locate_module_and_cells(
        data.datasets.poly10x6(1)[0], rows=6, cols=10, estimate_distortion=False
    )
    part = detection.segment_module_part(img, 0, 0, 3, 2, padding=0.5)
    part_det = detection.locate_module_and_cells(
        part, estimate_distortion=False, rows=3, cols=2, is_partial_module=True
    )

    x1_true = [part_det.shape[1] * 1 / 8, part_det.shape[0] * 1 / 6]
    x2_true = [part_det.shape[1] * 7 / 8, part_det.shape[0] * 5 / 6]
    x1 = part_det.get_meta("transform")(np.array([[0.0, 0.0]])).flatten()
    x2 = part_det.get_meta("transform")(np.array([[3.0, 2.0]])).flatten()
    eps = 0.05 * part_det.shape[1]
    print(img.get_meta("transform"))
    print(part_det.get_meta("transform"))

    assert_equal(x1[0], x1_true[0], eps)
    assert_equal(x1[1], x1_true[1], eps)
    assert_equal(x2[0], x2_true[0], eps)
    assert_equal(x2[1], x2_true[1], eps)


def test_detect_multiple_modules():
    anns, imgs = data.datasets.multi_module_detection(limit=2)
    _, boxes = detection.locate_multiple_modules(
        imgs, return_bounding_boxes=True, padding=0.0
    )  # type: ignore

    for p, prediction in boxes.items():
        _, _, recall = objdetect_metrics(anns[p], prediction, iou_thresh=[0.5])
        assert recall[0] > 0.5
