from pvinspect.common.util import *
from shapely import *
from test.utilities import *


def test_iou():
    p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p2 = affinity.translate(p1, -1, -1)
    assert_equal(iou(p1, p2), 1 / 7)


def test_mean_iou():
    p1_1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p1_2 = affinity.translate(p1_1, -1, -1)  # iou: 1/7
    p2_1 = affinity.translate(p1_1, 1.5, 1.5)
    p2_2 = affinity.translate(p2_1, 1, 1)  # iou: 1/7
    iou, _, _ = objdetect_metrics(
        [("Module", p1_1), ("Module", p2_1)], [("Module", p1_2), ("Module", p2_2)]
    )
    assert_equal(iou, (1 / 7 + 1 / 7) / 2)


def test_precision_recall1():
    p1_1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p1_2 = affinity.translate(p1_1, -1, -1)  # iou: 1/7
    p2_1 = affinity.translate(p1_1, 1.5, 1.5)
    p2_2 = affinity.translate(p2_1, 0, 0)  # iou: 1
    p3_2 = affinity.translate(p1_1, 10, 10)  # something far off
    _, prec, recall = objdetect_metrics(
        [("Module", p1_1), ("Module", p2_1)],
        [("Module", p1_2), ("Module", p2_2), ("Module", p3_2)],
        [1 / 8, 0.999],
    )
    assert_equal(prec[0], 2 / 3)
    assert_equal(prec[1], 1 / 3)
    assert_equal(recall[0], 1)
    assert_equal(recall[1], 1 / 2)


def test_precision_recall2():
    p1_1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p1_2 = affinity.translate(p1_1, -1, -1)  # iou: 1/7
    p2_1 = affinity.translate(p1_1, 1.5, 1.5)
    p2_2 = affinity.translate(p2_1, 0, 0)  # iou: 1
    p3_1 = affinity.translate(p1_1, 10, 10)  # something far off
    iou, prec, recall = objdetect_metrics(
        [("Module", p1_1), ("Module", p2_1), ("Module", p3_1)],
        [("Module", p1_2), ("Module", p2_2)],
        [1 / 8, 0.999],
    )
    assert_equal(prec[0], 1)
    assert_equal(prec[1], 1 / 2)
    assert_equal(recall[0], 2 / 3)
    assert_equal(recall[1], 1 / 3)
