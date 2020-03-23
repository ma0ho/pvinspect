from pvinspect.common.util import *
from shapely import *
from .utilities import *

def test_iou():
    p1 = Polygon([(0,0),(2,0),(2,2),(0,2)])
    p2 = affinity.translate(p1, -1, -1)
    assert_equal(iou(p1, p2), 1/7)

def test_mean_iou():
    p1_1 = Polygon([(0,0),(2,0),(2,2),(0,2)])
    p1_2 = affinity.translate(p1_1, -1, -1)     # iou: 1/7
    p2_1 = affinity.translate(p1_1, 1.5, 1.5)
    p2_2 = affinity.translate(p2_1, 1, 1)       # iou: 1/7
    assert_equal(mean_iou([p1_1, p2_1], [p1_2, p2_2]), (1/7+1/7)/2)
