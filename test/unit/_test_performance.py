from pvinspect.data.datasets import (
    multi_module_detection,
    caip_dataB,
    caip_dataC,
    caip_dataD,
)
from pvinspect.common.util import objdetect_metrics
from pvinspect.preproc.detection import locate_multiple_modules, locate_module_and_cells
import numpy as np
from sklearn import metrics


def test_locate_multiple_modules_performance():
    # load data and perform detection with default parameters
    anns, imgs = multi_module_detection()
    modules, boxes = locate_multiple_modules(
        imgs, padding=0.0, return_bounding_boxes=True
    )

    # compute mean IOU over all images
    ious = list()
    for img in imgs:
        iou, _, _ = objdetect_metrics(anns[img.path.name], boxes[img.path])
        ious.append(iou)
    mean_iou = np.mean(ious)
    print(ious)

    # make sure that mean IOU > 0.8
    assert mean_iou > 0.8


CAIP_THRESH = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]


def test_locate_module_and_cells_performance_caipB():
    # load data
    seq1, seq2, anns = caip_dataB()

    # perform detection
    seq1, boxes1 = locate_module_and_cells(
        seq1, estimate_distortion=False, return_bounding_boxes=True
    )
    seq2, boxes2 = locate_module_and_cells(
        seq2, estimate_distortion=False, return_bounding_boxes=True
    )

    # check IoU > 0.93
    ious = list()
    recalls = list()
    for img in seq1:
        iou, _, rec = objdetect_metrics(
            anns[img.path.name], boxes1[img.path.name], iou_thresh=CAIP_THRESH
        )
        ious.append(iou)
        recalls.append(rec)
    for img in seq2:
        iou, _, rec = objdetect_metrics(
            anns[img.path.name], boxes2[img.path.name], iou_thresh=CAIP_THRESH
        )
        ious.append(iou)
        recalls.append(rec)

    x = np.mean(recalls, axis=0)
    auc = metrics.auc(CAIP_THRESH, x)

    # TODO: Check, why this performs worse than in paper
    assert auc > 0.06


def test_locate_module_and_cells_performance_caipD():
    # load data
    seq, anns = caip_dataD()

    # perform detection
    seq, boxes = locate_module_and_cells(
        seq, estimate_distortion=False, return_bounding_boxes=True
    )

    # check IoU > 0.95
    ious = list()
    recalls = list()
    for img in seq:
        iou, _, rec = objdetect_metrics(
            anns[img.path.name], boxes[img.path.name], iou_thresh=CAIP_THRESH
        )
        ious.append(iou)
        recalls.append(rec)

    x = np.mean(recalls, axis=0)
    auc = metrics.auc(CAIP_THRESH, x)

    assert auc > 0.07
