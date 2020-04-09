from pvinspect.data.datasets import multi_module_detection
from pvinspect.common.util import objdetect_metrics
from pvinspect.preproc.detection import locate_multiple_modules
import numpy as np


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
