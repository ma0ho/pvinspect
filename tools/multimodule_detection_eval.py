from pvinspect.common.util import *
from pvinspect.data.io import *
from pvinspect.preproc.detection import *
import sys
from pathlib import Path
import optuna
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def objective(
    trial: optuna.Trial, imgs, labels, skip_pr=True, iou_thresh=None
) -> float:

    # perform detection
    _, boxes = locate_multiple_modules(
        imgs,
        return_bounding_boxes=True,
        padding=0.0,
        scale=trial.suggest_uniform("scale", 0.2, 1.0)
        if isinstance(trial, optuna.Trial)
        else trial.params["scale"],
        reject_size_thresh=trial.suggest_uniform("reject_size_thresh", 0.0, 1.0)
        if isinstance(trial, optuna.Trial)
        else trial.params["reject_size_thresh"],
        reject_fill_thresh=trial.suggest_uniform("reject_fill_thresh", 0.0, 1.0)
        if isinstance(trial, optuna.Trial)
        else trial.params["reject_fill_thresh"],
    )

    # perform per image evaluation
    iou_per_image = dict()
    precision_per_image = dict()
    recall_per_image = dict()
    for img in imgs:
        label = labels[img.path.name]
        if img.path in boxes.keys():
            pred = boxes[img.path]
            (
                iou_per_image[img.path.name],
                precision_per_image[img.path.name],
                recall_per_image[img.path.name],
            ) = objdetect_metrics(
                [polygon2boundingbox(x[1]) for x in label], pred, iou_thresh
            )
        else:
            # we assume that every image should have at least one object
            iou_per_image[img.path.name] = 0.0
            precision_per_image[img.path.name] = 0.0
            recall_per_image[img.path.name] = 0.0

    if skip_pr:
        return np.mean(list(iou_per_image.values()))
    else:
        return (
            np.mean(list(iou_per_image.values())),
            np.mean(list(precision_per_image.values()), axis=0),
            np.mean(list(recall_per_image.values()), axis=0),
        )


if __name__ == "__main__":

    path = Path(sys.argv[1])

    # read data
    imgs = read_images(path, same_camera=False, modality=EL_IMAGE, pattern=("**/*.png"))
    labels = load_json_object_masks(path / "labels.json")
    img_list = [x for x in imgs]
    obj_counts = [len(labels[img.path.name]) for img in img_list]

    # split
    imgs_train, imgs_test = train_test_split(
        img_list, train_size=0.5, stratify=obj_counts
    )
    imgs_train = ImageSequence.from_other(imgs, images=imgs_train)
    imgs_test = ImageSequence.from_other(imgs, images=imgs_test)

    # prepare objective
    thresholds = [(x + 1) / 20 for x in range(20)]
    obj_train = partial(
        objective, imgs=imgs_train, labels=labels, iou_thresh=thresholds
    )
    obj_test = partial(
        objective, imgs=imgs_test, labels=labels, skip_pr=False, iou_thresh=thresholds
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(obj_train, n_trials=50)

    # test
    print("Best configuration:")
    print(study.best_trial)

    print("Test performance:")
    iou, pr, rec = obj_test(study.best_trial)
    print("IOU: {:f}".format(iou))
    print("precision ({}): {}".format(thresholds, pr))
    print("recall ({}): {}".format(thresholds, rec))

    # save csv
    pd.DataFrame({"threshold": thresholds, "precision": pr, "recall": rec}).to_csv(
        "results.csv"
    )
