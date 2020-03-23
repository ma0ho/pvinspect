from pvinspect.common.util import *
from pvinspect.data.io import *
from pvinspect.preproc.detection import *
import sys
from pathlib import Path
import optuna
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split

def objective(trial: optuna.Trial, imgs, labels) -> float:

    # perform detection
    _, boxes = locate_multiple_modules(imgs, return_bounding_boxes=True, padding=0.0,
        #filter_type=trial.suggest_categorical('filter_type', ['gaussian', 'median']),
        filter_size=trial.suggest_discrete_uniform('filter_size', 0, 30, 2) if isinstance(trial, optuna.Trial) else trial.params['filter_size'],
        reject_size_thresh=trial.suggest_uniform('reject_size_thresh', 0.0, 1.0) if isinstance(trial, optuna.Trial) else trial.params['reject_size_thresh'],
        reject_fill_thresh=trial.suggest_uniform('reject_fill_thresh', 0.0, 1.0) if isinstance(trial, optuna.Trial) else trial.params['reject_fill_thresh']
    )

    # perform per image evaluation
    iou_per_image = dict()
    for img in imgs:
        label = labels[img.path.name]
        if img.path in boxes.keys():
            pred = boxes[img.path]
            iou_per_image[img.path.name] = mean_iou([polygon2boundingbox(x[1]) for x in label], pred)
        else:
            iou_per_image[img.path.name] = 0.0

    return np.mean(list(iou_per_image.values()))


if __name__ == '__main__':

    path = Path(sys.argv[1])

    # read data
    imgs = read_images(path, same_camera=False, modality=EL_IMAGE, pattern=('**/*.png'))
    labels = load_json_object_masks(path / 'labels.json')

    # split
    imgs_train, imgs_test = train_test_split([x for x in imgs], train_size=0.75)
    imgs_train = ImageSequence.from_other(imgs, images=imgs_train)
    imgs_test = ImageSequence.from_other(imgs, images=imgs_test)

    # prepare objective
    obj_train = partial(objective, imgs=imgs_train, labels=labels)
    obj_test = partial(objective, imgs=imgs_test, labels=labels)

    study = optuna.create_study(direction='maximize')
    study.optimize(obj_train, n_trials=30)
    
    print('Best configuration:')
    print(study.best_trial)

    print('Test performance:')
    print(obj_test(study.best_trial))
