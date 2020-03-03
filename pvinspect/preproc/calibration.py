'''Flatfield and lens calibration'''
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from typing import List
import numpy as np
from skimage import img_as_float
from tqdm.autonotebook import trange
import numpy as np

def _calibrate_flatfield(images: List[np.ndarray], targets: List[float], order: int = 0) -> np.array:
    assert images[0].dtype == np.float64 or images[0].dtype == np.float32

    imgs = [img.flatten() for img in images]
    
    assert order <= len(imgs)-1
    order = min(len(imgs)-1, 2) if order == 0 else order
    coeff = np.empty((order+1, imgs[0].shape[0]), dtype=images[0].dtype)

    if order == 1 and len(images) == 2:
        coeff[1] = (targets[1]-targets[0]) / (imgs[1]-imgs[0])
        coeff[0] = targets[0]-coeff[1]*imgs[0]
    else:
        for i in trange(imgs[0].shape[0]):
            coeff[:,i] = np.polynomial.polynomial.polyfit([img[i] for img in imgs], targets, order)
    
    #coeff = np.nan_to_num(coeff)
    return coeff.reshape((order+1, images[0].shape[0], images[0].shape[1]))

def calibrate_flatfield(images: ImageSequence, targets: List[float], order: int = 0, use_median: bool = True) -> np.array:
    '''Perform flat-field calibration. Note that there might be several calibration shots
    with the same normalization target. In that case, a least-squares estimate is computed.

    Args:
        images (ImageSequence): Sequence of calibration shots
        targets (List[float]): Corresponding list of normalization targets
        order (int): Order of compensation polynomial. The order is chosen automatically, if order == 0 (default)
        use_median (bool): Use the median of all images with the same target

    Returns:
        coeff: order+1 coefficients starting with the lowest order coefficient
    '''
    images = images.as_type(np.float64)

    if use_median:
        images_new = list()
        targets_new = list()
        
        for i1, t1 in enumerate(set(targets)):
            group = list()
            for i2, t2 in enumerate(targets):
                if t1 == t2:
                    group.append(images[i2].data)
            images_new.append(np.median(group, axis=0))
            targets_new.append(t1)
    
        return _calibrate_flatfield(images_new, targets_new, order)

    return _calibrate_flatfield([img.data for img in images], targets, order)