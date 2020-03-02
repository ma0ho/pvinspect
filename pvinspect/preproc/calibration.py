'''Flatfield and lens calibration'''
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from typing import List
import numpy as np
from skimage import img_as_float

def _calibrate_flatfield(images: ImageSequence, targets: List[float], order: int = 0) -> np.array:
    '''Low-level method to perform flat-field calibration

    Args:
        images (ImageSequence): Sequence of calibration shots (needs to be of type float)
        targets (List[float]): Corresponding list of normalization targets
        order (int): Order of compensation polynomial. The order is chosen automatically, if order == 0 (default)

    Returns:
        coeff: order+1 coefficients starting with the lowest order coefficient
    '''
    assert images.dtype == np.float

    imgs = [img._data.flatten() for img in images]
    
    assert order <= len(imgs)-1
    order = min(len(imgs)-1, 2) if order == 0 else order
    coeff = np.empty((order+1, imgs[0].shape[0]), dtype=np.float32)

    for i in range(imgs[0].shape[0]):
        x = [img[i] for img in imgs]
        if order == 1:
            coeff[0,i] = targets[0]-imgs[0][i]
            coeff[1,i] = (targets[1]-targets[0]) / (imgs[1][i]-imgs[0][i])
        else:
            coeff[:,i] = np.polynomial.polynomial.polyfit([img[i] for img in imgs], targets, order)
    
    return coeff.reshape((order+1, images.shape[0], images.shape[1]))

def calibrate_flatfield(images: ImageSequence, targets: List[float], order: int = 0) -> np.array:
    '''Perform flat-field calibration. Note that there might be several calibration shots
    with the same normalization target. In that case, a least-squares estimate is computed.

    Args:
        images (ImageSequence): Sequence of calibration shots
        targets (List[float]): Corresponding list of normalization targets
        order (int): Order of compensation polynomial. The order is chosen automatically, if order == 0 (default)

    Returns:
        coeff: order+1 coefficients starting with the lowest order coefficient
    '''
    images = images.as_type(np.float64)
    return _calibrate_flatfield(images, targets, order)