'''Flatfield and lens correction'''
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from typing import List
import numpy as np
from skimage import img_as_float, img_as_uint

@_sequence
def _compensate_flatfield(sequence: ModuleImageOrSequence, coeff: np.ndarray) -> ModuleImageOrSequence:
    '''Low level method to perform flat field correction

    Args:
        sequence (ModuleImageOrSequence): Sequence of images or single image (needs to be of type float)
        coeff (np.ndarray): Compensation coefficients

    Returns:
        sequence: The corrected images
    '''
    assert sequence.dtype == np.float

    def fn(data, coeff):
        res = coeff[0].copy()
        data = data.copy()
        for i in range(1, coeff.shape[0]):
            res += coeff[i]*data
            data *= data
        return np.clip(res, 0.0, 1.0)

    return sequence.apply_image_data(fn, coeff)

@_sequence
def compensate_flatfield(sequence: ModuleImageOrSequence, coeff: np.ndarray) -> ModuleImageOrSequence:
    '''Perform flat field correction

    Args:
        sequence (ModuleImageOrSequence): Sequence of images or single image
        coeff (np.ndarray): Compensation coefficients

    Returns:
        sequence: The corrected images
    '''
    sequence = sequence.as_type(np.float64)
    sequence = _compensate_flatfield(sequence, coeff)

    return sequence