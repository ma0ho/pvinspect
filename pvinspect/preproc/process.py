'''Flatfield and lens correction'''
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from typing import List
import numpy as np

@_sequence
def compensate_flatfield(sequence: ModuleImageOrSequence, coeff: np.ndarray) -> ModuleImageOrSequence:
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
        for i in range(1, coeff.shape[0]):
            res += coeff[i]*data
            data *= data
        return res

    return sequence.apply_image_data(fn, coeff)
