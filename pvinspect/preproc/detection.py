'''Detection, localization and segmentation of solar modules'''

from pvinspect.preproc._mdetect.locate import apply
from pvinspect.common.transform import HomographyTransform, warp_image, FullMultiTransform, FullTransform
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from pvinspect.data.exceptions import UnsupportedModalityException
from typing import Union
from tqdm.auto import tqdm
from copy import deepcopy
import logging
from pvinspect.common._ipy_exit import exit
import numpy as np

@_sequence
def locate_module_and_cells(sequence: ModuleImageOrSequence, estimate_distortion: bool = True, orientation: str = None) -> ModuleImageOrSequence:
    '''Locate a single module and its cells

    Note:
        This methods implements the following paper:
        Hoffmann, Mathis, et al. "Fast and robust detection of solar modules in electroluminescence images."
        International Conference on Computer Analysis of Images and Patterns. Springer, Cham, 2019.

    Args:
        sequence (ModuleImageOrSequence): A single module image or a sequence of module images 
        estimate_distortion (bool): Set True to estimate lens distortion, else False 
        orientation (str): Orientation of the module ('horizontal' or 'vertical' or None).
            If set to None (default), orientation is automatically determined

    Returns:
        images: The same image/sequence with location information added
    '''

    if sequence[0].modality != EL_IMAGE:
        logging.error('Module localization is not supporting given imaging modality')
        exit()

    result = list()
    failures = 0
    mcs = list()
    dts = list()
    flags = list()
    transforms = list()
    for img in tqdm(sequence.images):
        t, mc, dt, f = apply(img.data, img.cols, img.rows, is_module_detail=isinstance(img, PartialModuleImage), orientation=orientation)
        transforms.append(t)
        flags.append(f)
        mcs.append(mc)
        dts.append(dt)

    if estimate_distortion:
        if sequence.same_camera:
            # do joint estimation
            logging.info('Jointly estimating parameters for lens distortion. This might take some time..')

            mcs_new = list()
            dts_new = list()
            valid = list()
            for mc, dt, f in zip(mcs, dts, flags):
                if mc is not None and dt is not None:
                    mcs_new.append(mc[f])
                    dts_new.append(dt[f])
                    valid.append(True)
                else:
                    valid.append(False)
            transforms = FullMultiTransform(mcs_new, dts_new, image_width=sequence.shape[1], image_height=sequence.shape[0], n_dist_coeff=1)
            transforms_new = list()
            i = 0
            for v in valid:
                if v:
                    transforms_new.append(transforms[i])
                    i += 1
                else:
                    transforms_new.append(None)
            transforms = transforms_new

        else:
            transforms = list()
            for mc, dt, f, img in zip(mcs, dts, flags, sequence.images):
                if mc is not None and dt is not None:
                    t = FullTransform(mc[f], dt[f], image_width=img.shape[1], image_height=img.shape[0], n_dist_coeff=1)
                    transforms.append(t)
                else:
                    transforms.append(None)

    for t, img in zip(transforms, sequence.images):
        if t is not None and t.valid:
            if isinstance(img, ModuleImage):
                img_res = ModuleImage(img.data, img.modality, img.path, img.cols, img.rows, t)
            elif isinstance(img, PartialModuleImage):
                img_res = PartialModuleImage(img.data, img.modality, img.path, img.cols, img.rows, t)
            else:
                raise TypeError('Unsupported type {}'.format(type(img)))
            result.append(img_res)
        else:
            result.append(deepcopy(img))
            failures += 1
    if failures > 0:
        logging.warning('Module localization falied for {:d} images'.format(failures))

    return ModuleImageSequence(result, same_camera=sequence.same_camera, copy=False)

def segment_module_part(image: ModuleImage, first_col: int, first_row: int, cols: int, rows: int, size: int = None, padding: float = 0.0) -> PartialModuleImage:
    '''Segment a part of a module

    Args:
        image (ModuleImage): The corresponding module image
        first_col (int): First column to appear in the segment
        first_row (int): First row to appear in the segment
        cols (int): Number of columns of the segment
        rows (int): Number of rows of the segment
        size (int): Size of a cell in pixels (automatically chosen by default)
        padding (float): Optional padding around the given segment relative to the cell size
                         (must be in [0..1[ )

    Returns:
        segment: The resulting segment
    '''

    if image.transform is None or not image.transform.valid:
        logging.error('The ModuleImage does not have a valid transform. Did module localization succeed?')
        exit()

    if padding >= 1.0 or padding < 0.0:
        logging.error('padding needs to be in [0..1[')
        exit()
    
    last_col = first_col+cols
    last_row = first_row+rows
    if last_row > image.rows or last_col > image.cols:
        logging.error('The row or column index exceeds the module geometry')
        exit()

    size = image.transform.mean_scale() if size is None else size
    result = warp_image(image.data, image.transform, first_col-padding, first_row-padding,
        1/size, 1/size, cols+2*padding, rows+2*padding)
    result = result.astype(image.dtype)
    transform = HomographyTransform(np.array([
        [first_row-padding, first_col-padding], [first_row-padding, last_col-padding],
        [last_row-padding, first_col-padding], [last_row-padding, last_col-padding]]),
        np.array([[0.0, 0.0], [0.0, size*cols], [size*rows, 0.0], [size*rows, size*cols]]))
    return PartialModuleImage(result, image.modality, image.path, cols, rows, first_col, first_row, transform)

def segment_module(image: ModuleImage, size: int = None, padding: float = 0.0) -> ModuleImage:
    '''Obtain a rectified, cropped and undistorted module image

    Args:
        image (ModuleImage): A single module image
        size (int): Size of a cell in pixels (automatically chosen by default)
        padding (float): Optional padding around the given segment relative to the cell size
                         (must be in [0..1[ )

    Returns:
        module: The resulting module image
    '''

    result = segment_module_part(image, 0, 0, image.cols, image.rows, size, padding)
    return ModuleImage(result._data, image.modality, image.path, image.cols, image.rows, result.transform)

def segment_cell(image: ModuleImage, row: int, col: int, size: int = None, padding: float = 0.0) -> CellImage:
    '''Obtain a cell image from a module image

    Args:
        image (ModuleImageOrSequence): A single module image
        row (int): The row number (starting at 0)
        col (int): The column number (starting at 0)
        size (int): Size of the resulting cell image in pixels (automatically chosen by default)
        padding (float): Optional padding around the cell relative to the cell size
                         (must be in [0..1[ )

    Returns:
        cells: The segmented cell image
    '''


    result = segment_module_part(image, col, row, 1, 1, size, padding)
    return CellImage(result._data, image.modality, image.path, row, col)

@_sequence
def segment_modules(sequence: ModuleImageOrSequence, size: int = None) -> ModuleImageSequence:
    '''Obtain rectified, cropped and undistorted module images from a sequence. Note that images that do not have a valid transform,
    possibly because the detection step failed, are silently ignored.

    Args:
        sequence (ModuleImageOrSequence): A single module image or a sequence of module images
        size (int): Size of the resulting cell images in pixels (automatically chosen by default)

    Returns:
        module: The segmented module images
    '''

    scales = np.array([img.transform.mean_scale() for img in sequence.images if img.transform is not None and img.transform.valid])
    if scales.std() > 0.1*scales.mean() and size is None:
        logging.warning('The size of cells within the sequences varies by more than 10%. However, segment_modules, \
creates images of a fixed size. Please consider to split the sequence into multiple sequences \
with less variation in size.')
    if size is None:
        size = int(scales.mean())

    result = list()
    for img in tqdm(sequence.images):
        
        # for the moment, we silently ignore images without a valid transform
        if img.transform is not None and img.transform.valid:
            result.append(segment_module(img, size))

    return ModuleImageSequence(result, same_camera=False, copy=False)

@_sequence
def segment_cells(sequence: ModuleImageOrSequence, size: int = None) -> CellImageSequence:
    '''Obtain cell images from a sequence of module images. Note that images that do not have a valid transform,
    possibly because the detection step failed, are silently ignored.

    Args:
        sequence (ModuleImageOrSequence): A single module image or a sequence of module images
        size (int): Size of the resulting cell images in pixels (automatically chosen by default)

    Returns:
        cells: The segmented cell images
    '''

    scales = np.array([img.transform.mean_scale() for img in sequence.images if img.transform is not None and img.transform.valid])
    if scales.std() > 0.1*scales.mean() and size is None:
        logging.warning('The size of cells within the sequences varies by more than 10%. However, segment_cells, \
creates cell images of a fixed size. Please consider to split the sequence into multiple sequences \
with less variation in size.')
    if size is None:
        size = int(scales.mean())

    result = list()
    for img in tqdm(sequence.images):
        for row in range(img.rows):
            for col in range(img.cols):

                # for the moment, we silently ignore images without a valid transform
                if img.transform is not None and img.transform.valid:
                    result.append(segment_cell(img, row, col, size))

    return CellImageSequence(result, copy=False)

