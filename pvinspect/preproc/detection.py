from pvinspect.preproc._mdetect.locate import apply
from pvinspect.common.transform import HomographyTransform, warp_image, FullMultiTransform, FullTransform
from pvinspect.data.image import ModuleImageSequence, ModuleImage, ModuleImageOrSequence, _sequence, CellImage, CellImageSequence, EL_IMAGE
from pvinspect.data.exceptions import UnsupportedModalityException
from typing import Union
from tqdm.auto import tqdm
from copy import deepcopy
import logging
from pvinspect.common._ipy_exit import exit
import numpy as np

@_sequence
def locate_module_and_cells(sequence: ModuleImageOrSequence, estimate_distortion: bool = True) -> ModuleImageOrSequence:
    '''Locate a single module and its cells

    Args:
        sequence (ModuleImageOrSequence): A single module image or a sequence of module images 
        estimate_distortion (bool): Set True to estimate lens distortion, else False 

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
        t, mc, dt, f = apply(img.data, img.cols, img.rows)
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
            img_res = ModuleImage(img.data, img.modality, img.path, img.cols, img.rows, t)
            result.append(img_res)
        else:
            result.append(deepcopy(img))
            failures += 1
    if failures > 0:
        logging.warning('Module localization falied for {:d} images'.format(failures))

    return ModuleImageSequence(result, same_camera=sequence.same_camera, copy=False)

def segment_module(image: ModuleImage, size: int = None) -> ModuleImage:
    '''Obtain a rectified, cropped and undistorted module image

    Args:
        image (ModuleImage): A single module image
        size (int): Size of a cell in pixels (automatically chosen by default)

    Returns:
        module: The resulting module image
    '''

    if image.transform is None or not image.transform.valid:
        logging.error('The ModuleImage does not have a valid transform. Did module localization succeed?')
        exit()

    size = image.transform.mean_scale() if size is None else size
    result = warp_image(image.data, image.transform, 0, 0, 1/size, 1/size, image.cols, image.rows)
    result = result.astype(image.dtype)
    transform = HomographyTransform(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [0.0, size], [size, 0.0], [size, size]]))
    return ModuleImage(result, image.modality, image.path, image.cols, image.rows, transform)

def segment_cell(image: ModuleImage, row: int, col: int, size: int = None) -> CellImage:
    '''Obtain a cell image from a module image

    Args:
        image (ModuleImageOrSequence): A single module image
        row (int): The row number (starting at 0)
        col (int): The column number (starting at 0)
        size (int): Size of the resulting cell image in pixels (automatically chosen by default)

    Returns:
        cells: The segmented cell image
    '''

    if row >= image.rows or col >= image.cols:
        logging.error('The row or column index exceeds the module geometry')
        exit()
    if image.transform is None or not image.transform.valid:
        logging.error('The ModuleImage does not have a valid transform. Did module localization succeed?')
        exit()

    size = image.transform.mean_scale() if size is None else size
    cell = warp_image(image.data, image.transform, col, row, 1/size, 1/size, 1, 1)
    cell = cell.astype(image.dtype)
    return CellImage(cell, image.modality, image.path, row, col)

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

