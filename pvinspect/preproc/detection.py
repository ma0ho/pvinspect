"""Detection, localization and segmentation of solar modules"""

import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
from pvinspect.common.transform import (
    FullMultiTransform,
    FullTransform,
    HomographyTransform,
    warp_image,
)
from pvinspect.data.exceptions import UnsupportedModalityException
from pvinspect.data.image import *
from pvinspect.data.image.sequence import TImageOrSequence, sequence_no_unwrap
from pvinspect.data.io import ObjectAnnotations
from pvinspect.preproc._mdetect.locate import apply
from shapely.geometry import Polygon
from skimage import filters, measure, morphology, transform
from skimage.filters.thresholding import threshold_otsu
from tqdm import tqdm
from typing_extensions import Literal


@overload
def locate_module_and_cells(
    sequence: TImageOrSequence,
    rows: int = None,
    cols: int = None,
    is_partial_module: bool = False,
    estimate_distortion: bool = True,
    joint_distortion_estimation: bool = True,
    orientation: str = None,
    return_bounding_boxes: Literal[False] = False,
    drop_failed: bool = False,
) -> TImageOrSequence:
    ...


@overload
def locate_module_and_cells(
    sequence: TImageOrSequence,
    rows: int = None,
    cols: int = None,
    is_partial_module: bool = False,
    estimate_distortion: bool = True,
    joint_distortion_estimation: bool = True,
    orientation: str = None,
    return_bounding_boxes: Literal[True] = True,
    drop_failed: bool = False,
) -> Tuple[TImageOrSequence, ObjectAnnotations]:
    ...


@overload
def locate_module_and_cells(
    sequence: TImageOrSequence,
    rows: int = None,
    cols: int = None,
    is_partial_module: bool = False,
    estimate_distortion: bool = True,
    joint_distortion_estimation: bool = True,
    orientation: str = None,
    return_bounding_boxes: bool = False,
    drop_failed: bool = False,
) -> Union[Tuple[TImageOrSequence, ObjectAnnotations], TImageOrSequence]:
    ...


@sequence
def locate_module_and_cells(
    sequence: TImageOrSequence,
    rows: int = None,
    cols: int = None,
    is_partial_module: bool = False,
    estimate_distortion: bool = True,
    joint_distortion_estimation: bool = True,
    orientation: str = None,
    return_bounding_boxes: bool = False,
    drop_failed: bool = False,
) -> Union[Tuple[TImageOrSequence, ObjectAnnotations], TImageOrSequence]:
    """Locate a single module and its cells

    Note:
        This methods implements the following paper:
        Hoffmann, Mathis, et al. "Fast and robust detection of solar modules in electroluminescence images."
        International Conference on Computer Analysis of Images and Patterns. Springer, Cham, 2019.

    Args:
        sequence (ImageOrSequence): A single module image or a sequence of module images
        rows (int): Number of rows of cells (taken from corresponding meta argument if not specified here)
        cols (int): Number of columns of cells (taken from corresponding meta argument if not specified here)
        is_partial_module (bool): Locate entire or partial modules
        estimate_distortion (bool): Set True to estimate lens distortion, else False 
        joint_distortion_estimation (bool): Jointly estimates distortion for all images, if set to True. Joint
            estimation is more stable. However, it requires that images are taken with the same camera and settings.
        orientation (str): Orientation of the module ('horizontal' or 'vertical' or None).
            If set to None (default), orientation is automatically determined
        return_bounding_boxes (bool): Indicates, if bounding boxes of returned modules are returned
        enable_background_suppression (bool): Indicate, if background suppresion is enabled. This sometimes causes
            problems with PL images and disabling it may help.

    Returns:
        images: The same image/sequence with location information added
    """

    if not isinstance(sequence, ImageSequence):
        raise RuntimeError()

    result = list()
    failures = 0
    mcs = list()
    dts = list()
    flags = list()
    transforms = list()

    for img in tqdm(sequence):

        data = img.data.copy()

        # very simple background suppression
        if enable_background_suppresion:
            thresh = threshold_otsu(data)
            data[data < thresh] = 0

        t, mc, dt, f = apply(
            data,
            img.get_meta("cols") if cols is None else cols,
            img.get_meta("rows") if rows is None else rows,
            is_module_detail=is_partial_module,
            orientation=orientation,
        )
        transforms.append(t)
        flags.append(f)
        mcs.append(mc)
        dts.append(dt)

    if estimate_distortion:
        if joint_distortion_estimation:
            # do joint estimation
            logging.info(
                "Jointly estimating parameters for lens distortion. This might take some time.."
            )

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
            transforms = FullMultiTransform(
                mcs_new,
                dts_new,
                image_width=sequence[0].shape[1],
                image_height=sequence[0].shape[0],
                n_dist_coeff=1,
            )
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
            for mc, dt, f, img in zip(mcs, dts, flags, sequence):
                if mc is not None and dt is not None:
                    t = FullTransform(
                        mc[f],
                        dt[f],
                        image_width=img.shape[1],
                        image_height=img.shape[0],
                        n_dist_coeff=1,
                    )
                    transforms.append(t)
                else:
                    transforms.append(None)

    for t, img in zip(transforms, sequence):
        if t is not None and t.valid:
            img_res = img.from_self(transform=t)
            if rows is not None:
                img_res = img_res.from_self(rows=rows)
            if cols is not None:
                img_res = img_res.from_self(cols=cols)
            result.append(img_res)
        elif not drop_failed:
            result.append(deepcopy(img))
            failures += 1
        else:
            failures += 1
    if failures > 0:
        msg = "Module localization falied for {:d} images".format(failures)
        if drop_failed == False:
            msg += " You may drop failed images from the resulting sequence by setting drop_failed=True."
        logging.warning(msg)

    result = EagerImageSequence.from_images(result)

    if not return_bounding_boxes:
        return result
    else:
        boxes = dict()

        # compute polygon for every module and accumulate results
        for img in result:
            if img.has_meta("transform"):
                c = img.get_meta("cols") if cols is None else cols
                r = img.get_meta("rows") if rows is None else rows
                coords = np.array([[0.0, 0.0], [c, 0.0], [c, r], [0.0, r]])
                coords_transformed = img.get_meta("transform")(coords)
                poly = Polygon(
                    [
                        (x, y)
                        for x, y in zip(
                            coords_transformed[:, 0].tolist(),
                            coords_transformed[:, 1].tolist(),
                        )
                    ]
                )
                boxes[img.get_meta("original_filename")] = [("Module", poly)]
            else:
                boxes[img.get_meta("original_filename")] = []

        return result, boxes


def segment_module_part(
    image: Image,
    first_col: int,
    first_row: int,
    cols: int,
    rows: int,
    size: int = None,
    padding: float = 0.0,
) -> Image:
    """Segment a part of a module

    Args:
        image (Image): The corresponding image
        first_col (int): First column to appear in the segment
        first_row (int): First row to appear in the segment
        cols (int): Number of columns of the segment
        rows (int): Number of rows of the segment
        size (int): Size of a cell in pixels (automatically chosen by default)
        padding (float): Optional padding around the given segment relative to the cell size
                         (must be in [0..1[ )

    Returns:
        segment: The resulting segment
    """

    if not image.has_meta("transform") or not image.get_meta("transform").valid:
        logging.error(
            "The Image does not have a valid transform. Did module localization succeed?"
        )
        exit()

    t = image.get_meta("transform")

    if padding >= 1.0 or padding < 0.0:
        logging.error("padding needs to be in [0..1[")
        exit()

    last_col = first_col + cols
    last_row = first_row + rows

    size = t.mean_scale() if size is None else size
    result = warp_image(
        image.data,
        t,
        first_col - padding,
        first_row - padding,
        1 / size,
        1 / size,
        cols + 2 * padding,
        rows + 2 * padding,
    )
    result = result.astype(image.data.dtype)  # type: ignore

    # bounding box in original image coords
    bb = [
        [first_col - padding, first_row - padding],
        [first_col + cols + padding, first_row + rows + padding],
    ]
    bb = t(np.array(bb))
    bb = Polygon.from_bounds(bb[0][0], bb[0][1], bb[1][0], bb[1][1])
    original = image.from_other(image, segment_module_original_box=bb)

    return image.from_self(
        data=result,
        cols=cols + min(first_col, 0),
        rows=rows + min(first_row, 0),
        first_col=first_col if first_col >= 0 else None,
        first_row=first_row if first_row >= 0 else None,
        transform=None,
        segment_module_original=original,
    )


def segment_module(image: Image, size: int = None, padding: float = 0.0) -> Image:
    """Obtain a rectified, cropped and undistorted module image

    Args:
        image (Image): A single module image
        size (int): Size of a cell in pixels (automatically chosen by default)
        padding (float): Optional padding around the given segment relative to the cell size
                         (must be in [0..1[ )

    Returns:
        module: The resulting module image
    """

    return segment_module_part(
        image, 0, 0, image.get_meta("cols"), image.get_meta("rows"), size, padding
    )


def segment_cell(
    image: Image, row: int, col: int, size: int = None, padding: float = 0.0
) -> Image:
    """Obtain a cell image from a module image

    Args:
        image (ImageOrSequence): A single module image
        row (int): The row number (starting at 0)
        col (int): The column number (starting at 0)
        size (int): Size of the resulting cell image in pixels (automatically chosen by default)
        padding (float): Optional padding around the cell relative to the cell size
                         (must be in [0..1[ )

    Returns:
        cells: The segmented cell image
    """

    result = segment_module_part(image, col, row, 1, 1, size, padding)
    return result.from_self(row=row, col=col)


@sequence
def segment_modules(sequence: TImageOrSequence, size: int = None) -> TImageOrSequence:
    """Obtain rectified, cropped and undistorted module images from a sequence. Note that images that do not have a valid transform,
    possibly because the detection step failed, are silently ignored.

    Args:
        sequence (ImageOrSequence): A single module image or a sequence of module images
        size (int): Size of the resulting cell images in pixels (automatically chosen by default)

    Returns:
        module: The segmented module images
    """

    if not isinstance(sequence, ImageSequence):
        raise RuntimeError()

    scales = np.array(
        [
            img.get_meta("transform").mean_scale()
            for img in sequence
            if img.has_meta("transform") and img.get_meta("transform").valid
        ]
    )
    if scales.std() > 0.1 * scales.mean() and size is None:  # type: ignore
        logging.warning(
            "The size of cells within the sequences varies by more than 10%. However, segment_modules, \
creates images of a fixed size. Please consider to split the sequence into multiple sequences \
with less variation in size."
        )
    if size is None:
        size = int(scales.mean())

    result = list()
    img: Image
    for img in tqdm(sequence):

        # for the moment, we silently ignore images without a valid transform
        if img.has_meta("transform") and img.get_meta("transform").valid:
            result.append(segment_module(img, size))

    return EagerImageSequence.from_images(result)


@sequence(True)
def segment_cells(sequence: TImageOrSequence, size: int = None) -> TImageOrSequence:
    """Obtain cell images from a sequence of module images. Note that images that do not have a valid transform,
    possibly because the detection step failed, are silently ignored.

    Args:
        sequence (ImageOrSequence): A single module image or a sequence of module images
        size (int): Size of the resulting cell images in pixels (automatically chosen by default)

    Returns:
        cells: The segmented cell images
    """

    if not isinstance(sequence, ImageSequence):
        raise RuntimeError()

    scales = np.array(
        [
            img.get_meta("transform").mean_scale()
            for img in sequence
            if img.has_meta("transform") and img.get_meta("transform").valid
        ]
    )
    if scales.std() > 0.1 * scales.mean() and size is None:  # type: ignore
        logging.warning(
            "The size of cells within the sequences varies by more than 10%. However, segment_cells, \
creates cell images of a fixed size. Please consider to split the sequence into multiple sequences \
with less variation in size."
        )
    if size is None:
        size = int(scales.mean())

    result = list()
    for img in tqdm(sequence):
        for row in range(img.get_meta("rows")):
            for col in range(img.get_meta("cols")):

                # for the moment, we silently ignore images without a valid transform
                if (
                    img.has_meta("transform") is not None
                    and img.get_meta("transform").valid
                ):
                    result.append(segment_cell(img, row, col, size))

    return EagerImageSequence.from_images(result)


def _do_locate_multiple_modules(
    image: Image,
    scale: float,
    reject_size_thresh: float,
    reject_fill_thresh: float,
    padding: float,
    cols: Optional[int],
    rows: Optional[int],
    drop_clipped_modules: bool,
) -> Tuple[List[Image], List[Polygon]]:

    # filter + binarize
    # image_f = filters.gaussian(image._data, filter_size)
    image_f = transform.rescale(image.data, scale)
    image_f = image_f > filters.threshold_otsu(image_f)

    # find regions
    labeled = morphology.label(image_f)
    regions = measure.regionprops(labeled)

    # process regions
    # check if bbox is filled to 100*reject_fill_thres%
    regions = [r for r in regions if r.area / r.bbox_area >= reject_fill_thresh]
    if len(regions) == 0:
        return [], []

    max_area = int(np.max([r.bbox_area for r in regions]))
    results = []
    boxes = []
    i = 0
    for r in regions:
        # check size
        if r.bbox_area < reject_size_thresh * max_area:
            continue

        # check not touching boundary
        if (
            r.bbox[0] == 0
            or r.bbox[1] == 0
            or r.bbox[2] == labeled.shape[0]
            or r.bbox[3] == labeled.shape[1]
        ) and drop_clipped_modules:
            continue

        # transform bounding box to original size
        s = 1 / scale
        bbox = [int(r.bbox[i] * s) for i in range(4)]

        # crop module
        pad = int(np.sqrt(r.bbox_area) * padding * s)
        y0, x0 = max(0, bbox[0] - pad), max(0, bbox[1] - pad)
        y1, x1 = (
            min(image.shape[0], bbox[2] + pad),
            min(image.shape[1], bbox[3] + pad),
        )
        boxes.append(("Module", Polygon.from_bounds(x0, y0, x1, y1)))
        crop = image.data[y0:y1, x0:x1]
        if rows is not None and cols is not None:
            results.append(image.from_self(data=crop, cols=cols, rows=rows))
        else:
            results.append(image.from_self(data=crop))
        i += 1

    return results, boxes


@sequence_no_unwrap
def locate_multiple_modules(
    sequence: TImageOrSequence,
    scale: float = 0.31,
    reject_size_thresh: float = 0.26,
    reject_fill_thresh: float = 0.42,
    padding: float = 0.05,
    drop_clipped_modules: bool = True,
    cols: int = None,
    rows: int = None,
    return_bounding_boxes: bool = False,
) -> Union[
    Tuple[Optional[TImageOrSequence], ObjectAnnotations], Optional[TImageOrSequence]
]:
    """Perform localization and segmentation of multiple modules. The method is published in Hoffmann, Mathis, et al. 
    "Deep Learning-based Pipeline for Module Power Prediction from EL Measurements." arXiv preprint arXiv:2009.14712 (2020).

    Args:
        sequence (ImageOrSequence): Input images
        scale (float): Image is scaled to this size before processing
        reject_size_thresh (float): Detections smaller than this times the median size of detections are rejected
        reject_fill_thresh (float): Detections, where more that this parts of the area are black after thresholding are rejected
        padding (float): Detections are padded by this times the average size length of the bounding box
        drop_clipped_modules (bool): Indicate, if detections that touch the boundary are dropped
        cols (int): Number of columns of cells of a single module
        cols (rows): Number of rows of cells of a single module
        return_bounding_boxes (bool): If true, return the bounding boxes in addition to the crops

    Returns:
        The cropped modules as a ImageSequence as well as (optionally), the bounding boxes.
    """

    if not isinstance(sequence, ImageSequence):
        raise RuntimeError()

    # process sequence
    results = list()
    boxes: ObjectAnnotations = dict()
    # for img in tqdm(sequence):
    for img in tqdm(sequence):
        modules, b = _do_locate_multiple_modules(
            img,
            scale,
            reject_size_thresh,
            reject_fill_thresh,
            padding,
            cols,
            rows,
            drop_clipped_modules,
        )

        # add original images with box annotations as meta
        imgs_org = [
            EagerImage.from_other(img, multimodule_index=i, multimodule_boxes=b)
            for i in range(len(modules))
        ]
        modules = [
            EagerImage.from_other(m, multimodule_original=o)
            for m, o in zip(modules, imgs_org)
        ]

        results += modules
        boxes[img.get_meta("original_filename")] = b

    if return_bounding_boxes:
        if len(results) > 0:
            return EagerImageSequence.from_images(results), boxes
        else:
            return None, dict()
    else:
        if len(results) > 0:
            return EagerImageSequence.from_images(results)
        else:
            return None
