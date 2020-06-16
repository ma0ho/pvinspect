"""Detection, localization and segmentation of solar modules"""

from pvinspect.preproc._mdetect.locate import apply
from pvinspect.common.transform import (
    HomographyTransform,
    warp_image,
    FullMultiTransform,
    FullTransform,
)
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from pvinspect.data.exceptions import UnsupportedModalityException
from pvinspect.data.io import ObjectAnnotations
from typing import Union, List, Optional, Dict, Tuple
from tqdm.auto import tqdm
from copy import deepcopy
import logging
from pvinspect.common._ipy_exit import exit
import numpy as np
from skimage import measure, filters, morphology, transform
from shapely.geometry import Polygon


@_sequence
def locate_module_and_cells(
    sequence: ModuleImageOrSequence,
    estimate_distortion: bool = True,
    orientation: str = None,
    return_bounding_boxes: bool = False,
) -> Union[Tuple[ModuleImageOrSequence, ObjectAnnotations], ModuleImageSequence]:
    """Locate a single module and its cells

    Note:
        This methods implements the following paper:
        Hoffmann, Mathis, et al. "Fast and robust detection of solar modules in electroluminescence images."
        International Conference on Computer Analysis of Images and Patterns. Springer, Cham, 2019.

    Args:
        sequence (ModuleImageOrSequence): A single module image or a sequence of module images 
        estimate_distortion (bool): Set True to estimate lens distortion, else False 
        orientation (str): Orientation of the module ('horizontal' or 'vertical' or None).
            If set to None (default), orientation is automatically determined
        return_bounding_boxes (bool): Indicates, if bounding boxes of returned modules are returned

    Returns:
        images: The same image/sequence with location information added
    """

    if sequence[0].modality != EL_IMAGE:
        logging.error("Module localization is not supporting given imaging modality")
        exit()

    result = list()
    failures = 0
    mcs = list()
    dts = list()
    flags = list()
    transforms = list()
    for img in tqdm(sequence.images):
        t, mc, dt, f = apply(
            img.data,
            img.cols,
            img.rows,
            is_module_detail=isinstance(img, PartialModuleImage),
            orientation=orientation,
        )
        transforms.append(t)
        flags.append(f)
        mcs.append(mc)
        dts.append(dt)

    if estimate_distortion:
        if sequence.same_camera:
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
                image_width=sequence.shape[1],
                image_height=sequence.shape[0],
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
            for mc, dt, f, img in zip(mcs, dts, flags, sequence.images):
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

    for t, img in zip(transforms, sequence.images):
        if t is not None and t.valid:
            img_res = type(img).from_other(img, meta={"transform": t})
            result.append(img_res)
        else:
            result.append(deepcopy(img))
            failures += 1
    if failures > 0:
        logging.warning("Module localization falied for {:d} images".format(failures))

    result = ModuleImageSequence.from_other(sequence, images=result)

    if not return_bounding_boxes:
        return result
    else:
        boxes = dict()

        # compute polygon for every module and accumulate results
        for img in result:
            if img.has_meta("transform"):
                c = img.cols
                r = img.rows
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
                boxes[img.path.name] = [("Module", poly)]
            else:
                boxes[img.path.name] = []

        return result, boxes


def segment_module_part(
    image: ModuleImage,
    first_col: int,
    first_row: int,
    cols: int,
    rows: int,
    size: int = None,
    padding: float = 0.0,
) -> PartialModuleImage:
    """Segment a part of a module

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
    """

    if not image.has_meta("transform") or not image.get_meta("transform").valid:
        logging.error(
            "The ModuleImage does not have a valid transform. Did module localization succeed?"
        )
        exit()

    t = image.get_meta("transform")

    if padding >= 1.0 or padding < 0.0:
        logging.error("padding needs to be in [0..1[")
        exit()

    last_col = first_col + cols
    last_row = first_row + rows
    if last_row > image.rows or last_col > image.cols:
        logging.error("The row or column index exceeds the module geometry")
        exit()

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
    result = result.astype(image.data.dtype)
    transform = HomographyTransform(
        np.array(
            [
                [first_row - padding, first_col - padding],
                [first_row - padding, last_col - padding],
                [last_row - padding, first_col - padding],
                [last_row - padding, last_col - padding],
            ]
        ),
        np.array(
            [
                [0.0, 0.0],
                [0.0, size * cols],
                [size * rows, 0.0],
                [size * rows, size * cols],
            ]
        ),
    )

    # bounding box in original image coords
    bb = [
        [first_col - padding, first_row - padding],
        [first_col + cols + padding, first_row + rows + padding],
    ]
    bb = t(np.array(bb))
    bb = Polygon.from_bounds(bb[0][0], bb[0][1], bb[1][0], bb[1][1])
    original = image.from_other(image, meta={"segment_module_original_box": bb})

    return PartialModuleImage.from_other(
        image,
        drop_meta_types=[Polygon],  # geometric attributes are invalid now..
        data=result,
        cols=cols,
        rows=rows,
        first_col=first_col,
        first_row=first_row,
        meta={"transform": transform, "segment_module_original": original},
    )


def segment_module(
    image: ModuleImage, size: int = None, padding: float = 0.0
) -> ModuleImage:
    """Obtain a rectified, cropped and undistorted module image

    Args:
        image (ModuleImage): A single module image
        size (int): Size of a cell in pixels (automatically chosen by default)
        padding (float): Optional padding around the given segment relative to the cell size
                         (must be in [0..1[ )

    Returns:
        module: The resulting module image
    """

    result = segment_module_part(image, 0, 0, image.cols, image.rows, size, padding)
    return ModuleImage.from_other(result)


def segment_cell(
    image: ModuleImage, row: int, col: int, size: int = None, padding: float = 0.0
) -> CellImage:
    """Obtain a cell image from a module image

    Args:
        image (ModuleImageOrSequence): A single module image
        row (int): The row number (starting at 0)
        col (int): The column number (starting at 0)
        size (int): Size of the resulting cell image in pixels (automatically chosen by default)
        padding (float): Optional padding around the cell relative to the cell size
                         (must be in [0..1[ )

    Returns:
        cells: The segmented cell image
    """

    result = segment_module_part(image, col, row, 1, 1, size, padding)
    return CellImage.from_other(result, row=row, col=col)


@_sequence
def segment_modules(
    sequence: ModuleImageOrSequence, size: int = None
) -> ModuleImageSequence:
    """Obtain rectified, cropped and undistorted module images from a sequence. Note that images that do not have a valid transform,
    possibly because the detection step failed, are silently ignored.

    Args:
        sequence (ModuleImageOrSequence): A single module image or a sequence of module images
        size (int): Size of the resulting cell images in pixels (automatically chosen by default)

    Returns:
        module: The segmented module images
    """

    scales = np.array(
        [
            img.get_meta("transform").mean_scale()
            for img in sequence.images
            if img.has_meta("transform") and img.get_meta("transform").valid
        ]
    )
    if scales.std() > 0.1 * scales.mean() and size is None:
        logging.warning(
            "The size of cells within the sequences varies by more than 10%. However, segment_modules, \
creates images of a fixed size. Please consider to split the sequence into multiple sequences \
with less variation in size."
        )
    if size is None:
        size = int(scales.mean())

    result = list()
    for img in tqdm(sequence.images):

        # for the moment, we silently ignore images without a valid transform
        if img.has_meta("transform") and img.get_meta("transform").valid:
            result.append(segment_module(img, size))

    return type(sequence).from_other(sequence, images=result, same_camera=False)


@_sequence(True)
def segment_cells(
    sequence: ModuleImageOrSequence, size: int = None
) -> CellImageSequence:
    """Obtain cell images from a sequence of module images. Note that images that do not have a valid transform,
    possibly because the detection step failed, are silently ignored.

    Args:
        sequence (ModuleImageOrSequence): A single module image or a sequence of module images
        size (int): Size of the resulting cell images in pixels (automatically chosen by default)

    Returns:
        cells: The segmented cell images
    """

    scales = np.array(
        [
            img.get_meta("transform").mean_scale()
            for img in sequence.images
            if img.has_meta("transform") and img.get_meta("transform").valid
        ]
    )
    if scales.std() > 0.1 * scales.mean() and size is None:
        logging.warning(
            "The size of cells within the sequences varies by more than 10%. However, segment_cells, \
creates cell images of a fixed size. Please consider to split the sequence into multiple sequences \
with less variation in size."
        )
    if size is None:
        size = int(scales.mean())

    result = list()
    for img in tqdm(sequence.images):
        for row in range(img.rows):
            for col in range(img.cols):

                # for the moment, we silently ignore images without a valid transform
                if (
                    img.has_meta("transform") is not None
                    and img.get_meta("transform").valid
                ):
                    result.append(segment_cell(img, row, col, size))

    return CellImageSequence(result)


def _do_locate_multiple_modules(
    image: Image,
    scale: float,
    reject_size_thresh: float,
    reject_fill_thresh: float,
    padding: float,
    cols: int,
    rows: int,
    drop_clipped_modules: bool,
) -> Tuple[List[ModuleImage], List[Polygon]]:

    # filter + binarize
    # image_f = filters.gaussian(image._data, filter_size)
    image_f = transform.rescale(image._data, scale)
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
        crop = image._data[y0:y1, x0:x1]
        p = image.path.parent / "{}_module{:02d}{}".format(
            image.path.stem, i, image.path.suffix
        )
        results.append(
            ModuleImage(
                data=crop, modality=image.modality, path=p, cols=cols, rows=rows
            )
        )
        i += 1

    return results, boxes


@_sequence(True)
def locate_multiple_modules(
    sequence: ImageOrSequence,
    scale: float = 0.31,
    reject_size_thresh: float = 0.26,
    reject_fill_thresh: float = 0.42,
    padding: float = 0.05,
    drop_clipped_modules: bool = True,
    cols: int = None,
    rows: int = None,
    return_bounding_boxes: bool = False,
) -> Tuple[ModuleImageSequence, ObjectAnnotations]:
    """Perform localization and segmentation of multiple modules

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
        The cropped modules as a ModuleImageSequence as well as (optionally), the bounding boxes.
    """

    # process sequence
    results = list()
    boxes = dict()
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
            Image.from_other(img, meta={"multimodule_index": i, "multimodule_boxes": b})
            for i in range(len(modules))
        ]
        modules = [
            ModuleImage.from_other(m, meta={"multimodule_original": o})
            for m, o in zip(modules, imgs_org)
        ]

        results += modules
        boxes[img.path] = b

    if return_bounding_boxes:
        if len(results) > 0:
            return ModuleImageSequence(results, same_camera=False), boxes
        else:
            return None, dict()
    else:
        if len(results) > 0:
            return ModuleImageSequence(results, same_camera=False)
        else:
            return None
