from . import locate_corners
from . import locate_module
from pvinspect.common import transform
from .config import (
    CORNER_DETECTION_PATCH_SIZE,
    RANSAC_THRES,
    PREPROC_LOWER_PERCENTILE,
    PREPROC_UPPER_PERCENTILE,
)
import numpy as np
from typing import Tuple
from skimage import exposure
from scipy.ndimage.filters import median_filter


def apply(
    img: np.ndarray,
    n_cols: int,
    m_rows: int,
    is_module_detail: bool,
    orientation: str = None,
) -> Tuple[transform.Transform, np.ndarray, np.ndarray, np.ndarray]:
    """Locate the module in given EL image

    Args:
        img (np.array): The image
        n_cols (int): Number of columns of cells
        m_rows (int): Number of rows of cells
        is_module_detail (bool): Is this image showing only a part of a module?
        orientation (std): Orientation of the module (default: None = determine automatically)

    Returns:
        result: Returns a tuple (transform, model corners, detected corners, accepted flags)
    """

    # preprocessing
    img = img.copy()
    x = np.percentile(img.flatten(), PREPROC_UPPER_PERCENTILE, interpolation="lower")
    img[img > x] = x
    x = np.percentile(img.flatten(), PREPROC_LOWER_PERCENTILE, interpolation="lower")
    img[img < x] = x
    img = exposure.rescale_intensity(img)
    img = img / np.max(img)

    # locate module
    module_bb_fit = locate_module.locate_module(img, n_cols, m_rows)
    if module_bb_fit is None:
        return None, None, None, None  # failed
    module_bb = locate_module.module_boundingbox_model(
        module_bb_fit, n_cols, m_rows, orientation
    )
    locate_transform = transform.HomographyTransform(module_bb, module_bb_fit)

    # fit outer corners
    outer_corners = locate_corners.outer_corners_model(n_cols, m_rows)
    outer_corners_fit, outer_accepted = locate_corners.fit_outer_corners(
        img,
        outer_corners,
        locate_transform,
        CORNER_DETECTION_PATCH_SIZE,
        n_cols,
        m_rows,
        is_module_detail=is_module_detail,
    )
    outer_corners_fit = locate_transform(outer_corners_fit)

    # reestimate transform
    locate_transform = transform.HomographyTransform(
        outer_corners[outer_accepted], outer_corners_fit[outer_accepted], ransac=True
    )

    # fit inner corners
    # TODO:
    #   - fix inaccurate detection of x/y location if considered edges differ in length
    #   - refine location of module corners first
    inner_corners = locate_corners.inner_corners_model(n_cols, m_rows)
    inner_corners_fit, inner_accepted = locate_corners.fit_inner_corners(
        img, inner_corners, locate_transform, CORNER_DETECTION_PATCH_SIZE
    )
    inner_corners_fit = locate_transform(inner_corners_fit)
    locate_transform = transform.HomographyTransform(
        np.concatenate((module_bb, inner_corners[inner_accepted]), axis=0),
        np.concatenate((module_bb_fit, inner_corners_fit[inner_accepted]), axis=0),
    )

    # reestimate transform using all correspondences
    all_corners_model = np.concatenate((inner_corners, outer_corners), axis=0)
    all_corners_fit = np.concatenate((inner_corners_fit, outer_corners_fit), axis=0)
    all_accepted = np.concatenate((inner_accepted, outer_accepted), axis=0)
    locate_transform = transform.HomographyTransform(
        all_corners_model[all_accepted],
        all_corners_fit[all_accepted],
        img.shape[0],
        img.shape[1],
        ransac=True,
        ransac_thres=RANSAC_THRES,
    )
    all_accepted[all_accepted] = np.logical_and(
        all_accepted[all_accepted], locate_transform.mask
    )

    return locate_transform, all_corners_model, all_corners_fit, all_accepted
