"""Flatfield and lens calibration"""
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from pvinspect.common.exceptions import (
    MisconfigurationException,
    InvalidArgumentException,
)
from pvinspect.common.types import PathOrStr
from typing import List, Tuple
import numpy as np
from skimage import img_as_float
from tqdm.autonotebook import trange, tqdm
import numpy as np
import cv2
from skimage.exposure import rescale_intensity
import os
import logging
import pickle


def _calibrate_flatfield(
    images: List[np.ndarray], targets: List[float], order: int
) -> np.array:

    imgs = [img.flatten() for img in images]

    order = min(len(imgs) - 1, 2) if order == -1 else order
    coeff = np.empty((order + 1, imgs[0].shape[0]), dtype=images[0].dtype)

    if order == 1 and len(images) == 2:
        coeff[1] = (targets[1] - targets[0]) / (imgs[1] - imgs[0])
        coeff[0] = targets[0] - coeff[1] * imgs[0]
    else:
        for i in trange(imgs[0].shape[0]):
            coeff[:, i] = np.polynomial.polynomial.polyfit(
                [img[i] for img in imgs], targets, order
            )

    # coeff = np.nan_to_num(coeff)
    return coeff.reshape((order + 1, images[0].shape[0], images[0].shape[1]))


def calibrate_flatfield(
    images: ImageSequence,
    targets: List[float],
    order: int = -1,
    use_median: bool = True,
) -> np.array:
    """Perform flat-field calibration. Note that there might be several calibration shots
    with the same normalization target. In that case, a least-squares estimate is computed.

    Args:
        images (ImageSequence): Sequence of calibration shots
        targets (List[float]): Corresponding list of normalization targets
        order (int): Order of compensation polynomial. The order is chosen automatically, if order == -1 (default)
        use_median (bool): Use the median of all images with the same target

    Returns:
        coeff: order+1 coefficients starting with the lowest order coefficient
    """
    if order == 0:
        raise InvalidArgumentException("Order of flat-field-polynom must not be 0")
    if order > len(images) - 1:
        raise MisconfigurationException(
            "Need at least {:d} images to calibrate a flat-field-polynom of degree {:d} ({:d} given)".format(
                order + 1, order, len(images)
            )
        )

    images = images.as_type(DType.FLOAT)

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


def calibrate_distortion(
    images: ImageSequence, checkerboard_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Perform lens calibration

    Args:
        images (ImageSequence): Sequence of calibration shots
        checkerboard_size (Tuple[int, int]): Size of the checkerboard (outer corners do not count)

    Returns:
        mtx: Matrix of intrinsic camera parameters
        dist: Vectors of distortion coefficients (k1, k2, p1, p2, k3)
        newcameramtx: Matrix that performs additional scaling to account for black borders
        roi: ROI of valid pixels
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0 : checkerboard_size[0], 0 : checkerboard_size[1]
    ].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for img in tqdm(images):
        img = rescale_intensity(img.data, out_range=(0, 255)).astype(np.uint8)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCornersSB(
            img, checkerboard_size, flags=cv2.CALIB_CB_ACCURACY
        )

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, images[0].shape[::-1], None, None
    )
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, img.shape, 1, img.shape
    )

    return mtx, dist, newcameramtx, roi


@_sequence
def _compensate_flatfield(
    sequence: ModuleImageOrSequence, coeff: np.ndarray
) -> ModuleImageOrSequence:
    """Low level method to perform flat field correction

    Args:
        sequence (ModuleImageOrSequence): Sequence of images or single image (needs to be of type float)
        coeff (np.ndarray): Compensation coefficients

    Returns:
        sequence: The corrected images
    """
    if sequence.dtype != DType.FLOAT:
        raise InvalidArgumentException(
            "Images must be converted to DType.FLOAT, before applying flat-field-correction"
        )

    def fn(data, coeff):
        res = coeff[0].copy()
        data = data.copy()
        for i in range(1, coeff.shape[0]):
            res += coeff[i] * data
            data *= data
        return np.clip(res, 0.0, 1.0)

    return sequence.apply_image_data(fn, coeff)


@_sequence
def compensate_flatfield(
    sequence: ModuleImageOrSequence, coeff: np.ndarray
) -> ModuleImageOrSequence:
    """Perform flat field correction

    Args:
        sequence (ModuleImageOrSequence): Sequence of images or single image
        coeff (np.ndarray): Compensation coefficients

    Returns:
        sequence: The corrected images
    """
    sequence = sequence.as_type(DType.FLOAT)
    sequence = _compensate_flatfield(sequence, coeff)

    return sequence


@_sequence
def compensate_distortion(
    sequence: ModuleImageOrSequence,
    mtx: np.ndarray,
    dist: np.ndarray,
    newcameramtx: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> ModuleImageOrSequence:
    """Perform lens distortion correction

    Args:
        sequence (ModuleImageOrSequence): Sequence of images or single image
        mtx (np.ndarray): Matrix of instrinsic camera parameters
        dist (np.ndarray): Matrix of distortion coefficients
        newcameramtx (np.ndarray): Matrix that performs additional scaling to account for black borders
        roi (Tuple[int, int, int, int]): ROI of valid pixels

    Returns:
        sequence: The corrected images
    """

    def corr(x):
        dst = cv2.undistort(x, mtx, dist, None, newcameramtx)
        return dst[roi[0] : roi[2], roi[1] : roi[3]]

    return sequence.apply_image_data(corr)


class Calibration:
    """Handles camera calibration and processing of images"""

    def __init__(self, ff_poly_order: int = -1, ff_use_median: bool = True):
        """Initialize calibration object

        Args:
            ff_poly_order (int): Order of flat-field compensation polynomial. The order is chosen automatically, if order == -1 (default)
            ff_use_median (bool): Use the median of all images with the same target
        """

        self._ff_poly_order = ff_poly_order
        self._ff_use_median = ff_use_median
        self._ff_calibration = None
        self._dist_calibration = None

    def calibrate_flatfield(self, images: ImageSequence, targets: List[float]):
        """Perform flat-field calibration. Note that there might be several calibration shots
        with the same normalization target. In that case, a least-squares estimate is computed.

        Args:
            images (ImageSequence): Sequence of calibration shots
            targets (List[float]): Corresponding list of normalization targets
        """

        self._ff_calibration = calibrate_flatfield(
            images=images,
            targets=targets,
            order=self._ff_poly_order,
            use_median=self._ff_use_median,
        )

    def calibrate_distortion(
        self, images: ImageSequence, checkerboard_size: Tuple[int, int]
    ):
        """Perform lens calibration

        Args:
            images (ImageSequence): Sequence of calibration shots
            checkerboard_size (Tuple[int, int]): Size of the checkerboard (outer corners do not count)
        """

        self._dist_calibration = calibrate_distortion(
            images=images, checkerboard_size=checkerboard_size
        )

    def process(self, images: ImageOrSequence):
        """Process images and compensate camera artifacts

        Args:
            images (ImageOrSequence): Sequence of images or single image
        """

        if self._ff_calibration is not None:
            images = compensate_flatfield(images, self._ff_calibration)
        else:
            logging.warn(
                "No flat-field calibration data available. Use calibrate_flatfield to perform calibration. Skipping flat-field compensation.."
            )

        if self._dist_calibration is not None:
            images = compensate_distortion(images, *self._dist_calibration)
        else:
            logging.warn(
                "No distortion calibration data available. Use calibrate_distortion to perform calibration. Skipping distortion compensation.."
            )

        return images

    def save(self, path: PathOrStr):
        """Save calibration data to disk

        Args:
            path (PathOrStr): Target to save file to
        """

        with open(path, "wb") as f:
            pickle.dump(self, f)


def load_calibration(path: PathOrStr) -> Calibration:
    """Load calibration data from file

    Args:
        path (PathOrStr): Path to calibration file
    
    Returns:
        calibration (Calibration): The calibration object
    """

    with open(path, "rb") as f:
        return pickle.load(f)
