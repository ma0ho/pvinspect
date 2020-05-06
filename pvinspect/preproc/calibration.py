"""Flatfield and lens calibration"""
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from typing import List, Tuple
import numpy as np
from skimage import img_as_float
from tqdm.autonotebook import trange, tqdm
import numpy as np
import cv2
from skimage.exposure import rescale_intensity


def _calibrate_flatfield(
    images: List[np.ndarray], targets: List[float], order: int = 0
) -> np.array:
    assert images[0].dtype == np.float64 or images[0].dtype == np.float32

    imgs = [img.flatten() for img in images]

    assert order <= len(imgs) - 1
    order = min(len(imgs) - 1, 2) if order == 0 else order
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
    images: ImageSequence, targets: List[float], order: int = 0, use_median: bool = True
) -> np.array:
    """Perform flat-field calibration. Note that there might be several calibration shots
    with the same normalization target. In that case, a least-squares estimate is computed.

    Args:
        images (ImageSequence): Sequence of calibration shots
        targets (List[float]): Corresponding list of normalization targets
        order (int): Order of compensation polynomial. The order is chosen automatically, if order == 0 (default)
        use_median (bool): Use the median of all images with the same target

    Returns:
        coeff: order+1 coefficients starting with the lowest order coefficient
    """
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
