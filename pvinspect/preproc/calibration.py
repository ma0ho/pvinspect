"""Flatfield and lens calibration"""
from pvinspect.data.image import *
from pvinspect.data.image import _sequence
from pvinspect.common.exceptions import (
    MisconfigurationException,
    InvalidArgumentException,
)
from pvinspect.common.types import PathOrStr
from typing import List, Tuple, Union
import numpy as np
from skimage import img_as_float
from tqdm.autonotebook import trange, tqdm
import numpy as np
import cv2
from skimage.exposure import rescale_intensity
import os
import logging
import pickle
from shapely.geometry import Polygon
from shapely import affinity
from skimage import transform, morphology, measure, filters
from typing import Union
import logging


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

    # perserve mean of ff-image with highest intensity
    maxi = np.argmax(targets)
    mean = np.mean(images[maxi])
    coeff *= mean

    return coeff.reshape((order + 1, images[0].shape[0], images[0].shape[1]))


def calibrate_flatfield(
    images: Union[ImageSequence, List[ImageSequence]],
    targets: List[float],
    order: int = -1,
    use_median: bool = True,
) -> np.array:
    """Perform flat-field calibration. Note that there might be several calibration shots
    with the same normalization target. In that case, a least-squares estimate is computed.

    Args:
        images (Union[ImageSequence, List[ImageSequence]]): Sequence of calibration shots or list of sequences
        targets (List[float]): Corresponding list of calibration targets. If images is an ImageSequence, specify
            one target per element of the sequence. If images is a list of ImageSequence, specify one target
            per element of the list. The targets specify normalized intensity of the flatfield calibration image.
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

    # flatten multiple sequences
    if isinstance(images, list):
        images_new, targets_new = list(), list()
        for l, t in zip(images, targets):
            targets_new += [t] * len(l)
            images_new += l.images
        targets = targets_new
        images = ImageSequence(images=images_new, same_camera=True)

    # assure float type
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

        return _calibrate_flatfield(images=images_new, targets=targets_new, order=order)

    return _calibrate_flatfield(
        images=[img.data for img in images], targets=targets, order=order
    )


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

    # sanity check
    if len(objpoints) < 0.8 * len(images):
        logging.warn(
            "Checkerboard detection failed on more than 20% percent of the images (only {:d} succeeded). Please consider preprocessing the images, such that the checkerboard has a better contrast.".format(
                len(objpoints)
            )
        )

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
    """Low level method to perform flat field correction"""

    def fn(data, coeff):
        res = coeff[0].copy()
        data = data.copy()
        for i in range(1, coeff.shape[0]):
            res += coeff[i] * data
            data *= data

        if res.min() < 0.0 or res.max() > 1.0:
            logging.warn(
                "Image exceeds datatype limits after FF-compensation. Clipping to limits.."
            )
            res = np.clip(res, 0.0, 1.0)

        return res

    return sequence.apply_image_data(fn, coeff)


def _do_locate_reference_cell(
    image: Image, reference_area: int, scale: float = 0.1,
) -> Polygon:
    """Perform localization of reference cell"""

    # filter + binarize
    image_f = transform.rescale(image.data, scale)

    thresh = filters.threshold_multiotsu(image_f, classes=5)[0]
    image_t = image_f > thresh

    # find regions
    labeled = morphology.label(image_t)
    regions = measure.regionprops(labeled)

    # drop areas that are not approximately square
    regions = [
        r
        for r in regions
        if np.abs(r.bbox[2] - r.bbox[0]) / np.abs(r.bbox[3] - r.bbox[1]) > 0.8
        and np.abs(r.bbox[2] - r.bbox[0]) / np.abs(r.bbox[3] - r.bbox[1]) < 1.8
    ]

    # convert to Polygon
    tmp: List[Polygon] = []
    for r in regions:
        bbox = [int(r.bbox[i] * 1 / scale) for i in range(4)]
        y0, x0 = max(0, bbox[0]), max(0, bbox[1])
        y1, x1 = (
            min(image.shape[0], bbox[2]),
            min(image.shape[1], bbox[3]),
        )
        box = Polygon.from_bounds(x0, y0, x1, y1)
        tmp.append(box)

    # drop boxes that intersect with others
    regions = [
        r for r in tmp if not np.any([x.intersects(r) and x is not r for x in tmp])
    ]

    if len(regions) == 0:
        return None

    # process regions
    area_dev = [np.abs((r.area - reference_area) / reference_area) for r in regions]
    min_dev_idx = np.argmin(area_dev)

    if area_dev[min_dev_idx] < 2.0:
        return affinity.scale(regions[min_dev_idx], xfact=0.5, yfact=0.5)
    else:
        return None


def _do_reference_scaling(
    image: Image, area: Polygon, ref: Union[float, int],
) -> Image:
    """Compute mean intensity over area area and rescale image"""
    original_type = image.dtype

    # compute statistics and scale
    x, y = list(zip(*area.exterior.coords))
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    median = np.median(image.data[int(ymin) : int(ymax), int(xmin) : int(xmax)])
    data = (image.data * (ref / median)).astype(DTYPE_FLOAT)

    # convert to original datatype
    if original_type == DType.INT:
        data = data.astype(DTYPE_INT)
    elif original_type == DType.UNSIGNED_INT:
        data = data.astype(DTYPE_UNSIGNED_INT)

    return image.from_other(image, data=data, meta={"calibration_reference_box": area})


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
    otype = sequence.dtype
    sequence = sequence.as_type(DType.FLOAT)
    sequence = _compensate_flatfield(sequence, coeff)
    if otype is not None:
        sequence = sequence.as_type(otype)

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
        self._ff_dtype = None
        self._dist_calibration = None

    def calibrate_flatfield(
        self, images: Union[ImageSequence, List[ImageSequence]], targets: List[float]
    ):
        """Perform flat-field calibration. Note that there might be several calibration shots
        with the same normalization target. In that case, a least-squares estimate is computed.

        Args:
            images (Union[ImageSequence, List[ImageSequence]]): Sequence of calibration shots
            targets (List[float]): Corresponding list of calibration targets. If images is an ImageSequence, specify
                one target per element of the sequence. If images is a list of ImageSequence, specify one target
                per element of the list. The targets specify normalized intensity of the flatfield calibration image.
        """
        if np.max(targets) != 1.0:
            raise RuntimeError("One of the calibration targets must equal 1.0")
        if np.min(targets) != 0.0:
            raise RuntimeError("One of the calibration targets must equal 0.0")

        # store dtype for later checks
        self._ff_dtype = images[0].dtype

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

    def process(
        self,
        images: ImageOrSequence,
        flatfield: bool = True,
        distortion: bool = True,
        reference_cell_area: int = None,
        reference_intensity_key: str = None,
    ):
        """Process images and compensate camera artifacts, optionally normalize using reference cell

        Args:
            images (ImageOrSequence): Sequence of images or single image
            clip_result (bool): Clip the result such that all pixels are in the range [0,1]
            flatfield (bool): Perform flat-field compensation
            distortion (bool): Perform distortion compensation
            reference_cell_area (int): Approximate area of the reference cell in pixels. If set
                images are automatically normalizes such that the reference cell intensity
                matches values given by meta attribute reference_intensity_key
            reference_intensity_key (str): Meta key that holds the reference intensity
        """

        if self._ff_calibration is not None and flatfield:
            if images.dtype != self._ff_dtype:
                logging.warn(
                    "Datatype of images ({}) differs from calibration images {}. This might lead to incorrect results.".format(
                        images.dtype, self._ff_dtype
                    )
                )
            logging.info("Processing flatfield compensation..")
            images = compensate_flatfield(images, self._ff_calibration)
        elif flatfield:
            logging.warn(
                "No flat-field calibration data available. Use calibrate_flatfield to perform calibration. Skipping flat-field compensation.."
            )

        if self._dist_calibration is not None and distortion:
            logging.info("Processing distortion compensation..")
            images = compensate_distortion(images, *self._dist_calibration)
        elif distortion:
            logging.warn(
                "No distortion calibration data available. Use calibrate_distortion to perform calibration. Skipping distortion compensation.."
            )

        if reference_cell_area is not None and reference_intensity_key is not None:

            def fn(x: Image):
                box = _do_locate_reference_cell(x, reference_cell_area)
                if box is not None:
                    return _do_reference_scaling(
                        x, box, x.get_meta(reference_intensity_key)
                    )
                else:
                    logging.warn(
                        "Reference cell could not be detected for {}. Image is left unscaled!".format(
                            x.path.name
                        )
                    )
                    return x

            logging.info("Processing reference scaling..")
            images = images.apply(fn)

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
