from copy import deepcopy
from typing import Optional, Tuple

import cv2
import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from pvinspect.common import transform
from pvinspect.data.image.sequence import ImageSequence
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def _get_smoothed_image(img, transform, step):
    img_cell_scale = transform.mean_scale()
    new_scale = 1 / step
    subsampling = img_cell_scale / new_scale
    sigma = 3 * subsampling / (2 * np.pi)
    return gaussian_filter(img, sigma)


def _multiscale_registration(
    imgs,
    transforms,
    src_idxs,
    tgt_idx,
    extrinsic,
    intrinsic,
    distortion,
    similarity,
    region,
    initial_step,
):

    x0, y0, w, h = region

    def obj(x):
        transforms.update_parameters_by_handles(
            x, src_idxs, extrinsic, intrinsic, distortion
        )
        src_imgs_t = [
            transform.warp_image(img, transforms[i], x0, y0, step, step, w, h)
            for img, i in zip(src_imgs, src_idxs)
        ]

        if intrinsic or distortion:
            # target image is not fixed
            tgt_img_t2 = transform.warp_image(
                tgt_img, transforms[tgt_idx], x0, y0, step, step, w, h
            )
        else:
            tgt_img_t2 = tgt_img_t

        if similarity == "L2":
            return np.concatenate(
                [(src_img_t - tgt_img_t2).flatten() for src_img_t in src_imgs_t], axis=0
            )
        elif similarity == "CC":
            tgt_img_mean = np.mean(tgt_img_t2)
            tgt_img_std = np.std(tgt_img_t2)
            tgt_img_t_norm = tgt_img_t2 - tgt_img_mean
            nccs = []
            for src_img_t in src_imgs_t:
                nccs.append(
                    np.mean(
                        ((src_img_t - np.mean(src_img_t)) * tgt_img_t_norm)
                        / (np.std(src_img_t) * tgt_img_std)
                    )
                )
            return -np.mean(nccs)

    # maximum step size according to cell size in source image
    max_step = np.mean([1 / transforms[i].mean_scale() for i in src_idxs])
    step = initial_step

    while step > max_step:

        # low pass filter source and target images to avoid aliasing
        src_imgs = [_get_smoothed_image(imgs[i], transforms[i], step) for i in src_idxs]
        tgt_img = _get_smoothed_image(imgs[tgt_idx], transforms[tgt_idx], step)

        # initial parameters
        handles = transforms.get_parameter_handles(
            src_idxs, extrinsic, intrinsic, distortion
        )

        tgt_img_t = None
        if not intrinsic and not distortion:
            # target image is fixed
            tgt_img_t = transform.warp_image(
                tgt_img, transforms[tgt_idx], x0, y0, step, step, w, h
            )

        if similarity == "L2":
            res = scipy.optimize.least_squares(obj, handles, method="lm", x_scale="jac")
        elif similarity == "CC":
            res = scipy.optimize.minimize(obj, handles)

        # apply result
        transforms.update_parameters_by_handles(
            res.x, src_idxs, extrinsic, intrinsic, distortion
        )

        # decrease step size
        step = step / 2

    return transforms


def register_sequence(
    sequence: ImageSequence,
    ref_idx: int,
    register_extrinsic: bool = True,
    register_intrinsic: bool = False,
    register_distortion: bool = False,
    similarity: str = "L2",
    region: Optional[Tuple[int, int, int, int]] = None,
    initial_step: float = 0.1,
):
    """Register a sequence of images to a target image

    Args:
        imgs (List[np.array]): The input images
        transforms (List[transform.Transform]): The corresponding initial transforms
        ref_idx (int): Index of the reference images
        register_extrinsic (bool): If extrinsic parameters should be refined
        register_intrinsic (bool): If intrinsic parameters should be refined
        register_distortion (bool): If distortion parameters should be refined
        similarity (str): Cost function for registration (L2 for L2 norm or CC for cross correlation)
        region (Optional[Tuple[int, int, int, int]]): Region used for compution registration cost:
            (first_col, first_row, num_cols, num_rows). Defaults to the entire module
        initial_step (float): (1/initial_step)**2 equals to the initial number of pixels per cell used
            for the multi-scale registration

    Returns:
        sequence: The resulting sequence with transforms determined by registration
    """
    transforms = [img.get_meta("transform") for img in sequence]
    tmulti = transform.FullMultiTransform.from_parameters(
        [t._Rt for t in transforms], transforms[0]._A, transforms[0]._dist
    )
    region = (
        [0, 0, int(sequence[0].get_meta("cols")), int(sequence[0].get_meta("rows"))]
        if region is None
        else region
    )
    for i, (img, t) in enumerate(zip(tqdm(sequence), transforms)):
        if i != ref_idx:
            tmulti = _multiscale_registration(
                [x.data for x in sequence],
                tmulti,
                [i],
                ref_idx,
                register_extrinsic,
                register_intrinsic,
                register_distortion,
                similarity,
                region,
                initial_step,
            )

    return sequence.apply_meta_list("transform", [x for x in tmulti])
