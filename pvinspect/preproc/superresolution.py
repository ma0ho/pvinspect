import logging
from typing import List, Literal, Optional, Tuple, overload

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from pvinspect.common.transform import FullTransform, HomographyTransform, warp_image
from pvinspect.data import EagerImage, Image, ImageSequence
from pvinspect.preproc._msr.log import Logger
from skimage import img_as_float32
from skimage.exposure import rescale_intensity

from ._msr.sr import DummyLogger, SRParams
from ._msr.sr import superresolve as _msr_superresolve


def _prepare_motionvecs(
    transforms: List[FullTransform],
    image_height: int,
    image_width: int,
    module_cols: int,
    ref_width: int,
) -> t.Tensor:
    original_grid_y, original_grid_x = np.mgrid[:image_height, :image_width]
    original_grid = np.stack((original_grid_x, original_grid_y), axis=-1).reshape(-1, 2)

    motionvecs = list()

    for ti in transforms:
        transformed_grid = ti.inv()(original_grid)
        transformed_grid *= ref_width / module_cols
        motionvecs.append(
            (transformed_grid - original_grid).reshape(image_height, image_width, 2)
        )

    return t.flip(t.tensor(motionvecs).to(t.float32), [-1])


def _prepare_reference(
    image: Image, transform: FullTransform, module_cols: int, module_rows: int
):
    ref = warp_image(
        img_as_float32(image.data),
        transform,
        0,
        0,
        1 / transform.mean_scale(),
        1 / transform.mean_scale(),
        module_cols,
        module_rows,
    )
    return t.tensor(ref).to(t.float32)


SR_FAST = SRParams(
    max_cv_iter=0,
    initial_magnification=None,
    max_iter=5,
    image_reg_strength=1e-4,
    max_reco_iter=20,
    max_motion_iter=90,
)
SR_BALANCED = SRParams(
    max_cv_iter=10,
    initial_magnification=2,
    max_iter=8,
    max_reco_iter=25,
    max_motion_iter=140,
)
SR_BEST = SRParams()


def multiframe_sr(
    lr_images: ImageSequence,
    magnification: int = 4,
    ref_idx: Optional[int] = None,
    use_cuda: bool = False,
    params: SRParams = SR_BALANCED,
    logger=DummyLogger(),
) -> EagerImage:

    # automatically select reference image, if not set
    if ref_idx is None:
        ref_idx = len(lr_images) // 2

    # prepare transforms
    transforms = [img.get_meta("transform") for img in lr_images]

    # prepare reference image for initialization
    cols = lr_images[0].get_meta("cols")
    rows = lr_images[0].get_meta("rows")
    ref = _prepare_reference(
        lr_images[ref_idx], transforms[ref_idx], cols, rows
    ).unsqueeze(0)

    # prepare motionvecs
    h, w = lr_images[0].shape
    ref_w = ref.shape[2]
    motionvecs = _prepare_motionvecs(transforms, h, w, cols, ref_w)

    # lr images to torch tensor
    lr_images_torch = t.tensor(
        [img_as_float32(img.data) for img in lr_images]
    ).unsqueeze(1)

    # rescale images
    min = lr_images_torch.min()
    max = lr_images_torch.max()
    lr_images_torch = (lr_images_torch - min) / (max - min)
    ref = (ref - min) / (max - min)

    sr = _msr_superresolve(
        lr_images_torch,
        motionvecs,
        ref,
        use_cuda,
        magnification,
        params,
        results_logger=logger,
        log_level=logging.WARNING,
    )

    # undo scaling
    sr = sr * (max - min) + min

    # to numpy
    sr = sr.squeeze().numpy()

    # set up target transform
    ms = transforms[ref_idx].mean_scale() * magnification
    tfm = HomographyTransform(
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [0.0, ms], [ms, 0.0], [ms, ms]]),
        image_width=sr.shape[1],
        image_height=sr.shape[0],
    )

    return EagerImage.from_other(lr_images[ref_idx], data=sr, transform=tfm).as_type(
        lr_images[ref_idx].dtype
    )
