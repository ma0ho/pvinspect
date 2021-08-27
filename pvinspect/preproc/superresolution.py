from typing import List, Optional

import numpy as np
import torch as t
from pvinspect.common.transform import FullTransform, warp_image
from pvinspect.data import EagerImage, Image, ImageSequence
from skimage import img_as_float32

from ._msr.sr import SRParams, superresolve


def _prepare_motionvecs(
    transforms: List[FullTransform],
    image_height: int,
    image_width: int,
    module_cols: int,
) -> t.Tensor:
    original_grid_y, original_grid_x = np.mgrid[:image_height, :image_width]
    original_grid = np.stack((original_grid_x, original_grid_y), axis=-1).reshape(-1, 2)

    motionvecs = list()

    for ti in transforms:
        transformed_grid = ti.inv()(original_grid)
        transformed_grid *= image_width / module_cols
        motionvecs.append(
            (transformed_grid - original_grid).reshape(image_height, image_width, 2)
        )

    return t.tensor(motionvecs).to(t.float32)


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


def multiframe_sr(
    lr_images: ImageSequence,
    magnification: int = 4,
    ref_idx: Optional[int] = None,
    psf_width: float = 0.5,
    use_cuda: bool = False,
) -> Image:

    if ref_idx is None:
        ref_idx = len(lr_images) // 2

    transforms = [img.get_meta("transform") for img in lr_images]
    h, w = lr_images[0].shape
    cols = lr_images[0].get_meta("cols")
    rows = lr_images[0].get_meta("rows")
    motionvecs = _prepare_motionvecs(transforms, h, w, cols)
    ref = _prepare_reference(
        lr_images[ref_idx], transforms[ref_idx], cols, rows
    ).unsqueeze(0)
    lr_images_torch = t.tensor(
        [img_as_float32(img.data) for img in lr_images]
    ).unsqueeze(1)
    params = SRParams()
    params.use_cuda = use_cuda

    sr = superresolve(lr_images_torch, motionvecs, ref, params)

    return EagerImage.from_other(lr_images[ref_idx], data=sr.squeeze().numpy()).as_type(
        lr_images[ref_idx].dtype
    )
