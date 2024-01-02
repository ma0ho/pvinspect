from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch as t
from scipy.io import loadmat
from skimage import img_as_float32
from skimage.io import imread


def prepare_data(
    base: Path,
    clip_reference_x: Optional[int] = None,
    clip_reference_y: Optional[int] = None,
    N_lr: Optional[int] = None,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Load data

    Args:
        base (Path): Path to dataset

    Returns:
        A tuple consisting of the LR images, the motion vector fields and the refenrence
        image at magnification x1
    """
    image_paths = list((base / "images").glob("*.tif"))

    images = list()
    motionvecs = list()
    for p in image_paths:
        images.append(img_as_float32(imread(p, as_gray=True)))
        mat = loadmat(base / "{}_modelgrid.mat".format(p.stem))["v"]
        mat = mat[:, :, ::-1]  # matlab is row major
        motionvecs.append(mat.astype(np.float32))

    images = t.tensor(images).unsqueeze(1)  # 1 channel
    motionvecs = t.tensor(motionvecs)

    reference = img_as_float32(imread(base / "modelgrid_reference.tif", as_gray=True))
    reference = t.tensor(reference).unsqueeze(0)  # 1 channel

    if clip_reference_x is not None:
        reference = reference[:, :, :clip_reference_x]
    if clip_reference_y is not None:
        reference = reference[:, :clip_reference_y, :]

    if N_lr is not None:
        motionvecs = motionvecs[:N_lr]
        images = images[:N_lr]

    return images, motionvecs, reference


def prepare_data_homo(base: Path) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Load data

    Args:
        base (Path): Path to dataset

    Returns:
        A tuple consisting of the LR images, the motion vector fields and the refenrence
        image at magnification x1
    """
    image_paths = list((base / "images").glob("*.tif"))

    images = list()
    motionvecs = list()
    refimg = None
    for p in image_paths:
        images.append(img_as_float32(imread(p, as_gray=True)))
        mat = loadmat(base / "{}_refgrid.mat".format(p.stem))["v"]
        if mat.sum() < 1e-4:
            refimg = p
        mat = mat[:, :, ::-1]  # matlab is row major
        motionvecs.append(mat.astype(np.float32))

    images = t.tensor(images).unsqueeze(1)  # 1 channel
    motionvecs = t.tensor(motionvecs)

    reference = img_as_float32(imread(refimg, as_gray=True))
    reference = t.tensor(reference).unsqueeze(0)  # 1 channel

    return images, motionvecs, reference
