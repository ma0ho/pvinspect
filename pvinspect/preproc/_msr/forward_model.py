from math import pi
from typing import Optional

import numpy as np
import torch as t
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.filters import gaussian_filter
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d

from .gaussian import GaussianLayer


def gauss_kernel(size, dims, std) -> np.ndarray:
    assert dims < 3
    f = np.zeros([size] * dims, dtype=np.float32)
    if dims == 1:
        f[size // 2] = 1.0
    else:
        f[size // 2, size // 2] = 1.0
    f = gaussian_filter(f, std)
    return f / f.sum()


class GaussianBlur(nn.Module):

    blur_h: t.Tensor
    blur_v: t.Tensor

    def __init__(self, blur_std):
        """
        Args:
            blur_std (float): Standard deviation of blur kernel
        """
        super(GaussianBlur, self).__init__()
        # set up blur kernel
        kw = max(
            int(2 * 4 * blur_std) + 1 if int(2 * 4 * blur_std) % 2 == 1 else 0, 3
        )  # cut at 4*mu
        f = gauss_kernel(kw, 1, blur_std)
        self.register_buffer("blur_h", t.tensor(f).reshape([1, 1, kw, 1]))
        self.register_buffer("blur_v", t.tensor(f).reshape([1, 1, 1, kw]))

    def forward(self, x: t.Tensor) -> t.Tensor:
        pad_size = self.blur_v.size(-1) // 2
        C = x.shape[1]
        x = x.view(
            [-1, 1, x.shape[2], x.shape[3]]
        )  # we interpret channels as batch dim
        x = F.pad(x, [pad_size] * 4, mode="replicate")  # avoid darker borders
        x = F.conv2d(x, self.blur_h)  # apply LSI kernel
        x = F.conv2d(x, self.blur_v)
        x = x.view([-1, C, x.shape[2], x.shape[3]])  # undo dimension swap
        return x


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.transpose(self.dim0, self.dim1)


class ForwardModel(t.nn.Module):
    """Forward-model for multi-frame super-resolution

    This implements the forward model according to the following
    observation model

    $$y_i = HDM^i x$$,

    where \(y_i\) is the resulting low resultion frame, \(H\) is the
    system blur, \(D\) the downsampling, \(M^i\) the motion resulting
    in frame \(i\) and \(x\) the high resultion image. Note that this differs
    from common formulations in the blur, which is applied after downsampling.

    Furthermore, we do not require that \(M\) is a quadratic matrix.

    """

    _magn_factor: t.Tensor
    psf: nn.Module

    def init_antialiasing(self):
        std = 2 * self.magn_factor.numpy() / (2 * pi)
        self.antialiasing_kernel = GaussianBlur(std).to(self.grid.device)

    def __init__(
        self,
        magn_factor: int,
        blur_std: Optional[float],
        kernel_width: int,
        H_lr: int,
        W_lr: int,
        N_lr: int,
        separable_blur: bool = False,
    ):
        """
        Args:
            magn_factor (int): The magnification factor, specifying \(D\)
            blur_std (Optional[float]): Standard deviation of the blur in LR pixels
                defaults to a trainable kernel, if not set
            kernel_width (int): Width of the psf kernel. This is ignored, if blur_std is set
            H_lr (int): Height of LR images \(H_{lr}\)
            W_lr (int): Width of LR images \(W_{lr}\)
            N_lr (int): Number of LR images \(N_{lr}\)
            separable_blur (bool): Determines, if PSF is separable into 2x 1D
        """
        super(ForwardModel, self).__init__()

        # check that kernel_width is uneven
        if kernel_width % 2 != 1:
            raise RuntimeError("kernel_width must be uneven")

        self.register_buffer("_magn_factor", t.tensor(magn_factor))

        # construct a 1 x H_inter x W_inter x 2 tensor of pixel indices
        self.register_buffer(
            "grid",
            t.stack(t.meshgrid(t.arange(0, H_lr), t.arange(0, W_lr)), dim=-1)
            .view(1, H_lr, W_lr, 2)
            .to(t.float),
        )

        # initialize antialiasing
        self.antialiasing_kernel = None
        self.init_antialiasing()

        # PSF in LR domain
        if blur_std is not None:
            self.psf = GaussianBlur(blur_std)
        else:
            self.psf = GaussianLayer(1, kernel_width, "replicate", std=0.4)
            # conv = nn.Conv2d(
            #            in_channels=N_lr,
            #            out_channels=N_lr,
            #            kernel_size=kernel_width,
            #            groups=N_lr,
            #            bias=False,
            #            padding_mode="replicate",
            #            padding=kernel_width//2,
            #        )
            ## weight: N_lr, 1, kw, kw
            # conv.weight.data[:] = t.from_numpy(gauss_kernel(kernel_width, 2, 0.5).reshape((1, 1, kernel_width, kernel_width)))
            # self.psf = nn.Sequential(
            #    Transpose(0, 1),
            #    conv,
            #    Transpose(0, 1),
            # )

    @property
    def magn_factor(self) -> t.Tensor:
        return self._magn_factor

    @magn_factor.setter
    def magn_factor(self, value):
        self._magn_factor = t.tensor(value)
        self.init_antialiasing()

    def forward(self, x: t.Tensor, M: t.Tensor) -> t.Tensor:
        """Computes the forward model

        Args:
            x (t.Tensor): The current SR estimate \((C, H_{in}, W_{in})\)
            M (t.Tensor): The motion vector fields \((N, H_{lr}, W_{lr}, 2)\)

        Result:
            The resulting LR images \((N, C, H_{out}, W_{out})\)
        """
        # constants
        C, H_sr, W_sr = x.shape
        N, H_int, W_int, _ = M.shape

        # construct motion vectors
        thisgrid = (self.grid + M) * self.magn_factor

        # add batch dimension
        x = x.unsqueeze(0)

        # apply blur before downsampling to avoid aliasing and sparse gradients
        x = self.antialiasing_kernel(x)

        # grid_sample accepts motion vectors relative to the input image,
        # scaled between (-1,-1) (upper left) and (1,1) (lower right)
        scale = t.tensor([1 / H_sr, 1 / W_sr]).to(thisgrid)
        thisgrid = thisgrid * scale
        thisgrid = thisgrid * 2 - 1.0

        # repeat N times along the batch dimension
        x = x.expand((N, -1, -1, -1))

        # warp according to DM^i
        x = F.grid_sample(x, t.flip(thisgrid, dims=[3]), align_corners=False)

        # apply PSF
        x = self.psf(x)

        # outputs must be non-negative
        return F.relu(x)
