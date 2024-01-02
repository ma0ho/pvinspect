from typing import List

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage._shared.utils import check_nD
from torch import nn
from torch.nn import functional as F

from .thirdpary.resize_right.resize_right import resize
from .util import *


class BTV(nn.Module):
    def __init__(self, win_size: int, channels: int, decay: float = 0.7):
        super().__init__()
        assert win_size % 2 == 1

        self.win_size = win_size

        kernels = list()

        for i in range(win_size):
            for j in range(win_size):
                if i != j:
                    # we implement the derivatives by convolution with win_size^2 - 1 kernels
                    k = np.zeros((win_size, win_size), dtype=np.float32)
                    k[i, j] = 1.0
                    k[win_size // 2, win_size // 2] = -1.0

                    # kernels are weighted by distance to the center (cf. Eq. (7))
                    m = i - (win_size // 2)
                    n = j - (win_size // 2)
                    k *= decay ** (abs(m) + abs(n))

                    kernels.append(k)

        # TODO: In the paper (p. 49) it is mentioned that p=0.5 for the WBTV prior, but I cannot find
        # it here..

        self.register_buffer(
            "kernels", t.tensor(kernels).unsqueeze(1).expand([-1, channels, -1, -1])
        )

    def pixelwise_prior(self, x: t.Tensor) -> t.Tensor:
        assert x.ndim == 3
        x = x.unsqueeze(0)  # insert batch dim

        x = F.conv2d(x, self.kernels, padding=self.win_size // 2)
        x = t.abs(x)
        return x.sum(dim=1)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.pixelwise_prior(x).mean()

    def update_weights(self, sr, sr_previous) -> None:
        pass


class WeightedBTV(BTV):

    weights: t.Tensor

    def __init__(
        self,
        win_size: int,
        channels: int,
        weights_shape: List[int],
        decay: float = 0.7,
        sparsity_param: float = 0.5,
        tuning_constant: float = 2,
    ):
        super(WeightedBTV, self).__init__(win_size, channels, decay)
        self.register_buffer("weights", t.ones(weights_shape, dtype=t.float))
        self.sparsity_param = sparsity_param
        self.tuning_constant = tuning_constant

    def forward(self, x: t.Tensor) -> t.Tensor:
        return (self.pixelwise_prior(x) * self.weights).mean()

    def update_weights(self, sr, sr_previous) -> None:
        super().update_weights(sr, sr_previous)

        # compute BTV
        btv = self.pixelwise_prior(sr)

        # estimate scale parameter
        # scale_param = weighted_median_absolute_deviation(btv, resize(self.weights, out_shape=btv.shape))
        scale_param = median_absolute_deviation(btv)

        # esimate weights
        weights = (
            self.sparsity_param
            * (self.tuning_constant * scale_param) ** (1 - self.sparsity_param)
        ) / t.abs(btv) ** (1 - self.sparsity_param)
        weights[
            t.abs(btv) <= t.tensor([self.tuning_constant * scale_param]).to(weights)
        ] = 1.0

        self.weights = weights

    # @staticmethod
    # def _adaptive_scale_parameter(z: t.Tensor, weights: t.Tensor) -> t.Tensor:
    #    idx = weights > 0.0
    #    z = z[idx]
    #    weights = weights[idx]

    #    weighted_median_residual = weighted_median(z, weights)
    #    absolute_shifted_residual = t.abs(z - weighted_median_residual)
    #    return weighted_median(
    #        absolute_shifted_residual, weights
    #    )  # TODO: Maybe the 1.4... is missing here?

    # def update_weights(
    #    self,
    #    sr: t.Tensor,
    #    sr_previous: t.Tensor,
    #    sparsity_param: float = 0.5,
    #    tuning_constant: float = 2.0,
    # ) -> None:
    #    """Update the weight map

    #    Args:
    #        sr (t.Tensor): The new weight map will be computed according to this
    #        sr_previous (t.Tensor): This is used to compute the scale factor. Hence, this
    #            needs to match the size of the previous weight map
    #        sparsity_param (float): Sparsity parameter [0..1], which is set to 0.5 according to the paper p. 46
    #        tuning_constant (float): Tuning constant, defaults to 2 according to the paper
    #    """
    #    # compute prior level according to Eq. 24
    #    btv = self.pixelwise_prior(sr_previous)
    #    scale_param = self._adaptive_scale_parameter(
    #        btv.flatten(), self.weights.flatten()
    #    )
    #    scale_param = t.max(t.tensor([scale_param, 1e-8]))

    #    btv = self.pixelwise_prior(sr)
    #    show_pytorch_image(btv)
    #    weights = (
    #        sparsity_param * (tuning_constant * scale_param) ** (1 - sparsity_param)
    #    ) / (t.abs(btv) ** (1 - sparsity_param))
    #    weights[t.abs(btv) <= t.tensor([tuning_constant * scale_param]).to(weights)] = 1.0
    #    self.weights = weights
