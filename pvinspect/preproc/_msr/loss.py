from typing import List, Optional

import torch as t
from torch.nn.modules.loss import L1Loss, MSELoss

from .thirdpary.resize_right.resize_right import resize
from .util import *

# class WeightedMSELoss(MSELoss):
#    def __init__(self, weights_shape: List[int]):
#        super(WeightedMSELoss, self).__init__(reduction="none")
#        self.register_buffer("weights", t.ones(weights_shape))
#
#    def forward(self, input: t.Tensor, target: t.Tensor) -> t.Tensor:
#        return (self.weights * super().forward(input, target)).mean()
#
#    def update_weights(self, lr_images: t.Tensor, lr_images_est: t.Tensor) -> None:
#        # compute weighted residual error
#        residual = lr_images - lr_images_est
#
#        # compute noise level according to Eq. 23
#        sigma_noise = 1.4826 * median_absolute_deviation(residual)
#
#        # update pixel confidence according to Eq. 17-19
#        bias = (
#            (
#                t.abs(residual.view(residual.size(0), -1).median(dim=1)[0])
#                <= residual.max()
#            )
#            .to(t.float)
#            .unsqueeze(1)
#            .unsqueeze(2)
#            .unsqueeze(3)
#        )
#        local = t.ones_like(lr_images)
#        idx = t.abs(residual) > 2 * sigma_noise
#        local[idx] = (2 * sigma_noise / t.abs(residual))[idx]
#        self.weights = bias * local


# This is the one implemented in Thomas' Toolbox. However, that performs a little worse..
# Main difference is the use of weighted median instead of median for MAD
class WeightedLoss:
    def __init__():
        super().__init__()

    @staticmethod
    def _adaptive_scale_parameter(residual: t.Tensor, weights: t.Tensor) -> t.Tensor:
        idx = weights > 0.0
        residual = residual[idx]
        weights = weights[idx]

        return 1.4826 * weighted_median_absolute_deviation(residual, weights)

    def update_weights(
        self,
        lr_images: t.Tensor,
        lr_images_est: t.Tensor,
        masks: Optional[t.Tensor] = None,
        rmax: float = 0.02,
        tuning_const: float = 1.0,
    ) -> None:
        # compute weighted residual error
        residual = self.weights * (lr_images - lr_images_est)

        # if set, apply masks to residual
        if masks is not None:
            residual = residual * masks
        else:
            masks = t.ones_like(residual)

        # frame-wise confidence (bias)
        bias = t.abs(residual.view(residual.size(0), -1).median(dim=1)[0]) < rmax
        bias = bias.to(t.float).unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # update weights
        # self.weights = self.weights*bias

        # determine scale parameter: here, we need to mask the weights, if mask
        # is given, to make sure that weighted median is not 0
        scale = self._adaptive_scale_parameter(residual, self.weights * masks)

        # compute pixel-wise weights
        weights_local = t.ones_like(self.weights) / (t.abs(residual) + 1e-8)
        idx = t.abs(residual) < tuning_const * scale
        weights_local[idx] = t.tensor(1 / (tuning_const * scale)).to(self.weights)
        weights_local = tuning_const * scale * weights_local

        # combine
        self.weights = weights_local * bias


class WeightedMSELoss(MSELoss, WeightedLoss):

    weights: t.Tensor

    def __init__(self, weights_shape: List[int]):
        super(WeightedMSELoss, self).__init__(reduction="none")
        self.register_buffer("weights", t.ones(weights_shape))

    def forward(self, input: t.Tensor, target: t.Tensor) -> t.Tensor:
        return (self.weights * super().forward(input, target)).mean()


class WeightedL1Loss(L1Loss, WeightedLoss):

    weights: t.Tensor

    def __init__(self, weights_shape: List[int]):
        super(WeightedL1Loss, self).__init__(reduction="none")
        self.register_buffer("weights", t.ones(weights_shape))

    def forward(self, input: t.Tensor, target: t.Tensor) -> t.Tensor:
        return (self.weights * super().forward(input, target)).mean()
