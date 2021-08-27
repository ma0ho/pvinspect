import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch as t
from torch import nn
from torch.nn.modules.loss import L1Loss, MSELoss

from .forward_model import ForwardModel
from .prior import BTV
from .util import *


class RecoParams:
    def __init__(self, lambd: float, n_iter: int, lr: float, min_diff: float):
        self.lambd = lambd
        self.n_iter = n_iter
        self.lr = lr
        self.min_diff = min_diff


def run_reconstruction(
    model: ForwardModel,
    prior: BTV,
    image_loss: Union[MSELoss, L1Loss],
    sr_init: t.Tensor,
    lr_images: t.Tensor,
    motion: t.Tensor,
    params: RecoParams,
    idx: Optional[t.Tensor] = None,
) -> Tuple[t.Tensor, float, pd.DataFrame]:
    # if only using a subset specified by idx, image_loss weights needs to be adapted
    image_loss_weights = image_loss.weights
    if idx is not None:
        image_loss.weights = image_loss_weights[idx]

    # set trainable parameter
    sr = nn.Parameter(sr_init.clone(), requires_grad=True)

    # convert to tensor
    lambd_t = t.tensor(params.lambd).to(sr_init)

    # optimizer
    opt = t.optim.SGD(params=[sr], lr=params.lr)

    # scale loss by this factor
    if isinstance(image_loss, MSELoss):
        loss_factor = 1e3
    else:
        loss_factor = 1e2

    metrics = list()

    for i in range(params.n_iter):
        with t.enable_grad():
            opt.zero_grad()

            y = model(sr, motion)
            if idx is None:
                Li = loss_factor * image_loss(y, lr_images)
            else:
                # only use a subset specified by idx
                Li = loss_factor * image_loss(y[idx], lr_images[idx])

            Lp = loss_factor * prior(sr)
            L = Li + lambd_t * Lp
            L.backward()

            sr_prev = sr.detach().clone()
            opt.step()

        with t.no_grad():
            # logging
            metrics.append(
                dict(
                    image_loss=Li.item(),
                    prior=Lp.item(),
                    reco_loss=L.item(),
                    reco_grad=np.linalg.norm(sr.grad.detach().cpu().numpy().flatten()),
                    max_image_diff=t.max(t.abs(sr - sr_prev)).item(),
                )
            )

        if metrics[-1]["max_image_diff"] < params.min_diff:
            logging.info("Optimization terminated by min diff criterion")
            break

    # reset loss weights
    image_loss.weights = image_loss_weights

    metrics_pd = pd.DataFrame(metrics)
    return sr.data.detach(), metrics_pd.iloc[-1]["reco_loss"], metrics_pd
