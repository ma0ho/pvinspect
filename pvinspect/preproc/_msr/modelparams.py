import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch as t
from torch import nn
from torch.nn.modules.loss import L1Loss, MSELoss

from .data import prepare_data
from .forward_model import ForwardModel
from .thirdpary.resize_right.resize_right import resize
from .util import *


class PsfParams:
    def __init__(self, n_iter: int, lr: float):
        self.n_iter = n_iter
        self.lr = lr


def run_psf_estimation(
    model: ForwardModel,
    loss: Union[MSELoss, L1Loss],
    sr: t.Tensor,
    lr_images: t.Tensor,
    motion: t.Tensor,
    params: PsfParams,
) -> Tuple[float, pd.DataFrame]:
    # optimizer
    opt = t.optim.SGD(params=model.psf.parameters(), lr=params.lr)

    # scale loss by this factor
    if isinstance(loss, MSELoss):
        loss_factor = 1e3
    else:
        loss_factor = 1e2

    metrics = list()

    for i in range(params.n_iter):
        with t.enable_grad():
            opt.zero_grad()
            y = model(sr, motion)
            L = loss_factor * loss(y, lr_images)
            L.backward()
            opt.step()

        # project onto valid solution
        # weight = model.psf[1].weight.data
        # with t.no_grad():
        #    weight[weight < 0.0] = 0.0
        #
        #    for j in range(weight.shape[0]):
        #        weight[j] /= weight[j].sum()
        # weight = model.psf[2].weight.data
        # with t.no_grad():
        #    weight[weight < 0.0] = 0.0
        #
        #    for j in range(weight.shape[0]):
        #        weight[j] /= weight[j].sum()

        with t.no_grad():
            # logging
            metrics.append(dict(psf_loss=L.item(),))

        if i > 0 and metrics[-1]["psf_loss"] > metrics[-2]["psf_loss"]:
            logging.info("Optimization terminated by loss criterion")
            break

    metrics_pd = pd.DataFrame(metrics)

    return metrics_pd.iloc[-1]["psf_loss"], metrics_pd


class MotionParams:
    def __init__(self, n_iter: int, lr: float, min_diff: float):
        self.n_iter = n_iter
        self.lr = lr
        self.min_diff = min_diff


def run_motion_estimation(
    model: ForwardModel,
    motion_loss: Union[MSELoss, L1Loss],
    sr: t.Tensor,
    lr_images: t.Tensor,
    motion: t.Tensor,
    scale: int,
    params: MotionParams,
) -> Tuple[t.Tensor, float, pd.DataFrame]:
    # set trainable parameter
    motion_diff = nn.Parameter(
        resize(t.zeros_like(motion), scale_factors=[1, 1 / scale, 1 / scale, 1])
    )

    # optimizer
    opt = t.optim.SGD(params=[motion_diff], lr=params.lr)

    # scale loss by this factor
    if isinstance(motion_loss, MSELoss):
        loss_factor = 1e5
    else:
        loss_factor = 1e4

    metrics = list()

    for i in range(params.n_iter):
        with t.enable_grad():
            opt.zero_grad()
            y = model(sr, motion + resize(motion_diff, out_shape=motion.shape))
            L = loss_factor * motion_loss(y, lr_images)
            L.backward()

            motion_diff_prev = motion_diff.detach().clone()
            opt.step()

        with t.no_grad():
            # logging
            metrics.append(
                dict(
                    motion_loss=L.item(),
                    motion_grad=np.linalg.norm(
                        motion_diff.grad.detach().cpu().numpy().flatten()
                    ),
                    mean_motion_diff=t.mean(
                        t.abs(motion_diff - motion_diff_prev)
                    ).item(),
                )
            )

        m = pd.DataFrame(metrics)
        if i > 0 and metrics[-1]["motion_loss"] > metrics[-2]["motion_loss"]:
            logging.info("Optimization terminated by loss criterion")
            break

    metrics_pd = pd.DataFrame(metrics)

    return (
        resize(motion_diff.data, out_shape=motion.shape).detach(),
        metrics_pd.iloc[-1]["motion_loss"],
        metrics_pd,
    )


def run_motion_and_psf_estimation(
    model: ForwardModel,
    loss: Union[MSELoss, L1Loss],
    sr: t.Tensor,
    lr_images: t.Tensor,
    motion: t.Tensor,
    scale: int,
    motion_params: MotionParams,
    psf_params: PsfParams,
) -> Tuple[t.Tensor, float, pd.DataFrame]:
    # set trainable parameter
    motion_diff = nn.Parameter(
        resize(t.zeros_like(motion), scale_factors=[1, 1 / scale, 1 / scale, 1])
    )

    # optimizer
    opt = t.optim.SGD(
        params=[
            {"params": motion_diff, "lr": motion_params.lr},
            {"params": model.psf.parameters(), "lr": psf_params.lr},
        ],
        momentum=0.9,
    )

    # scale loss by this factor
    if isinstance(loss, MSELoss):
        loss_factor = 1e5
    else:
        loss_factor = 1e4

    # collect metrics during estimation
    metrics = list()

    for i in range(motion_params.n_iter):
        with t.enable_grad():
            opt.zero_grad()
            y = model(sr, motion + resize(motion_diff, out_shape=motion.shape))
            L = loss_factor * loss(y, lr_images)

            with t.no_grad():
                # logging
                metrics.append(dict(motion_and_psf_loss=L.item(),))

            if (
                i > 0
                and metrics[-1]["motion_and_psf_loss"]
                > metrics[-2]["motion_and_psf_loss"]
            ):
                logging.info("Optimization terminated by loss criterion")
                break

            # only step, if loss is smaller than previous
            L.backward()
            opt.step()

    metrics_pd = pd.DataFrame(metrics)

    return (
        resize(motion_diff.data, out_shape=motion.shape).detach(),
        metrics_pd.iloc[-1]["motion_and_psf_loss"],
        metrics_pd,
    )
