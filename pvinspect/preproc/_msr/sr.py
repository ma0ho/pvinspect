import logging
from argparse import ArgumentParser
from datetime import datetime
from math import log
from os import rmdir
from pathlib import Path
from shutil import rmtree
from typing import Union

import pandas as pd
import torch as t
from torch.nn.modules.loss import L1Loss, MSELoss

from .cv import run_cv
from .data import prepare_data
from .forward_model import ForwardModel
from .log import Logger
from .loss import WeightedL1Loss, WeightedMSELoss
from .modelparams import *
from .prior import WeightedBTV
from .reconstruction import RecoParams, run_reconstruction
from .thirdpary.resize_right.resize_right import resize
from .util import *


# a logger that does nothing
class DummyLogger:
    def dummymethod(self, *args, **kwargs):
        return

    def __init__(self):
        super(DummyLogger).__init__()

    def __getattr__(self, name):
        return self.dummymethod


class SRParams:
    @initializer
    def __init__(
        self,
        max_cv_iter: int = 20,
        cv_val_image_p: float = 0.3,
        initial_motion_scale: int = 32,
        final_motion_scale: int = 1,
        blur_std: float = 0.5,
        image_loss_type: str = "WeightedL1",
        motion_loss_type: str = "L1",
        use_cuda: bool = True,
        image_reg_strength: float = None,
        max_reco_iter: int = 30,
        lr_image: float = 50.0,
        min_diff: float = 5e-5,
        max_motion_iter: int = 180,
        lr_motion: float = 1e-1,
        motion_min_diff: float = 0.0005,
        magnification: int = 4,
        initial_magnification: int = 1,
        enable_bias_compensation: bool = True,
        max_iter: int = 10,
    ):
        pass


def superresolve(
    lr_images: t.Tensor,
    motion: t.Tensor,
    ref: t.Tensor,
    params: SRParams = SRParams(),
    log_level=logging.INFO,
    results_logger=DummyLogger(),
):

    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    # disable gradient computation to save GPU memory
    with t.no_grad():

        # init params
        current_magnification: int = 1
        cv_iter: int = params.max_cv_iter
        initial_motion_scale: int = params.initial_motion_scale

        # prepare initialization
        sr = ref

        # set up
        logger.info("Set up model..")
        model = ForwardModel(
            magn_factor=1,
            blur_std=params.blur_std,
            kernel_width=3,  # ignored later on
            H_lr=lr_images.size(2),
            W_lr=lr_images.size(3),
            N_lr=lr_images.size(0),
        )

        # init losses
        image_loss: Union[MSELoss, L1Loss]
        if params.image_loss_type == "WeightedMSE":
            image_loss = WeightedMSELoss(weights_shape=list(lr_images.shape))
        else:
            image_loss = WeightedL1Loss(weights_shape=list(lr_images.shape))

        motion_loss: Union[MSELoss, L1Loss]
        if params.motion_loss_type == "MSE":
            motion_loss = MSELoss()
        else:
            motion_loss = L1Loss()

        prior = WeightedBTV(
            win_size=5, channels=lr_images.size(1), weights_shape=list(sr.shape)
        )

        # use cuda?
        if params.use_cuda:
            logger.info("Using CUDA")
            model.cuda()
            prior.cuda()
            image_loss.cuda()

            sr = sr.cuda()
            lr_images = lr_images.cuda()
            motion = motion.cuda()

        # init reco parameters
        reco_params = RecoParams(
            lambd=0.001
            if params.image_reg_strength is None
            else params.image_reg_strength,
            n_iter=params.max_reco_iter,
            lr=params.lr_image,
            min_diff=params.min_diff,
        )

        # init motion params
        motion_params = MotionParams(
            n_iter=params.max_motion_iter,
            lr=params.lr_motion,
            min_diff=params.motion_min_diff,
        )

        # init
        sr_previous = sr

        # log initial image
        results_logger.log_image(sr, "init", 0)

        # motion is estimated as a residual to the initial motion
        motion_diff = t.zeros_like(motion)

        # timing
        start_t = datetime.now()
        reco_params = run_cv(
            model,
            prior,
            image_loss,
            sr,
            lr_images,
            motion + motion_diff,
            reco_params,
            -12,
            0,
            cv_iter,
            params.cv_val_image_p,
            search_lambda=False,
            search_lr=True,
        )
        reco_params = run_cv(
            model,
            prior,
            image_loss,
            sr,
            lr_images,
            motion + motion_diff,
            reco_params,
            -12,
            0,
            cv_iter,
            params.cv_val_image_p,
            search_lambda=True,
            search_lr=False,
        )

        if params.enable_bias_compensation:
            logger.info("Estimate and compensate sensor bias..")

            # run an initial reconstruction
            tmp, _, _ = run_reconstruction(
                model,
                prior,
                image_loss,
                sr,
                lr_images,
                motion + motion_diff,
                reco_params,
            )

            # estimate and compensate sensor bias
            lr_images -= (lr_images - model.forward(tmp, motion)).median(
                0, keepdim=True
            )[0]

        # log initial images
        lr_est = model.forward(sr, motion)
        results_logger.log_image(lr_est[0], "lr_estim_initial_first", 0)
        results_logger.log_image(lr_est[-1], "lr_estim_initial_last", 0)

        # BEGIN OUTER LOOP ==================================================================================
        for i in range(params.max_iter):

            # set magnification and rescale sr
            if (
                current_magnification < params.magnification
                and i > 0
                or current_magnification < params.magnification
                and params.initial_magnification > 1
            ):
                increment = max(params.initial_magnification - current_magnification, 1)
                sr_previous = sr
                factor = (current_magnification + increment) / current_magnification
                sr = resize(sr, scale_factors=t.tensor(factor).to(sr))
                model.magn_factor = t.tensor(
                    current_magnification + increment
                )  # model needs to know this as well
                current_magnification = current_magnification + increment

            logger.info(
                "Running at {:d}x magnification [{:02d}/{:02d}]".format(
                    current_magnification, i + 1, params.max_iter
                )
            )

            # motion estimation
            mscale = params.initial_motion_scale
            mi = 0
            all_failed = True

            while mscale >= params.final_motion_scale and mi <= i:
                logger.info("Running motion estimation at scale {:d}".format(mscale))

                motion_diff_i, loss, metrics = run_motion_estimation(
                    model,
                    motion_loss,
                    sr,
                    lr_images,
                    motion + motion_diff,
                    mscale,
                    motion_params,
                )
                # log metrics with every step
                results_logger.log_metrics(metrics, i)

                # restore initial psf and motion diff, if loss increased
                if metrics.iloc[0]["motion_loss"] < metrics.iloc[-1]["motion_loss"]:
                    motion_diff_i[:] = 0.0
                else:
                    all_failed = False

                mi += 1
                motion_diff += motion_diff_i
                mscale = mscale // 2

            if all_failed:
                logger.info("Reducing motion and PSF learning rate..")
                # lr too large
                motion_params.lr /= 2

            # logging
            results_logger.log_plot(
                motion_diff[0, :, :, 0].unsqueeze(0), "motion_diff_y", i
            )
            results_logger.log_plot(
                motion_diff[0, :, :, 1].unsqueeze(0), "motion_diff_x", i
            )

            # update weight maps
            image_loss.update_weights(
                lr_images,
                model.forward(sr, motion + motion_diff),
                model.forward(t.ones_like(sr), motion + motion_diff),
            )
            prior.update_weights(sr, sr_previous)

            # prepare CV
            if cv_iter > 1 and params.image_reg_strength is None:
                # determine CV parameters
                if i == 0:
                    log_lambd_l = -12
                    log_lambd_u = 0
                else:
                    # search within local neighborhood of previous lambda
                    log_lambd_l = log(reco_params.lambd) - 0.5 / (i + 1)
                    log_lambd_u = log(reco_params.lambd) + 0.5 / (i + 1)

                # run CV
                reco_params = run_cv(
                    model,
                    prior,
                    image_loss,
                    sr,
                    lr_images,
                    motion + motion_diff,
                    reco_params,
                    log_lambd_l,
                    log_lambd_u,
                    cv_iter,
                    params.cv_val_image_p,
                    search_lambda=True,
                    search_lr=False,
                )

                # reduce CV iterations
                cv_iter = cv_iter // 2

            # log CV result
            results_logger.log_metric("lambda", reco_params.lambd, i)
            results_logger.log_metric("lr", reco_params.lr, i)

            # run reconstruction using new parameters
            sr_previous = sr
            sr, _, metrics = run_reconstruction(
                model,
                prior,
                image_loss,
                sr,
                lr_images,
                motion + motion_diff,
                reco_params,
            )

            # decay lr
            reco_params.lr *= 0.75

            # save and log
            results_logger.log_metrics(metrics, i)
            results_logger.log_image(sr, "sr", i)
            results_logger.log_plot(image_loss.weights[0], "loss_weights_first", i)
            results_logger.log_plot(image_loss.weights[-1], "loss_weights_last", i)
            results_logger.log_plot(prior.weights, "prior_weights", i)
            results_logger.log_metric(
                "loss_weights_magnitude", image_loss.weights.sum().item(), i
            )
            results_logger.log_metric(
                "prior_weights_magnitude", prior.weights.sum().item(), i
            )

            lr_est = model.forward(sr, motion + motion_diff)
            results_logger.log_image(lr_est[0], "lr_estim_first", i)
            results_logger.log_image(lr_est[-1], "lr_estim_last", i)

            # finally, check termination
            diff = t.max(t.abs(sr - sr_previous)).item()
            if (
                diff < params.min_diff
                and current_magnification == magnification
                and params.initial_motion_scale / 2 ** i <= params.final_motion_scale
            ):
                logger.info("Outer loop terminated by min diff criterion")
                break

    # END OUTER LOOP =======================================================================

    diff_t = datetime.now() - start_t
    logger.info("Computation took {:f} seconds".format(diff_t.total_seconds()))

    return sr.detach().cpu()
