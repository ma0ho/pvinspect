from math import exp
from typing import Union

import optuna
import torch as t
from torch.nn.modules.loss import L1Loss, MSELoss

from .forward_model import ForwardModel
from .loss import WeightedMSELoss
from .prior import WeightedBTV
from .reconstruction import RecoParams, run_reconstruction
from .util import *

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_cv(
    model: ForwardModel,
    prior: WeightedBTV,
    image_loss: Union[MSELoss, L1Loss],
    sr_init: t.Tensor,
    lr_images: t.Tensor,
    motion: t.Tensor,
    params: RecoParams,
    log_lambd_l: float,
    log_lambd_u: float,
    cv_iter: int,
    val_image_p: float,
    search_lr: bool,
    search_lambda: bool,
) -> RecoParams:
    idx_val = t.rand(lr_images.size(0)) < val_image_p
    idx_train = t.bitwise_not(idx_val)

    def objective(trial: optuna.Trial) -> float:
        if search_lambda:
            params.lambd = trial.suggest_loguniform(
                "lambd", exp(log_lambd_l), exp(log_lambd_u)
            )
        if search_lr:
            params.lr = trial.suggest_loguniform("lr", 0.1, 200)
            # print(params.lr)
            # params.lr = trial.suggest_loguniform("lr", 0.1, 200)

        # run reco
        train_sr, _, _ = run_reconstruction(
            model, prior, image_loss, sr_init, lr_images, motion, params, idx_train
        )

        # set val weights and compute validation error
        tmp = image_loss.weights
        image_loss.weights = tmp[idx_val]
        lr_images_val = model.forward(train_sr, motion)[idx_val]
        err = image_loss.forward(lr_images_val, lr_images[idx_val]).item()
        image_loss.weights = tmp

        return err

    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"lambd": params.lambd, "lr": params.lr})  # initial guess
    study.optimize(objective, n_trials=cv_iter)

    if search_lambda:
        params.lambd = study.best_params.get("lambd")
    if search_lr:
        params.lr = study.best_params.get("lr")

    return params
