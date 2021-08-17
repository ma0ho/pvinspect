import logging
from argparse import Namespace
from pathlib import Path
from random import randrange

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import skimage
import torch as t
import torchvision as tv
from albumentations.pytorch.transforms import ToTensorV2
from pvinspect.analysis.model import InspectModel
from pvinspect.data.image.sequence import ImageSequence, sequence
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from scipy.special import expit
from skimage import img_as_ubyte
from skimage.color import gray2rgb
from sklearn.metrics import f1_score


class DefectModule(pl.LightningModule):

    hparams = dict()

    def __init__(self, hparams=None):
        super().__init__()

        if isinstance(hparams, dict):
            self.hparams.update(hparams)

        # set up model
        if self.hparams["model"] == "resnet50_seg":

            if "pool" in self.hparams.keys():
                # choice of pooling depending on hparams.pool
                if self.hparams["pool"] == "max":
                    pool = t.nn.AdaptiveMaxPool2d((1, 1))
                elif self.hparams["pool"] == "avg":
                    pool = t.nn.AdaptiveAvgPool2d((1, 1))
            else:
                # fallback for old checkpoints
                logging.warning(
                    "Pooling not specified. Using AdaptiveMaxPool2d as fallback."
                )
                pool = t.nn.AdaptiveMaxPool2d((1, 1))

            from_scratch = True
            if "from_scratch" in self.hparams.keys():
                from_scratch = self.hparams["from_scratch"]
            self._net = tv.models.resnet50(pretrained=not from_scratch)
            self._net.layer4 = t.nn.Conv2d(
                1024, len(self.hparams["classes"]), 1
            )  # squeeze into num-classes channels
            self._fc = t.nn.Sequential(pool, t.nn.Flatten())

            def _forward_impl(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                return x

            # override forward
            setattr(
                self._net,
                "_forward_impl",
                _forward_impl.__get__(self._net, self._net.__class__),
            )

        if self.hparams["model"] == "resnet50":

            if "pool" in self.hparams.keys():
                # choice of pooling depending on hparams.pool
                if self.hparams["pool"] == "max":
                    pool = t.nn.AdaptiveMaxPool2d((1, 1))
                elif self.hparams["pool"] == "avg":
                    pool = t.nn.AdaptiveAvgPool2d((1, 1))
            else:
                # fallback for old checkpoints
                pool = t.nn.AdaptiveAvgPool2d((1, 1))

            self._net = tv.models.resnet50(pretrained=not self.hparams["from_scratch"])
            self._net.avgpool = pool
            self._net.fc = t.nn.Identity()
            self._fc = t.nn.Linear(2048, len(self.hparams["classes"]))

        # initialized in prepare_data()
        self._sample_weights = None

        # log training sample results
        self._val_sample_paths = list()
        self._val_sample_cams = None

        # log val metrics
        self._val_loss_and_metrics = list()

    def forward(self, x):
        cams = self._net(x)
        pred = self._fc(cams)
        return pred, t.sigmoid(cams)

    def configure_optimizers(self):
        opt = t.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        sched = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt, patience=self.hparams["scheduler_patience"], verbose=True,
        )
        return dict(optimizer=opt, lr_scheduler=sched, monitor="val_loss",)

    def training_step(self, batch, batch_idx):
        logits, _ = self.forward(batch[0])
        sample_losses = t.nn.functional.binary_cross_entropy_with_logits(
            logits, batch[1], reduction="none"
        )
        loss = sample_losses.mean()
        self.log("loss", loss)

        return {
            "loss": loss,
            "logits": logits,
            "targets": batch[1],
        }

    def training_epoch_end(self, outputs):
        avg_loss = np.mean([x["loss"].item() for x in outputs])
        self.log("avg_train_loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        logits, cams = self.forward(batch[0])
        loss = t.nn.functional.binary_cross_entropy_with_logits(logits, batch[1])

        # log first step of validation epoch
        if self._val_sample_cams is None:
            self._val_sample_paths = batch[2]
            self._val_sample_cams = cams.detach().cpu()

        return {
            "loss": loss,
            "logits": logits,
            "targets": batch[1],
            "dataset": batch[3],
        }

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([x["loss"].item() for x in outputs])
        predictions = (
            np.concatenate([x["logits"].cpu().numpy() for x in outputs], axis=0) > 0.0
        )
        targets = (
            np.concatenate([x["targets"].cpu().numpy() for x in outputs], axis=0) > 0.5
        )
        dataset = []
        for x in outputs:
            dataset += x["dataset"]

        # compute evaluation metrics
        metrics = self.compute_metrics(predictions, targets, dataset)

        # keep logging here..
        self._val_loss_and_metrics.append((avg_loss, metrics))

        # log cams
        if self.hparams["model"].endswith("_seg"):
            imgs = np.array(
                [
                    skimage.transform.resize(
                        skimage.img_as_float(
                            skimage.exposure.rescale_intensity(
                                skimage.io.imread(p, as_gray=False)
                            )
                        ),
                        (170, 170),
                    )
                    for p in self._val_sample_paths
                ]
            )
            imgs = t.tensor(imgs).reshape(-1, 1, 170, 170).repeat(1, 3, 1, 1)
            self.logger.experiment.add_images(
                "validation_images", imgs, global_step=self.global_step
            )
            for i, c in enumerate(self.hparams["classes"]):
                self.logger.experiment.add_images(
                    "validation_cams_{}".format(c),
                    self._val_sample_cams[:, i].unsqueeze(1),
                    global_step=self.global_step,
                )

        # reset
        self._val_sample_paths = list()
        self._val_sample_cams = None

        # logging
        self.log("val_loss", avg_loss)

        for k, v in metrics.items():
            self.log("val_{}".format(k), v)

    def compute_metrics(self, predictions, targets, datasets):
        unique_datasets = tuple(datasets)
        scores = dict()

        # go over datasets
        for ds in unique_datasets:
            ind = np.array([x == ds for x in datasets])

            # compute f1 per class
            values = list()
            for i, c in enumerate(self.hparams["classes"]):
                v = f1_score(targets[ind, i], predictions[ind, i])
                scores["{}_f1_{}".format(ds, c)] = v
                values.append(v)

            # compute unweighted mean
            scores["{}_f1_unweighted_mean".format(ds)] = np.mean(values)

        return scores


class DefectModel(InspectModel):
    def __init__(self, checkpoint: Path, prefix: str = "pred_", use_cuda: bool = False):
        data_transform = self.get_default_test_albumentations(
            [0.59675165], [0.16061852]
        )
        data_transform_wrap = (
            lambda x: data_transform(image=np.expand_dims(img_as_ubyte(x), axis=-1))[
                "image"
            ]
            .repeat_interleave(3, axis=0)
            .unsqueeze(0)
        )
        result_names = ["class", "cam"]
        wrapped_module = DefectModule
        hparams = (
            pd.read_csv(checkpoint.parent / "meta_tags.csv")
            .set_index("key")["value"]
            .to_dict()
        )
        hparams["classes"] = ["crack", "inactive"]
        super().__init__(
            wrapped_module,
            checkpoint,
            result_names,
            prefix=prefix,
            data_transform=data_transform_wrap,
            use_cuda=True,
            hparams=hparams,
        )

    def get_default_test_albumentations(self, mean, std):
        return A.Compose(
            [
                A.Resize(300, 300),
                A.Normalize(mean=mean, std=std),
                A.pytorch.transforms.ToTensorV2(),
            ]
        )

    def apply(self, data: ImageSequence) -> ImageSequence:
        data = super().apply(data)

        def label(x: pd.Series) -> pd.Series:
            x = x.copy()
            x[self.prefix + "crack_p"] = expit(x[self.prefix + "class"][0][0])
            x[self.prefix + "inactive_p"] = expit(x[self.prefix + "class"][0][1])
            x[self.prefix + "crack_cam"] = x[self.prefix + "cam"][0][0]
            x[self.prefix + "inactive_cam"] = x[self.prefix + "cam"][0][1]
            del x[self.prefix + "class"]
            del x[self.prefix + "cam"]
            return x

        return data.apply_meta(label)
