import onnxruntime
import numpy as np
from scipy import special
from typing import List, Tuple
from pvinspect.data.image import ImageOrSequence, _sequence, Image
from skimage.transform import rescale
from pathlib import Path
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(
        self, input_shape: Tuple[int, int],
    ):
        self._input_shape = input_shape

    @abstractmethod
    def predict(
        self, images: ImageOrSequence, norm_mean: float = None, norm_std: float = None,
    ) -> ImageOrSequence:
        pass


class ONNXModel(BaseModel):
    def __init__(self, file: Path, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        # init session
        self._session = onnxruntime.InferenceSession(file)
        self._output_names = [x.name for x in self._session.get_outputs()]

    def predict(
        self, images: ImageOrSequence, norm_mean: float = None, norm_std: float = None,
    ) -> ImageOrSequence:

        if (norm_mean is None or norm_std is None) and len(images) < 20:
            raise RuntimeError(
                "Please provide a sufficient amount of images to automatically compute statistics"
            )
        elif norm_mean is None or norm_std is None:
            # automatically compute statistics using all images
            imgs = np.stack(
                [images[i].data.astype(np.float) for i in range(len(images))]
            )
            norm_mean = np.mean(imgs)
            norm_std = np.std(imgs)

        def apply(x: Image):
            data = x.data.astype(np.float)

            # conditionally resize
            if self._input_shape is not None:
                tgt = [
                    s1 / s2 for s1, s2 in zip(list(self._input_shape), list(data.shape))
                ]
                data = rescale(data, scale=tgt)

            # normalize
            data = (data - norm_mean) / norm_std

            # predict
            data = np.tile(data, (1, 3, 1, 1)).astype(np.float32)
            ort_inputs = {self._session.get_inputs()[0].name: data}
            pred = self._session.run(None, ort_inputs)
            return {"onnx_{}".format(k): v for k, v in zip(self._output_names, pred)}

        x = images.meta_from_fn(apply, progress_bar=True)
        return x


class ClassificationModel:
    def __init__(
        self,
        base: BaseModel,
        classes: List[str],
        logistic_transform: bool,
        probability_postfix: str = "_p_predicted",
        prediction_postfix: str = "_predicted",
    ):
        self._base = base
        self._classes = classes
        self._logistic_transform = logistic_transform
        self._probability_postfix = probability_postfix
        self._prediction_postfix = prediction_postfix

    def predict(
        self,
        images: ImageOrSequence,
        norm_mean: float = None,
        norm_std: float = None,
        thresh: float = 0.5,
    ) -> ImageOrSequence:

        # run base model
        images = self._base.predict(
            images=images, norm_mean=norm_mean, norm_std=norm_std
        )

        def apply(x: Image):
            pred = x.get_meta("onnx_prediction").flatten()

            # sigmoid
            if self._logistic_transform:
                pred = special.expit(pred)

            d1 = {
                k + self._probability_postfix: pred[i]
                for k, i in zip(self._classes, range(len(self._classes)))
            }
            d2 = {
                k + self._prediction_postfix: pred[i] > thresh
                for k, i in zip(self._classes, range(len(self._classes)))
            }
            d1.update(d2)
            return d1

        return images.meta_from_fn(apply, progress_bar=False)
