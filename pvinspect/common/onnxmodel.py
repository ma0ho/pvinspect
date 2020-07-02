import onnxruntime
import numpy as np
from scipy import special
from typing import List, Tuple
from pvinspect.data.image import ImageOrSequence, _sequence, Image
from skimage.transform import rescale
from pathlib import Path


class ONNXModel:
    def __init__(
        self,
        file: Path,
        classes: List[str],
        logistic_transform: bool,
        input_shape: Tuple[int, int],
        predictions_postfix: str = "_prediction",
    ):
        self._classes = classes
        self._logistic_transform = logistic_transform
        self._input_shape = input_shape
        self._predictions_postfix = predictions_postfix

        # init session
        self._session = onnxruntime.InferenceSession(file)

    def predict(
        self, images: ImageOrSequence, norm_mean: float = None, norm_std: float = None
    ) -> ImageOrSequence:

        if (norm_mean is None or norm_std is None) and len(images) < 20:
            raise RuntimeError(
                "Please provide a sufficient amount of images to automatically compute statistics"
            )
        elif norm_mean is None or norm_std is None:
            # automatically compute statistics using first 20 images
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
            pred = self._session.run(None, ort_inputs)[0].flatten()

            # sigmoid
            if self._logistic_transform:
                pred = special.expit(pred)

            return {
                k + self._predictions_postfix: pred[i]
                for k, i in zip(self._classes, range(len(self._classes)))
            }

        return images.meta_from_fn(apply)
