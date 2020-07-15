import onnxruntime
import numpy as np
from scipy import special
from typing import List, Tuple
from pvinspect.data.image import ImageOrSequence, _sequence, Image, ImageSequence
from skimage.transform import rescale
from pathlib import Path
from abc import ABC, abstractmethod
import random
from tqdm.autonotebook import trange
import multiprocessing.pool


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

        # thread pool for data loading
        self._pool = multiprocessing.pool.ThreadPool(processes=1)

    def predict(
        self,
        images: ImageOrSequence,
        norm_mean: float = None,
        norm_std: float = None,
        batch_size: int = 8,
    ) -> ImageOrSequence:

        if (norm_mean is None or norm_std is None) and len(images) < 20:
            raise RuntimeError(
                "Please provide a sufficient amount of images to automatically compute statistics"
            )
        elif norm_mean is None or norm_std is None:
            # automatically compute statistics using 20 or 200 images
            samples = random.choices(images, k=20 if len(images) < 200 else 200)
            imgs = np.stack([image.data.astype(np.float) for image in samples])
            norm_mean = np.mean(imgs)
            norm_std = np.std(imgs)

        def apply(x: ImageSequence):
            data = [item.data.astype(np.float32) for item in x]

            # conditionally resize
            if self._input_shape is not None:
                tgt = [
                    s1 / s2
                    for s1, s2 in zip(list(self._input_shape), list(data[0].shape))
                ]
                data = [rescale(x, scale=tgt) for x in data]
            data = np.stack(data, axis=0)

            # normalize + reshape
            data = (data - norm_mean) / norm_std
            data = np.tile(np.expand_dims(data, axis=1), (1, 3, 1, 1))

            # predict
            ort_inputs = {self._session.get_inputs()[0].name: data}
            pred = self._session.run(None, ort_inputs)
            return [
                {
                    "onnx_{}".format(k): pred[i][b]
                    for i, k in enumerate(self._output_names)
                }
                for b in range(len(x))
            ]

        results = []
        cur_batch_thread = self._pool.apply_async(
            lambda: images.pandas.iloc[0 * batch_size : (0 + 1) * batch_size]
        )
        for i in trange(
            len(images) // batch_size + (0 if len(images) % batch_size == 0 else 1)
        ):
            next_batch_thread = self._pool.apply_async(
                lambda: images.pandas.iloc[i * batch_size : (i + 1) * batch_size]
            )
            batch = cur_batch_thread.get()
            results.extend(apply(batch))
            cur_batch_thread = next_batch_thread

        images_new = []
        for img, anns in zip(images, results):
            images_new.append(img.from_self(meta=anns))

        return images.from_self(images=images_new)


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
        self, images: ImageOrSequence, thresh: float = 0.5, **kwargs,
    ) -> ImageOrSequence:

        # run base model
        images = self._base.predict(images=images, **kwargs)

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
