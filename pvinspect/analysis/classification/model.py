import numpy as np
from scipy import special
from typing import List, Tuple
from pvinspect.data.image import ImageOrSequence, Image
from pvinspect.analysis.common.model import BaseModel


class ClassificationModel:
    def __init__(
        self,
        base: BaseModel,
        classes: List[str],
        logistic_transform: bool,
        prediction_name: str,
        probability_postfix: str = "_p_predicted",
        prediction_postfix: str = "_predicted",
    ):
        self._base = base
        self._classes = classes
        self._logistic_transform = logistic_transform
        self._probability_postfix = probability_postfix
        self._prediction_postfix = prediction_postfix
        self._prediction_name = prediction_name

    def predict(
        self, images: ImageOrSequence, thresh: float = 0.5, **kwargs,
    ) -> ImageOrSequence:

        # run base model
        images = self._base.predict(images=images, **kwargs)

        def apply(x: Image):
            pred = x.get_meta(self._prediction_name).flatten()

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
