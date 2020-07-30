from typing import Tuple
from pvinspect.data.image import ImageOrSequence
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
