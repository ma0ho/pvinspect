from __future__ import annotations

from pvinspect.data.image.image import Image
from pvinspect.data.image.type import DType

"""Provides classes to store and visualize images with metadata"""

import copy
import logging
import math
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from matplotlib import markers as markers  # type: ignore
from matplotlib import pyplot as plt

MetaType = Union[pd.DataFrame, List[pd.Series], List[Dict[str, Any]]]
LoadSeqItemFnType = Callable[[pd.Series], Image]

TImageSequence = TypeVar("TImageSequence", bound="ImageSequence")


class ImageSequence(Generic[TImageSequence], Iterable, metaclass=ABCMeta):
    """An immutable sequence of images, allowing for access to single images as well as analysis of the sequence"""

    def _show(self, imgs: ImageSequence, cols: int, *args, **kwargs):
        N = len(imgs)
        rows = math.ceil(N / cols)

        # adjust the figure size
        shape = imgs[0].shape
        aspect = shape[0] / shape[1]
        plt.figure(figsize=(6 * cols, 6 * rows * aspect))

        for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i + 1)  # type: ignore
            img.show(*args, **kwargs)

    class _PandasHandler:
        def __init__(self, parent: ImageSequence):
            self._parent = parent
            pass

        class _Sub:
            @overload
            def _result(self, pandas_result: pd.DataFrame) -> ImageSequence:
                ...

            @overload
            def _result(self, pandas_result: pd.Series) -> Image:
                ...

            def _result(
                self, pandas_result: Union[pd.DataFrame, pd.Series]
            ) -> Union[ImageSequence, Image]:
                if isinstance(pandas_result, pd.DataFrame):
                    idx = list(pandas_result.index)
                else:
                    idx = pandas_result.name
                return self._parent[idx]

            def __init__(self, parent: ImageSequence, attr):
                self._parent = parent
                self._attr = attr

            def __call__(self, *argv, **kwargs):
                pandas_result = self._attr(*argv, **kwargs)
                return self._result(pandas_result)

            def __getitem__(self, arg):
                pandas_result = self._attr[arg]
                return self._result(pandas_result)

        def __getattr__(self, name):
            attr = getattr(self._parent._meta, name)
            return self._Sub(self._parent, attr)

    """ this is used to store meta data """
    _meta: pd.DataFrame

    def __init__(
        self, meta: MetaType,
    ):
        """Initialize an image sequence
        
        Args:
            images (List[Image]): The list of images
        """

        self._meta = (
            pd.DataFrame(meta, copy=True)  # type: ignore
            if not isinstance(meta, pd.DataFrame)
            else meta.copy(deep=True)
        )
        if len(self._meta) == 0:
            logging.error("Creation of an empty sequence is not supported")
            exit()

        # namespace for accessing pandas methods
        self.pandas = self._PandasHandler(self)

    def head(self, N: int = 4, cols: int = 2, *args, **kwargs):
        """Show the first N images

        Args:
            N (int): Number of images to show
            cols (int): How many images to show in a column
        """
        self._show(self[:N], cols, *args, **kwargs)

    def tail(self, N: int = 4, cols: int = 2, *args, **kwargs):
        """Show the last N images

        Args:
            N (int): Number of images to show
            cols (int): How many images to show in a column
        """
        self._show(self[-N:], cols, *args, **kwargs)

    def apply_meta(
        self: TImageSequence, fn: Callable[[pd.Series], pd.Series]
    ) -> TImageSequence:
        res = copy.copy(self)
        res._meta = self._meta.apply(fn)  # type: ignore
        return res

    @abstractmethod
    def apply_image_data(
        self: TImageSequence, fn: Callable[[np.ndarray], np.ndarray]
    ) -> TImageSequence:
        pass

    @abstractmethod
    def as_type(self: TImageSequence, dtype: DType) -> TImageSequence:
        pass

    def __len__(self) -> int:
        return len(self._meta)

    @abstractmethod
    @overload
    def _get_image(self, idx: int, meta: pd.Series) -> Image:
        ...

    @abstractmethod
    @overload
    def _get_image(
        self, idx: Union[slice, List[int]], meta: pd.DataFrame
    ) -> TImageSequence:
        ...

    @abstractmethod
    def _get_image(
        self, idx: Union[int, slice, List[int]], meta: Union[pd.Series, pd.DataFrame]
    ) -> Union[Image, TImageSequence]:
        pass

    @overload
    def __getitem__(self, idx: int) -> Image:
        ...

    @overload
    def __getitem__(
        self: TImageSequence, idx: Union[slice, List[int]]
    ) -> TImageSequence:
        ...

    def __getitem__(
        self: TImageSequence, idx: Union[int, slice, List[int]]
    ) -> Union[Image, TImageSequence]:
        if isinstance(idx, int):
            image = self._get_image(idx, self._meta.iloc[idx])
            return image
        else:
            images = self._get_image(idx, self._meta.iloc[idx])
            images._meta = self._meta.iloc[idx]
            return images

    class _Iterator(Iterator):
        def __init__(self, iterable: ImageSequence):
            self.iterable = iterable
            self.pos = 0

        def __next__(self) -> Image:
            if self.pos < len(self.iterable):
                res = self.iterable[self.pos]
                self.pos += 1
                return res
            else:
                raise StopIteration()

    def __iter__(self) -> Iterator[Image]:
        return self._Iterator(self)

    @abstractmethod
    def __add__(self: TImageSequence, other: TImageSequence) -> TImageSequence:
        """ Concatenate two image sequences """
        pass

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta.copy(deep=True)


class EagerImageSequence(ImageSequence):

    _images: List[Image]

    def __init__(self, images: List[Image], meta: MetaType):
        super(EagerImageSequence, self).__init__(meta)

        self._images = images

    def apply_image_data(
        self, fn: Callable[[np.ndarray], np.ndarray]
    ) -> EagerImageSequence:
        res = copy.copy(self)
        res._images = [img.apply_data(fn) for img in res._images]
        return res

    def as_type(self, dtype: DType) -> EagerImageSequence:
        res = copy.copy(self)
        res._images = [img.as_type(dtype) for img in res._images]
        return res

    def _get_image(
        self, idx: Union[int, slice, List[int]], meta: Union[pd.DataFrame, pd.Series]
    ) -> Union[Image, EagerImageSequence]:
        if isinstance(idx, int) and isinstance(meta, pd.Series):
            return self._images[idx].from_self(meta=meta)
        elif isinstance(idx, slice) and isinstance(meta, pd.DataFrame):
            return EagerImageSequence(self._images[idx], meta)
        elif isinstance(idx, list) and isinstance(meta, pd.DataFrame):
            imgs = [self._images[i] for i in idx]
            return EagerImageSequence(imgs, meta)
        else:
            raise RuntimeError()

    def __add__(self, other: EagerImageSequence) -> EagerImageSequence:
        imgs = self._images + other._images
        meta = self.meta.append(other.meta)
        return EagerImageSequence(imgs, meta)


class LazyImageSequence(ImageSequence):

    """ this is used to store meta data """

    """ used to load data """
    _load_fn: LoadSeqItemFnType

    """ stack of functions that is applied to an image on load """
    _apply_fns: List[Callable[[Image], Image]]

    def _apply_all(self, image: Image) -> Image:
        for f in self._apply_fns:
            image = f(image)
        return image

    def __init__(self, meta: MetaType, load_fn: LoadSeqItemFnType):
        super(LazyImageSequence, self).__init__(meta)

        self._load_fn = load_fn
        self._apply_fns = list()

    def apply_image_data(
        self, fn: Callable[[np.ndarray], np.ndarray]
    ) -> LazyImageSequence:
        res = copy.copy(self)
        res._apply_fns = copy.deepcopy(self._apply_fns)
        res._apply_fns.append(lambda x: x.apply_data(fn))
        return res

    def as_type(self, dtype: DType) -> LazyImageSequence:
        res = copy.copy(self)
        res._apply_fns = copy.deepcopy(self._apply_fns)
        res._apply_fns.append(lambda x: x.as_type(dtype))
        return res

    def _get_image(
        self, idx: Union[int, slice, List[int]], meta: Union[pd.DataFrame, pd.Series]
    ) -> Union[Image, LazyImageSequence]:
        if isinstance(idx, int) and isinstance(meta, pd.Series):
            return self._apply_all(self._load_fn(meta))
        elif isinstance(meta, pd.DataFrame):
            return LazyImageSequence(meta, self._load_fn)
        else:
            raise RuntimeError()

    def __add__(self, other: LazyImageSequence) -> LazyImageSequence:
        raise NotImplementedError(
            "Concatenation of lazy image sequences not supported. Hint: Use `seq.to_eager()` to convert to an eager sequence before"
        )

    def to_eager(self) -> EagerImageSequence:
        imgs = [img for img in self]
        return EagerImageSequence(imgs, self.meta)
