from __future__ import annotations

from functools import wraps

from pvinspect.data.image.image import EagerImage, Image, LazyImage
from pvinspect.data.image.type import DType

"""Provides classes to store and visualize images with metadata"""

import copy
import logging
import math
from abc import ABCMeta, abstractclassmethod, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Type,
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
        scale = kwargs["figscale"] if "figscale" in kwargs.keys() else 3
        fig, axs = plt.subplots(
            rows, cols, figsize=(scale * cols, scale * rows * aspect)
        )

        for i, img in enumerate(imgs):
            x = i // cols
            y = i % cols
            if rows > 1:
                img.show(axs[x][y], *args, **kwargs)
            else:
                img.show(axs[y], *args, **kwargs)

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

    @property
    def pandas(self):
        return self._PandasHandler(self)

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
        res._meta = self._meta.apply(fn, axis=1)  # type: ignore
        return res

    def apply_meta_list(
        self: TImageSequence, column_name: str, meta_list: List[Any]
    ) -> TImageSequence:
        def fn(x: pd.Series) -> pd.Series:
            x = x.copy()
            x[column_name] = meta_list[x.name]
            return x

        return self.apply_meta(fn)

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
            images._meta = self._meta.iloc[idx].reset_index(drop=True)
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

    @classmethod
    @abstractmethod
    def from_images(cls: Type[TImageSequence], images: List[Image]) -> TImageSequence:
        pass


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

    @classmethod
    def from_images(cls, images: List[Image]) -> EagerImageSequence:
        meta = pd.DataFrame(
            [img.meta if img.meta is not None else pd.Series({}) for img in images]
        ).reset_index(drop=True)
        return cls(images, meta)


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

    @classmethod
    def from_images(cls, images: List[Image]) -> LazyImage:
        raise NotImplementedError(
            "Creating a LazyImageSequence from a list of images is currently not supported"
        )


ImageOrSequence = Union[Image, ImageSequence]
TImageOrSequence = TypeVar("TImageOrSequence", bound=Union[ImageSequence, Image])


def sequence(*args):
    """Assure that the first argument is a sequence and handle the first return value accordingly"""

    def decorator_sequence(func):
        @wraps(func)
        def wrapper_sequence(*args, **kwargs):
            if not isinstance(args[0], ImageSequence):
                args = list(args)
                args[0] = EagerImageSequence.from_images([args[0]])
                unwrap = True
            else:
                unwrap = False
            res = func(*tuple(args), **kwargs)
            if unwrap and not disable_unwrap:
                if isinstance(res, tuple) and isinstance(res[0], ImageSequence):
                    res = list(res)
                    res[0] = res[0][0]
                elif isinstance(res, ImageSequence):
                    res = res[0]
            return res

        return wrapper_sequence

    if len(args) == 1 and callable(args[0]):
        disable_unwrap = False
        return decorator_sequence(args[0])
    else:
        disable_unwrap = args[0] if len(args) == 1 else False
        return decorator_sequence


def sequence_no_unwrap(*args):
    """Assure that the first argument is a sequence but do not unwrap the result accordingly"""

    def decorator_sequence(func):
        @wraps(func)
        def wrapper_sequence(*args, **kwargs):
            if not isinstance(args[0], ImageSequence):
                args = list(args)
                args[0] = EagerImageSequence.from_images([args[0]])
            res = func(*tuple(args), **kwargs)
            return res

        return wrapper_sequence

    if len(args) == 1 and callable(args[0]):
        return decorator_sequence(args[0])
    else:
        return decorator_sequence
