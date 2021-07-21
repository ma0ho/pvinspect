from __future__ import annotations

from os import write

import matplotlib
from matplotlib.axes import Axes
from pandas.core.frame import DataFrame
from pvinspect.data.image.image import Image, LoadFnType, TImage
from pvinspect.data.image.show_plugin import (
    PluginOption,
    get_active_show_plugins,
    invoke_show_plugins,
)
from pvinspect.data.image.type import DType

"""Provides classes to store and visualize images with metadata"""

import copy
import inspect
import logging
import math
import re
import sys
from abc import ABCMeta, abstractclassmethod, abstractmethod, abstractproperty
from enum import Enum
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from matplotlib import markers as markers  # type: ignore
from matplotlib import pyplot as plt
from skimage import img_as_float64, img_as_int, img_as_uint
from tqdm.autonotebook import tqdm

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
            plt.subplot(rows, cols, i + 1)
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
    _meta: Optional[pd.DataFrame]

    def __init__(
        self, meta: Optional[MetaType],
    ):
        """Initialize an image sequence
        
        Args:
            images (List[Image]): The list of images
        """

        self._meta = (
            pd.DataFrame(meta, copy=True)
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

    @abstractmethod
    def apply(self: TImageSequence, fn: Callable[[Image], Image]) -> TImageSequence:
        pass

    def apply_image_data(
        self: TImageSequence, fn: Callable[[np.ndarray], np.ndarray]
    ) -> TImageSequence:
        return self.apply(lambda x: x.from_self(data=fn(x.data)))

    def as_type(self: TImageSequence, dtype: DType) -> TImageSequence:
        return self.apply(lambda x: x.as_type(dtype))

    def meta_from_fn(
        self, fn: Callable[[Image], Dict[str, Any]], **kwargs
    ) -> ImageSequence:
        """Extract meta information using given function

        Args:
            fn (Callable[[Image], Dict[str, Any]]): Function that is applied on every element of the sequence
        """
        return self.apply(fn=lambda x: x.meta_from_fn(fn), **kwargs)

    def meta_from_meta(
        self,
        pattern: str,
        source_key: str,
        target_key: str,
        target_type: Type,
        group_n: int = 1,
        transform: Callable[[Any], Any] = None,
    ) -> ImageSequence:
        """Extract meta information from path of individual aimges. The group_n'th matching group
        from pattern is used as meta value

        Args:
            pattern (str): Regular expression used to parse meta
            source_key (str): Key of the source meta attribute
            target_key (str): Key of the target meta attribute
            target_type (Type): Result is converted to this datatype
            group_n (int): Index of matching group
            transform (Callable[[Any], Any]): Optional function that is applied on the value
                before datatype conversion

        Returns:
            images (ImageSequence): Resulting ImageSequence
        """

        def fn(x: Image) -> Image:
            return x.meta_from_meta(
                pattern=pattern,
                source_key=source_key,
                target_key=target_key,
                target_type=target_type,
                group_n=group_n,
                transform=transform,
            )

        return self.apply(fn)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    @overload
    def __getitem__(self, idx: int) -> Image:
        ...

    @abstractmethod
    @overload
    def __getitem__(
        self: TImageSequence, idx: Union[slice, List[int]]
    ) -> TImageSequence:
        ...

    @abstractmethod
    def __getitem__(
        self: TImageSequence, idx: Union[int, slice, List[int]]
    ) -> Union[Image, TImageSequence]:
        pass

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
    @abstractmethod
    def meta(self) -> pd.DataFrame:
        pass


class LazyImageSequence(ImageSequence):

    """ this is used to store meta data """

    _meta: pd.DataFrame

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

    def apply(
        self: LazyImageSequence, fn: Callable[[Image], Image]
    ) -> LazyImageSequence:
        res = copy.copy(self)
        res._apply_fns = copy.copy(self._apply_fns)
        res._apply_fns.append(fn)
        return res

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(
        self, idx: Union[int, slice, List[int]]
    ) -> Union[Image, LazyImageSequence]:
        if isinstance(idx, int):
            return self._apply_all(self._load_fn(self._meta.iloc[idx]))
        else:
            return LazyImageSequence(self._meta.iloc[idx], self._load_fn)

    def __add__(self, other: ImageSequence) -> LazyImageSequence:
        """ Concatenate two image sequences """
        return LazyImageSequence(
            pd.concat([self._meta, other.meta], ignore_index=True, copy=False),
            self._load_fn,
        )

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta.copy()


class EagerImageSequence(ImageSequence):

    _images: List[Image]

    def __init__(self, images: List[Image], meta: Optional[MetaType]):
        super(EagerImageSequence, self).__init__(meta)

        self._images = images

    @abstractmethod
    def apply(self: TImageSequence, fn: Callable[[Image], Image]) -> TImageSequence:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    @overload
    def __getitem__(self, idx: int) -> Image:
        ...

    @abstractmethod
    @overload
    def __getitem__(
        self: TImageSequence, idx: Union[slice, List[int]]
    ) -> TImageSequence:
        ...

    @abstractmethod
    def __getitem__(
        self: TImageSequence, idx: Union[int, slice, List[int]]
    ) -> Union[Image, TImageSequence]:
        pass

    @abstractmethod
    def __add__(self: TImageSequence, other: TImageSequence) -> TImageSequence:
        """ Concatenate two image sequences """
        pass

    @property
    @abstractmethod
    def meta(self) -> pd.DataFrame:
        pass
