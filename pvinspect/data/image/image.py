from __future__ import annotations

from os import write

from matplotlib.axes import Axes
from pvinspect.data.image.show_plugin import (
    PluginOption,
    get_active_show_plugins,
    invoke_show_plugins,
)

"""Provides classes to store and visualize images with metadata"""

import copy
import inspect
import re
from abc import ABCMeta, abstractclassmethod, abstractmethod
from enum import Enum
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from matplotlib import markers as markers  # type: ignore
from matplotlib import pyplot as plt
from pvinspect.data.image.type import (
    DType,
    _convert_numpy_image,
    _map_numpy_dtype,
    _unify_dtypes,
)

# caching
SEQUENCE_MAX_CACHE_SIZE = 5000

# Types
LoadFnType = Callable[[], np.ndarray]
MetaType = Union[pd.Series, Dict[str, Any]]
DataFunction = Callable[[np.ndarray], np.ndarray]

# Type vars
TImage = TypeVar("TImage", bound="Image")


class Image(metaclass=ABCMeta):
    """A general image"""

    _meta: pd.Series

    def __init__(self, meta: Optional[pd.Series] = None, **kwargs):
        """Create a new image. Additional meta data can be given to kwargs.

        Args:
            meta (Optional[pd.Series]): Meta attributes of this image
        """

        kwargs_meta = {k: v for k, v in kwargs.items()}

        # force meta data type
        if meta is not None and len(kwargs_meta) > 0:
            self._meta = pd.concat([meta, pd.Series(kwargs_meta, d)])  # type: ignore
        elif meta is not None:
            self._meta = meta
        else:
            self._meta = (
                pd.Series(kwargs_meta)
                if len(kwargs_meta) > 0
                else pd.Series({}, dtype=object)
            )

    @abstractmethod
    def _data_ref(self) -> Any:
        pass

    @classmethod
    @abstractmethod
    def from_other(cls: Type[TImage], other: TImage, **kwargs) -> TImage:
        """Create a new image by partially overwriting the properties of another. This also merges meta attributes.

        Args:
            other (Image): The other image
            **kwargs: Arguments that should be overwritten
        """
        pass

    def from_self(self: TImage, **kwargs) -> TImage:
        """Create a new image by partially overwriting the properties of this image. This also merges meta attributes.

        Args:
            **kwargs: Arguments that should be overwritten
        """
        return type(self).from_other(self, **kwargs)

    def show(self, ax: Optional[Axes] = None, **kwargs):
        """Show this image
        
        Args:
            ax (Optional[Axes]): If given, image will be shown here
        """
        # create a figure, if none is given
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1)

        # invoke plugins
        invoke_show_plugins(self, ax, **kwargs)

    def show_options(self) -> List[Tuple[str, List[PluginOption]]]:
        """Determine currently active plugin options"""
        res: List[Tuple[str, List[PluginOption]]] = list()

        for p in get_active_show_plugins(self):
            res.append((p.title, p.options()))

        return res

    @abstractmethod
    def apply_data(self: TImage, fn: DataFunction) -> TImage:
        """Apply a callable to the image data returning new `Image`. This is evaluated lazyly for
        `LazyImage`

        Args:
            fn (DataFunction): This callable is applied to the image data

        Returns:
            image (TImage): The resulting image
        """
        pass

    def as_type(self: TImage, dtype: DType) -> TImage:
        """Return this image with the image datatype converted according to dtype

        Args:
            dtype (DType): Type of the resulting image
        """
        f = partial(_convert_numpy_image, dtype=dtype)
        return self.apply_data(f)

    def __deepcopy__(self, memo) -> Image:
        # let behavior be determined by overridden attributes
        return self.from_self()

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """The underlying image data"""
        pass

    @property
    def dtype(self) -> DType:
        """Datatype of the image"""
        return _map_numpy_dtype(self.data.dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the image"""
        return self.data.shape

    def get_meta(self, key: str) -> Any:
        """Access a meta attribute"""
        if isinstance(self._meta[key], np.ndarray):
            v = self._meta[key].view()
            v.setflags(write=False)
            return v
        else:
            return copy.copy(self._meta[key])

    def has_meta(self, key: str) -> bool:
        """Check if a meta attribute is set"""
        return key in self._meta.index

    def list_meta(self) -> List[str]:
        """List avaliable meta keys"""
        return list(self._meta.index)

    def meta_from_meta(
        self: TImage,
        pattern: str,
        source_key: str,
        target_key: str,
        target_type: Type,
        group_n: int = 0,
        transform: Callable[[Any], Any] = None,
    ) -> TImage:
        """Extract meta information from path. The group_n'th matching group
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
            image (Image): Resulting Image
        """
        s = str(self.get_meta(source_key).absolute())
        res = re.search(pattern, s)

        if res is not None:
            v = res.group(group_n)
            if transform is not None:
                v = transform(v)
            v = target_type(v)
            return self.from_self(meta={target_key: v})
        else:
            return self.from_self()

    def meta_from_fn(self: TImage, fn: Callable[[Image], Dict[str, Any]]) -> TImage:
        """Extract meta data using given callable

        Args:
            fn (Callable[[Image], Dict[str, Any]]): Function used to extract meta data
        
        Returns
            image (Image): Resulting Image
        """
        return self.from_other(self, meta=fn(self))

    @property
    def meta(self) -> Optional[pd.Series]:
        """Convert (compatible) meta data to pandas series"""
        return copy.deepcopy(self._meta)


class EagerImage(Image):
    def __init__(self, data: np.ndarray, meta: Optional[pd.Series] = None, **kwargs):
        """Create a new image.

        Args:
            data (np.ndarray): The image data
            meta (Optional[pd.Series]): Meta attributes of this image
        """
        super().__init__(meta, **kwargs)
        self._data = _unify_dtypes(data)
        self._data.setflags(write=False)

    def _data_ref(self) -> np.ndarray:
        return self._data

    def apply_data(self: TImage, fn: DataFunction) -> TImage:
        return self.from_self(data=fn(self.data))

    @property
    def data(self) -> np.ndarray:
        return self._data.view()

    @classmethod
    def from_other(
        cls,
        other: EagerImage,
        data: Optional[np.ndarray] = None,
        meta: Optional[pd.Series] = None,
        **kwargs
    ) -> EagerImage:
        if not isinstance(other, EagerImage) and data is None:
            raise NotImplementedError(
                "Cannot create an EagerImage from a non-eager image without specifying the data"
            )

        args = dict()

        args["data"] = data if data is not None else other._data

        args["meta"] = other.meta
        if meta is not None:
            for k, v in meta.iteritems():
                args["meta"][k] = v  # type: ignore

        for k, v in kwargs.items():
            args["meta"][k] = v  # type: ignore

        return EagerImage(**args)


class LazyImage(Image):
    class LazyData:
        @classmethod
        @lru_cache(maxsize=SEQUENCE_MAX_CACHE_SIZE, typed=True)
        def _load(
            cls, load_fn: LoadFnType, checks: Tuple[Callable[[np.ndarray], np.ndarray]],
        ) -> np.ndarray:
            data = load_fn()

            # perform data checks/conversions
            for check in checks:
                data = check(data)

            # make it immutable
            data.setflags(write=False)

            return data

        def __init__(self, load_fn: LoadFnType):
            self._load_fn = load_fn
            self._checks: List[Callable[[np.ndarray], np.ndarray]] = list()

        def __getattr__(self, name: str):
            # forward to numpy
            data = self._load(self._load_fn, tuple(self._checks))
            return getattr(data, name)

        def __getitem__(self, s):
            data = self._load(self._load_fn, tuple(self._checks))
            return data[s]

        def push_check(self, fn: Callable[[np.ndarray], np.ndarray]):
            self._checks.append(fn)

        def load(self) -> np.ndarray:
            return self._load(self._load_fn, tuple(self._checks))

    def __init__(self, data: LazyData, meta: Optional[pd.Series] = None, **kwargs):
        """Create a new image.

        Args:
            data (LazyData): The lazy loaded image data
            meta (Optional[pd.Series]): Meta attributes of this image
        """
        super().__init__(meta, **kwargs)
        self._data = data

        # make sure that data types are correctly handled after loading the data
        self._data.push_check(_unify_dtypes)

    def _data_ref(self) -> LazyData:
        return copy.deepcopy(self._data)

    def apply_data(self: LazyImage, fn: DataFunction) -> LazyImage:
        # get a referencing copy
        res = self.from_self()
        res._data.push_check(fn)
        return res

    @property
    def data(self) -> np.ndarray:
        return self._data.load()

    @classmethod
    def from_other(
        cls,
        other: Image,
        data: Optional[LazyData] = None,
        meta: Optional[pd.Series] = None,
        **kwargs
    ) -> LazyImage:
        if not isinstance(other, LazyImage) and data is None:
            raise NotImplementedError(
                "Cannot create a LazyImage from a non-lazy image without specifying the data"
            )

        args = dict()

        args["data"] = data if data is not None else other._data

        args["meta"] = other.meta
        if meta is not None:
            for k, v in meta.item:
                args["meta"][k] = v  # type: ignore

        for k, v in kwargs.items():
            args["meta"][k] = v  # type: ignore

        return LazyImage(**args)
