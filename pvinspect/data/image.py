"""Provides classes to store and visualize images with metadata"""

import numpy as np
from skimage import io, img_as_uint, img_as_float64, img_as_int
from pvinspect.common.transform import Transform
from matplotlib import pyplot as plt
from pathlib import Path
from typing import List, Tuple, Union, Callable, Type, TypeVar, Any, Dict
from copy import deepcopy
import math
from functools import wraps
import logging
from pvinspect.common._ipy_exit import exit
import inspect
import sys
from enum import Enum

# this is a pointer to the module object instance itself
this = sys.modules[__name__]


# modality
EL_IMAGE = 0
"""Indicate an electroluminescense (EL) image"""
PL_IMAGE = 1
"""Indicate a photoluminescense (PL) image"""


# datatypes
DTYPE_INT = np.int64
DTYPE_UNSIGNED_INT = np.uint32
DTYPE_FLOAT = np.float64
img_as_float = img_as_float64


class DType(Enum):
    INT = (0,)
    UNSIGNED_INT = 1
    FLOAT = 2


# global list of plugins that are called on every .show()
this.show_plugins = list()


def register_show_plugin(callable, priority: int = 0):
    """Register a new plugin that is called on every .show()

    Args:
        callable: Callable that receives the image as a first argument and variable arguments to .show() next
        priority (int): Plugins are invoked in the order of increasing priority (highest priority is invoked
            last and hence appears on top)
    """
    this.show_plugins.append((priority, callable))
    this.show_plugins = sorted(this.show_plugins, key=lambda x: x[0])


def _invoke_show_plugins(image, **kwargs):
    for p in this.show_plugins:
        p[1](image, **kwargs)


def _register_default_plugins():
    def show_cell_crossings(
        image: ModuleImage, show_cell_crossings: bool = True, **kwargs
    ):
        if (
            show_cell_crossings
            and isinstance(image, ModuleImage)
            and image.has_meta("transform")
        ):
            grid = image.grid()
            coords = image.get_meta("transform").__call__(grid)
            plt.scatter(coords[:, 0], coords[:, 1], c="yellow", marker="+")

    register_show_plugin(show_cell_crossings)

    def multimodule_show_boxes(
        image: Image,
        multimodule_show_boxes: bool = True,
        multimodule_highlight_selection: bool = True,
        multimodule_boxes_linewidth: int = 2,
        **kwargs
    ):
        if (
            multimodule_show_boxes
            and isinstance(image, Image)
            and image.has_meta("multimodule_boxes")
        ):
            for i, box in enumerate(image.get_meta("multimodule_boxes")):
                color = (
                    "red"
                    if i == image.get_meta("multimodule_index")
                    and multimodule_highlight_selection
                    else "yellow"
                )
                plt.plot(
                    *box[1].exterior.xy,
                    linewidth=multimodule_boxes_linewidth,
                    color=color
                )

    register_show_plugin(multimodule_show_boxes)

    def multimodule_show_numbers(
        image: Image,
        multimodule_show_numbers: bool = True,
        multimodule_highlight_selection: bool = True,
        multimodule_numbers_fontsize: int = 20,
        **kwargs
    ):
        if (
            multimodule_show_numbers
            and isinstance(image, Image)
            and image.has_meta("multimodule_boxes")
        ):
            for i, box in enumerate(image.get_meta("multimodule_boxes")):
                bgcolor = (
                    "red"
                    if i == image.get_meta("multimodule_index")
                    and multimodule_highlight_selection
                    else "white"
                )
                textcolor = (
                    "white"
                    if i == image.get_meta("multimodule_index")
                    and multimodule_highlight_selection
                    else "black"
                )
                plt.text(
                    box[1].centroid.x,
                    box[1].centroid.y,
                    s=str(i),
                    color=textcolor,
                    fontsize=multimodule_numbers_fontsize,
                    bbox=dict(facecolor=bgcolor, alpha=0.8),
                    ha="center",
                    va="center",
                )

    register_show_plugin(multimodule_show_numbers)

    def show_image(
        image: Image,
        clip_low: float = 0.001,
        clip_high: float = 99.999,
        colorbar: bool = True,
        **kwargs
    ):
        clip_low = clip_low if clip_low is not None else 0.0
        clip_high = clip_high if clip_high is not None else 100.0
        p = np.percentile(image._data, [clip_low, clip_high])
        d = np.clip(image._data, p[0], p[1])
        plt.imshow(d, cmap="gray")
        if colorbar:
            plt.colorbar()

    register_show_plugin(show_image, -100)

    def axis_options(
        image: Image, show_axis: bool = True, show_title: bool = True, **kwargs
    ):
        if not show_axis:
            plt.axis("off")
        if show_title:
            plt.title(str(image.path.name))

    register_show_plugin(axis_options, -200)


class _Base:

    T = TypeVar("T")

    @classmethod
    def from_other(cls: Type[T], other: T, **kwargs) -> T:
        """Create a new image by partially overwriting the properties of another

        Args:
            other (Image): The other image
            **kwargs: Arguments that should be overwritten
        """
        required = inspect.getfullargspec(cls.__init__)[0]

        other_args = dict()
        for name in required:
            if name == "meta" and "meta" in kwargs.keys():
                # joint meta dictionaries
                tmp = deepcopy(other._meta)
                tmp.update(kwargs["meta"])
                kwargs["meta"] = tmp
            if name not in kwargs.keys() and name != "self":
                other_args[name] = getattr(other, "_" + name)

        return cls(**kwargs, **other_args)


class Image(_Base):
    """A general image"""

    def __init__(
        self,
        data: np.ndarray,
        path: Path,
        modality: int = None,
        meta: Dict[str, Any] = {},
    ):
        """Create a new image. All non-float images as automatically converted to uint.

        Args:
            data (np.ndarray): The image data
            path (Path): Path to the image
            modality (int): The imaging modality (EL_IMAGE or PL_IMAGE) or None
            meta (Dict[str, Any]): Meta attributes of this image
        """
        self._data = data
        self._path = path
        self._modality = modality
        self._meta = meta

    def show(self, **kwargs):
        """Show this image"""
        _invoke_show_plugins(self, **kwargs)

    _T = TypeVar("T")

    def as_type(self: _T, dtype: DType) -> _T:
        if dtype == DType.FLOAT:
            return type(self).from_other(self, data=img_as_float(self._data))
        elif dtype == DType.UNSIGNED_INT:
            return type(self).from_other(self, data=img_as_uint(self._data))
        elif dtype == DType.INT:
            return type(self).from_other(self, data=img_as_int(self._data))

    def __add__(self: _T, other: _T) -> _T:
        if self.dtype != other.dtype:
            raise RuntimeError("Images must have the same datatype")
        return type(self).from_other(self, data=self._data + other._data)

    def __sub__(self: _T, other: _T) -> _T:
        if self.dtype != other.dtype:
            raise RuntimeError("Images must have the same datatype")
        if self.dtype == DType.UNSIGNED_INT:
            res = self._data.astype(DTYPE_INT) - other._data.astype(DTYPE_INT)
            iinfo = np.iinfo(DTYPE_UNSIGNED_INT)
            res = np.clip(res, 0, iinfo.max)
            return type(self).from_other(self, data=res)
        else:
            return type(self).from_other(self, data=self._data - other._data)

    def __mul__(self: _T, other: _T) -> _T:
        if self.dtype != other.dtype:
            raise RuntimeError("Images must have the same datatype")
        return type(self).from_other(self, data=self._data * other._data)

    def __truediv__(self: _T, other: _T) -> _T:
        if self.dtype != DType.FLOAT or other.dtype != DType.FLOAT:
            raise RuntimeError("Images must be of type float")
        return type(self).from_other(self, data=self._data / other._data)

    def __floordiv__(self: _T, other: _T) -> _T:
        if self.dtype != other.dtype:
            raise RuntimeError("Images must have the same datatype")
        return type(self).from_other(self, data=self._data // other._data)

    def __mod__(self: _T, other: _T) -> _T:
        if self.dtype != other.dtype:
            raise RuntimeError("Images must have the same datatype")
        return type(self).from_other(self, data=self._data % other._data)

    def __pow__(self: _T, other: _T) -> _T:
        if self.dtype != other.dtype:
            raise RuntimeError("Images must have the same datatype")
        return type(self).from_other(self, data=self._data ** other._data)

    @property
    def data(self) -> np.ndarray:
        """The underlying image data"""
        return deepcopy(self._data)

    @property
    def path(self) -> Path:
        """Path to the original image"""
        return deepcopy(self._path)

    @property
    def dtype(self) -> DType:
        """Datatype of the image"""
        if self.data.dtype == np.float32 or self.data.dtype == np.float64:
            return DType.FLOAT
        elif (
            self.data.dtype == np.uint8
            or self.data.dtype == np.uint16
            or self.data.dtype == np.uint32
            or self.data.dtype == np.uint64
        ):
            return DType.UNSIGNED_INT
        elif (
            self.data.dtype == np.int8
            or self.data.dtype == np.int16
            or self.data.dtype == np.int32
            or self.data.dtype == np.int64
        ):
            return DType.INT

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the image"""
        return deepcopy(self.data.shape)

    @property
    def modality(self) -> int:
        """The imaging modality"""
        return self._modality

    def get_meta(self, key: str) -> Any:
        """Access a meta attribute"""
        return self._meta[key]

    def has_meta(self, key: str) -> bool:
        """Check if a meta attribute is set"""
        return key in self._meta.keys()

    def list_meta(self) -> List[str]:
        """List avaliable meta keys"""
        return list(self._meta.keys())


class ImageSequence(_Base):
    """An immutable sequence of images, allowing for access to single images as well as analysis of the sequence"""

    def _show(self, imgs: List[Image], cols: int, *args, **kwargs):
        N = len(imgs)
        rows = math.ceil(N / cols)

        # adjust the figure size
        if self.shape is not None:
            aspect = self.shape[0] / self.shape[1]
        else:
            aspect = 1.0
        plt.figure(figsize=(6 * cols, 6 * rows * aspect))

        for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i + 1)
            img.show(*args, **kwargs)

    def __init__(
        self, images: List[Image], same_camera: bool, allow_different_dtypes=False
    ):
        """Initialize a module image sequence
        
        Args:
            images (List[Image]): The list of images
            came_camera (bool): Indicates, if all images are from the same camera and hence share the same intrinsic parameters
            allow_different_dtypes (bool): Allow images to have different datatypes?
        """

        self._images = images
        self._same_camera = same_camera
        self._allow_different_dtypes = allow_different_dtypes
        if len(self.images) == 0:
            logging.error("Creation of an empty sequence is not supported")
            exit()

        # check that all have the same modality, dimension, dtype and module configuration
        shape = self.images[0].shape
        dtype = self.images[0].dtype
        modality = self.images[0].modality
        for img in self.images:
            if img.dtype != dtype and not allow_different_dtypes:
                logging.error(
                    'Cannot create sequence from mixed dtypes. Consider using the "allow_different_dtypes" argument, when reading images.'
                )
                exit()
            if img.shape != shape and same_camera:
                logging.error(
                    'Cannot create sequence from mixed shapes. Consider using the "same_camera" argument, when reading images.'
                )
                exit()
            if img.modality != modality:
                logging.error("Cannot create a sequence from mixed modalities.")
                exit()

    def head(self, N: int = 4, cols: int = 2, *args, **kwargs):
        """Show the first N images

        Args:
            N (int): Number of images to show
            cols (int): How many images to show in a column
        """
        self._show(self.images[:N], cols, *args, **kwargs)

    def tail(self, N: int = 4, cols: int = 2, *args, **kwargs):
        """Show the last N images

        Args:
            N (int): Number of images to show
            cols (int): How many images to show in a column
        """
        self._show(self.images[-N:], cols, *args, **kwargs)

    _T = TypeVar("T")

    def apply_image_data(
        self: _T, fn: Callable[[np.ndarray], np.ndarray], *argv, **kwargs
    ) -> _T:
        """Apply the given callable on every image data."""
        result = []
        for img in self._images:
            data = img.data
            res = fn(data, *argv, **kwargs)
            result.append(type(img).from_other(img, data=res))
        return type(self).from_other(self, images=result)

    def as_type(self: _T, dtype: DType) -> _T:
        """Convert sequence to specified dtype"""
        result = [img.as_type(dtype) for img in self._images]
        return type(self).from_other(self, images=result)

    def __add__(self: _T, other: _T) -> _T:
        res = [x + y for x, y in zip(self.images, other.images)]
        return type(self).from_other(self, images=res)

    def __sub__(self: _T, other: _T) -> _T:
        res = [x - y for x, y in zip(self.images, other.images)]
        return type(self).from_other(self, images=res)

    def __mul__(self: _T, other: _T) -> _T:
        res = [x * y for x, y in zip(self.images, other.images)]
        return type(self).from_other(self, images=res)

    def __truediv__(self: _T, other: _T) -> _T:
        res = [x / y for x, y in zip(self.images, other.images)]
        return type(self).from_other(self, images=res)

    def __floordiv__(self: _T, other: _T) -> _T:
        res = [x // y for x, y in zip(self.images, other.images)]
        return type(self).from_other(self, images=res)

    def __mod__(self: _T, other: _T) -> _T:
        res = [x % y for x, y in zip(self.images, other.images)]
        return type(self).from_other(self, images=res)

    def __pow__(self: _T, other: _T) -> _T:
        res = [x ** y for x, y in zip(self.images, other.images)]
        return type(self).from_other(self, images=res)

    @property
    def images(self) -> List[Image]:
        """Access the list of images"""
        return deepcopy(self._images)

    @property
    def dtype(self) -> DType:
        """Access the image datatype"""
        return self.images[0].dtype if not self._allow_different_dtypes else None

    @property
    def shape(self) -> Tuple[int, int]:
        """Access the image shape"""
        return self.images[0].shape if self._same_camera else None

    @property
    def modality(self) -> int:
        """Access the imaging modaility"""
        return self.images[0].modality

    @property
    def same_camera(self) -> bool:
        """Indicate, if the images originate from the same camera"""
        return self._same_camera

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> Image:
        return deepcopy(self.images[i])


ImageOrSequence = Union[Image, ImageSequence]


class CellImage(Image):
    """An image of a solar cell with additional meta data"""

    def __init__(
        self,
        data: np.ndarray,
        modality: int,
        path: Path,
        row: int,
        col: int,
        meta: Dict[str, Any] = {},
    ):
        """Initialize a cell image

        Args:
            data (np.ndarray): The image data
            modality (int): The imaging modality
            path (Path): Path to the image
            row (int): Row index (zero-based)
            col (int): Cell index (zero-based)
            meta (Dict[str, Any]): Meta data
        """

        super().__init__(data, path=path, modality=modality, meta=meta)
        self._row = row
        self._col = col

    @property
    def row(self) -> int:
        """0-based row index of the cell in the original module"""
        return self._row

    @property
    def col(self) -> int:
        """0-based column index of the cell in the original module"""
        return self._col

    def show(self, *argv, **kwargs):
        """Show this image"""
        super().show(*argv, **kwargs)
        plt.title(
            "{}: (row: {:d}, col: {:d})".format(self._path.name, self._row, self._col)
        )


class CellImageSequence(ImageSequence):
    """An immutable sequence of cell images, allowing for access to single images as well as analysis of the sequence"""

    def __init__(self, images: List[CellImage], copy=True):
        """Initialize a module image sequence
        
        Args:
            images (List[CellImage]): The list of images
            copy (bool): Copy the images?
        """

        super().__init__(images, False, copy)


class ModuleImage(Image):
    """An image of a solar module with additional meta data"""

    def __init__(
        self,
        data: np.ndarray,
        modality: int,
        path: Path,
        cols: int = None,
        rows: int = None,
        meta: dict = {},
    ):
        """Initialize a module image

        Args:
            data (np.ndarray): The image data
            modality (int): The imaging modality
            path (Path): Path to the image
            cols (int): Number of cells in a column
            rows (int): Number of cells in a row
        """

        super().__init__(data, path, modality, meta)
        self._cols = cols
        self._rows = rows

    def grid(self) -> np.ndarray:
        """Create a grid of corners according to the module geometry
        
        Returns:
            grid: (cols*rows, 2)-array of coordinates on a regular grid
        """

        if self._cols is not None and self._rows is not None:
            x, y = np.mgrid[0 : self.cols + 1 : 1, 0 : self.rows + 1 : 1]
            grid = np.stack([x.flatten(), y.flatten()], axis=1)
            return grid
        else:
            logging.error("Module geometry is not initialized")
            exit()

    @property
    def cols(self):
        """Number of cell-columns"""
        return self._cols

    @property
    def rows(self):
        """Number of row-columns"""
        return self._rows


class PartialModuleImage(ModuleImage):
    """An image of a solar module with additional meta data"""

    def __init__(
        self,
        data: np.ndarray,
        modality: int,
        path: Path,
        cols: int = None,
        rows: int = None,
        first_col: int = None,
        first_row: int = None,
        meta: dict = {},
    ):
        """Initialize a module image

        Args:
            data (np.ndarray): The image data
            modality (int): The imaging modality
            path (Path): Path to the image
            cols (int): Number of completely visible cells in a column
            rows (int): Number of completely visible cells in a row
            first_col (int): Index of the first complete column shown
            first_row (int): Index of the first complete row shown
        """

        super().__init__(data, modality, path, cols, rows, meta)

        self._first_col = first_col
        self._first_row = first_row


ModuleOrPartialModuleImage = Union[ModuleImage, PartialModuleImage]


class ModuleImageSequence(ImageSequence):
    """An immutable sequence of module images, allowing for access to single images as well as analysis of the sequence"""

    def __init__(
        self,
        images: List[ModuleOrPartialModuleImage],
        same_camera: bool,
        allow_different_dtypes=False,
    ):
        """Initialize a module image sequence
        
        Args:
            images (List[ModuleImage]): The list of images
            same_camera (bool): Indicates if all images are from the same camera
            allow_different_dtypes (bool): Allow images to have different datatypes?
        """

        cols = images[0].cols
        rows = images[0].rows
        for img in images:
            if img.cols != cols:
                logging.error(
                    "Cannot create sequence from different module configurations"
                )
                exit()
            if img.rows != rows:
                logging.error(
                    "Cannot create sequence from different module configurations"
                )
                exit()

        super().__init__(images, same_camera, allow_different_dtypes)


ModuleImageOrSequence = Union[
    ModuleImageSequence, ModuleImage, PartialModuleImage, Image
]


def _sequence(*args):
    """Assure that the first argument is a sequence and handle the first return value accordingly"""

    def decorator_sequence(func):
        @wraps(func)
        def wrapper_sequence(*args, **kwargs):
            if not isinstance(args[0], ImageSequence):
                args = list(args)
                args[0] = (
                    ModuleImageSequence([args[0]], same_camera=False)
                    if type(args[0]) == ModuleImage
                    else ImageSequence([args[0]], same_camera=False)
                )
                unwrap = True
            else:
                unwrap = False
            res = func(*tuple(args), **kwargs)
            if unwrap and not disable_unwrap:
                if isinstance(res, tuple) and isinstance(res[0], ImageSequence):
                    res[0] = res[0].images[0]
                elif isinstance(res, ImageSequence):
                    res = res.images[0]
            return res

        return wrapper_sequence

    if len(args) == 1 and callable(args[0]):
        disable_unwrap = False
        return decorator_sequence(args[0])
    else:
        disable_unwrap = args[0] if len(args) == 1 else False
        return decorator_sequence
