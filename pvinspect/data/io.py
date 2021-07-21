"""Read and write images"""

import json
import logging
import pickle
import urllib.parse
from datetime import date, datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.lib.npyio import save
from pvinspect.common import util
from pvinspect.common.types import ObjectAnnotations, PathOrStr
from pvinspect.data.image import Modality
from pyparsing import ParseException
from shapely.errors import WKTReadingError
from shapely.geometry import Point, Polygon
from shapely.wkt import dumps as shapely_dumps
from shapely.wkt import loads as shapely_loads
from skimage import color, img_as_float, img_as_uint, io
from tqdm.auto import tqdm

from .exceptions import UnsupportedModalityException
from .image import *


def _prepare_json_meta(meta):

    # numpy scalar -> builtin
    if type(meta).__module__ == np.__name__ and meta.ndim == 0:
        meta = np.asscalar(meta)

    if isinstance(meta, dict):
        return {k: _prepare_json_meta(v) for k, v in meta.items()}
    elif isinstance(meta, list):
        return [_prepare_json_meta(v) for v in meta]
    elif isinstance(meta, (Polygon, Point)):
        return shapely_dumps(meta)
    elif isinstance(meta, (date, datetime)):
        return meta.isoformat()
    elif isinstance(meta, Modality):
        return str(meta)
    elif isinstance(meta, (str, float, int, bool)):
        return meta
    else:
        return None


def _load_json_meta_hook(pairs):
    result = dict()
    for k, v in pairs.items():
        if isinstance(v, str):
            if v == "Modality.EL_IMAGE":
                result[k] = Modality.EL_IMAGE
                continue
            if v == "Modality.PL_IMAGE":
                result[k] = Modality.PL_IMAGE
                continue

            try:
                result[k] = datetime.fromisoformat(v)
                continue
            except ValueError:
                result[k] = v
            try:
                with util.suppress_stderr():
                    result[k] = shapely_loads(v)
                continue
            except WKTReadingError:
                result[k] = v
        else:
            result[k] = v
    return result


def __assurePath(p: PathOrStr) -> Path:
    if isinstance(p, str):
        return Path(p)
    else:
        return p


def _get_meta_path(img_path: Path) -> Path:
    return img_path.parent / "{}.json".format(img_path.name)


def _read_image(
    path: PathOrStr,
    is_module_image: bool,
    is_partial_module: bool,
    lazy: bool,
    modality: int = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    first_col: Optional[int] = None,
    first_row: Optional[int] = None,
    force_dtype: Union[DType, None] = None,
    meta=None,
):
    assert (
        is_module_image
        and not is_partial_module
        or is_partial_module
        and not is_module_image
        or not is_partial_module
        and not is_module_image
    )

    path = __assurePath(path)

    def _read(path):
        img = io.imread(path)

        if img.dtype == ">u2":
            # big endian -> little endian
            img = img.astype(np.uint16)

        if img.ndim == 3 and (img.dtype == np.float32 or img.dtype == np.float64):
            img = color.rgb2gray(img)
        elif img.ndim == 3:
            img = img_as_uint(color.rgb2gray(img))

        if force_dtype == DType.FLOAT:
            img = img.astype(DTYPE_FLOAT)
        elif force_dtype == DType.UNSIGNED_INT:
            img = img.astype(DTYPE_UNSIGNED_INT)
        elif force_dtype == DType.INT:
            img = img.astype(DTYPE_INT)

        if (img.dtype == np.float32 or img.dtype == np.float64) and (
            img.min() < 0.0 or img.max() > 1.0
        ):
            raise RuntimeWarning(
                'Image "{}" is of type float but not scaled between 0 and 1. This might cause trouble later. You might want to force conversion to another datatype using force_dtype=DType.UNSIGNED_INT (for example).'.format(
                    path
                )
            )
        return img

    # conditionally enable lazy loading
    img = _read(path) if not lazy else Image.LazyData(partial(_read, path))

    # try to read meta file
    meta_path = _get_meta_path(path)
    meta_series: pd.Series = None
    if meta is not None:
        meta_series = meta
    elif meta_path.is_file():
        with open(meta_path, "r") as f:
            meta_series = pd.Series(json.load(f, object_hook=_load_json_meta_hook))

    # merge modality, rows and cols
    if (
        meta_series is not None
        and "modality" in meta_series.keys()
        and modality is None
    ):
        modality = meta_series["modality"]
    if meta_series is not None and "rows" in meta_series.keys() and rows is None:
        rows = meta_series["rows"]
    if meta_series is not None and "cols" in meta_series.keys() and cols is None:
        cols = meta_series["cols"]

    if is_partial_module:
        return PartialModuleImage(
            img,
            modality,
            path,
            cols,
            rows,
            first_row=first_row,
            first_col=first_col,
            meta=meta,
        )
    elif is_module_image:
        return ModuleImage(img, modality, path, cols, rows, meta=meta_series)
    else:
        return Image(img, path, modality, meta=meta_series)


def _get_meta_cache_path(sequence_path: Path) -> Path:
    return sequence_path / ".cached_meta.pck"


def _read_images(
    path: PathOrStr,
    is_module_image: bool,
    is_partial_module: bool,
    lazy: bool,
    same_camera: bool = False,
    modality: int = None,
    cols: int = None,
    rows: int = None,
    N: int = 0,
    pattern: Union[str, Tuple[str]] = ("*.png", "*.tif", "*.tiff", "*.bmp"),
    allow_different_dtypes=False,
    force_dtype: DType = None,
) -> ImageSequence:
    path = __assurePath(path)

    if isinstance(pattern, str):
        pattern = [pattern]

    # find files and skip if more than N
    imgpaths: List[Path] = list(
        reduce(lambda x, y: x + y, [list(path.glob(pat)) for pat in pattern])
    )
    imgpaths.sort()
    if N > 0 and N < len(imgpaths):
        imgpaths = imgpaths[:N]

    # check if cached meta data is available
    cache_file = _get_meta_cache_path(path)
    cached_meta: pd.DataFrame = None
    if cache_file.is_file():
        with open(cache_file, "rb") as f:
            try:
                cached_meta = pd.read_pickle(f)
            except:
                cache_file.unlink()

    imgs = list()
    missing_meta: Dict[str, pd.Series] = dict()
    for fn in tqdm(imgpaths):
        if cached_meta is not None and fn.name in cached_meta.keys():
            img = _read_image(
                fn,
                is_module_image,
                is_partial_module,
                lazy,
                modality,
                cols,
                rows,
                force_dtype=force_dtype,
                meta=cached_meta.loc[fn.name],
            )
        else:
            img = _read_image(
                fn,
                is_module_image,
                is_partial_module,
                lazy,
                modality,
                cols,
                rows,
                force_dtype=force_dtype,
            )
            missing_meta[fn.name] = img.meta_to_pandas()
        imgs.append(img)

    # update/create meta cache file
    if len(missing_meta.keys()) > 0:
        missing_meta_df = pd.DataFrame(missing_meta).T
        new_meta = (
            pd.concat([cached_meta, missing_meta_df])
            if cached_meta is not None
            else missing_meta_df
        )
        new_meta.to_pickle(cache_file)

    # TODO: How to handle this with lazy loading?
    # if not same_camera:
    #    homogeneous_types = np.all(
    #        np.array([img.dtype == imgs[0].dtype for img in imgs])
    #    )
    #    shapes = [img.shape for img in imgs]
    #    homogeneous_shapes = np.all(np.array([s == shapes[0] for s in shapes]))
    #    target_shape = np.max(shapes, axis=0)

    #    if not homogeneous_shapes:
    #        logging.warning(
    #            "The original images are of different shape. They might not be suited for all applications (for example superresolution)."
    #        )

    #    if not homogeneous_types:
    #        # target type is determined by the first image
    #        logging.warning(
    #            "The original images are of different type. They are converted to the type of the first image ({})".format(
    #                imgs[0].dtype
    #            )
    #        )
    #        conv = (
    #            img_as_float
    #            if imgs[0].dtype == np.float32 or imgs[0].dtype == np.float64
    #            else img_as_uint
    #        )
    #        imgs = [conv(img) for img in imgs]

    if is_module_image or is_partial_module:
        res = ModuleImageSequence(
            imgs, same_camera=same_camera, allow_different_dtypes=allow_different_dtypes
        )
    else:
        res = ImageSequence(imgs, same_camera, allow_different_dtypes)

    return res


def read_image(
    path: PathOrStr, modality: int = None, force_dtype: DType = None, lazy: bool = True,
) -> Image:
    """Read a single image of a solar module and return it

    Args:
        path (PathOrStr): Path to the file to be read
        modality (int): The imaging modality
        force_dtype (DType): Force images to have this datatype
        lazy (bool): Enable lazy loading for this image

    Returns:
        image: The module image
    """

    return _read_image(
        path=path,
        is_module_image=False,
        is_partial_module=False,
        lazy=lazy,
        modality=modality,
        force_dtype=force_dtype,
    )


def read_images(
    path: PathOrStr,
    same_camera: bool = False,
    modality: int = None,
    N: int = 0,
    pattern: Union[str, Tuple[str]] = ("*.png", "*.tif", "*.tiff", "*.bmp"),
    allow_different_dtypes=False,
    force_dtype: DType = None,
    lazy: bool = True,
) -> ImageSequence:
    """Read a sequence of images and return it

    Args:
        path (PathOrStr): Path to the sequence
        same_camera (bool): Indicate, if all images are from the same camera and hence share the same intrinsic parameters
        modality (int): The imaging modality
        N (int): Only read first N images
        pattern (Union[str, Tuple[str]]): Files must match any of the given pattern
        allow_different_dtypes (bool): Allow images to have different datatypes?
        force_dtype (DType): Force images to have this datatype
        lazy (bool): Enable lazy loading for these images

    Returns:
        image: The image sequence
    """

    return _read_images(
        path,
        is_module_image=False,
        same_camera=same_camera,
        is_partial_module=False,
        lazy=lazy,
        modality=modality,
        pattern=pattern,
        allow_different_dtypes=allow_different_dtypes,
        N=N,
        force_dtype=force_dtype,
    )


def read_module_image(
    path: PathOrStr,
    modality: Modality,
    cols: int = None,
    rows: int = None,
    lazy: bool = True,
) -> ModuleImage:
    """Read a single image of a solar module and return it

    Args:
        path (PathOrStr): Path to the file to be read
        modality (int): The imaging modality
        cols (int): Number of columns of cells
        rows (int): Number of rows of cells
        lazy (bool): Enable lazy loading for this image

    Returns:
        image: The module image
    """

    return _read_image(
        path=path,
        is_module_image=True,
        is_partial_module=False,
        lazy=lazy,
        modality=modality,
        cols=cols,
        rows=rows,
    )


def read_module_images(
    path: PathOrStr,
    modality: int,
    same_camera: bool = False,
    cols: int = None,
    rows: int = None,
    N: int = 0,
    pattern: Union[str, Tuple[str]] = ("*.png", "*.tif", "*.tiff", "*.bmp"),
    allow_different_dtypes=False,
    force_dtype: DType = None,
    lazy: bool = True,
) -> ModuleImageSequence:
    """Read a sequence of module images and return it

    Args:
        path (PathOrStr): Path to the sequence
        modality (int): The imaging modality
        same_camera (bool): Indicate, if all images are from the same camera and hence share the same intrinsic parameters
        cols (int): Number of columns of cells
        rows (int): Number of rows of cells
        N (int): Only read first N images
        pattern (Union[str, Tuple[str]]): Files must match any of the given pattern
        allow_different_dtypes (bool): Allow images to have different datatypes?
        force_dtype (DType): Force images to have this datatype
        lazy (bool): Enable lazy loading for these images

    Returns:
        image: The module image sequence
    """

    return _read_images(
        path=path,
        modality=modality,
        same_camera=same_camera,
        is_partial_module=False,
        is_module_image=True,
        lazy=lazy,
        cols=cols,
        rows=rows,
        N=N,
        pattern=pattern,
        allow_different_dtypes=allow_different_dtypes,
        force_dtype=force_dtype,
    )


def read_partial_module_image(
    path: PathOrStr,
    modality: int,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    first_col: Optional[int] = None,
    first_row: Optional[int] = None,
    lazy: bool = True,
) -> ModuleImage:
    """Read a single partial view of a solar module and return it

    Args:
        path (PathOrStr): Path to the file to be read
        modality (int): The imaging modality
        cols (int): Number of completely visible columns of cells
        rows (int): Number of completely visible rows of cells
        lazy (bool): Enable lazy loading for this image

    Returns:
        image: The module image
    """

    return _read_image(
        path=path,
        is_module_image=False,
        is_partial_module=True,
        lazy=lazy,
        modality=modality,
        cols=cols,
        rows=rows,
        first_col=first_col,
        first_row=first_row,
    )


def read_partial_module_images(
    path: PathOrStr,
    modality: int,
    same_camera: bool = False,
    cols: int = None,
    rows: int = None,
    N: int = 0,
    pattern: Union[str, Tuple[str]] = ("*.png", "*.tif", "*.tiff", "*.bmp"),
    allow_different_dtypes=False,
    force_dtype: DType = None,
    lazy: bool = True,
) -> ModuleImageSequence:
    """Read a sequence of partial views of solar modules and return it

    Args:
        path (PathOrStr): Path to the sequence
        modality (int): The imaging modality
        same_camera (bool): Indicate, if all images are from the same camera and hence share the same intrinsic parameters
        cols (int): Number of completely visible columns of cells
        rows (int): Number of completely visible rows of cells
        N (int): Only read first N images
        pattern (Union[str, Tuple[str]]): Files must match any of the given pattern
        allow_different_dtypes (bool): Allow images to have different datatypes?
        force_dtype (DType): Force images to have this datatype
        lazy (bool): Enable lazy loading for these images

    Returns:
        image: The module image sequence
    """

    return _read_images(
        path=path,
        is_module_image=False,
        is_partial_module=True,
        same_camera=same_camera,
        lazy=lazy,
        modality=modality,
        cols=cols,
        rows=rows,
        N=N,
        pattern=pattern,
        allow_different_dtypes=allow_different_dtypes,
        force_dtype=force_dtype,
    )


def save_image(
    filename: PathOrStr,
    image: Image,
    with_visusalization: bool = False,
    save_meta: bool = False,
    **kwargs
):
    """Write an image to disk. Float64 is automatically converted to float32 in order to be compatible to ImageJ.

    Args:
        filename (PathOrStr): Filename of the resulting image
        image (Image): The image
        with_visualization (bool): Include the same visualizations as with image.show() or sequence.head()
        save_meta (bool): Save meta data to separate json file (cannot be used with visualization)
    """
    filename = __assurePath(filename)

    if with_visusalization and save_meta:
        logging.error("Cannot save meta data for image with visualization")
        return

    if with_visusalization:
        plt.clf()
        image.show(**kwargs)
        plt.savefig(filename, **kwargs)
    else:
        if image.dtype == DType.FLOAT:
            io.imsave(filename, image.data.astype(np.float32), check_contrast=False)
        else:
            io.imsave(filename, image.data, check_contrast=False)
        if save_meta and len(image.list_meta()) > 0:
            meta = {k: image.get_meta(k) for k in image.list_meta()}
            meta = _prepare_json_meta(meta)
            meta_path = _get_meta_path(filename)
            with open(meta_path, "w") as f:
                json.dump(meta, f)

    # delete cached entry
    cache_file = _get_meta_cache_path(filename.parent)
    if save_meta and cache_file.is_file():
        cached_meta: pd.DataFrame = pd.read_pickle(cache_file)
        cached_meta.drop(filename.name, inplace=True)
        cached_meta.to_pickle(cache_file)


def save_images(
    path: PathOrStr,
    sequence: ImageSequence,
    mkdir: bool = True,
    with_visualization: bool = False,
    hierarchical: List[str] = None,
    include_meta_keys: bool = True,
    save_meta: bool = False,
    filename_prefix: str = "",
    filename_suffix: str = "",
    filename_hook: Callable[[Image], Path] = lambda image: Path(image.path.name),
    **kwargs
):
    """Write a sequence of images to disk

    Args:
        path (PathOrStr): Target directory
        sequence (ImageSequence): The sequence of images
        mkdir (bool): Automatically create missing directories
        with_visualization (bool): Include the same visualizations as with image.show() or sequence.head()
        hierarchical (List[str]): Create a directory hierarchy using given meta keys
        include_meta_keys (bool): Indicate, if meta keys should be included in the folder names
        save_meta (bool): Save image meta in json file
        filename_prefix (str): Prepend every filename with this
        filename_suffix (str): Add this suffix to every filename
        filename_hook (Callable[[Image], Path]): Callable that is used to determine the filename
    """

    path = __assurePath(path)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    # check if meta cache exists and delete if necessary
    cache_file = _get_meta_cache_path(path)
    if save_meta and cache_file.is_file():
        cache_file.unlink()

    for image in tqdm(sequence.images):

        # determine filename
        fn = filename_hook(image)
        fn = Path(filename_prefix + fn.stem + filename_suffix + fn.suffix)

        fpath = path
        if hierarchical is not None:
            for mk in hierarchical:
                if not include_meta_keys:
                    fpath /= str(image.get_meta(mk))
                else:
                    fpath /= "{}_{}".format(mk, str(image.get_meta(mk)))

        if mkdir:
            fpath.mkdir(parents=True, exist_ok=True)

        fpath /= fn
        save_image(
            fpath,
            image,
            with_visusalization=with_visualization,
            save_meta=save_meta,
            **kwargs
        )


def load_json_object_masks(path: PathOrStr) -> ObjectAnnotations:
    """Load object annotations from file

    Args:
        path (PathOrStr): Path to the annotations file
    
    Returns:
        annotations: Dict with filenames a keys and a list of annotations, where every
            annotation is a tuple of "classname" and a Polygon
    """

    path = __assurePath(path)

    with open(path, "r") as f:
        js = json.load(f)

    if isinstance(js, list):
        result = dict()
        for item in js:
            anns = list()
            for k, v in item["Label"].items():
                for ann in v:
                    poly = Polygon([(x["x"], x["y"]) for x in ann["geometry"]])
                    anns.append((k, poly))
            result[item["External ID"]] = anns
        return result
    elif isinstance(js, dict):
        result = dict()
        prefix_len = 124

        # id -> category name
        catbyid = dict()
        for item in js["categories"]:
            catbyid[item["id"]] = item["name"]

        for img in js["images"]:
            fn = urllib.parse.unquote(img["file_name"][prefix_len:])
            result[fn] = list()
            for item in js["annotations"]:
                if item["image_id"] == img["id"]:
                    x = item["segmentation"][0]
                    poly = Polygon(
                        [(x[2 * i + 0], x[2 * i + 1]) for i in range(len(x) // 2)]
                    )
                    result[fn].append((catbyid[item["category_id"]], poly))

        return result
