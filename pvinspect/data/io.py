import json
import urllib
from pathlib import Path
from typing import List, Optional, Type, Union

import pandas as pd
from numpy import ceil, empty, log10
from pvinspect.common.types import ObjectAnnotations
from pvinspect.data.image.image import EagerImage, Image, LazyImage, MetaType
from pvinspect.data.image.sequence import (
    EagerImageSequence,
    ImageSequence,
    LazyImageSequence,
)
from pvinspect.data.meta import MetaDriver, PandasMetaDriver
from shapely.geometry import Polygon
from skimage import io as skio


def read_image(
    path: Path,
    lazy: bool = False,
    with_meta: bool = False,
    meta_driver: Type[MetaDriver] = PandasMetaDriver,
) -> Image:
    meta = meta_driver.read_image_meta(path) if with_meta else None

    # make sure that the original filename is set in the meta data
    if meta is None:
        meta = pd.Series({"original_filename": path.name})

    if not lazy:
        data = skio.imread(path)
        return EagerImage(data, meta)
    else:
        return LazyImage(LazyImage.LazyData(lambda: skio.imread(path)), meta)


def read_images(
    path: Path,
    lazy: bool = False,
    limit: Optional[int] = None,
    with_meta: bool = False,
    meta_driver: Type[MetaDriver] = PandasMetaDriver,
    common_meta: Optional[MetaType] = None,
    pattern: Union[str, List[str]] = ["*.png", "*,jpg", "*.tif"],
) -> ImageSequence:
    meta = meta_driver.read_sequence_meta(path) if with_meta else None

    # if meta does not exist or is not loaded, we resort to listing all images from path
    if meta is None:
        fns = []
        if isinstance(pattern, list):
            for pat in pattern:
                fns += list(path.glob(pat))
        else:
            fns = list(path.glob(pattern))
        fns = [fn.name for fn in fns]
        meta = pd.DataFrame({"original_filename": fns})

    # apply limit
    if limit is not None and len(meta) > limit:
        meta = meta.iloc[:limit]

    # set common meta
    if common_meta is not None:
        if isinstance(common_meta, dict):
            common_meta = pd.Series(common_meta)
        meta.loc[:, common_meta.keys()] = common_meta.values  # type: ignore

    if not lazy:
        images = [
            EagerImage(
                skio.imread(path / meta.iloc[i]["original_filename"]), meta.iloc[i]
            )
            for i in meta.index
        ]
        return EagerImageSequence(images, meta)
    else:
        load_seq_item_fn = lambda x: LazyImage(
            LazyImage.LazyData(lambda: skio.imread(path / x["original_filename"])), x
        )
        return LazyImageSequence(meta, load_seq_item_fn)


def save_images(
    path: Path,
    data: ImageSequence,
    with_meta: bool = False,
    meta_driver: Type[MetaDriver] = PandasMetaDriver,
    default_filetype: str = "tif",
) -> None:
    # check if directory is empty
    try:
        next(path.glob("*"))
        is_empty = False
    except StopIteration:
        is_empty = True

    if not is_empty:
        raise RuntimeError(
            "Cannot save a sequence to a non-empty directory ({})".format(str(path))
        )

    fmtstr = "{:0" + str(int(ceil(log10(len(data))))) + "d}." + default_filetype

    # save images
    for i, image in enumerate(data):
        fn = (
            fmtstr.format(i)
            if not image.has_meta("original_filename")
            else image.get_meta("original_filename")
        )
        skio.imsave(path / fn, image.data, check_contrast=False)

    # save meta
    if with_meta:
        meta_driver.save_sequence_meta(path, data)


def save_image(
    path: Path,
    data: Image,
    with_meta: bool = False,
    meta_driver: Type[MetaDriver] = PandasMetaDriver,
) -> None:
    # check if file exists
    if path.is_file():
        raise RuntimeError("File ({}) already exists".format(str(path)))

    # save image
    skio.imsave(path, data.data, check_contrast=False)

    # save meta
    if with_meta:
        meta_driver.save_image_meta(path, data)


def load_json_object_masks(path: Path) -> ObjectAnnotations:
    """Load object annotations from file
    Args:
        path (Path): Path to the annotations file
    
    Returns:
        annotations: Dict with filenames a keys and a list of annotations, where every
            annotation is a tuple of "classname" and a Polygon
    """

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
    else:
        raise RuntimeError("Unknown format")
