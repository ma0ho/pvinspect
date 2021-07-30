from pathlib import Path
from typing import Optional, Type

import pandas as pd
from numpy import ceil, empty, log10
from pvinspect.data.image.image import EagerImage, Image, LazyImage
from pvinspect.data.image.sequence import (
    EagerImageSequence,
    ImageSequence,
    LazyImageSequence,
)
from pvinspect.data.meta import MetaDriver, PandasMetaDriver
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
) -> ImageSequence:
    meta = meta_driver.read_sequence_meta(path) if with_meta else None

    # if meta does not exist or is not loaded, we resort to listing all images from path
    if meta is None:
        fns = (
            list(path.glob("*.png"))
            + list(path.glob("*.jpg"))
            + list(path.glob("*.tif"))
        )
        fns = [fn.name for fn in fns]
        meta = pd.DataFrame({"original_filename": fns})

    # apply limit
    if limit is not None and len(meta) > limit:
        meta = meta.iloc[:limit]

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
