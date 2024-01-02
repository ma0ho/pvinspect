"""Provides access to demo datasets"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

import requests
import gdown
from pvinspect.common.types import ObjectAnnotations
from pvinspect.data import Image, ImageSequence
from pvinspect.data.image.type import DType
from pvinspect.data.io import *
from skimage.color import rgb2gray
from skimage.transform import rescale

_DS_PATH = Path(__file__).parent.absolute() / "datasets"
_DS_KEYS = {
    "20191219_poly10x6": "1B5fQPLvStuMvuYJ5CxbzyxfwuWQdfNVE",
    "20200728_elpv_labels": "1hK_hViiZ1-rHhvI3yGxpC6DSCXAyAFiJ",
}
_ZIP_DS_URLS = {"elpv": "https://github.com/zae-bayern/elpv-dataset/archive/master.zip"}


def _get_dataset_key(name: str):
    if name in _DS_KEYS.keys():
        return _DS_KEYS[name]
    else:
        keys = os.getenv("PVINSPECT_KEYS").split(";")
        keys = {x.split(",")[0]: x.split(",")[1] for x in keys}
        if name in keys.keys():
            return keys[name]
        else:
            raise RuntimeError(
                'The specified dataset "{}" could not be found. Maybe you tried \
                to access a protected dataset and didn\'t set PVINSPECT_KEYS environment variable?'
            )


def _check_and_download_ds(name: str):
    ds_path = Path(__file__).parent.absolute() / "datasets" / name
    if not ds_path.is_dir():
        logging.info("Data is being downloaded..")
        k = _get_dataset_key(name)
        ds_path.mkdir(parents=True, exist_ok=False)
        gdown.cached_download(f"https://drive.google.com/uc?id={k}", str(ds_path / "data.zip"), postprocess=gdown.extractall)
    return ds_path


def _check_and_download_zip_ds(name: str) -> Path:
    url = _ZIP_DS_URLS[name]
    target = _DS_PATH / name

    if not target.is_dir():
        logging.info("Data is being downloaded..")
        target.mkdir(parents=True)
        r = requests.get(url, allow_redirects=True)
        open(target / "data.zip", "wb").write(r.content)
        zipf = ZipFile(target / "data.zip")
        zipf.extractall(target)

    return target


def poly10x6(N: Optional[int] = None) -> ImageSequence:
    """Read sequence of 10x6 poly modules
    
    Args:
        N (int): Only read first N images
    """
    p = _check_and_download_ds("20191219_poly10x6")
    return read_images(
        p, limit=N, common_meta={"modality": "EL_IMAGE", "cols": 10, "rows": 6}
    )


def elpv(N: Optional[int] = None, lazy: bool = True) -> ImageSequence:
    """Read images from ELPV dataset

        Note:
            This dataset is part of the following publication:
            Deitsch, Sergiu, et al. "Automatic classification of defective photovoltaic module cells in electroluminescence images."
            Solar Energy, Elsevier BV, 2019, 185, 455-468. 
            
            Additional labels for defect types are provided by the author of this toolbox.
        
        Args:
            N (int): Number of images to return. Defaults to using all images.

        Returns:
            images: Images from the ELPV dataset with defect type annotations as `Image` meta data
    """
    # download and read images
    images_path = _check_and_download_zip_ds("elpv") / "elpv-dataset-master" / "images"
    seq = read_images(
        images_path, common_meta={"modality": "EL_IMAGE"}, limit=N, lazy=lazy
    )

    # download and read labels
    labels_path = _check_and_download_ds("20200728_elpv_labels") / "labels.csv"

    logging.info("Loading labels..")
    labels = pd.read_csv(
        labels_path,
        delimiter=";",
        index_col="filename",
        dtype={
            "defect probability": float,
            "wafer": str,
            "crack": bool,
            "inactive": bool,
            "blob": bool,
            "finger": bool,
            "testset": bool,
        },
    ).rename(columns={"defect probability": "defect_probability"})

    # associate images with labels
    def label(meta: pd.Series) -> pd.Series:
        l = labels.loc["images/{}".format(meta["original_filename"])]
        return pd.concat([meta, l])  # type: ignore

    # read images and labels
    seq = seq.apply_meta(label)

    return seq


def caip_dataB() -> Tuple[ImageSequence, ImageSequence, ObjectAnnotations]:
    """Read DataB from CAIP paper (private dataset)
    
    Note:
        This dataset is from the following publication:
        Hoffmann, Mathis, et al. "Fast and robust detection of solar modules in electroluminescence images."
        International Conference on Computer Analysis of Images and Patterns. Springer, Cham, 2019.

    Returns:
        images1: All modules with shape 10x6
        images2: All modules with shape 9x4
        annot: Annotations specifying the position of modules
    """
    p = _check_and_download_ds("20200616_caip_v2")
    images1 = read_images(
        p / "deitsch_testset" / "10x6",
        common_meta={"modality": "EL_IMAGE", "cols": 10, "rows": 6},
    ).apply_image_data(rgb2gray)
    images2 = read_images(
        p / "deitsch_testset" / "9x4",
        common_meta={"modality": "EL_IMAGE", "cols": 9, "rows": 4},
    ).apply_image_data(rgb2gray)
    annot = load_json_object_masks(p / "deitsch_testset" / "module_locations.json")
    return images1, images2, annot


def caip_dataC() -> Tuple[ImageSequence, ObjectAnnotations]:
    """Read DataC from CAIP paper (private dataset)
    
    Note:
        This dataset is from the following publication:
        Hoffmann, Mathis, et al. "Fast and robust detection of solar modules in electroluminescence images."
        International Conference on Computer Analysis of Images and Patterns. Springer, Cham, 2019.

    Returns:
        images: All modules images
        annot: Annotations specifying the position of modules
    """
    p = _check_and_download_ds("20200616_caip_v2")
    annot = load_json_object_masks(p / "multiple" / "module_locations.json")
    imgs = read_images(
        p / "multiple",
        common_meta={"modality": "EL_IMAGE", "cols": 10, "rows": 6},
        pattern="*.bmp",
    ).apply_image_data(rgb2gray)
    return imgs, annot


def caip_dataD() -> Tuple[ImageSequence, ObjectAnnotations]:
    """Read DataC from CAIP paper (private dataset)
    
    Note:
        This dataset is from the following publication:
        Hoffmann, Mathis, et al. "Fast and robust detection of solar modules in electroluminescence images."
        International Conference on Computer Analysis of Images and Patterns. Springer, Cham, 2019.

    Returns:
        images: All modules images
        annot: Annotations specifying the position of modules
    """
    p = _check_and_download_ds("20200616_caip_v2")
    annot = load_json_object_masks(p / "rotated" / "module_locations.json")
    imgs = read_images(
        p / "rotated", common_meta={"modality": "EL_IMAGE", "cols": 10, "rows": 6}
    ).apply_image_data(rgb2gray)
    return imgs, annot


def stitching_demo(N: Optional[int] = None) -> List[Tuple[Image, Image]]:
    """Data to demonstrate stitching capabilities

    Args:
        N (int): Number of image pairs that is returened

    Returns:
        images: List of image pairs
    """
    result = list()
    images = poly10x6(N)

    for image in images:
        height = image.shape[0]

        # split
        img0 = EagerImage(image.data[0 : int(height * 2 // 3)])
        img1 = EagerImage(image.data[int(height // 3) :])
        result.append((img0, img1))

    return result


def calibration_ipv40CCD_FF(N: Optional[int] = None) -> Dict[str, ImageSequence]:
    """Flat-field calibration data for ipv40CCD (private dataset)

    Args:
        N (int): Number of images per excitation

    Returns:
        images: Dict with excitation as key and images
    """
    p = _check_and_download_ds("20200303_calibration_iPV40CCD")
    res = dict()
    for d in p.glob("FF*"):
        key = d.name.split("_")[1]
        seq = read_images(d, limit=N)
        res[key] = seq
    return res


def calibration_ipv40CCD_distortion(N: Optional[int] = None) -> ImageSequence:
    """Lens calibration data for ipv40CCD (private dataset)

    Args:
        N (int): Number of images

    Returns:
        images: Sequence of images
    """
    p = _check_and_download_ds("20200303_calibration_iPV40CCD")
    return read_images(path=p / "distortion", limit=N)


def multi_module_detection(limit=None) -> Tuple[ObjectAnnotations, ImageSequence]:
    """Dataset for multi module detection (private dataset)

    Returns:
        anns: Dict of annotations by image
        imgs: Sequence of images
    """
    p = _check_and_download_ds("20200331_multi_module_detection")

    imgs = None

    for sub in p.iterdir():
        if sub.is_dir():
            tmp = read_images(
                path=p / sub,
                pattern="*.png",
                common_meta={"modality": "EL_IMAGE"},
                limit=limit,
            )

            if imgs is not None:
                imgs += tmp
            else:
                imgs = tmp
    anns = load_json_object_masks(path=p / "labels.json")
    return anns, imgs


def sr_demo(N: int, magnification: int, poly_idx: int = 5) -> ImageSequence:
    try:
        import torchvision as tv
    except ImportError as e:
        logging.error("You need to have torchvision installed: pip install torchvision")
        raise e

    # get image
    img = poly10x6(poly_idx - 1)[-1]

    # transforms
    tfms = tv.transforms.Compose(
        [
            tv.transforms.ToPILImage(),
            tv.transforms.RandomAffine(3, (0.05, 0.05), (0.7, 0.9), 3),
            tv.transforms.ToTensor(),
        ]
    )

    # apply
    return (
        EagerImageSequence.from_images([img for _ in range(N)])
        .as_type(DType.FLOAT)
        .apply_image_data(lambda x: tfms(x).squeeze().numpy())
        .apply_image_data(lambda x: rescale(x, 1 / magnification, order=0))
        .as_type(DType.UNSIGNED_INT)
    )
