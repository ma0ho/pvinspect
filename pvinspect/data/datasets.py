"""Provides access to demo datasets"""

from .image import (
    ModuleImageSequence,
    ModuleImage,
    EL_IMAGE,
    ImageSequence,
    CellImage,
    CellImageSequence,
)
from .io import *
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from typing import Tuple, Dict
import os
import requests
from zipfile import ZipFile
import logging

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
        gdd.download_file_from_google_drive(k, str(ds_path / "data.zip"), unzip=True)
    return ds_path


def _check_and_download_zip_ds(name: str) -> Path:
    url = _ZIP_DS_URLS[name]
    target = _DS_PATH / name

    if not target.is_dir():
        logging.info("Data is being downloaded..")
        target.mkdir()
        r = requests.get(url, allow_redirects=True)
        open(target / "data.zip", "wb").write(r.content)
        zipf = ZipFile(target / "data.zip")
        zipf.extractall(target)

    return target


def poly10x6(N: int = 0) -> ModuleImageSequence:
    """Read sequence of 10x6 poly modules
    
    Args:
        N (int): Only read first N images
    """
    p = _check_and_download_ds("20191219_poly10x6")
    return read_module_images(p, EL_IMAGE, True, 10, 6, N=N)


def elpv(N: int = 0) -> ImageSequence:
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
    seq = read_images(images_path, same_camera=False, modality=EL_IMAGE, N=N)

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
    def label(img: Image):
        l = labels.loc["images/{}".format(img.path.name)]
        return l.to_dict()

    # read images and labels
    seq = seq.meta_from_fn(label, progress_bar=False)

    return seq


def caip_dataB() -> Tuple[ModuleImageSequence, ModuleImageSequence, ObjectAnnotations]:
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
    images1 = read_module_images(
        p / "deitsch_testset" / "10x6",
        EL_IMAGE,
        False,
        10,
        6,
        allow_different_dtypes=True,
    )
    images2 = read_module_images(
        p / "deitsch_testset" / "9x4",
        EL_IMAGE,
        False,
        9,
        4,
        allow_different_dtypes=True,
    )
    annot = load_json_object_masks(p / "deitsch_testset" / "module_locations.json")
    return images1, images2, annot


def caip_dataC() -> Tuple[ModuleImageSequence, ObjectAnnotations]:
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
    return read_module_images(p / "multiple", EL_IMAGE, True, 10, 6), annot


def caip_dataD() -> Tuple[ModuleImageSequence, ObjectAnnotations]:
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
    return read_module_images(p / "rotated", EL_IMAGE, True, 10, 6), annot


def calibration_ipv40CCD_FF(N: int = 0) -> Dict[str, ImageSequence]:
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
        seq = read_images(path=d, same_camera=False, N=N)
        res[key] = seq
    return res


def calibration_ipv40CCD_distortion(N: int = 0) -> ImageSequence:
    """Lens calibration data for ipv40CCD (private dataset)

    Args:
        N (int): Number of images

    Returns:
        images: Sequence of images
    """
    p = _check_and_download_ds("20200303_calibration_iPV40CCD")
    return read_images(path=p / "distortion", same_camera=True, N=N)


def multi_module_detection(N: int = 0) -> Tuple[ObjectAnnotations, ImageSequence]:
    """Dataset for multi module detection (private dataset)

    Args:
        N (int): Number of images

    Returns:
        anns: Dict of annotations by image
        imgs: Sequence of images
    """
    p = _check_and_download_ds("20200331_multi_module_detection")
    imgs = read_images(
        path=p, same_camera=False, N=N, pattern="**/*.png", modality=EL_IMAGE
    )
    anns = load_json_object_masks(path=p / "labels.json")
    return anns, imgs
