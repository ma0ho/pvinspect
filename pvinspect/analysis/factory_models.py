import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

import requests
import gdown
from pvinspect.analysis.defects import DefectModel
from pvinspect.common.types import ObjectAnnotations

_MODEL_PATH = Path(__file__).parent.absolute() / "factory_models"
_MODEL_KEYS = {
    "20210817_defects": "1Hz20WbJvOp9hfiiAiiJ_yfpmL3O3lD33",
}
_ZIP_MODEL_URLS = {}


def _get_model_key(name: str):
    if name in _MODEL_KEYS.keys():
        return _MODEL_KEYS[name]
    else:
        keys = os.getenv("PVINSPECT_KEYS").split(";")
        keys = {x.split(",")[0]: x.split(",")[1] for x in keys}
        if name in keys.keys():
            return keys[name]
        else:
            raise RuntimeError(
                'The specified model "{}" could not be found. Maybe you tried \
                to access a protected model and didn\'t set PVINSPECT_KEYS environment variable?'
            )


def _check_and_download_model(name: str):
    model_path = _MODEL_PATH / name
    print(model_path)
    if not model_path.is_dir():
        logging.info("Data is being downloaded..")
        k = _get_model_key(name)
        model_path.mkdir(parents=True, exist_ok=False)
        gdown.cached_download(f"https://drive.google.com/uc?id={k}", str(model_path / "data.zip"), postprocess=gdown.extractall)
    return model_path


def _check_and_download_zip_model(name: str) -> Path:
    url = _ZIP_MODEL_URLS[name]
    target = _MODEL_PATH / name

    if not target.is_dir():
        logging.info("Data is being downloaded..")
        target.mkdir()
        r = requests.get(url, allow_redirects=True)
        open(target / "data.zip", "wb").write(r.content)
        zipf = ZipFile(target / "data.zip")
        zipf.extractall(target)

    return target


def defects(use_cuda: bool = False) -> DefectModel:
    path = _check_and_download_model("20210817_defects")
    return DefectModel(path / "model.ckpt", use_cuda=use_cuda)
