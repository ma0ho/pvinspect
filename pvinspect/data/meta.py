from abc import ABC, abstractclassmethod, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
from pvinspect.data.image.image import Image
from pvinspect.data.image.sequence import ImageSequence


class MetaDriver(ABC):
    @classmethod
    @abstractmethod
    def read_sequence_meta(cls, path: Path) -> Optional[pd.DataFrame]:
        pass

    @classmethod
    @abstractmethod
    def save_sequence_meta(cls, path: Path, sequence: ImageSequence) -> None:
        pass

    @classmethod
    @abstractmethod
    def read_image_meta(cls, path: Path) -> Optional[pd.Series]:
        pass

    @classmethod
    @abstractmethod
    def save_image_meta(cls, path: Path, image: Image) -> None:
        pass


class PandasMetaDriver(MetaDriver):
    @classmethod
    def read_sequence_meta(cls, path: Path) -> Optional[pd.DataFrame]:
        if (path / "meta.pck").is_file():
            return pd.read_pickle(path / "meta.pck")  # type: ignore
        else:
            return None

    @classmethod
    def save_sequence_meta(cls, path: Path, sequence: ImageSequence) -> None:
        if not (path / "meta.pck").is_file():
            sequence.meta.to_pickle(path / "meta.pck")  # type: ignore
        else:
            raise RuntimeError(
                "Meta data already exists ({}). Please delete this file if you want to \
                override the previous data.".format(
                    str(path / "meta.pck")
                )
            )

    @classmethod
    def read_image_meta(cls, path: Path) -> Optional[pd.Series]:
        meta = cls.read_sequence_meta(path.parent)
        if meta is not None:
            return meta.query("original_filename == '{}'".format(path.name)).iloc[0]
        else:
            return None

    @classmethod
    def save_image_meta(cls, path: Path, image: Image) -> None:
        raise NotImplementedError(
            "Cannot save a single image meta datum with the PandasMetaDriver. Please always save the \
            meta data for a complete sequence."
        )
