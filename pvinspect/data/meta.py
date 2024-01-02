from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
from pvinspect.data.image.image import Image
from pvinspect.data.image.sequence import ImageSequence


class MetaDriver(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def read_sequence_meta(self, path: Path) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def save_sequence_meta(self, path: Path, sequence: ImageSequence) -> None:
        pass

    @abstractmethod
    def read_image_meta(self, path: Path) -> Optional[pd.Series]:
        pass

    @abstractmethod
    def save_image_meta(self, path: Path, image: Image) -> None:
        pass


class PandasMetaDriver(MetaDriver):
    def read_sequence_meta(self, path: Path) -> Optional[pd.DataFrame]:
        if (path / "meta.pck").is_file():
            return pd.read_pickle(path / "meta.pck")  # type: ignore
        else:
            return None

    def save_sequence_meta(self, path: Path, sequence: ImageSequence) -> None:
        if not (path / "meta.pck").is_file():
            sequence.meta.to_pickle(path / "meta.pck")  # type: ignore
        else:
            raise RuntimeError(
                "Meta data already exists ({}). Please delete this file if you want to \
                override the previous data.".format(
                    str(path / "meta.pck")
                )
            )

    def read_image_meta(self, path: Path) -> Optional[pd.Series]:
        meta = self.read_sequence_meta(path.parent)
        if meta is not None:
            return meta.query("original_filename == '{}'".format(path.name)).iloc[0]
        else:
            return None

    def save_image_meta(self, path: Path, image: Image) -> None:
        raise NotImplementedError(
            "Cannot save a single image meta datum with the PandasMetaDriver. Please always save the \
            meta data for a complete sequence."
        )
