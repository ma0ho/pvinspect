import logging
from pvinspect.data import ImageSequence
from typing import Tuple, List, Optional, Callable, Any, Dict, Union, TypeVar
import numpy as np

try:
    import torch as t
except ImportError as e:
    logging.error("You need to install PyTorch in order to use the PyTorch-integration")
    raise e


class Dataset(t.utils.data.Dataset):

    D = TypeVar("D")

    def __init__(
        self,
        data: ImageSequence,
        data_transform: Optional[Callable[[np.ndarray], D]] = None,
        meta_attrs: Optional[List[str]] = None,
        meta_transforms: Optional[Dict[str, Callable[[Any], Any]]] = None,
    ):
        super().__init__()
        self._data = data
        self._meta_attrs = meta_attrs

        # make sure that data_transform is callable
        identity = lambda x: x
        self._data_transform = (
            data_transform if data_transform is not None else identity
        )

        # make sure that for every meta attribute there is a callable transform
        n_meta_transforms = len(meta_transforms) if meta_transforms is not None else 0
        if meta_attrs is not None:
            self._meta_transforms = [
                meta_transforms[k] if k in meta_transforms.keys() else identity
                for k in meta_attrs
            ]
        else:
            self._meta_transforms = []

        if not data[0].lazy:
            logging.warn(
                "The ImageSequence used to construct this Dataset is not lazy loaded. "
                "We recommend using lazy loaded data in combination with PyTorch."
            )

    def __getitem__(self, index: int) -> Union[D, Tuple[Any]]:
        # image data
        d = self._data_transform(self._data[index].data)

        # meta -> labels
        if self._meta_attrs is not None:
            meta = [
                self._meta_transforms[i](self._data[index].get_meta(k))
                for i, k in enumerate(self._meta_attrs)
            ]
        else:
            meta = []

        ret = tuple([d] + meta)
        if len(ret) > 1:
            return ret
        else:
            return ret[0]

    def __len__(self) -> int:
        return len(self._data)
