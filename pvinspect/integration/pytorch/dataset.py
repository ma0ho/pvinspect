import logging
from pvinspect.data import ImageSequence, Image
from typing import Tuple, List, Optional, Callable, Any, Dict, Union, TypeVar
import numpy as np

try:
    import torch as t
    from torch.utils.data import Dataset as TorchDataset  # need this for mypy
except ImportError as e:
    logging.error("You need to install PyTorch in order to use the PyTorch-integration")
    raise e


# return type of data transform
D = TypeVar("D")


class Dataset(TorchDataset):
    def __init__(
        self,
        data: ImageSequence,
        data_transform: Optional[Callable[[np.ndarray], D]] = None,
        meta_attrs: List[str] = list(),
        meta_transforms: Dict[str, Callable[[Any], Any]] = dict(),
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
            meta_transforms_keys = (
                meta_transforms.keys() if meta_transforms is not None else []
            )
            self._meta_transforms = [
                meta_transforms[k] if k in meta_transforms_keys else identity
                for k in meta_attrs
            ]
        else:
            self._meta_transforms = []

        if not data[0].lazy:
            logging.warn(
                "The ImageSequence used to construct this Dataset is not lazy loaded. "
                "We recommend using lazy loaded data in combination with PyTorch."
            )

    def __getitem__(self, index: int) -> Union[D, Tuple[Any, ...]]:
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


class ClassificationDataset(Dataset):

    ClassificationMeta = Dict[str, bool]

    def meta_to_tensor(self, meta: List[Any]) -> t.Tensor:
        return t.tensor([meta[i] for i in self._meta_classes_idx], dtype=t.float32)

    def tensor_to_meta_dict(self, tensor: t.Tensor, prefix: str) -> ClassificationMeta:
        l = tensor.to(t.bool).tolist()
        return {prefix + k: v for k, v in zip(self._meta_classes, l)}

    def __init__(
        self,
        data: ImageSequence,
        meta_classes: List[str],
        data_transform: Optional[Callable[[np.ndarray], D]] = None,
        meta_attrs: List[str] = list(),
        meta_transforms: Dict[str, Callable[[Any], Any]] = dict(),
    ):
        meta_attrs = meta_classes + meta_attrs
        super().__init__(
            data=data,
            data_transform=data_transform,
            meta_attrs=meta_attrs,
            meta_transforms=meta_transforms,
        )
        self._meta_classes = meta_classes
        self._meta_classes_idx = [self._meta_attrs.index(k) for k in meta_classes]

    def __getitem__(self, index: int) -> Union[D, Tuple[D, t.Tensor], Tuple[Any, ...]]:
        res = list(super().__getitem__(index))
        x = res[0]
        y = self.meta_to_tensor(res[1:])
        if len(res) > len(self._meta_classes) + 1:
            other = res[len(self._meta_classes) + 1 :]
        else:
            other = []
        res = [x, y] + other
        return tuple(res)

    def result_sequence(
        self, results: List[t.Tensor], prefix: str = "pred_"
    ) -> ImageSequence:
        assert len(results) == len(self._data)

        # build an inverse map from Image to index
        imap = {k: v for v, k in enumerate(self._data.images)}

        # assign meta data
        def meta_fn(x: Image) -> ClassificationDataset.ClassificationMeta:
            idx = imap[x]
            return self.tensor_to_meta_dict(results[idx], prefix=prefix)

        return self._data.meta_from_fn(meta_fn)
