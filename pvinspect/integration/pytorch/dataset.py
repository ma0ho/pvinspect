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
        """A PyTorch-compatible Dataset implementation for ImageSequence

        Args:
            data (ImageSequence): The ImageSequence that should be wrapped in the dataset
            data_transform (Optional[Callable[[np.ndarray], D]]): Callable that transforms data, use 
                this for online data augmentation/conversion etc.
            meta_attrs (List[str]): Meta attributes that should be returned in addition to the data
                on accessing single elements of the dataset
            meta_transforms (Dict[str, Callable[[Any], Any]]): Use this to specify additional transforms
                for individual meta attributes
        """
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
        # PyTorch does not like uint>8 -> perform automatic conversion
        array = self._data[index].data
        if array.dtype in (np.uint16, np.uint32):
            if np.max(array) > 255:
                logging.error(
                    "Uint-images with more than 8 bits are incompatible to PyTorch. "
                    "However, image exceeds 8 bit and cannot be automatically converted. "
                    "Please consider to convert to float images in advance."
                )
            else:
                array = array.astype(np.uint8)

        # image data
        d = self._data_transform(array)

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

    def _meta_to_tensor(self, meta: List[Any]) -> t.Tensor:
        return t.tensor([meta[i] for i in self._meta_classes_idx], dtype=t.float32)

    def _tensor_to_meta_dict(self, tensor: t.Tensor, prefix: str) -> ClassificationMeta:
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
        """A PyTorch-compatible Dataset implementation for ImageSequence

        Args:
            data (ImageSequence): The ImageSequence that should be wrapped in the dataset
            meta_classes (List[str]): List of meta attributes that should become part of the
                one-hot encoded target vector. These meta attributes must be boolean.
            data_transform (Optional[Callable[[np.ndarray], D]]): Callable that transforms data, use 
                this for online data augmentation/conversion etc.
            meta_attrs (List[str]): Meta attributes that should be returned in addition to the data
                and the one-hot encoded target on accessing single elements of the dataset
            meta_transforms (Dict[str, Callable[[Any], Any]]): Use this to specify additional transforms
                for individual meta attributes. These apply to members of meta_classes as well
        """
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
        y = self._meta_to_tensor(res[1:])
        if len(res) > len(self._meta_classes) + 1:
            other = res[len(self._meta_classes) + 1 :]
        else:
            other = []
        res = [x, y] + other
        return tuple(res)

    def result_sequence(
        self, results: List[t.Tensor], prefix: str = "pred_"
    ) -> ImageSequence:
        """Feed classification results back to obtain an ImageSequence with predictions

        Args:
            results (List[torch.Tensor]): List of predictions from the network. Convert to
                boolean before calling this. The order of elements must be the same as in
                the underlying ImageSequence.
            prefix (str): Prefix prediction result meta attributes by this
        
        Returns:
            result (ImageSequence): The ImageSequence with additional meta data
        """
        assert len(results) == len(self._data)

        # build an inverse map from Image to index
        imap = {k: v for v, k in enumerate(self._data.images)}

        # assign meta data
        def meta_fn(x: Image) -> ClassificationDataset.ClassificationMeta:
            idx = imap[x]
            return self._tensor_to_meta_dict(results[idx], prefix=prefix)

        return self._data.meta_from_fn(meta_fn)
