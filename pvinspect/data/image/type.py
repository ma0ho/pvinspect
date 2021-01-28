import numpy as np
from skimage import img_as_float64
from enum import Enum

from skimage.util.dtype import img_as_float32, img_as_int, img_as_ubyte, img_as_uint

# datatypes
DTYPE_INT = np.int32
DTYPE_UNSIGNED_INT = np.uint16
DTYPE_UNSIGNED_BYTE = np.uint8
DTYPE_FLOAT = np.float32
img_as_float = img_as_float32


class DType(Enum):
    INT = 0
    UNSIGNED_INT = 1
    FLOAT = 2
    UNSIGNED_BYTE = 3


def _map_numpy_dtype(dtype) -> DType:
    if dtype == np.float32 or dtype == np.float64:
        return DType.FLOAT
    elif dtype == np.uint16 or dtype == np.uint32 or dtype == np.uint64:
        return DType.UNSIGNED_INT
    elif dtype == np.uint8:
        return DType.UNSIGNED_BYTE
    else:
        return DType.INT


def _convert_numpy_image(image: np.ndarray, dtype: DType):
    if dtype == DType.INT:
        return image.astype(DTYPE_INT)
    elif dtype == DType.UNSIGNED_INT:
        return img_as_uint(image)
    elif dtype == DType.FLOAT:
        return img_as_float(image)
    return img_as_ubyte(image)


def _type_min(type) -> int:
    return np.iinfo(type).min  # type: ignore


def _type_max(type) -> int:
    return np.iinfo(type).max  # type: ignore


def _unify_dtypes(array: np.ndarray) -> np.ndarray:
    if (
        _map_numpy_dtype(array.dtype) == DType.UNSIGNED_INT
        and array.dtype != DTYPE_UNSIGNED_INT
    ):
        if array.max() > _type_max(DTYPE_UNSIGNED_INT) or array.min() < _type_min(
            DTYPE_UNSIGNED_INT
        ):
            raise RuntimeError(
                "Datatype conversion to {} failed, since original data exceeds dtype limits.".format(
                    DTYPE_UNSIGNED_INT
                )
            )
        return array.astype(DTYPE_UNSIGNED_INT)
    if _map_numpy_dtype(array.dtype) == DType.INT and array.dtype != DTYPE_INT:
        if array.max() > _type_max(DTYPE_INT) or array.min() < _type_min(DTYPE_INT):
            raise RuntimeError(
                "Datatype conversion to {} failed, since original data exceeds dtype limits.".format(
                    DTYPE_INT
                )
            )
        return array.astype(DTYPE_INT)
    if _map_numpy_dtype(array.dtype) == DType.FLOAT and array.dtype != DTYPE_FLOAT:
        return array.astype(DTYPE_FLOAT)

    # default
    return array
