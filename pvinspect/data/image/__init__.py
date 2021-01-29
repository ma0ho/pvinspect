from pvinspect.data.image.default_plugins import register_default_plugins
from . import image
from .image import (
    DType,
    Image,
)
from .default_plugins import _register_default_plugins


# register default image viewer plugins
_register_default_plugins()
