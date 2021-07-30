from pvinspect.data.image.default_plugins import _register_default_plugins

from . import image
from .default_plugins import _register_default_plugins
from .image import DType, EagerImage, Image, LazyImage
from .sequence import EagerImageSequence, ImageSequence, LazyImageSequence

# register default image viewer plugins
_register_default_plugins()
