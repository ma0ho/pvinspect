from pvinspect.data.image.default_plugins import _register_default_plugins

from .default_plugins import _register_default_plugins
from .image import *
from .sequence import *
from .show_plugin import *

# register default image viewer plugins
_register_default_plugins()
