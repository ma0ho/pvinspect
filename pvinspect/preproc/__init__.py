"""Methods for preprocessing data"""

from . import detection, stitching

# from . import calibration
from .default_plugins import _register_default_plugins

_register_default_plugins()
