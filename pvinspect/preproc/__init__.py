"""Methods for preprocessing data"""

from . import calibration, detection, stitching
from .default_plugins import _register_default_plugins

_register_default_plugins()
